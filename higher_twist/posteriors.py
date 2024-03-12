from validphys.api import API
import numpy as np
import pandas as pd

import sys
from pathlib import Path

# Yaml loaders and dumpers
from ruamel.yaml.main import \
    round_trip_load as yaml_load, \
    round_trip_dump as yaml_dump

# Globals
# Number of spaces for an indent 
INDENTATION = 2 
# Used to reset comment objects
tsRESET_COMMENT_LIST = [None, [], None, None]

def ComputePosterior(fitname):
    thcovmat_dict = API.fit(fit=fitname).as_input()["theorycovmatconfig"]

    if "ht_version" in thcovmat_dict:
        version = thcovmat_dict["ht_version"]
    else:
        version = 1

    if version == 1:
        ht_coeff = thcovmat_dict["ht_coeff"]
    elif version == 2 or version == 3:
        ht_coeff_1 = thcovmat_dict["ht_coeff_1"]
        ht_coeff_2 = thcovmat_dict["ht_coeff_2"]

    # dict used to produce theory predictions to construct the theory covmat as well as to produce
    # theory predictions from the fit performed using the ht covmat (i.e. the predicitons that should
    # be compared to data)
    common_dict = dict(
        dataset_inputs={"from_": "fit"},
        fit=fitname,
        fits=[fitname],
        use_cuts="fromfit",
        metadata_group="nnpdf31_process",
        theory={"from_": "fit"},
        theoryid={"from_": "theory"},
    )

    # collect the information (predictions + kinematics) needed for the computation of the HT covmat

    # Calculate theory predictions of the input PDF
    S_dict = dict(
        theorycovmatconfig={"from_": "fit"},
        pdf={"from_": "theorycovmatconfig"},
        use_t0=True,
        datacuts={"from_": "fit"},
        t0pdfset={"from_": "datacuts"},
    )
    preds_ht_cov_construction = API.group_result_table_no_table(**(S_dict | common_dict))

    # collect the corresponding kinemacs
    process_info = API.combine_by_type_ht(**(S_dict | common_dict))
    kinematics_DIS = np.concatenate([v for v in [process_info.data["DIS NC"], process_info.data["DIS CC"]]]).T
    # TO CHECK: IS preds[][1] THE THEORY PREDICTION?
    #preds_DIS = np.concatenate([v for v in [process_info.preds["DIS NC"][1], process_info.preds["DIS CC"][1]]]).T
    xvals_DIS = kinematics_DIS[0]
    q2vals_DIS = kinematics_DIS[1]

    # Calculate theory predictions of the fit with ht covmat - this will be compared to data
    preds = API.group_result_table_no_table(pdf={"from_": "fit"}, **common_dict)

    # compute the matrix X encoding the PDF uncertainties of the predictions
    preds_onlyreplicas = preds.iloc[:, 2:].to_numpy()
    mean_prediction = np.mean(preds_onlyreplicas, axis=1)
    
    X = np.zeros((preds.shape[0], preds.shape[0]))
    for i in range(preds_onlyreplicas.shape[1]):
        X += np.outer(
            (preds_onlyreplicas[:, i] - mean_prediction), (preds_onlyreplicas[:, i] - mean_prediction)
        )
    X *= 1 / preds_onlyreplicas.shape[1]

    preds_ht = pd.DataFrame(preds_ht_cov_construction['theory_central'])
    # compute the delta of the theory prediction
    if version == 1:
        preds_ht["higher twist"] = 0
        preds_ht.loc[['DIS NC', 'DIS CC'],'higher twist'] = ht_coeff * (
            preds_ht.loc[['DIS NC', 'DIS CC'], 'theory_central'] / q2vals_DIS/ (1 - xvals_DIS)
        )
    elif version == 2:
        preds_ht["higher twist"] = 0
        preds_ht.loc[['DIS NC', 'DIS CC'],'higher twist'] = preds_ht.loc[['DIS NC', 'DIS CC'], 'theory_central'] * ( 
             ht_coeff_1 + ht_coeff_2 * xvals_DIS / (1 - xvals_DIS)
        ) / q2vals_DIS
    elif version == 3:
        preds_ht["higher twist (±,0)"] = 0
        preds_ht["higher twist (0,±)"] = 0
        # Compute beta for shift (±,0)
        preds_ht.loc[['DIS NC', 'DIS CC'],'higher twist (±,0)'] = preds_ht.loc[['DIS NC', 'DIS CC'], 'theory_central'] * (
            ht_coeff_1 / q2vals_DIS
        )
        preds_ht.loc[['DIS NC', 'DIS CC'],'higher twist (0,±)'] = preds_ht.loc[['DIS NC', 'DIS CC'], 'theory_central'] * (
            ht_coeff_2 * xvals_DIS / (1 - xvals_DIS) / q2vals_DIS
        )
    
    delta_pred = []

    if version == 1 or version == 2:
        delta_pred.append(preds_ht['higher twist'])
    elif version == 3:
        delta_pred.append(preds_ht['higher twist (±,0)'])
        delta_pred.append(preds_ht['higher twist (0,±)'])


    # Theory covariance matrix
    S = np.ndarray((delta_pred[0].size, delta_pred[0].size))
    for delta in delta_pred:
        S += np.outer(delta, delta)

    S = pd.DataFrame(S, index=delta_pred.index, columns=delta_pred.index)
    
    # Experimental covariance matrix
    C = API.groups_covmat_no_table(**common_dict)
    
    # Ensure that S anc C are ordered in the same way (in practice they already are)
    S = S.reindex(C.index).T.reindex(C.index)

    # Load the central value of the pseudodata
    # this is needed to compute the distance between prediction and data
    pseudodata = API.read_pdf_pseudodata(**common_dict)
    dat_central = np.mean(
        [i.pseudodata.reindex(preds.index.to_list()).to_numpy().flatten() for i in pseudodata],
        axis=0,
    )

    # Compute delta_T_tilde (Eq. 3.37) and P_tilde (Eq. 3.38) of arXiv:2105.05114

    # The factors 1/sqrt(2) are to normalize for the fact that beta provides information about
    # theoretical uncertainties along two directions
    # CHECK THIS PART
    # b_tilde SHOULD BE INDEPENDENT OF THE PRIOR THAT WE USE TO MODEL HT CORRECTIONS.
    if version == 1:
        central_ht_coeffs = [0] # central prediction for ht_coeff
        beta_tilde = [ht_coeff]
    
    elif version == 2:
        central_ht_coeffs = [0, 0] # central prediction for ht_coeff_1 and ht_coeff_2
        beta_tilde = [ht_coeff_1, ht_coeff_2]
    elif version == 3:
        central_ht_coeffs = [0, 0] # central prediction for ht_coeff_1 and ht_coeff_2
        beta_tilde = [[ht_coeff_1, 0], [0, ht_coeff_2]]

    S_tilde = np.zeros((len(beta_tilde[0]), len(beta_tilde[0])))
    for tilde in beta_tilde:
        S_tilde += np.outer(tilde,tilde)
    
    beta = delta_pred
    S_hat = np.zeros((len(beta_tilde[0]),delta_pred[0].size))
    for b in zip(beta_tilde, beta):
        S_hat += np.outer(b[0], b[1])
    
    invcov = np.linalg.inv(C + S)
    
    delta_T_tilde = -S_hat @ invcov @ (mean_prediction - dat_central)
    # where are the X_tilde and X_hat terms in P_tilde?
    # Maybe not present because we don't have correlations between theory parameters
    P_tilde = S_hat @ invcov @ X @ invcov @ S_hat.T + (S_tilde - S_hat @ invcov @ S_hat.T)
    preds = central_ht_coeffs + delta_T_tilde

    # check if the stored covmat is equal to S we recomputed above
    fitpath = API.fit(fit=fitname).path
    try:
        stored_covmat = pd.read_csv(
            fitpath / "tables/datacuts_theory_theorycovmatconfig_user_covmat.csv",
            sep="\t",
            encoding="utf-8",
            index_col=2,
            header=3,
            skip_blank_lines=False,
        )
    except FileNotFoundError:
        stored_covmat = pd.read_csv(
            fitpath / "tables/datacuts_theory_theorycovmatconfig_theory_covmat_custom.csv",
            index_col=[0, 1, 2],
            header=[0, 1, 2],
            sep="\t|,",
            engine="python",
        ).fillna(0)
        storedcovmat_index = pd.MultiIndex.from_tuples(
            [(aa, bb, np.int64(cc)) for aa, bb, cc in stored_covmat.index],
            names=["group", "dataset", "id"],
        )
        stored_covmat = pd.DataFrame(
            stored_covmat.values, index=storedcovmat_index, columns=storedcovmat_index
        )
        stored_covmat = stored_covmat.reindex(S.index).T.reindex(S.index)

    # print the final result
    if np.allclose(S, stored_covmat):
        if version == 1:
            print(
                f"Prediction for ht_coeff: {preds[0]:.5f} ± {np.sqrt(P_tilde[0,0]):.5f} \n"
                f"ht_coeff : {ht_coeff}"
                )
        elif version == 2 or version == 3:
            print(
                f"Prediction for \n ht_coeff_1: {preds[0]:.5f} ± {np.sqrt(P_tilde[0,0]):.5f} \n ht_coeff_2: {preds[1]:.5f} ± {np.sqrt(P_tilde[1,1]):.5f}"
                )
    else:
        print("Reconstructed theory covmat, S, is not the same as the stored covmat!")

    return preds, P_tilde

if __name__ == "__main__":
    
    if (args_count := len(sys.argv)) > 3:
        print(f"Two arguments expected, got {args_count - 1}")
        raise SystemExit(2)
    elif args_count < 3:
        print("You must specify the the fit name and the target directory")
        raise SystemExit(2)
    
    # Check if the target folder already exists
    dir = sys.argv[2] + "/" + sys.argv[1]
    target_dir = Path(dir)
    if not target_dir.is_dir():
        target_dir.mkdir(parents=True, exist_ok=True)

    preds, P_tilde = ComputePosterior(sys.argv[1])

    results_list = []
    for i, pred in enumerate(preds):
        print(i)
        result = {f'h_{i}' : f'{preds[0]:.5f}', 'unc' : f'{np.sqrt(P_tilde[i][i]):.5f}'}
        results_list.append(result)

    results = {
        f'Posterior(s) for {sys.argv[1]}': results_list
    }

    with open(dir + '/result.yml', 'w') as outfile:
        yaml_dump(results, outfile)

