from validphys.api import API 
import numpy as np
import pandas as pd
from scipy import interpolate as scint
import matplotlib.pyplot as plt
from collections import defaultdict, namedtuple

import sys
from pathlib import Path

# Yaml loaders and dumpers
from ruamel.yaml.main import round_trip_dump as yaml_dump


# ABMP parametrisation
x_abmp = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]

pd.options.mode.chained_assignment = None


def compute_normalisation_by_experiment(experiment_name, x, y, Q2):
    N_2 = np.zeros(shape=y.shape)
    N_L = np.zeros(shape=y.shape)

    if "HERA_NC" in experiment_name or "HERA_CC" in experiment_name or "NMC" in experiment_name:
        yp = 1 + np.power(1 - y, 2)
        yL = np.power(y, 2)

        if "HERA_NC" in experiment_name or "NMC" in experiment_name:
            N_2 = 1
            N_L = - yL / yp

        elif "HERA_CC" in experiment_name:
            N_2 = 1 / 4 * yp
            N_L = - N_2 * yL / yp

    if "CHORUS_CC" in experiment_name:
        yL = np.power(y, 2)
        Gf = 1.1663787e-05
        Mh = 0.938
        MW2 = 80.398 ** 2
        yp = 1 + np.power(1 - y, 2) - 2 * np.power(x * y * Mh, 2) / Q2
        N_2 = Gf**2 * Mh * yp / ( 2 * np.pi * np.power( 1 + Q2 / MW2, 2) )
        N_L = - N_2 * yL / yp

    return N_2, N_L

def ComputePosterior(fitname):
    thcovmat_dict = API.fit(fit=fitname).as_input()["theorycovmatconfig"]

    H2_coeff_list = thcovmat_dict["H2_list"]
    HL_coeff_list = thcovmat_dict["HL_list"]

    def wrapper_to_splines(i):
        if not thcovmat_dict["reverse"]:
            shifted_H2_list = [0 for k in range(len(x_abmp))]
            shifted_HL_list = [0 for k in range(len(x_abmp))]
            shifted_H2_list[i] = H2_coeff_list[i]
            shifted_HL_list[i] = HL_coeff_list[i]
        else:
            shifted_H2_list = H2_coeff_list.copy()
            shifted_HL_list = HL_coeff_list.copy()
            shifted_H2_list[i] = 0
            shifted_HL_list[i] = 0

        H_2 = scint.CubicSpline(x_abmp, shifted_H2_list)
        H_L = scint.CubicSpline(x_abmp, shifted_HL_list)
        H_2 = np.vectorize(H_2)
        H_L = np.vectorize(H_L)
        return H_2, H_L

    common_dict = dict(
        dataset_inputs={"from_": "fit"},
        fit=fitname,
        fits=[fitname],
        use_cuts="fromfit",
        metadata_group="nnpdf31_process",
        theory={"from_": "fit"},
        theoryid={"from_": "theory"},
    )

    # Calculate theory predictions of the input PDF
    S_dict = dict(
        theorycovmatconfig={"from_": "fit"},
        pdf={"from_": "theorycovmatconfig"},
        use_t0=True,
        datacuts={"from_": "fit"},
        t0pdfset={"from_": "datacuts"},
    )
    preds_ht_cov_construction = API.group_result_central_table_no_table(**(S_dict | common_dict))
    preds_ht = pd.DataFrame(preds_ht_cov_construction['theory_central'])

    # collect the corresponding kinemacs
    process_info = API.combine_by_type_ht(**(S_dict | common_dict))
    N_full_data = np.sum([i for i in process_info.sizes.values()])
    kinematics_DIS = np.concatenate([v for v in [process_info.data["DIS NC"], process_info.data["DIS CC"]]]).T
    preds_DIS = np.concatenate([v for v in [process_info.preds["DIS NC"][1], process_info.preds["DIS CC"][1]]]).T
    xvals_DIS = kinematics_DIS[0]
    q2vals_DIS = kinematics_DIS[1]
    yvals_DIS = kinematics_DIS[2]

    # Calculate theory predictions of the fit with ht covmat - this will be compared to data
    preds = API.group_result_table_no_table(pdf={"from_": "fit"}, **common_dict)

    # compute the matrix X encoding the PDF uncertainties of the predictions
    preds_onlyreplicas = preds.iloc[:, 2:].to_numpy()
    mean_prediction = np.mean(preds_onlyreplicas, axis=1)
    
    preds_onlyreplicas = preds.iloc[:, 2:].to_numpy()
    mean_prediction = np.mean(preds_onlyreplicas, axis=1)

    X = np.zeros((preds.shape[0], preds.shape[0]))
    for i in range(preds_onlyreplicas.shape[1]):
        X += np.outer(
            (preds_onlyreplicas[:, i] - mean_prediction), (preds_onlyreplicas[:, i] - mean_prediction)
        )
    X *= 1 / preds_onlyreplicas.shape[1]

    included_proc = ["DIS NC"]
    excluded_exp = {"DIS NC" : ["NMC_NC_NOTFIXED_DW_EM-F2"]}
    included_exp = {}
    for proc in included_proc:
        aux = []
        for exp in process_info.namelist[proc]:
            if exp not in excluded_exp[proc]:
                aux.append(exp)
        included_exp[proc] = aux

    preds_ht.loc[['DIS NC', 'DIS CC'], 'x'] = xvals_DIS
    preds_ht.loc[['DIS NC', 'DIS CC'], 'q2'] = q2vals_DIS
    preds_ht.loc[['DIS NC', 'DIS CC'], 'y'] = yvals_DIS

    # Initialise dataframe
    for i in range(len(x_abmp)):
        preds_ht[f"({i+1}+,0)"] = 0
        preds_ht[f"(0,{i+1}+)"] = 0

    deltas = defaultdict(list)

    for proc in process_info.namelist.keys():
            for exp in process_info.namelist[proc]:
                if proc in included_proc and exp in included_exp[proc]:
                    x  = np.array(preds_ht.xs(exp, level=1, drop_level=False).loc[:,"x"])
                    Q2 = np.array(preds_ht.xs(exp, level=1, drop_level=False).loc[:,"q2"])
                    y  = np.array(preds_ht.xs(exp, level=1, drop_level=False).loc[:,"y"])
                    N = np.array([])

                    if "SIGMA" in exp:
                        N_2, N_L = compute_normalisation_by_experiment(exp, x, y, Q2)
                    elif "F2" in exp:
                        N_2 = np.ones(shape=x.shape)
                        N_L = np.zeros(shape=x.shape)
                    else:
                        raise ValueError(f"The normalisation for the observable is not known.")

                    for i in range(len(x_abmp)):
                        H_L, H_2 = wrapper_to_splines(i)
                        deltas[f"({i+1}+,0)"] += [N_2 * H_2(x) / Q2]
                        deltas[f"(0,{i+1}+)"] += [N_L * H_L(x) / Q2]
                else:
                    for i in range(len(x_abmp)):
                        deltas[f"({i+1}+,0)"] += [np.zeros(preds_ht.xs(exp, level=1, drop_level=False).shape[0])]
                        deltas[f"(0,{i+1}+)"] += [np.zeros(preds_ht.xs(exp, level=1, drop_level=False).shape[0])]

    delta_pred = []
    for i in range(len(x_abmp)):
        temp_1 = np.array([])
        temp_2 = np.array([])
        for vec in zip(deltas[f"({i+1}+,0)"], deltas[f"(0,{i+1}+)"]):
            temp_1 = np.concatenate((temp_1, vec[0]))
            temp_2 = np.concatenate((temp_2, vec[1]))
        
        preds_ht[f"({i+1}+,0)"] = temp_1
        preds_ht[f"(0,{i+1}+)"] = temp_2
        delta_pred.append(preds_ht[f"({i+1}+,0)"])
        delta_pred.append(preds_ht[f"(0,{i+1}+)"])
    

    # Theory covariance matrix
    S = np.zeros((delta_pred[0].size, delta_pred[0].size))
    for delta in delta_pred:
        S += np.outer(delta, delta)

    S = pd.DataFrame(S, index=delta_pred[0].index, columns=delta_pred[0].index)
    
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
    central_ht_coeffs = np.zeros(len(H2_coeff_list) + len(HL_coeff_list))
    # Construct beta tilde
    H_single_list = np.concatenate((H2_coeff_list, HL_coeff_list))
    beta_tilde = []
    for i, par in enumerate(H_single_list):
        aux = np.zeros(H_single_list.size)
        aux[i] = par
        beta_tilde.append(aux) 

    S_tilde = np.zeros((len(beta_tilde[0]), len(beta_tilde[0])))
    for tilde in beta_tilde:
        S_tilde += np.outer(tilde,tilde)
    
    beta = delta_pred
    S_hat = np.zeros((len(beta_tilde[0]),len(delta_pred[0])))
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
    # print the final result
    if np.allclose(S, stored_covmat):
        print(
            f"Reversed 5pt\n"
            f"-----------------------------\n"
            )
        for i, pred in enumerate(preds):
            tpye = "2" if i < len(H2_coeff_list) else "L"
            n = i%7
            print(
                f"H_{tpye} node {n+1} = {preds[i]:.5f} ± {np.sqrt(P_tilde[i,i]):.5f} \n"
            )
            if i == len(H2_coeff_list)-1:
                print("-----------------------------\n")
    else:
        print("Reconstructed theory covmat, S, is not the same as the stored covmat!")

    return preds, P_tilde


def make_plots(fitname, savedir, preds, P_tilde):
    length = int((len(preds))/2)
    y2_central = [preds[i] for i in range(length)]
    y2_sigma = [np.sqrt(P_tilde[i,i]) for i in range(length)]
    y2_plus = [x1 + x2 for x1,x2 in zip(y2_central, y2_sigma)]
    y2_minus = [x1 - x2 for x1,x2 in zip(y2_central, y2_sigma)]

    yL_central = [preds[i] for i in range(length, length*2)]
    yL_sigma = [np.sqrt(P_tilde[i,i]) for i in range(length, length*2)]
    yL_plus = [x1 + x2 for x1,x2 in zip(yL_central, yL_sigma)]
    yL_minus = [x1 - x2 for x1,x2 in zip(yL_central, yL_sigma)]

    H2 = scint.CubicSpline(x_abmp, y2_central)
    H2_plus = scint.CubicSpline(x_abmp, y2_plus)
    H2_minus = scint.CubicSpline(x_abmp, y2_minus)
    H2_color = "sandybrown"

    HL = scint.CubicSpline(x_abmp, yL_central)
    HL_plus = scint.CubicSpline(x_abmp, yL_plus)
    HL_minus = scint.CubicSpline(x_abmp, yL_minus)
    HL_color = "green"

    fig, axs = plt.subplots(1,2, figsize=(15, 7))
    H2_dict = {
    "func" : H2,
    "func_plus_std": H2_plus,
    "func_minus_std": H2_minus,
    "knots": y2_central,
    "label": r"$H_2 \pm \sigma$",
    "ylabel": r"$H_2$",
    "color": H2_color
    }

    HL_dict = {
    "func" : HL,
    "func_plus_std": HL_plus,
    "func_minus_std": HL_minus,
    "knots": yL_central,
    "label": r"$H_L \pm \sigma$",
    "ylabel": r"$H_L$",
    "color": HL_color
    }
    dicts = [H2_dict, HL_dict]

    def plot_wrapper(H, H_p, H_m, y, label, color, ylabel, ax):
        xv = np.logspace(-5, -0.0001, 100)
        legends = []
        legend_name = [label, "knots"]
        knots = ax.plot(x_abmp, y, 'o', label='data')
        pl = ax.plot(xv, H(xv), ls = "-", lw = 1, color=color)
        pl_lg= ax.fill(np.NaN, np.NaN, color = color, alpha = 0.3) # Necessary for fancy legend
        legends.append((pl[0], pl_lg[0]))
        legends.append(knots[0])
        ax.fill_between(xv, H_p(xv), H_m(xv), color = color, alpha = 0.3)
        ax.set_xscale("log")
        ax.set_xlabel(f'$x$')
        ax.set_ylabel(ylabel)
        ax.legend(legends, legend_name, loc=[0.1,0.15], fontsize=15)

    for i, HTdict in enumerate(dicts):
        plot_wrapper(H=HTdict["func"],
                    H_p=HTdict["func_plus_std"],
                    H_m=HTdict["func_minus_std"],
                    y=HTdict["knots"],
                    label=HTdict["label"],
                    color=HTdict["color"],
                    ylabel=HTdict["ylabel"],
                    ax=axs[i])

    axs[0].text(5.e-5, 0.06, fitname, fontsize=20)
    fig.savefig(savedir + "/" + fitname)

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
    make_plots(sys.argv[1], dir, preds, P_tilde)

    results_list = []
    for i, pred in enumerate(preds):
        result = {f'h_{i+1}' : f'{preds[i]:.5f}', 'unc' : f'{np.sqrt(P_tilde[i][i]):.5f}'}
        results_list.append(result)


    results = {
        f'Posterior(s) for {sys.argv[1]}': results_list
    }

    with open(dir + '/result.yml', 'w') as outfile:
        yaml_dump(results, outfile)

