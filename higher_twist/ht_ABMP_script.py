from validphys.api import API
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import functools
from scipy import interpolate as scint
from collections import defaultdict, namedtuple
import operator
from pathlib import Path
import sys
import ipdb

from validphys.theorycovariance.construction import extract_target, compute_ratio_delta, compute_ht_parametrisation

# Yaml loaders and dumpers
from ruamel.yaml.main import round_trip_dump as yaml_dump


# ABMP parametrisation
x_abmp = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]


POSTERIOR_FOLDER = 'posteriors'

pd.options.mode.chained_assignment = None

def store_dict(name='result.yml'):
   def store_dict_decorator(func):
      @functools.wraps(func)
      def wrapper(*args, **kwargs):

        if not args[0].save:
          return

        dict_result = func(*args, **kwargs)
        for key in dict_result.keys():
           for key2 in dict_result[key].keys():
              for pred in dict_result[key][key2]:
                 pred = pred.string_rep()

        if args[0].save_dir is None: 
            dir = POSTERIOR_FOLDER + '/' +  args[0].fitname
        else:
            dir = args[0].save_dir + '/' +  args[0].fitname

        target_dir = Path(dir)
        if not target_dir.is_dir():
          target_dir.mkdir(parents=True, exist_ok=True)

        with open(dir + '/' + name, 'w') as saved_dict:
           yaml_dump(dict_result, saved_dict)
      return wrapper
   return store_dict_decorator


class Prediction:
  def __init__(self, central, sigma):
    self.central = central
    self.sigma = sigma
    self.central_plus_sigma = central + sigma
    self.central_minus_sigma = central - sigma

  def string_rep(self):
      return f"{self.central:.5f} ± {self.sigma:.5f} \n"

  def __str__(self) -> str:
    return self.string_rep


class Posterior:
  def __init__(self, fitname, save = True, save_dir = None):
    
    self.thcovmat_dict = API.fit(fit=fitname).as_input()["theorycovmatconfig"]
    self.H2_coeff_list = self.thcovmat_dict["H2_list"]
    self.HL_coeff_list = self.thcovmat_dict["HL_list"]
    self.lenH2 = len(self.H2_coeff_list)
    self.lenHL = len(self.HL_coeff_list)
    self.fitname = fitname
    self.x_knots = self.thcovmat_dict.get('ht_knots', x_abmp)
    self.save_dir = save_dir
    self.save = save

    self.common_dict = dict(
        dataset_inputs={"from_": "fit"},
        fit=fitname,
        fits=[fitname],
        use_cuts="fromfit",
        metadata_group="nnpdf31_process",
        theory={"from_": "fit"},
        theoryid={"from_": "theory"},
    )

    # Calculate theory predictions of the input PDF
    self.S_dict = dict(
        theorycovmatconfig={"from_": "fit"},
        pdf={"from_": "theorycovmatconfig"},
        use_t0=True,
        datacuts={"from_": "fit"},
        t0pdfset={"from_": "datacuts"},
    )
    self.process_info = API.combine_by_type_ht(**(self.S_dict | self.common_dict))
    self.predictions = API.group_result_table_no_table(pdf={"from_": "fit"}, **self.common_dict)

  def compute_posteriors_and_unc(self):
    delta_predictions = self.compute_delta_preds()
    S = self.compute_theory_covmat()
    X = self.compute_PDF_unc(self.predictions)
    S_hat = self.compute_S_hat(delta_predictions)
    S_tilde = self.compute_S_tilde()
    C = self.collect_exp_covmat()
    invcov = np.linalg.inv(C + S)

    mean_prediction = self.compute_mean_prediction(self.predictions)
    dat_central = self.compute_dat_central(self.predictions)

    central_ht_coeffs = np.zeros(2*len(self.H2_coeff_list) + 2*len(self.HL_coeff_list)) 

    delta_T_tilde = -S_hat @ invcov @ (mean_prediction - dat_central)
    P_tilde = S_hat @ invcov @ X @ invcov @ S_hat.T + (S_tilde - S_hat @ invcov @ S_hat.T)
    posteriors = central_ht_coeffs + delta_T_tilde
    posteriors_sigma = np.sqrt(P_tilde)
    return posteriors, posteriors_sigma

  def compute_dat_central(self, preds):
    pseudodata = API.read_pdf_pseudodata(**self.common_dict)
    dat_central = np.mean(
        [i.pseudodata.reindex(preds.index.to_list()).to_numpy().flatten() for i in pseudodata],
        axis=0,
    )
    return dat_central

  def collect_theory_preds_input_pdf(self):
    preds_input_cov_construction = API.group_result_central_table_no_table(**(self.S_dict | self.common_dict))
    return pd.DataFrame(preds_input_cov_construction['theory_central'])
  
  def compute_preds_onlyreplica(self, predictions):
     return predictions.iloc[:, 2:].to_numpy()
  
  def compute_mean_prediction(self, predictions):
    preds_onlyreplicas = self.compute_preds_onlyreplica(predictions)
    mean_prediction = np.mean(preds_onlyreplicas, axis=1)
    return mean_prediction 
  
  def compute_PDF_unc(self, predictions):
    preds_onlyreplicas = self.compute_preds_onlyreplica(predictions)
    mean_prediction = self.compute_mean_prediction(predictions)
    X = np.zeros((predictions.shape[0], predictions.shape[0]))
    for i in range(preds_onlyreplicas.shape[1]):
        X += np.outer(
            (preds_onlyreplicas[:, i] - mean_prediction), (preds_onlyreplicas[:, i] - mean_prediction)
        )
    X *= 1 / preds_onlyreplicas.shape[1]
    return X
  
  def compute_beta_tilde(self):
    H_single_list = np.concatenate((self.H2_coeff_list, self.HL_coeff_list, self.H2_coeff_list, self.HL_coeff_list))
    beta_tilde = []
    for i, par in enumerate(H_single_list):
      aux = np.zeros(H_single_list.size)
      aux[i] = par
      beta_tilde.append(aux)
    return beta_tilde
  
  def compute_S_tilde(self):
    beta_tilde = self.compute_beta_tilde()
    S_tilde = np.zeros((len(beta_tilde[0]), len(beta_tilde[0])))
    for tilde in beta_tilde:
      S_tilde += np.outer(tilde,tilde)
    return S_tilde

  def compute_S_hat(self, delta_pred):
    beta_tilde = self.compute_beta_tilde()
    beta = delta_pred
    S_hat = np.zeros((len(beta_tilde[0]),delta_pred[0].size))
    for b in zip(beta_tilde, beta):
      S_hat += np.outer(b[0], b[1])   
    return S_hat
  
  def collect_kinematics(self):
    kinematics_DIS = np.concatenate([v for v in [self.process_info.data["DIS NC"], self.process_info.data["DIS CC"]]]).T
    xvals_DIS = kinematics_DIS[0]
    q2vals_DIS = kinematics_DIS[1]
    yvals_DIS = kinematics_DIS[2]
    return xvals_DIS, q2vals_DIS, yvals_DIS
  
  def collect_exp_covmat(self):
    # Experimental covariance matrix
    C = API.groups_covmat_no_table(**self.common_dict)
    return C
  
  def compute_theory_covmat(self):
    delta_predictions = self.compute_delta_preds()
    S = np.zeros((delta_predictions[0].size, delta_predictions[0].size))
    for delta in delta_predictions:
        S += np.outer(delta, delta)
    S = pd.DataFrame(S, index=delta_predictions[0].index, columns=delta_predictions[0].index)

    # Ensure that S anc C are ordered in the same way (in practice they already are)
    C = self.collect_exp_covmat()
    S = S.reindex(C.index).T.reindex(C.index)
    return S

  def compute_delta_preds(self):
    preds_input = self.collect_theory_preds_input_pdf()
    data_by_process = API.groups_data_by_process(**(self.S_dict | self.common_dict))
    PDF_thcovmat = API.pdf(**(self.S_dict | self.common_dict))

    included_proc = ["DIS NC"]
    excluded_exp = {"DIS NC" : []}
    included_exp = {}
    for proc in included_proc:
        aux = []
        for exp in self.process_info.namelist[proc]:
            if exp not in excluded_exp[proc]:
                aux.append(exp)
        included_exp[proc] = aux

    xvals_DIS, q2vals_DIS, yvals_DIS = self.collect_kinematics()
    preds_input.loc[['DIS NC', 'DIS CC'], 'x'] =  xvals_DIS
    preds_input.loc[['DIS NC', 'DIS CC'], 'q2'] = q2vals_DIS
    preds_input.loc[['DIS NC', 'DIS CC'], 'y'] =  yvals_DIS

    # Initialise dataframe
    for i in range(len(self.x_knots)):
        preds_input[f"p({i+1}+,0)"] = 0
        preds_input[f"p(0,{i+1}+)"] = 0
        preds_input[f"d({i+1}+,0)"] = 0
        preds_input[f"d(0,{i+1}+)"] = 0

    deltas = defaultdict(list)

    for i_proc, proc in enumerate(self.process_info.namelist.keys()):
            for i_exp, exp in enumerate(self.process_info.namelist[proc]):
                dataset = data_by_process[i_proc].datasets[i_exp]
                kin_dict = {}

                if proc in included_proc and exp in included_exp[proc]:
                    kin_dict['x']  = np.array(preds_input.xs(exp, level=1, drop_level=False).loc[:,"x"])
                    kin_dict['Q2'] = np.array(preds_input.xs(exp, level=1, drop_level=False).loc[:,"q2"])
                    kin_dict['y']  = np.array(preds_input.xs(exp, level=1, drop_level=False).loc[:,"y"])
                    kin_size =  kin_dict['x'].size
                    target = extract_target(dataset)


                    # Loop over the parameter
                    for i in range(len(self.x_knots)):
                        PC_2, PC_L = compute_ht_parametrisation(i, self.x_knots, kin_dict, exp, self.H2_coeff_list, self.HL_coeff_list)
                        if target == 'proton':
                          deltas[f"p({i+1}+,0)"] += [PC_2]
                          deltas[f"p(0,{i+1}+)"] += [PC_L]
                          deltas[f"d({i+1}+,0)"] += [np.zeros(kin_size)]
                          deltas[f"d(0,{i+1}+)"] += [np.zeros(kin_size)]
                        elif target == 'deuteron':
                          deltas[f"p({i+1}+,0)"] += [np.zeros(kin_size)]
                          deltas[f"p(0,{i+1}+)"] += [np.zeros(kin_size)]
                          deltas[f"d({i+1}+,0)"] += [PC_2]
                          deltas[f"d(0,{i+1}+)"] += [PC_L]
                        elif target == 'ratio':
                          deltas[f"p({i+1}+,0)"] += [compute_ratio_delta(dataset, PDF_thcovmat, "p", PC_2) - compute_ratio_delta(dataset, PDF_thcovmat)]
                          deltas[f"p(0,{i+1}+)"] += [compute_ratio_delta(dataset, PDF_thcovmat, "p", PC_L) - compute_ratio_delta(dataset, PDF_thcovmat)]
                          deltas[f"d({i+1}+,0)"] += [compute_ratio_delta(dataset, PDF_thcovmat, "d", PC_2) - compute_ratio_delta(dataset, PDF_thcovmat)]
                          deltas[f"d(0,{i+1}+)"] += [compute_ratio_delta(dataset, PDF_thcovmat, "d", PC_L) - compute_ratio_delta(dataset, PDF_thcovmat)]
                        else:
                            raise ValueError("Could not detect target.")
                else:
                    for i in range(len(self.x_knots)):
                        deltas[f"p({i+1}+,0)"] += [np.zeros(preds_input.xs(exp, level=1, drop_level=False).shape[0])]
                        deltas[f"p(0,{i+1}+)"] += [np.zeros(preds_input.xs(exp, level=1, drop_level=False).shape[0])]
                        deltas[f"d({i+1}+,0)"] += [np.zeros(preds_input.xs(exp, level=1, drop_level=False).shape[0])]
                        deltas[f"d(0,{i+1}+)"] += [np.zeros(preds_input.xs(exp, level=1, drop_level=False).shape[0])]

    delta_pred = []
    for i in range(len(self.x_knots)):
        temp_1 = np.array([])
        temp_2 = np.array([])
        temp_3 = np.array([])
        temp_4 = np.array([])
        for vec in zip(deltas[f"p({i+1}+,0)"], deltas[f"p(0,{i+1}+)"], deltas[f"d({i+1}+,0)"], deltas[f"d(0,{i+1}+)"]):
            temp_1 = np.concatenate((temp_1, vec[0]))
            temp_2 = np.concatenate((temp_2, vec[1]))
            temp_3 = np.concatenate((temp_3, vec[2]))
            temp_4 = np.concatenate((temp_4, vec[3]))
        
        preds_input[f"p({i+1}+,0)"] = temp_1
        preds_input[f"p(0,{i+1}+)"] = temp_2
        preds_input[f"d({i+1}+,0)"] = temp_3
        preds_input[f"d(0,{i+1}+)"] = temp_4
        delta_pred.append(preds_input[f"p({i+1}+,0)"])
        delta_pred.append(preds_input[f"p(0,{i+1}+)"])
        delta_pred.append(preds_input[f"d({i+1}+,0)"])
        delta_pred.append(preds_input[f"d(0,{i+1}+)"])
    return delta_pred
  
  #@store_dict()
  def create_posterior_dict(self):
    posteriors, posteriors_sigma = self.compute_posteriors_and_unc()
    predictions_list = [Prediction(central, sigma) for central, sigma in zip(posteriors, posteriors_sigma.diagonal())]
    preds_dict = defaultdict(list)
    preds_dict['proton'] = {"H2": predictions_list[0:self.lenH2], 
                            "HL": predictions_list[self.lenH2:self.lenHL + self.lenH2]}
    preds_dict['deuteron'] = {"H2": predictions_list[self.lenHL + self.lenH2 : self.lenHL + 2*self.lenH2], 
                          "HL": predictions_list[self.lenHL + 2*self.lenH2 : 2*self.lenHL + 2*self.lenH2]}
    return preds_dict
  
  def check_against_stored_covmat(self, S):
     # check if the stored covmat is equal to S we recomputed above
    fitpath = API.fit(fit=self.fitname).path
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

      return np.allclose(stored_covmat, S)



class PlotHT:
  def __init__(self, preds, x_nodes, color, type, target, fitname):
    self.preds = preds
    self.color = color
    self.type = type
    self.target = target
    self.fitname = fitname
    self.x_knots = x_nodes
    self.HT = sp.interpolate.CubicSpline(x_nodes, [pred.central for pred in self.preds])
    self.HT_plus = sp.interpolate.CubicSpline(x_nodes, [pred.central_plus_sigma for pred in self.preds])
    self.HT_minus = sp.interpolate.CubicSpline(x_nodes, [pred.central_minus_sigma for pred in self.preds])

  def plot_wrapper(self, save=False):
    xv = np.logspace(-5, -0.0001, 100)
    legends = []
    legend_label = rf"$H^{self.target}_{self.type} \pm \sigma$"
    legend_name = [legend_label, "knots"]
    fig, ax = plt.subplots(figsize=(12.5, 8))
    knots = ax.plot(self.x_knots, [pred.central for pred in self.preds], 'o', label='data')
    pl = ax.plot(xv, self.HT(xv), ls = "-", lw = 1, color = self.color)
    pl_lg= ax.fill(np.NaN, np.NaN, color = self.color, alpha = 0.3) # Necessary for fancy legend
    legends.append((pl[0], pl_lg[0]))
    legends.append(knots[0])
    ax.fill_between(xv, self.HT_plus(xv), self.HT_minus(xv), color = self.color, alpha = 0.3)
    ax.set_xscale("log")
    ax.set_xlabel(f'$x$')
    ax.set_ylabel(rf"$H^{self.target}_{self.type}$", fontsize = 20)
    ax.set_title(rf"$H^{self.target}_{self.type}$", x = 0.15, y=0.85, fontsize=30)
    fig.legend(legends, legend_name, loc=[0.1,0.15], fontsize=15)

    if save:
      save_dir = f"./figs/{self.fitname}"
      self.make_dir(save_dir)
      plt.savefig(save_dir + "/" + f"H_{self.type}_{self.target}.pdf")

  def make_dir(self, path):
    target_dir = Path(path)
    if not target_dir.is_dir():
        target_dir.mkdir(parents=True, exist_ok=True)



if __name__ == "__main__":
    
    if (args_count := len(sys.argv)) > 2:
        print(f"Two arguments expected, got {args_count - 1}")
        raise SystemExit(2)
    elif args_count < 2:
        print("You must specify the target directory")
        raise SystemExit(2)
    
    fitnames = [
       "240805-03-ABMP",
    ]
    
    # Check if the target folder already exists
    for fitname in fitnames:
      dir = sys.argv[1] + "/" + fitname
      target_dir = Path(dir)
      if not target_dir.is_dir():
          target_dir.mkdir(parents=True, exist_ok=True)

      fit_posteriors = Posterior(fitname)
      preds_dict = fit_posteriors.create_posterior_dict()

      
      proton_H2 = PlotHT(preds_dict['proton']['H2'], fit_posteriors.x_knots, 'red', "2", 'p', fitname)
      proton_HL = PlotHT(preds_dict['proton']['HL'], fit_posteriors.x_knots, 'green', "L", 'p', fitname)
      deuteron_H2 = PlotHT(preds_dict['deuteron']['H2'], fit_posteriors.x_knots, 'blue', "2", 'd', fitname)
      deuteron_HL = PlotHT(preds_dict['deuteron']['HL'], fit_posteriors.x_knots, 'purple', "L", 'd', fitname)

      proton_H2.plot_wrapper(save=True)
      proton_HL.plot_wrapper(save=True)
      deuteron_H2.plot_wrapper(save=True)
      deuteron_HL.plot_wrapper(save=True)

