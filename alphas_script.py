#!/usr/bin/env python
# coding: utf-8

import pathlib

import numpy as np
import numpy.linalg as la
import pandas as pd
import seaborn as sns
from validphys.loader import Loader

import matplotlib.pyplot as plt
import scipy.stats as stats



sns.set()
sns.set_style("ticks")
from validphys.api import API

l = Loader()

# Configuration
fit = "221203-ern-001"
pdf_ori = "NNPDF40_nnlo_as_01180"
output_file_name = "alphas.csv"
nbins = 20

path_this_module = pathlib.Path(__file__).parent.resolve()
output_file = path_this_module / output_file_name

if not output_file.exists():
    # Inputs for central theory: alphas=0.118
    inps_central = dict(
        dataset_inputs={"from_": "fit"}, fit=fit, theoryid=200, use_cuts="fromfit", pdf=pdf_ori
    )

    # Inputs for plus theory: alphas=0.120
    inps_plus = dict(
        dataset_inputs={"from_": "fit"}, fit=fit, theoryid=207, use_cuts="fromfit", pdf=pdf_ori
    )

    # Inputs for minus theory: alphas=0.116
    inps_minus = dict(
        dataset_inputs={"from_": "fit"}, fit=fit, theoryid=201, use_cuts="fromfit", pdf=pdf_ori
    )

    # Inputs for central theory: alphas=0.118
    inps_central_fit = dict(
        dataset_inputs={"from_": "fit"}, fit=fit, theoryid=200, use_cuts="fromfit", pdf={"from_": "fit"}
    )

    # Experimental covariance matrix
    C = API.groups_covmat(**inps_central)

    # Indexes
    dsindex = API.groups_index(**inps_central)
    datth_central = API.group_result_table_no_table(**inps_central)
    datth_plus = API.group_result_table_no_table(**inps_plus)
    datth_minus = API.group_result_table_no_table(**inps_minus)
    datth_central_fit = API.group_result_table_no_table(**inps_central_fit)

    dat_central = datth_central["data_central"]
    th_central_fit = datth_central_fit["theory_central"]
    th_replicas_fit = datth_central_fit.iloc[:, 2:102].to_numpy()
    th_central_and_replcas_fit = np.concatenate(
        [th_central_fit.to_numpy().reshape(-1, 1), th_replicas_fit], axis=1
    )

    # Computation of Eqs.(3.37)-(3.38) in [arXiv:2105.05114](https://arxiv.org/pdf/2105.05114.pdf)
    beta_tilde = (0.002 / np.sqrt(2)) * np.array([1, -1])
    S_tilde = beta_tilde.T @ beta_tilde

    datth_plus_np = datth_plus.to_numpy()
    datth_central_np = datth_central.to_numpy()
    datth_minus_np = datth_minus.to_numpy()

    delta_plus = (1 / np.sqrt(2)) * (datth_plus_np[:, 1:] - datth_central_np[:, 1:])
    delta_minus = (1 / np.sqrt(2)) * (datth_minus_np[:, 1:] - datth_central_np[:, 1:])

    preds = []
    for dp, dm, th_pred in zip(delta_plus.T, delta_minus.T, th_central_and_replcas_fit.T):
        beta = [dp, dm]
        S_hat = beta_tilde.T @ beta

        S = np.outer(dp, dp) + np.outer(dm, dm)
        S = pd.DataFrame(S, index=dsindex, columns=dsindex)
        S = pd.DataFrame(S.values, index=C.index, columns=C.index)

        invcov = la.inv(C + S)

        # Final result
        delta_T_tilde = S_hat @ invcov @ (dat_central - th_pred)
        preds.append(0.118 + delta_T_tilde)

    np.savetxt(f"{output_file}", np.array(preds))

preds = np.loadtxt(output_file) - 0.118

std = np.std(np.array(preds))
cv = np.mean(np.array(preds))
print(rf"Prediction for $\alpha_s$: {cv} +/- {std}")

plt.hist(preds, bins=nbins)
bin_width = (preds.max() - preds.min())/nbins
xaxis_plotpoints = np.linspace(cv - 3*std, cv + 3*std, 100)
gaussian = np.exp(-(xaxis_plotpoints-cv)**2/(2*std**2))/(np.sqrt(2*np.pi*std**2))*preds.size*bin_width
plt.plot(xaxis_plotpoints, gaussian, label=fr"$\delta\alpha_s = ${cv:.4f} $\pm$ {std:.4f}")

plt.xlabel(r"$\delta\alpha_s$")
plt.ylabel("counts")
plt.legend()
plt.savefig("histplot.pdf")
