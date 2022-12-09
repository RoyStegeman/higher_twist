#!/usr/bin/env python
# coding: utf-8

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import pandas as pd
import seaborn as sns
from validphys.loader import Loader

sns.set()
sns.set_style("ticks")
from validphys.api import API

l = Loader()

# Configuration
fit = "221203-ern-001"
pdf_ori = "NNPDF40_nnlo_as_01180"
output_file_name = "alphas.csv"
nbins = 20

inp = {
    # 'dataset_inputs': [{'dataset': 'NMCPD'}, {'dataset': 'NMC'}],
    "dataset_inputs": {"from_": "fit"},
    "fit": fit,
    "use_cuts": "fromfit",
}


path_this_module = pathlib.Path(__file__).parent.resolve()
output_file = path_this_module / output_file_name

if not output_file.exists():
    # Inputs for central theory: alphas=0.118
    inps_central = dict(theoryid=200, pdf=pdf_ori, **inp)

    # Inputs for plus theory: alphas=0.120
    inps_plus = dict(theoryid=207, pdf=pdf_ori, **inp)

    # Inputs for minus theory: alphas=0.116
    inps_minus = dict(theoryid=201, pdf=pdf_ori, **inp)

    # Inputs for central theory: alphas=0.118
    inps_central_fit = dict(theoryid=200, pdf={"from_": "fit"}, **inp)

    # Experimental covariance matrix
    C = API.groups_covmat(**inps_central)

    # Indexes
    dsindex = API.groups_index(**inps_central)
    datth_central = API.group_result_table_no_table(**inps_central)
    datth_plus = API.group_result_table_no_table(**inps_plus)
    datth_minus = API.group_result_table_no_table(**inps_minus)
    datth_central_fit = API.group_result_table_no_table(**inps_central_fit)

    dat_central = datth_central["data_central"]
    th_central_and_replicas_fit = datth_central_fit.iloc[:, 1:].to_numpy()

    # Computation of Eqs.(3.37)-(3.38) in [arXiv:2105.05114](https://arxiv.org/pdf/2105.05114.pdf)
    beta_tilde = (0.002 / np.sqrt(2)) * np.array([1, -1])

    delta_plus = (1 / np.sqrt(2)) * (
        datth_plus["theory_central"] - datth_central["theory_central"]
    ).to_numpy()
    delta_minus = (1 / np.sqrt(2)) * (
        datth_minus["theory_central"] - datth_central["theory_central"]
    ).to_numpy()
    beta = [delta_plus, delta_minus]
    S_hat = beta_tilde.T @ beta

    S = np.outer(delta_plus, delta_plus) + np.outer(delta_minus, delta_minus)
    S = pd.DataFrame(S, index=dsindex, columns=dsindex)
    S = pd.DataFrame(S.values, index=C.index, columns=C.index)

    invcov = la.inv(C + S)

    preds = []
    for pred in th_central_and_replicas_fit.T:
        delta_T_tilde = S_hat @ invcov @ (dat_central - pred)
        preds.append(delta_T_tilde + 0.118)

    np.savetxt(f"{output_file}", np.array(preds))

preds = np.loadtxt(output_file)

pred_cv = preds[0]
preds_replicas = preds[1:]

std = np.std(np.array(preds_replicas))
cv = np.mean(np.array(preds_replicas))
print(rf"Prediction for $\alpha_s$: {cv} +/- {std}")

plt.hist(preds_replicas, bins=nbins)
bin_width = (preds_replicas.max() - preds_replicas.min()) / nbins
xaxis_plotpoints = np.linspace(cv - 3 * std, cv + 3 * std, 100)

gaussian = (
    np.exp(-((xaxis_plotpoints - cv) ** 2) / (2 * std**2))
    / (np.sqrt(2 * np.pi * std**2))
    * preds_replicas.size
    * bin_width
)
plt.plot(xaxis_plotpoints, gaussian, label=rf"$\alpha_s = ${cv:.4f} $\pm$ {std:.4f}")

plt.axvline(x=pred_cv, color="green", label=f"replica0 prediction: {pred_cv:.4f}")

plt.xlabel(r"$\alpha_s$")
plt.ylabel("counts")
plt.legend()
plt.savefig("histplot.pdf")
