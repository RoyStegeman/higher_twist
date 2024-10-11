import argparse
import pandas as pd
import numpy as np
from scipy import interpolate as scint
import matplotlib.pyplot as plt


# Default option if nodes are not provided
PATH_DIR = "./"

def compute_ht_parametrisation(
        index: int,
        nodes: list,
        x: list,
        Q2: list,
        h_prior: list,
        reverse: bool = False
):
    if not reverse:
        shifted_H_list = [0 for k in range(len(nodes))]
        shifted_H_list[index] = h_prior[index]
    else:
        shifted_H_list = h_prior.copy()
        shifted_H_list[index] = 0

    H = scint.CubicSpline(nodes, shifted_H_list)
    H = np.vectorize(H)

    PC = H(x) / Q2
    return PC


class TCM:

  def __init__(self, mean_prediction,
               preds_onlyreplicas,
               index,
               invcov,
               dat_central,
               prior_central_value):
    self.preds_onlyreplicas = preds_onlyreplicas
    self.mean_prediction = mean_prediction
    self.index = index
    self.invcov = invcov
    self.dat_central = dat_central
    self.prior_central_value = prior_central_value

  @classmethod   
  def construct_X(cls, preds_onlyreplicas, mean_prediction):
    '''Construct the matrix X given the mean predictions and the predictions for
    each replica.

    Parameters
    ----------
        preds_onlyreplicas: array-like 
          Multi-dimensional array with the predictions for each replica. The shape of
          the array is (predictions_size, replica).
        mean_prediction: array
          Array of mean predictions with shape (predictions_size,)
    
    Returns
    ------
        numpy array representing the matrix X.

    '''
    X = np.zeros((mean_prediction.shape[0], mean_prediction.shape[0]))

    for i in range(preds_onlyreplicas.shape[1]):
      X += np.outer(
          (preds_onlyreplicas[:, i] - mean_prediction), (preds_onlyreplicas[:, i] - mean_prediction)
      )
    X *= 1 / preds_onlyreplicas.shape[1]
    return X
  

  @classmethod
  def construct_S_tilde(cls, beta_tilde):
    '''Construct S_tilde providing the shifts for the genuine predictions.
    
    Parameters
    ----------
        beta_tilde: pandas.DataFrame
          Pandas DataFrame containing the size and directions of the theoretical
          uncertainties in the genuine predictions.

    Returns
    -------
        pandas.DataFrame representing the matrix S_tilde with the index inherited
        from beta_tilde.
    '''
    S_tilde = np.zeros(beta_tilde.shape)
    for shift in beta_tilde.columns:
      S_tilde += np.outer(beta_tilde[shift],beta_tilde[shift])
    S_tilde = pd.DataFrame(S_tilde, index=beta_tilde.index, columns=beta_tilde.columns)
    return S_tilde


  @classmethod
  def construct_S_hat(cls, beta_tilde, beta):
    '''Construct the matri S_hat defined as the outer product of the genuine uncertainty
    vector beta_tilde with the uncertainty vector beta'''
    S_hat = np.zeros((beta_tilde.shape[0], beta.shape[0]))
    for shift in beta.columns:
      S_hat += np.outer(beta_tilde.droplevel(level="HT", axis=1)[shift], beta[shift])
    S_hat = pd.DataFrame(S_hat, index=beta_tilde.index, columns=beta.index)
    return S_hat
  

  @classmethod
  def construct_delta_T_tilde(cls, invcov,
                              S_hat,
                              mean_prediction,
                              dat_central):
    '''Compute the the shifts for the genuine predictions

    Parameters
    ----------
        invcov: numpy.array
          Matrix representing the inverse of the full covariance matrix used in
          the fit.
        S_hat: 
          The matrix S_hat
        mean_prediction: array
          The mean predictions
        dat_central: array
          The central values of the data to be compared with the predictions
    
    Returns
    -------
        pandas.core.series.Series containing the shifts for the parameters
        used to model the uncertainty.
    '''
    return - S_hat @ invcov @ (mean_prediction - dat_central)
  
  @classmethod
  def calculate_posterior(cls, invcov,
                          S_hat,
                          mean_prediction,
                          dat_central,
                          prior_central_value):

    delta_T_tilde = - S_hat @ invcov @ (mean_prediction - dat_central)
    posteriors = prior_central_value + delta_T_tilde
    return posteriors
  

  @classmethod
  def calculate_correaltions(cls, S_hat, invcov, X, S_tilde):
    
    # Necessary for DataFrame compatibility in the matrix products
    invcov = pd.DataFrame(invcov, columns = S_hat.columns, index = S_hat.columns)
    X = pd.DataFrame(X, columns = S_hat.columns, index = S_hat.columns)

    P_tilde = S_hat @ invcov @ X @ invcov @ S_hat.T + (S_tilde - S_hat @ invcov @ S_hat.T)
    return P_tilde
  

  def apply_tcm(self, beta, beta_tilde):
    S_tilde = self.construct_S_tilde(beta_tilde)
    X = self.construct_X(self.preds_onlyreplicas, self.mean_prediction)
    S_hat = self.construct_S_hat(beta_tilde, beta)
    posteriors = self.calculate_posterior(self.invcov, S_hat, self.mean_prediction, self.dat_central, self.prior_central_value)
    P_tilde = self.calculate_correaltions(S_hat, self.invcov, X, S_tilde)
    return posteriors, P_tilde
  
class GenerationError(Exception): pass

class HTset:
  def __init__(self, posteriors, 
                covmat, 
                x_nodes, 
                drop_last_node = True):
    # The last node of the parametrisation is usually constrained to zero. If that is the case,
    # the covmat should be reduced by removing these zero nodes, otherwise it would have zero
    # eigenvalues and the Cholesky decomposition cannot be applied.
    if drop_last_node:
      self.central_nodes = posteriors.groupby(level=0, sort=False).apply(lambda x: x.iloc[:-1]).reset_index(level=0, drop=True)
      self.covmat = covmat.groupby(level=0, sort=False).apply(lambda x: x.iloc[:-1]).reset_index(level=0, drop=True).T.groupby(level=0, sort=False).apply(lambda x: x.iloc[:-1]).reset_index(level=0, drop=True)
      self.index = self.central_nodes.index
    else:
      self.central_nodes = posteriors
      self.covmat = covmat
      self.index = posteriors.index
    self.x_nodes = x_nodes

    self.generated = False

  def __make_pseudonodes(self, number_of_replicas, seed):
    # Compute Cholesky decomposition
    L = np.linalg.cholesky(self.covmat.to_numpy())
    central_posteriors = self.central_nodes.to_numpy().reshape((self.central_nodes.to_numpy().shape[0],))
    fluctuated_nodes = pd.DataFrame(self.central_nodes.copy(), columns=['central'])
    for k in range(number_of_replicas):
      rng = np.random.default_rng(seed = seed + k)
      pseudo_nodes = central_posteriors + L @ rng.normal(size=central_posteriors.shape[0])
      fluctuated_nodes = pd.concat([fluctuated_nodes, pd.DataFrame(pseudo_nodes, index = self.index, columns=[f'rep_{k+1}'])], axis=1)
    return fluctuated_nodes
    
  def __make_pseudo_functions(self, fluctuated_nodes):
    number_of_replicas = fluctuated_nodes.shape[1] - 1
    # Construct dictionary
    pseudo_ht_func = {}
    for name, df in fluctuated_nodes.groupby(level = "HT"):
      aux = []
      for k in range(number_of_replicas):
        aux.append(scint.CubicSpline(self.x_nodes[name], np.concatenate([df.iloc[:,1+k].to_numpy(), [0]])))
      pseudo_ht_func[name] = aux
    return pseudo_ht_func
  
  def generate_set(self, number_of_replicas, seed, check = False):
    pseudonodes = self.__make_pseudonodes(number_of_replicas, seed)
    if check:
      self.plot_pseudodata_average(pseudonodes)
    self.set = self.__make_pseudo_functions(pseudonodes)
    self.generated = True

  @classmethod
  def plot_pseudodata_average(cls, fluctuated_nodes):
    x = np.linspace(-0.15, 0.15,100)
    plt.scatter(fluctuated_nodes['central'].to_numpy(), fluctuated_nodes.drop(columns='central').mean(axis=1))
    plt.plot(x, x, color='lightblue', ls="-", alpha=0.3)
    plt.xlabel('Central velues')
    plt.ylabel('Average of pseudodata')
    plt.show()
  

  @classmethod
  def compute_central_value_and_std(cls, input, function_replicas):
    function_replicas = np.array(function_replicas)
    input = np.array(input)
    mat = np.zeros(shape=(input.size, function_replicas.size))

    for k, func in enumerate(function_replicas):
      mat[:,k] = func(input)

    mean_values = mat.mean(axis=1)
    stddev = mat.std(axis=1)
    
    return mean_values, stddev
  
  # Override __getitem__ to enable getting an item with square brackets
  def __getitem__(self, key):
      if not self.generated:
        raise GenerationError(f"The set has not been generated yet. Please, run the method `generate_set` first.")
      return self.set[key]

  def __call__(self, input, key):
    if not self.generated:
        raise GenerationError(f"The set has not been generated yet. Please, run the method `generate_set` first.")
    ht_func_replicas = np.array(self[key])
    input = np.array(input)
    mat = np.zeros(shape=(input.size, ht_func_replicas.size))
    for k, func in enumerate(ht_func_replicas):
      mat[:,k] = func(input)

    return mat


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Apply the theory covariance method to extract posterior of the\
                                                coefficients used to parametrised the higher twist. It also stores\
                                                a plot with the posterior parametrisation, a comparison against ABMP,\
                                                and the heat map for the correlated uncertainties.")
  parser.add_argument('fitname', type=str, help='The name of the fit to analyse.')
  parser.add_argument('-d', '--dir_path', type=str, default='./', help='Path of the directory to store the results.')
  

  args = parser.parse_args()
  fitname = args.fitname
  PATH_DIR = args.dir_path


x_abmp = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
y_h2_abmp = [0.023, -0.032, -0.005, 0.025, 0.051, 0.003, 0.0]
y_ht_abmp = [-0.319, -0.134, -0.052, 0.071, 0.030, 0.003, 0.0]
h2_sigma_abmp = [0.019, 0.013, 0.009, 0.006, 0.005, 0.004, 0.0]
ht_sigma_abmp = [0.126, 0.040, 0.030, 0.025, 0.012, 0.007, 0.0]
H_2_abmp = scint.CubicSpline(x_abmp, y_h2_abmp)
H_T_abmp = scint.CubicSpline(x_abmp, y_ht_abmp)

# Reconstruct HL from HT and H2
def H_L(x, h2, ht):
    return (h2 - np.power(x, 0.05) * ht)


H_2_abmp = np.vectorize(H_2_abmp)

H2_plus_abmp = scint.CubicSpline(x_abmp, np.add(y_h2_abmp, h2_sigma_abmp))
H2_minus_abmp = scint.CubicSpline(x_abmp, np.add(y_h2_abmp, np.multiply(h2_sigma_abmp, -1)))
Ht_plus_abmp = scint.CubicSpline(x_abmp, np.add(y_ht_abmp, ht_sigma_abmp))
Ht_minus_abmp = scint.CubicSpline(x_abmp, np.add(y_ht_abmp, np.multiply(ht_sigma_abmp, -1)))