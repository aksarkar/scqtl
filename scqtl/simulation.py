import numpy as np
import scipy.special as sp
import scqtl.diagnostic

def simulate(num_samples, size=None, log_mu=None, log_phi=None, logodds=None, seed=None, design=None, fold=None):
  if seed is None:
    seed = 0
  np.random.seed(seed)
  if log_mu is None:
    log_mu = np.random.uniform(low=-12, high=-8)
  if log_phi is None:
    log_phi = np.random.uniform(low=-6, high=0)
  if size is None:
    size = 1e5
  if logodds is None:
    prob = np.random.uniform()
  else:
    prob = sp.expit(logodds)
  if design is None:
    design = np.random.normal(size=(num_samples, 1))
  else:
    assert design.shape[0] == num_samples
  if fold is None or np.isclose(fold, 1):
    beta = np.array([[0]])
  else:
    assert fold > 1
    beta = np.random.normal(size=(design.shape[1], 1), scale=2 * np.log(fold) / (1 - 2 * np.log(fold)))

  n = np.exp(-log_phi)
  p = 1 / (1 + size * np.exp(log_mu + design.dot(beta) + log_phi)).ravel()
  x = np.where(np.random.uniform(size=num_samples) < prob,
               0,
               np.random.negative_binomial(n=n, p=p, size=num_samples))
  return np.vstack((x, size * np.ones(num_samples))).T, design

def batch_design_matrix(num_samples, num_batches):
  """Return a matrix of binary indicators representing batch assignment"""
  design = np.zeros((num_samples, num_batches))
  design[np.arange(num_samples), np.random.choice(num_batches, size=num_samples)] = 1
  return design

def evaluate(num_samples, num_mols, log_mu, log_phi, logodds, fold, trial):
  x, design = simulate(num_samples=num_samples, size=num_mols,
                       log_mu=log_mu, log_phi=log_phi,
                       logodds=logodds, design=None, fold=fold, seed=trial)
  onehot = np.ones((num_samples, 1))
  keys = ['num_samples', 'num_mols', 'log_mu', 'log_phi', 'logodds', 'trial',
          'fold', 'log_mu_hat', 'log_phi_hat', 'logodds_hat', 'mean', 'var']
  result = [num_samples, num_mols, log_mu, log_phi, logodds, trial, fold] + [param[0] for param in _fit_gene(x, onehot, design)]
  result = {k: v for k, v in zip(keys, result)}
  eps = .5 / num_mols
  log_cpm = (np.log(np.ma.masked_values(x[:,0], 0) + eps) -
             np.log(x[:,1] + 2 * eps) +
             6 * np.log(10)).compressed()
  result['mean_log_cpm'] = log_cpm.mean()
  result['var_log_cpm'] = log_cpm.var()
  d, p = scqtl.diagnostic.diagnostic_test(x[:,0],
                         np.atleast_1d(result['log_mu_hat']),
                         np.atleast_1d(result['log_phi_hat']),
                         np.atleast_1d(-result['logodds_hat']),
                         num_mols,
                         onehot)
  result['ks_d'] = d
  result['ks_p'] = p
  return result
