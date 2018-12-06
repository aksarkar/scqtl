import numpy as np
import scipy.optimize as so
import scipy.special as sp

def log(x):
  """Numerically safe log"""
  return np.log(x + 1e-8)

def sigmoid(x):
  """Numerically safe sigmoid"""
  lim = np.log(np.finfo(np.float64).resolution)
  return np.clip(sp.expit(x), lim, -lim)

def nb(theta, x, size, onehot, design):
  """Return the per-data point log likelihood

  x ~ Poisson(size .* design' * theta[2 * m:k] * exp(onehot * theta[:m]) * u)
  u ~ Gamma(exp(onehot * theta[m:2 * m]), exp(onehot * theta[m:2 * m]))

  theta - (2 * m + k, 1)
  x - (n, 1)
  size - (n, 1)
  onehot - (n, m)
  design - (n, k)

  """
  n, m = onehot.shape
  assert x.shape == (n,)
  assert size.shape == (n,)
  assert design.shape[0] == n
  assert theta.shape == (2 * m + design.shape[1],)
  mean = size * np.exp(onehot.dot(theta[:m]) + design.dot(theta[2 * m:]))
  assert mean.shape == (n,)
  inv_disp = onehot.dot(np.exp(theta[m:2 * m]))
  assert inv_disp.shape == (n,)
  return (x * log(mean / inv_disp) -
          x * log(1 + mean / inv_disp) -
          inv_disp * log(1 + mean / inv_disp) +
          sp.gammaln(x + inv_disp) -
          sp.gammaln(inv_disp) -
          sp.gammaln(x + 1))

def _nb(theta, x, size, onehot, design=None):
  """Return the mean negative log likelihood of x"""
  return -nb(theta, x, size, onehot, design).mean()

def zinb(theta, x, size, onehot, design=None):
  """Return the mean negative log likelihood of x"""
  n, m = onehot.shape
  logodds, theta = theta[:m], theta[m:]
  case_non_zero = -np.log1p(np.exp(onehot.dot(logodds))) + nb(theta, x, size, onehot, design)
  case_zero = np.logaddexp(onehot.dot(logodds - np.log1p(np.exp(logodds))), case_non_zero)
  return -np.where(x < 1, case_zero, case_non_zero).mean()

def _fit_gene(chunk, onehot, design=None):
  n, m = onehot.shape
  assert chunk.shape[0] == n
  # We need to take care here to initialize mu=-inf for all zero observations
  x0 = np.log((onehot * chunk[:,:1]).sum(axis=0) / onehot.sum(axis=0)) - np.log(np.ma.masked_values(onehot, 0) * chunk[:,1:]).mean(axis=0).compressed()
  x0 = np.hstack((x0, np.zeros(m)))
  if design is not None:
    assert design.shape[0] == n
    design -= design.mean(axis=0)
    x0 = np.hstack((x0, np.zeros(design.shape[1])))
  res0 = so.minimize(_nb, x0=x0, args=(chunk[:,0], chunk[:,1], onehot, design))
  res = so.minimize(zinb, x0=list(np.zeros(m)) + list(res0.x), args=(chunk[:,0], chunk[:,1], onehot, design))
  if res0.fun < res.fun:
    # This isn't a likelihood ratio test. Numerically, our implementation of
    # ZINB can't represent pi = 0, so we need to use a separate implementation
    # for it
    log_mu = res0.x[:m]
    neg_log_phi = res0.x[m:2 * m]
    logit_pi = np.zeros(m)
    logit_pi.fill(-np.inf)
  else:
    logit_pi = res.x[:m]
    log_mu = res.x[m:2 * m]
    neg_log_phi = res.x[2 * m:3 * m]
  mean_by_sample = chunk[:,1] * onehot.dot(np.exp(log_mu))
  var_by_sample = mean_by_sample + np.square(mean_by_sample) * onehot.dot(np.exp(-neg_log_phi))
  mean_by_ind = np.ma.masked_equal(onehot * mean_by_sample.reshape(-1, 1), 0).mean(axis=0).filled(0)
  var_by_ind = np.ma.masked_equal(onehot * (np.square(mean_by_sample - onehot.dot(mean_by_ind)) + var_by_sample).reshape(-1, 1), 0).mean(axis=0).filled(0)
  return [log_mu, -neg_log_phi, logit_pi, mean_by_ind, var_by_ind]

def fit_gene(chunk, bootstraps=100):
  orig = _fit_gene(chunk)
  B = []
  for _ in range(bootstraps):
    B.append(_fit_gene(chunk[np.random.choice(chunk.shape[0], chunk.shape[0], replace=True)]))
  se = np.array(B)[:,:2].std(axis=0)
  return orig + list(se.ravel())
