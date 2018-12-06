import numpy as np
import scipy.special as sp
import scipy.stats as st

def rpp(x, log_mu, log_phi, logodds, size, onehot, n_samples=100):
  n = onehot.dot(np.exp(-log_phi))
  pi0 = onehot.dot(sp.expit(-logodds))
  p = 1 / (1 + (size * onehot.dot(np.exp(log_mu + log_phi))))

  cdf = st.nbinom(n=n, p=p).cdf(x - 1)
  # Important: this excludes the right endpoint, so we need to special case x =
  # 0
  cdf = np.where(x > 0, pi0 + (1 - pi0) * cdf, cdf)
  pmf = st.nbinom(n=n, p=p).pmf(x)
  pmf *= (1 - pi0)
  pmf[x == 0] += pi0[x == 0]
  rpp = cdf + np.random.uniform(size=(n_samples, x.shape[0])) * pmf
  return rpp

def diagnostic_test(x, log_mu, log_phi, logodds, size, onehot, n_samples=100):
  vals = rpp(x, log_mu, log_phi, logodds, size, onehot, n_samples)
  return st.kstest(vals.ravel(), 'uniform')
