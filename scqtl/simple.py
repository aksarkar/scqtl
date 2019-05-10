import numpy as np
import scipy.optimize as so
import scipy.special as sp
import scipy.stats as st

def check_args(x, size):
  n = x.shape[0]
  if x.shape != (n,):
    raise ValueError
  size = np.array(size)
  if size.shape != () and size.shape != x.shape:
    raise ValueError
  if size.shape == ():
    size = np.ones(x.shape) * size
  return x, size

def fit_pois(x, size):
  x, size = check_args(x, size)
  mean = x.sum() / size.sum()
  return mean, st.poisson(mu=size * mean).logpmf(x).sum()

def nb_obj(theta, x, size):
  mean = np.exp(theta[0])
  inv_disp = np.exp(theta[1])
  return -st.nbinom(n=inv_disp, p=1 / (1 + size * mean / inv_disp)).logpmf(x).sum()

def fit_nb(x, size):
  x, size = check_args(x, size)
  opt = so.minimize(nb_obj, x0=[np.log(x.sum() / size.sum()), 10], args=(x, size), method='Nelder-Mead')
  if not opt.success:
    raise RuntimeError(opt.message)
  mean = np.exp(opt.x[0])
  inv_disp = np.exp(opt.x[1])
  nll = opt.fun
  return mean, inv_disp, -nll

def zinb_obj(theta, x, size):
  mean = np.exp(theta[0])
  inv_disp = np.exp(theta[1])
  logodds = theta[2]
  nb = st.nbinom(n=inv_disp, p=1 / (1 + size * mean / inv_disp)).logpmf(x)
  case_zero = -np.log1p(np.exp(-logodds)) + np.log1p(np.exp(nb - logodds))
  case_nonzero = -np.log1p(np.exp(logodds)) + nb
  return -np.where(x < 1, case_zero, case_nonzero).sum()

def fit_zinb(x, size):
  x, size = check_args(x, size)
  init = so.minimize(nb_obj, x0=[np.log(x.sum() / size.sum()), 10], args=(x, size), method='Nelder-Mead')
  if not init.success:
    raise RuntimeError(init.message)
  opt = so.minimize(zinb_obj, x0=[init.x[0], init.x[1], -8], args=(x, size), method='Nelder-Mead')
  if not opt.success:
    raise RuntimeError(opt.message)
  mean = np.exp(opt.x[0])
  inv_disp = np.exp(opt.x[1])
  logodds = opt.x[2]
  nll = opt.fun
  return mean, inv_disp, logodds, -nll
