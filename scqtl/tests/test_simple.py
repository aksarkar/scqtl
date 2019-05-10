import numpy as np
import pytest
import scipy.special as sp
import scipy.stats as st
import scqtl.simple

@pytest.fixture
def simulate_pois():
  np.random.seed(1)
  x = np.random.poisson(lam=100, size=1000)
  return x

@pytest.fixture
def simulate_nb():
  np.random.seed(1)
  x = st.nbinom(n=3, p=1e-3).rvs(size=1000)
  return x

@pytest.fixture
def simulate_pois_gam():
  np.random.seed(3)
  # n = 3
  # p = 1e-3
  # true_mean = n * (1 - p) / p
  # true_inv_disp = n

  # scipy uses shape (a), scale parameterization
  # V[u] = 1 / true_inv_disp = shape * (scale ** 2)
  u = st.gamma(a=3, scale=1/3).rvs(size=1000)
  mu = 2997
  x = np.random.poisson(lam=mu * u)
  return x

@pytest.fixture
def simulate_pois_size():
  np.random.seed(1)
  s = np.random.lognormal(size=1000)
  x = np.random.poisson(lam=s * 100, size=1000)
  return x, s

@pytest.fixture
def simulate_pois_gam_size():
  np.random.seed(4)
  s = np.random.lognormal(size=1000)
  mu = 100
  u = st.gamma(a=3, scale=1/3).rvs(size=1000)
  x = np.random.poisson(lam=s * mu * u)
  return x, s

@pytest.fixture
def simulate_pois_zig():
  np.random.seed(4)
  mu = 100
  u = st.gamma(a=3, scale=1/3).rvs(size=1000)
  y = (np.random.uniform(size=1000) < 0.95).astype(float)
  x = np.random.poisson(lam=mu * u * y)
  return x

@pytest.fixture
def simulate_pois_zig_size():
  np.random.seed(4)
  s = np.random.lognormal(size=1000)
  mu = 200
  u = st.gamma(a=3, scale=1/3).rvs(size=1000)
  y = (np.random.uniform(size=1000) < 0.95).astype(float)
  x = np.random.poisson(lam=s * mu * u * y)
  return x, s

def test_fit_pois_pois_data(simulate_pois):
  x = simulate_pois
  size = 1
  mean, llik = scqtl.simple.fit_pois(x, size)
  assert np.isclose(mean, x.mean())
  assert llik < 0

def test_fit_nb_pois_data(simulate_pois):
  x = simulate_pois
  size = 1
  mean0, llik0 = scqtl.simple.fit_pois(x, size)
  mean1, inv_disp, llik1 = scqtl.simple.fit_nb(x, size)
  assert np.isclose(mean0, mean1, atol=1e-2)
  assert np.isclose(1 / inv_disp, 0)
  assert np.isclose(llik0, llik1)

def test_fit_nb_nb_data(simulate_nb):
  x = simulate_nb
  size = 1
  # n = 3
  # p = 1e-3
  # true_mean = n * (1 - p) / p
  # true_inv_disp = n
  _, llik0 = scqtl.simple.fit_pois(x, size)
  mean, inv_disp, llik1 = scqtl.simple.fit_nb(x, size)
  assert llik1 > llik0
  assert np.isclose(mean, x.mean(), atol=1)
  assert np.isclose(inv_disp, 3, atol=0.1)

def test_fit_nb_pois_gam_data(simulate_pois_gam):
  x = simulate_pois_gam
  size = 1
  mean, inv_disp, _ = scqtl.simple.fit_nb(x, size)
  assert np.isclose(mean, x.mean(), atol=1)
  assert np.isclose(inv_disp, 3, atol=0.1)

def test_fit_zinb_pois_data(simulate_pois):
  x = simulate_pois
  size = 1
  mean0, llik0 = scqtl.simple.fit_pois(x, size)
  mean1, inv_disp, logodds, llik1 = scqtl.simple.fit_zinb(x, size)
  assert np.isclose(mean0, mean1, atol=1e-2)
  assert np.isclose(1 / inv_disp, 0)
  assert np.isclose(llik0, llik1)

def test_fit_zinb_nb_data(simulate_nb):
  x = simulate_nb
  size = 1
  mean0, inv_disp0, llik0 = scqtl.simple.fit_nb(x, size)
  mean1, inv_disp1, logodds, llik1 = scqtl.simple.fit_zinb(x, size)
  assert np.isclose(llik1, llik0)
  assert np.isclose(mean0, mean1)
  assert np.isclose(inv_disp0, inv_disp1, atol=1e-3)

def test_fit_zinb_zinb_data(simulate_pois_zig):
  x = simulate_pois_zig
  size = 1
  mean, inv_disp, logodds, llik = scqtl.simple.fit_zinb(x, size)
  assert np.isclose(mean, 100, atol=2e-1)
  assert np.isclose(inv_disp, 3, atol=1e-1)
  assert np.isclose(logodds, sp.logit(.05), atol=2e-1)

def test_fit_pois_size(simulate_pois_size):
  x, s = simulate_pois_size
  mean, _ = scqtl.simple.fit_pois(x, s)
  assert np.isclose(mean, 100, atol=1)

def test_fit_nb_pois_size(simulate_pois_size):
  x, s = simulate_pois_size
  mean, inv_disp, _ = scqtl.simple.fit_nb(x, s)
  assert np.isclose(mean, 100, atol=1)
  assert np.isclose(1 / inv_disp, 0, atol=1e-3)

def test_fit_nb_nb_size(simulate_pois_gam_size):
  x, s = simulate_pois_gam_size
  mean, inv_disp, _ = scqtl.simple.fit_nb(x, s)
  assert np.isclose(mean, 100, atol=3)
  assert np.isclose(inv_disp, 3, atol=.25)

@pytest.mark.xfail()
def test_fit_zinb_zinb_size(simulate_pois_zig_size):
  x, s = simulate_pois_zig_size
  mean, inv_disp, logodds, _ = scqtl.simple.fit_zinb(x, s)
  assert np.isclose(mean, 200, atol=3)
  assert np.isclose(inv_disp, 3, atol=.25)
  assert np.isclose(logodds, sp.logit(.05))
