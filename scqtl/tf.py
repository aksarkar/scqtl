import numpy as np
import tensorflow as tf
import scqtl.amsgrad

def nb_llik(x, mean, inv_disp):
  """Log likelihood of x distributed as NB

  See Hilbe 2012, eq. 8.10

  mean - mean (> 0)
  inv_disp - inverse dispersion (> 0)

  """
  assert x.shape == mean.shape == inv_disp.shape
  return (x * tf.log(mean / inv_disp) -
          x * tf.log(1 + mean / inv_disp) -
          inv_disp * tf.log(1 + mean / inv_disp) +
          tf.lgamma(x + inv_disp) -
          tf.lgamma(inv_disp) -
          tf.lgamma(x + 1))

def zinb_llik(x, mean, inv_disp, logodds):
  """Log likelihood of x distributed as ZINB

  See Hilbe 2012, eq. 11.12, 11.13

  mean - mean (> 0)
  inv_disp - inverse dispersion (> 0)
  logodds - logit proportion of excess zeros

  """
  # Important identities:
  # log(x + y) = log(x) + softplus(y - x)
  # log(sigmoid(x)) = -softplus(-x)
  case_zero = -tf.nn.softplus(-logodds) + tf.nn.softplus(nb_llik(x, mean, inv_disp) + tf.nn.softplus(-logodds))
  case_non_zero = -tf.nn.softplus(logodds) + nb_llik(x, mean, inv_disp)
  return tf.where(tf.less(x, 1), case_zero, case_non_zero)

def fit(umi, onehot, size_factor, design=None, learning_rate=1e-2,
        max_epochs=100000, return_beta=False, warm_start=None, verbose=False):
  """Return estimated ZINB parameters

  umi - count matrix (n x p; float32)
  onehot - mapping of individuals to cells (m x n; float32)
  size_factor - size factor vector (n x 1; float32)
  design - confounder matrix (n x q; float32)
  warm_start - tuple of (log_mean, log_disp, logodds)

  Returns:

  log_mean - log mean parameter (m x p)
  log_disp - log dispersion parameter (m x p)
  logodds - logit proportion of excess zeros (m x p)

  """
  assert onehot.shape[0] == umi.shape[0]
  if design is not None:
    assert onehot.shape[0] == design.shape[0]
  n, p = umi.shape
  _, m = onehot.shape

  graph = tf.Graph()
  with graph.as_default(), graph.device('/gpu:*'):
    size_factor = tf.Variable(size_factor, trainable=False)
    umi = tf.Variable(umi, trainable=False)
    onehot = tf.Variable(onehot, trainable=False)
    if design is not None:
      _, k = design.shape
      design = tf.Variable(design, trainable=False)
      beta = tf.Variable(tf.zeros([k, p]))

    if warm_start is not None:
      log_mean, log_disp, logodds = warm_start
      assert log_mean.shape == (m, p)
      assert log_disp.shape == (m, p)
      assert logodds.shape == (m, p)
      mean = tf.exp(tf.Variable(log_mean))
      inv_disp = tf.exp(tf.Variable(-log_disp))
      logodds = tf.Variable(logodds)
    else:
      mean = tf.exp(tf.Variable(tf.zeros([m, p])))
      inv_disp = tf.exp(tf.Variable(tf.zeros([m, p])))
      logodds = tf.Variable(tf.zeros([m, p]))

    lam = size_factor * tf.matmul(onehot, mean)
    if design is not None:
      lam *= tf.exp(tf.matmul(design, beta))

    nb_nll = -tf.reduce_sum(nb_llik(umi, lam, tf.matmul(onehot, inv_disp)))
    zinb_nll = -tf.reduce_sum(zinb_llik(umi, lam, tf.matmul(onehot, inv_disp), tf.matmul(onehot, logodds)))

    train_nb = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(nb_nll)
    train_zinb = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(zinb_nll)

    opt = [tf.log(mean), -tf.log(inv_disp), logodds]
    if return_beta:
      opt.append(beta)

    obj = float('-inf')
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for i in range(max_epochs):
        if i < max_epochs // 2 and warm_start is None:
          _, update = sess.run([train_nb, nb_nll])
        else:
          _, update = sess.run([train_zinb, zinb_nll])          
        if not np.isfinite(update):
          raise tf.train.NanLossDuringTrainingError
        if verbose and not i % 500:
          print(i, update, end='\r')
      if verbose:
        print(i, update)
      return sess.run(opt)
