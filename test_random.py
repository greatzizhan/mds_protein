
import numpy as np

from metric import compute_rmsd
from mds import closed_form_mds, smacof

from logger import Logger

logger = Logger.logger


n = 100
k = 3
np.random.seed(0)

X = np.random.rand(n, k)
D = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1))
logger.info(f"avg={np.mean(D)}, max={np.max(D)}, min={np.min(D)}")

pred_X_cf = closed_form_mds(D, k=k)
dist_cf = np.sqrt(np.sum((pred_X_cf[:, None, :] - pred_X_cf[None, :, :]) ** 2, axis=-1))
diff_cf = (D - dist_cf) ** 2
rmsd_cf = compute_rmsd(X, pred_X_cf)
logger.info(f"closed form: rmsd={rmsd_cf}, avg={np.mean(diff_cf)}, max={np.max(diff_cf)}, min={np.min(diff_cf)}")

pred_X_sma, stress, n_iter_ = smacof(D, n_components=k, max_iter=300, eps=1e-8)
dist_sma = np.sqrt(np.sum((pred_X_sma[:, None, :] - pred_X_sma[None, :, :]) ** 2, axis=-1))
diff_sma = (D - dist_sma) ** 2
rmsd_sma = compute_rmsd(X, pred_X_cf)
logger.info(f"SMACOF: rmsd={rmsd_sma}, avg={np.mean(diff_sma)}, max={np.max(diff_sma)}, min={np.min(diff_sma)}")

pred_X_sma_init, stress, n_iter_ = smacof(D, n_components=k, init=pred_X_cf, max_iter=300, eps=1e-8)
dist_sma_init = np.sqrt(np.sum((pred_X_sma_init[:, None, :] - pred_X_sma_init[None, :, :]) ** 2, axis=-1))
diff_sma_init = (D - dist_sma_init) ** 2
rmsd_sma_init = compute_rmsd(X, pred_X_sma_init)
logger.info(f"SMACOF_init: rmsd={rmsd_sma_init}, avg={np.mean(diff_sma_init)}, max={np.max(diff_sma_init)}, min={np.min(diff_sma_init)}")
