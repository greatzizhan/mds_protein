

import numpy as np

from metric import compute_rmsd
from mds import closed_form_mds, smacof

from logger import Logger

logger = Logger.logger


def read_pdb(chain_id, filename):
    from Bio.PDB import PDBParser

    # Parse the PDB file
    parser = PDBParser()
    structure = parser.get_structure("none", filename)

    # Get the coordinates of the CA atoms
    ca_coords = []
    chain = None
    for model in structure:
        for c in model:
            if c.id.strip() == chain_id.strip():
                chain = c
    if chain is None:
        raise f"the specific chain={chain} is not existed in {filename}."

    for residue in chain:
        if residue.has_id("CA"):
            ca_coords.append(residue["CA"].get_coord())

    return np.array(ca_coords)


k = 3
num_seeds = 1
chain_id = "B"
filename = "data/5b8c_assembly1.pdb"
np.random.seed(1)

X = read_pdb(chain_id, filename)
n = X.shape[0]

D = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1))
logger.info(f"avg={np.mean(D)}, max={np.max(D)}, min={np.min(D)}")

pred_X_cf = closed_form_mds(D)
dist_cf = np.sqrt(np.sum((pred_X_cf[:, None, :] - pred_X_cf[None, :, :]) ** 2, axis=-1))
diff_cf = (D - dist_cf) ** 2
rmsd_cf = compute_rmsd(X, pred_X_cf)
logger.info(f"closed form: rmsd={rmsd_cf}, avg={np.mean(diff_cf)}, max={np.max(diff_cf)}, min={np.min(diff_cf)}")

pred_X_sma, stress, n_iter_ = smacof(D, max_iter=3000, eps=0)
dist_sma = np.sqrt(np.sum((pred_X_sma[:, None, :] - pred_X_sma[None, :, :]) ** 2, axis=-1))
diff_sma = (D - dist_sma) ** 2
rmsd_sma = compute_rmsd(X, pred_X_sma)
logger.info(f"SMACOF: rmsd={rmsd_sma}, avg={np.mean(diff_sma)}, max={np.max(diff_sma)}, min={np.min(diff_sma)}")

pred_X_sma_init, stress, n_iter_ = smacof(D, init=pred_X_cf, max_iter=3000, eps=0)
dist_sma_init = np.sqrt(np.sum((pred_X_sma_init[:, None, :] - pred_X_sma_init[None, :, :]) ** 2, axis=-1))
diff_sma_init = (D - dist_sma_init) ** 2
rmsd_sma_init = compute_rmsd(X, pred_X_sma_init)
logger.info(f"SMACOF_init: rmsd={rmsd_sma_init}, avg={np.mean(diff_sma_init)}, max={np.max(diff_sma_init)}, min={np.min(diff_sma_init)}")



