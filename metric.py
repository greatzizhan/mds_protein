
from Bio.SVDSuperimposer import SVDSuperimposer


def compute_rmsd(reference, coordinate):
    """
    Superimposes coordinates onto a reference by minimizing RMSD using SVD.

    Args:
        reference:
            [N, 3] reference array
        coordinate:
            [N, 3] array
    Returns:
        RMSD.
    """
    sup = SVDSuperimposer()
    sup.set(reference, coordinate)
    sup.run()
    return sup.get_rms()

