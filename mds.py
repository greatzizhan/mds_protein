
import numpy as np


from logger import Logger

logger = Logger.logger


def closed_form_mds(D, k=3):
    D_Sqaure = D ** 2
    n = D_Sqaure.shape[0]
    H = np.eye(n) * 1.0 - np.ones((n, n)) * 1.0/n
    M = -0.5 * np.matmul(np.matmul(H, D_Sqaure), H)

    U, S, VT = np.linalg.svd(M)  # M = U * S * VT
    embedding = np.matmul(U, np.sqrt(np.diag(S)))

    return embedding[:, 0:k]


def smacof(
    dissimilarities,
    n_components=3,
    init=None,
    max_iter=300,
    eps=1e-3,
):
    """Computes multidimensional scaling using SMACOF algorithm.

    Parameters
    ----------
    dissimilarities : ndarray of shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Must be symmetric.

    n_components : int, default=3
        Number of dimensions in which to immerse the dissimilarities. If an
        ``init`` array is provided, this option is overridden and the shape of
        ``init`` is used to determine the dimensionality of the embedding
        space.

    init : ndarray of shape (n_samples, n_components), default=None
        Starting configuration of the embedding to initialize the algorithm. By
        default, the algorithm is initialized with a randomly chosen array.

    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    eps : float, default=1e-3
        Relative tolerance with respect to stress at which to declare
        convergence. The value of `eps` should be tuned separately depending
        on whether or not `normalized_stress` is being used.

    Returns
    -------
    X : ndarray of shape (n_samples, n_components)
        Coordinates of the points in a ``n_components``-space.

    stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).
        If `normalized_stress=True`, and `metric=False` returns Stress-1.
        A value of 0 indicates "perfect" fit, 0.025 excellent, 0.05 good,
        0.1 fair, and 0.2 poor [1]_.

    n_iter : int
        The number of iterations corresponding to the best stress.

    References
    ----------
    .. [1] "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
           Psychometrika, 29 (1964)

    .. [2] "Multidimensional scaling by optimizing goodness of fit to a nonmetric
           hypothesis" Kruskal, J. Psychometrika, 29, (1964)

    .. [3] "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
           Groenen P. Springer Series in Statistics (1997)

    .. [4] scikit-learn
    """
    n_samples = dissimilarities.shape[0]
    if init is None:
        X = np.random.uniform(size=n_samples * n_components)
        X = X.reshape((n_samples, n_components))
    else:
        n_components = init.shape[1]
        if n_samples != init.shape[0]:
            raise ValueError(
                "init matrix should be of shape (%d, %d)" % (n_samples, n_components)
            )
        X = init

    old_stress = None
    for it in range(max_iter):
        dis = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1))
        disparities = dissimilarities

        # Compute stress
        stress = ((dis.ravel() - disparities.ravel()) ** 2).sum() / 2
        # Update X using the Guttman transform
        dis[dis == 0] = 1e-5
        ratio = disparities / dis
        B = -ratio
        B[np.arange(len(B)), np.arange(len(B))] += ratio.sum(axis=1)
        X = 1.0 / n_samples * np.dot(B, X)

        dis = np.sqrt((X**2).sum(axis=1)).sum()
        if old_stress is not None:
            if (old_stress - stress / dis) < eps:
                break
        old_stress = stress / dis

    return X, stress, it + 1
