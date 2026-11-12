"""PCA of the random-effects variance-covariance estimates (rePCA.R).

``rePCA(x)`` performs a principal components analysis of the relative
(scaled by sigma) random-effects covariance matrices of a fitted model: for
each grouping factor the lower Cholesky factors of its terms (``getME(x,
"Tlist")``) are put block-diagonally together and their singular value
decomposition gives the standard deviations (``sdev``) and the rotation
(left singular vectors) of the orthogonal components.  A component with
standard deviation (close to) zero indicates a singular fit.
"""
import numpy as np
import pandas as pd
from scipy.linalg import block_diag

__all__ = ["rePCA", "PrComp", "PrCompList"]


class PrComp:
    """Mimics R's ``prcomp`` object as built by ``rePCA``'s ``svals()``:
    ``sdev`` (singular values), ``rotation`` (left singular vectors),
    ``center = FALSE``, ``scale = FALSE``."""

    def __init__(self, sdev, rotation):
        self.sdev = np.asarray(sdev, dtype=float)
        self.rotation = np.asarray(rotation, dtype=float)
        self.center = False
        self.scale = False

    def summary(self):
        """summary.prcomp: importance of the components (rounded as in R)."""
        vars_ = self.sdev ** 2
        vars_ = vars_ / vars_.sum() if vars_.sum() > 0 else vars_
        # R: colnames(importance) <- colnames(rotation), NULL for rePCA's svd()
        cols = [f"[,{i + 1}]" for i in range(len(self.sdev))]
        return pd.DataFrame([self.sdev, np.round(vars_, 5), np.round(np.cumsum(vars_), 5)],
                            index=["Standard deviation", "Proportion of Variance",
                                   "Cumulative Proportion"], columns=cols)

    def __str__(self):
        p = len(self.sdev)
        n, k = self.rotation.shape
        sd = " ".join(f"{v:.{7}g}" for v in self.sdev)
        rot = pd.DataFrame(self.rotation, index=[f"[{i + 1},]" for i in range(n)],
                           columns=[f"[,{j + 1}]" for j in range(k)])
        return (f"Standard deviations (1, .., p={p}):\n[1] {sd}\n\n"
                f"Rotation (n x k) = ({n} x {k}):\n{rot.to_string()}")

    __repr__ = __str__


class PrCompList(dict):
    """class ``prcomplist``: a named list of :class:`PrComp`, one per grouping factor."""

    def summary(self):
        """summary.prcomplist: ``lapply(object, summary)``."""
        return {k: v.summary() for k, v in self.items()}

    def __str__(self):
        return "\n\n".join(f"${k}\n{v}" for k, v in self.items())

    __repr__ = __str__


def _svals(m):
    # svd(m, nv = 0L): d and u (LAPACK dgesdd, as numpy)
    u, d, _ = np.linalg.svd(np.asarray(m, dtype=float), full_matrices=False)
    return PrComp(d, u)


def rePCA(x):
    """rePCA.merMod: PCA of the random-effects covariance of a fitted model.

    Returns a :class:`PrCompList` keyed by grouping factor; terms sharing a
    grouping factor (e.g. ``(1|g) + (0 + x|g)``) are combined block-diagonally,
    as ``Matrix::bdiag(chfs[which(nms == m)])`` in R."""
    # getME(x, "Tlist") is a list whose names may repeat; build it from the
    # covariance objects so that repeated grouping factors are not lost
    chfs = [np.atleast_2d(np.asarray(c.getLambda().toarray() if hasattr(c.getLambda(), "toarray")
                                     else c.getLambda(), dtype=float))
            for c in x.getReCovs()]
    nms = [g for g, _ in x.cnms]
    unms = list(dict.fromkeys(nms))
    return PrCompList((m, _svals(block_diag(*[chfs[i] for i, n in enumerate(nms) if n == m])))
                      for m in unms)
