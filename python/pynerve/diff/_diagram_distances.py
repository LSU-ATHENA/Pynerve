"""Differentiable persistence diagram distances and kernels."""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812

import torch
from torch import nn

from .._validation import validate_finite_tensor as _validate_finite_tensor
from ._loss_helpers import _persistence_values, _validate_positive_scalar


class PersistenceLoss(nn.Module):
    """Smooth persistence-diagram distances and kernels."""

    @staticmethod
    def softmin(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
        """Differentiable softmin using temperature-scaled softmax."""
        _validate_finite_tensor(x, "x")
        temperature = _validate_positive_scalar("temperature", temperature)
        return -temperature * torch.logsumexp(-x / temperature, dim=dim)

    @staticmethod
    def softmax(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
        """Temperature-scaled softmax for soft assignments."""
        _validate_finite_tensor(x, "x")
        temperature = _validate_positive_scalar("temperature", temperature)
        return F.softmax(x / temperature, dim=dim)

    @classmethod
    def diagram_wasserstein(
        cls,
        diagram1: torch.Tensor,
        diagram2: torch.Tensor,
        p: float = 2.0,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """Differentiable Sinkhorn approximation of the p-Wasserstein distance.

        :param diagram1: First persistence diagram of shape ``(N, 2)`` or ``(N, 3)``.
        :param diagram2: Second persistence diagram of shape ``(M, 2)`` or ``(M, 3)``.
        :param p: Order of the Wasserstein distance (``p >= 1``).
        :param temperature: Softness parameter controlling the entropy regularization
            strength. Lower values yield a tighter approximation.
        :returns: Scalar tensor approximating the p-Wasserstein distance.
        :raises ValueError: If ``p`` or ``temperature`` is non-positive.
        :raises ValueError: If any diagram birth/death coordinate is non-finite
            or any death is less than its birth.
        """
        p = _validate_positive_scalar("p", p)
        temperature = _validate_positive_scalar("temperature", temperature)
        pers1 = _persistence_values(diagram1)
        pers2 = _persistence_values(diagram2)

        n1, n2 = diagram1.shape[0], diagram2.shape[0]
        if n1 == 0 and n2 == 0:
            return diagram1.new_zeros(())
        if n1 == 0:
            return torch.abs(diagram2[:, 1] - diagram2[:, 0]).sum() / 2
        if n2 == 0:
            return torch.abs(diagram1[:, 1] - diagram1[:, 0]).sum() / 2

        diff = diagram1[:, :2].unsqueeze(1) - diagram2[:, :2].unsqueeze(0)

        if p == 1:
            C = diff.abs().sum(dim=-1)  # noqa: N806  # [n1, n2]
        elif p == 2:
            C = (diff**2).sum(dim=-1).sqrt()  # noqa: N806  # [n1, n2]
        else:
            C = (diff.abs() ** p).sum(dim=-1) ** (1.0 / p)  # noqa: N806

        diag_costs1 = pers1 / 2
        diag_costs2 = pers2 / 2

        mu = diagram1.new_ones(n1) / n1
        nu = diagram1.new_ones(n2) / n2

        K = torch.exp(-C / temperature)  # noqa: N806

        u = diagram1.new_ones(n1)
        v = diagram1.new_ones(n2)
        u = mu / (K @ v + 1e-8)
        v = nu / (K.T @ u + 1e-8)

        P = u.unsqueeze(1) * K * v.unsqueeze(0)  # noqa: N806
        w_dist = (P * C).sum()

        diag_cost = (diag_costs1.sum() + diag_costs2.sum()) / max(n1, n2)
        return w_dist + diag_cost * 0.1

    @classmethod
    def diagram_bottleneck(
        cls, diagram1: torch.Tensor, diagram2: torch.Tensor, temperature: float = 0.01
    ) -> torch.Tensor:
        """Differentiable approximation of the bottleneck distance.

        Uses a soft-min/max relaxation to produce a smooth, differentiable
        surrogate of the classical bottleneck distance.

        :param diagram1: First persistence diagram of shape ``(N, 2)`` or ``(N, 3)``.
        :param diagram2: Second persistence diagram of shape ``(M, 2)`` or ``(M, 3)``.
        :param temperature: Softness parameter controlling the min/max relaxation.
            Lower values yield a tighter approximation.
        :returns: Scalar tensor approximating the bottleneck distance.
        :raises ValueError: If ``temperature`` is non-positive.
        :raises ValueError: If any diagram birth/death coordinate is non-finite
            or any death is less than its birth.
        """
        temperature = _validate_positive_scalar("temperature", temperature)
        _persistence_values(diagram1)
        _persistence_values(diagram2)
        if diagram1.shape[0] == 0 and diagram2.shape[0] == 0:
            return diagram1.new_zeros(())
        if diagram1.shape[0] == 0:
            return torch.abs(diagram2[:, 1] - diagram2[:, 0]).max() / 2
        if diagram2.shape[0] == 0:
            return torch.abs(diagram1[:, 1] - diagram1[:, 0]).max() / 2

        diff = diagram1[:, :2].unsqueeze(1) - diagram2[:, :2].unsqueeze(0)
        dists = (diff**2).sum(dim=-1).sqrt()

        min_dists = -cls.softmin(-dists, dim=1, temperature=temperature)
        return min_dists.max()

    @staticmethod
    def persistence_kernel(
        diagram1: torch.Tensor, diagram2: torch.Tensor, sigma: float = 1.0
    ) -> torch.Tensor:
        """Persistence-weighted Gaussian kernel between two diagrams.

        Computes :math:`\\sum_{a \\in D_1} \\sum_{b \\in D_2}
        \\sqrt{\\mathrm{pers}(a) \\cdot \\mathrm{pers}(b)} \\cdot
        \\exp(-\\frac{\\|a-b\\|^2}{2\\sigma^2})`.

        :param diagram1: First persistence diagram of shape ``(N, 2)`` or ``(N, 3)``.
        :param diagram2: Second persistence diagram of shape ``(M, 2)`` or ``(M, 3)``.
        :param sigma: Bandwidth of the Gaussian kernel.
        :returns: Scalar kernel value. Returns zero when either diagram is empty.
        :raises ValueError: If ``sigma`` is non-positive.
        :raises ValueError: If any diagram birth/death coordinate is non-finite
            or any death is less than its birth.
        """
        sigma = _validate_positive_scalar("sigma", sigma)
        pers1 = _persistence_values(diagram1)
        pers2 = _persistence_values(diagram2)
        if diagram1.shape[0] == 0 or diagram2.shape[0] == 0:
            return diagram1.new_zeros(())

        diff = diagram1[:, :2].unsqueeze(1) - diagram2[:, :2].unsqueeze(0)
        sq_dist = (diff**2).sum(dim=-1)

        K = torch.exp(-sq_dist / (2 * sigma**2))  # noqa: N806
        W = (pers1.unsqueeze(1) * pers2.unsqueeze(0)).sqrt()  # noqa: N806
        return (K * W).sum()
