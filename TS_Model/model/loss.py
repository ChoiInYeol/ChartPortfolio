import torch
import numpy as np
import logging
from typing import Optional, Tuple
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def batch_covariance(x: torch.Tensor) -> torch.Tensor:
    """
    배치 공분산 계산

    Args:
        x: (batch_size, n_samples, n_features)

    Returns:
        cov: (batch_size, n_features, n_features)
    """
    x_mean = x.mean(dim=1, keepdim=True)
    x_centered = x - x_mean
    cov = torch.matmul(x_centered.transpose(1, 2), x_centered) / (x.size(1) - 1)
    return cov

def max_sharpe(
    returns: torch.Tensor,
    weights: torch.Tensor,
    risk_free_rate: float = 0.02,
    annualization_factor: float = 12
) -> torch.Tensor:
    """Sharpe ratio 손실 함수"""
    weights = weights.unsqueeze(1)  # (batch_size, 1, n_assets)
    mean_return = returns.mean(dim=1).unsqueeze(2)  # (batch_size, n_assets, 1)
    covmat = batch_covariance(returns)  # (batch_size, n_assets, n_assets)

    port_return = torch.matmul(weights, mean_return)  # (batch_size, 1, 1)
    port_vol = torch.matmul(
        weights, torch.matmul(covmat, weights.transpose(2, 1))
    )  # (batch_size, 1, 1)

    sharpe_ratio = (port_return * annualization_factor - risk_free_rate) / (
        torch.sqrt(port_vol * annualization_factor)
    )
    return -sharpe_ratio.mean()


def mean_variance(
    returns: torch.Tensor,
    weights: torch.Tensor,
    risk_aversion: float = 1.0
) -> torch.Tensor:
    """평균-분산 손실 함수"""
    weights = weights.unsqueeze(1)  # (batch_size, 1, n_assets)
    mean_return = returns.mean(dim=1).unsqueeze(2)  # (batch_size, n_assets, 1)
    covmat = batch_covariance(returns)  # (batch_size, n_assets, n_assets)

    port_return = torch.matmul(weights, mean_return)  # (batch_size, 1, 1)
    port_var = torch.matmul(
        weights, torch.matmul(covmat, weights.transpose(2, 1))
    )  # (batch_size, 1, 1)

    utility = port_return - (risk_aversion / 2) * port_var
    return -utility.mean()


def min_variance(
    returns: torch.Tensor,
    weights: torch.Tensor
) -> torch.Tensor:
    """최소 분산 손실 함수"""
    weights = weights.unsqueeze(1)  # (batch_size, 1, n_assets)
    covmat = batch_covariance(returns)  # (batch_size, n_assets, n_assets)

    port_var = torch.matmul(
        weights, torch.matmul(covmat, weights.transpose(2, 1))
    )  # (batch_size, 1, 1)
    return port_var.mean()

def equal_risk_contribution_loss(
    returns: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """동일 위험 기여도 손실 함수"""
    batch_size, n_samples, n_assets = returns.size()
    weights = weights.unsqueeze(1)  # (batch_size, 1, n_assets)
    covmat = batch_covariance(returns)  # (batch_size, n_assets, n_assets)

    port_vol = torch.sqrt(
        torch.matmul(weights, torch.matmul(covmat, weights.transpose(2, 1)))
    )  # (batch_size, 1, 1)
    mrc = (1 / port_vol) * torch.matmul(covmat, weights.transpose(2, 1))  # (batch_size, n_assets, 1)
    rc = weights.transpose(2, 1) * mrc  # (batch_size, n_assets, 1)
    rc = rc.squeeze(2)  # (batch_size, n_assets)

    target = torch.ones_like(rc) * (1 / n_assets)  # (batch_size, n_assets)
    risk_diffs = rc - target
    sum_risk_diffs_squared = torch.mean(risk_diffs.pow(2))
    return sum_risk_diffs_squared

# 기존 인터페이스와의 호환성을 위한 별칭
sharpe_ratio_loss = max_sharpe
mean_variance_loss = mean_variance
minimum_variance_loss = min_variance

if __name__ == "__main__":
    pass