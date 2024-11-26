import torch
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def batch_covariance(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    배치의 공분산 행렬을 계산합니다.

    Args:
        returns (torch.Tensor): 수익률 텐서 [batch_size, time_steps, n_stocks]
        eps (float): 수치 안정성을 위한 작은 값

    Returns:
        torch.Tensor: 공분산 행렬 [batch_size, n_stocks, n_stocks]
    """
    B, T, N = returns.shape
    returns_centered = returns - returns.mean(dim=1, keepdim=True)
    cov = torch.bmm(returns_centered.transpose(1, 2), returns_centered) / (T - 1)
    # 수치 안정성을 위해 대각선에 작은 값 추가
    cov = cov + torch.eye(N, device=returns.device) * eps
    return cov

def max_sharpe(returns: torch.Tensor, weights: torch.Tensor, risk_free_rate: float = 0.02) -> torch.Tensor:
    """
    Sharpe ratio를 최대화하는 손실 함수입니다.

    Args:
        returns (torch.Tensor): 수익률 텐서 [batch_size, time_steps, n_stocks]
        weights (torch.Tensor): 포트폴리오 가중치 [batch_size, n_stocks]
        risk_free_rate (float): 무위험 수익률 (연율화)

    Returns:
        torch.Tensor: negative Sharpe ratio (손실값)
    """
    # 입력값 검증
    if torch.isnan(returns).any() or torch.isnan(weights).any():
        logger.warning("NaN detected in inputs")
        return torch.tensor(1e6, device=returns.device, dtype=returns.dtype, requires_grad=True)

    weights = weights.unsqueeze(1)  # [B, 1, N]
    
    # 연간 기대수익률 계산
    mean_return = returns.mean(dim=1, keepdim=True).transpose(1, 2)  # [B, N, 1]
    port_return = torch.bmm(weights, mean_return).squeeze()  # [B, 1]
    annual_return = port_return * 12  # 월간 -> 연간
    
    # 연간 변동성 계산
    covmat = batch_covariance(returns)  # [B, N, N]
    port_var = torch.bmm(weights, torch.bmm(covmat, weights.transpose(1, 2))).squeeze()  # [B, 1]
    port_vol = torch.sqrt(port_var * 12 + 1e-8)  # 월간 -> 연간, 수치 안정성
    
    # Sharpe ratio 계산
    sharpe = (annual_return - risk_free_rate) / port_vol
    
    return -sharpe.mean()  # 최대화를 위해 음수 반환

def mean_variance(returns: torch.Tensor, weights: torch.Tensor, risk_aversion: float = 1.0) -> torch.Tensor:
    """
    평균-분산 최적화를 위한 손실 함수입니다.

    Args:
        returns (torch.Tensor): 수익률 텐서 [batch_size, time_steps, n_stocks]
        weights (torch.Tensor): 포트폴리오 가중치 [batch_size, n_stocks]
        risk_aversion (float): 위험 회피 계수

    Returns:
        torch.Tensor: negative utility (손실값)
    """
    weights = weights.unsqueeze(1)  # [B, 1, N]
    
    # 기대수익률 계산
    mean_return = returns.mean(dim=1, keepdim=True).transpose(1, 2)  # [B, N, 1]
    port_return = torch.bmm(weights, mean_return).squeeze()  # [B, 1]
    
    # 포트폴리오 분산 계산
    covmat = batch_covariance(returns)  # [B, N, N]
    port_var = torch.bmm(weights, torch.bmm(covmat, weights.transpose(1, 2))).squeeze()  # [B, 1]
    
    # 효용함수 = 기대수익률 - (위험회피계수/2) * 분산
    utility = port_return - (risk_aversion / 2) * port_var
    
    return -utility.mean()

# 기존 인터페이스와의 호환성을 위한 별칭
sharpe_ratio_loss = max_sharpe
mean_variance_loss = mean_variance
minimum_variance_loss = lambda returns, weights: mean_variance(returns, weights, risk_aversion=1e6)

if __name__ == "__main__":
    pass