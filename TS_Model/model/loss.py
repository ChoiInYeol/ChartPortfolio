import torch
import numpy as np
import logging

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

def max_sharpe(y_return: torch.Tensor, weights: torch.Tensor, risk_free_rate: float = 0.02) -> torch.Tensor:
    """
    Sharpe ratio를 최대화하는 손실 함수입니다.

    Args:
        y_return (torch.Tensor): 수익률 텐서 [batch_size, time_steps, n_stocks]
        weights (torch.Tensor): 포트폴리오 가중치 [batch_size, n_stocks]
        risk_free_rate (float): 무위험 수익률 (연율화)

    Returns:
        torch.Tensor: negative Sharpe ratio (손실값)
    """
    # 입력값 검증
    if torch.isnan(y_return).any() or torch.isnan(weights).any():
        logger.warning("NaN detected in inputs")
        return torch.tensor(1e6, device=y_return.device, dtype=y_return.dtype, requires_grad=True)

    weights = weights.unsqueeze(1)  # [B, 1, N]
    
    # 연간 기대수익률 계산
    mean_return = y_return.mean(dim=1, keepdim=True).transpose(1, 2)  # [B, N, 1]
    port_return = torch.bmm(weights, mean_return).squeeze()  # [B, 1]
    annual_return = port_return * 12  # 월간 -> 연간
    
    # 연간 변동성 계산
    covmat = batch_covariance(y_return)  # [B, N, N]
    port_var = torch.bmm(weights, torch.bmm(covmat, weights.transpose(1, 2))).squeeze()  # [B, 1]
    port_vol = torch.sqrt(port_var * 12 + 1e-8)  # 월간 -> 연간, 수치 안정성
    
    # Sharpe ratio 계산
    sharpe = (annual_return - risk_free_rate) / port_vol
    
    return -sharpe.mean()  # 최대화를 위해 음수 반환

def equal_risk_parity(y_return: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    동일 위험 기여도를 목표로 하는 손실 함수입니다.

    Args:
        y_return (torch.Tensor): 수익률 텐서 [batch_size, time_steps, n_stocks]
        weights (torch.Tensor): 포트폴리오 가중치 [batch_size, n_stocks]

    Returns:
        torch.Tensor: 위험 기여도 차이의 MSE
    """
    B, T, N = y_return.shape
    weights = weights.unsqueeze(1)  # [B, 1, N]
    
    # 공분산 행렬 계산
    covmat = batch_covariance(y_return)  # [B, N, N]
    
    # 포트폴리오 변동성
    port_var = torch.bmm(weights, torch.bmm(covmat, weights.transpose(1, 2)))  # [B, 1, 1]
    port_vol = torch.sqrt(port_var + 1e-8)  # 수치 안정성
    
    # 한계 위험 기여도
    mrc = torch.bmm(covmat, weights.transpose(1, 2)) / port_vol  # [B, N, 1]
    
    # 위험 기여도
    rc = (weights * mrc.transpose(1, 2)).squeeze()  # [B, N]
    target_rc = torch.full_like(rc, 1.0 / N)  # 목표 위험 기여도
    
    return torch.mean((rc - target_rc) ** 2)

def mean_variance(y_return: torch.Tensor, weights: torch.Tensor, risk_aversion: float = 1.0) -> torch.Tensor:
    """
    평균-분산 최적화를 위한 손실 함수입니다.

    Args:
        y_return (torch.Tensor): 수익률 텐서 [batch_size, time_steps, n_stocks]
        weights (torch.Tensor): 포트폴리오 가중치 [batch_size, n_stocks]
        risk_aversion (float): 위험 회피 계수

    Returns:
        torch.Tensor: negative utility (손실값)
    """
    weights = weights.unsqueeze(1)  # [B, 1, N]
    
    # 기대수익률 계산
    mean_return = y_return.mean(dim=1, keepdim=True).transpose(1, 2)  # [B, N, 1]
    port_return = torch.bmm(weights, mean_return).squeeze()  # [B, 1]
    
    # 포트폴리오 분산 계산
    covmat = batch_covariance(y_return)  # [B, N, N]
    port_var = torch.bmm(weights, torch.bmm(covmat, weights.transpose(1, 2))).squeeze()  # [B, 1]
    
    # 효용함수 = 기대수익률 - (위험회피계수/2) * 분산
    utility = port_return - (risk_aversion / 2) * port_var
    
    return -utility.mean()

def portfolio_return(returns: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """포트폴리오 수익률을 계산합니다.
    
    Args:
        returns: 수익률 텐서 [batch_size, time_steps, n_stocks]
        weights: 포트폴리오 가중치 [batch_size, n_stocks]
        
    Returns:
        포트폴리오 수익률 [batch_size]
    """
    # weights를 [batch_size, 1, n_stocks]로 확장
    weights = weights.unsqueeze(1)
    
    # 포트폴리오 수익률 계산 [batch_size, time_steps]
    portfolio_rets = torch.bmm(weights, returns.transpose(1, 2)).squeeze(1)
    
    return portfolio_rets

def portfolio_variance(returns: torch.Tensor, weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """포트폴리오 분산을 계산합니다.
    
    Args:
        returns: 수익률 텐서 [batch_size, time_steps, n_stocks]
        weights: 포트폴리오 가중치 [batch_size, n_stocks]
        eps: 수치 안정성을 위한 작은 값
        
    Returns:
        포트폴리오 분산 [batch_size]
    """
    # 공분산 행렬 계산
    B, T, N = returns.shape
    returns_centered = returns - returns.mean(dim=1, keepdim=True)
    cov = torch.bmm(returns_centered.transpose(1, 2), returns_centered) / (T - 1)
    
    # 수치 안정성을 위해 대각선에 작은 값 추가
    cov = cov + torch.eye(N, device=returns.device) * eps
    
    # 포트폴리오 분산 계산
    weights = weights.unsqueeze(2)  # [B, N, 1]
    port_var = torch.bmm(torch.bmm(weights.transpose(1, 2), cov), weights).squeeze()
    
    return port_var

def mean_variance_objective(
    returns: torch.Tensor,
    weights: torch.Tensor,
    risk_aversion: float = 1.0
) -> torch.Tensor:
    """평균-분산 목적함수를 계산합니다.
    
    Args:
        returns: 수익률 텐서 [batch_size, time_steps, n_stocks]
        weights: 포트폴리오 가중치 [batch_size, n_stocks]
        risk_aversion: 위험 회피 계수
        
    Returns:
        negative utility (손실값)
    """
    port_return = portfolio_return(returns, weights).mean(dim=1)
    port_var = portfolio_variance(returns, weights)
    
    utility = port_return - (risk_aversion / 2) * port_var
    return -utility.mean()

def sharpe_ratio_objective(
    returns: torch.Tensor,
    weights: torch.Tensor,
    risk_free_rate: float = 0.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """Sharpe ratio 목적함수를 계산합니다.
    
    Args:
        returns: 수익률 텐서 [batch_size, time_steps, n_stocks]
        weights: 포트폴리오 가중치 [batch_size, n_stocks]
        risk_free_rate: 무위험 수익률
        eps: 수치 안정성을 위한 작은 값
        
    Returns:
        negative Sharpe ratio (손실값)
    """
    port_return = portfolio_return(returns, weights).mean(dim=1)
    port_std = torch.sqrt(portfolio_variance(returns, weights) + eps)
    
    sharpe = (port_return - risk_free_rate) / port_std
    return -sharpe.mean()

def minimum_variance_objective(
    returns: torch.Tensor,
    weights: torch.Tensor
) -> torch.Tensor:
    """최소 분산 목적함수를 계산합니다.
    
    Args:
        returns: 수익률 텐서 [batch_size, time_steps, n_stocks]
        weights: 포트폴리오 가중치 [batch_size, n_stocks]
        
    Returns:
        포트폴리오 분산 (손실값)
    """
    return portfolio_variance(returns, weights).mean()

if __name__ == "__main__":
    pass