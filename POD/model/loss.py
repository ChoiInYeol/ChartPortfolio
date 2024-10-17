# loss.py
import torch
import torch.nn.functional as F
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

import numpy as np
import logging

logger = logging.getLogger(__name__)

def batch_covariance(y_return):
    # y_return: [batch_size, time_steps, num_features]
    y_return = y_return - y_return.mean(dim=1, keepdim=True)  # Centering
    cov = torch.einsum('bti,btj->bij', y_return, y_return) / (y_return.shape[1] - 1)
    # cov: [batch_size, num_features, num_features]
    return cov

def elastic_net_selection(y_return, alpha=0.1, l1_ratio=0.5, target_k=50, max_iter=10000):
    """
    Elastic Net을 사용하여 상위 K개의 자산을 선택합니다.

    Args:
        y_return (torch.Tensor): [batch_size, time_steps, num_features]
        alpha (float): 규제 강도
        l1_ratio (float): L1과 L2 규제의 비율
        target_k (int): 선택할 자산 수
        max_iter (int): 최대 반복 횟수

    Returns:
        torch.Tensor: 선택된 자산의 인덱스 [batch_size, k]
    """
    batch_size, time_steps, num_features = y_return.shape
    top_k_indices = []

    scaler = StandardScaler()  # 스케일러 초기화

    for i in range(batch_size):
        # 스케일링 적용
        X = scaler.fit_transform(y_return[i].cpu().numpy())  # [time_steps, num_features]
        y = np.ones(time_steps)  # 타겟 벡터

        # Elastic Net 모델 학습
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, fit_intercept=False)
        model.fit(X, y)

        # 중요도가 높은 자산 상위 K개 선택
        abs_coef = np.abs(model.coef_)
        top_k = np.argsort(abs_coef)[-target_k:]  # 상위 K개 인덱스
        top_k_indices.append(top_k)

    return torch.tensor(top_k_indices, device=y_return.device)

def max_sharpe(y_return, weights, alpha=0.1, l1_ratio=0.5, k=50, max_iter=10000):
    """
    Elastic Net을 사용하여 상위 K개의 자산에 대해 Sharpe 비율 최적화.

    Args:
        y_return (torch.Tensor): [batch_size, time_steps, 4586]
        weights (torch.Tensor): [batch_size, k]

    Returns:
        torch.Tensor: 손실 (negative Sharpe ratio)
    """
    # Elastic Net을 사용해 상위 K개 자산 선택
    top_k_indices = elastic_net_selection(y_return, alpha=alpha, l1_ratio=l1_ratio, target_k=k, max_iter=max_iter)

    # 상위 K개 자산의 수익률 추출
    batch_size = y_return.shape[0]
    top_k_returns = torch.stack([y_return[i, :, top_k_indices[i]] for i in range(batch_size)])  # [batch_size, time_steps, k]

    # 평균 수익률과 공분산 계산
    mean_return = top_k_returns.mean(dim=1)  # [batch_size, k]
    covmat = batch_covariance(top_k_returns)  # [batch_size, k, k]

    # 포트폴리오 수익률과 변동성 계산
    port_return = (weights * mean_return).sum(dim=1)  # [batch_size]
    weights_unsqueezed = weights.unsqueeze(1)  # [batch_size, 1, k]
    port_vol = torch.bmm(weights_unsqueezed, torch.bmm(covmat, weights_unsqueezed.transpose(1, 2))).squeeze()  # [batch_size]
    port_vol = torch.sqrt(port_vol + 1e-8)  # [batch_size]

    # 연율화 조정
    scale = 12  # Assuming monthly returns
    port_return = port_return * scale
    port_vol = port_vol * np.sqrt(scale)

    # 샤프 비율 계산
    risk_free_rate = 0.02
    sharpe_ratio = (port_return - risk_free_rate) / port_vol  # [batch_size]

    # 손실은 샤프 비율의 음수
    loss = -sharpe_ratio.mean()
    return loss


def equal_risk_parity(y_return, weights):
    # y_return: [batch_size, time_steps, num_features]
    # weights: [batch_size, num_features]

    covmat = batch_covariance(y_return)  # [batch_size, num_features, num_features]
    weights_unsqueezed = weights.unsqueeze(1)  # [batch_size, 1, num_features]

    # Portfolio variance
    port_var = torch.bmm(weights_unsqueezed, torch.bmm(covmat, weights_unsqueezed.transpose(1, 2))).squeeze()  # [batch_size]
    sigma_p = torch.sqrt(port_var + 1e-8)  # [batch_size]

    # Marginal Risk Contribution
    sigma_w = torch.bmm(covmat, weights_unsqueezed.transpose(1, 2)).squeeze()  # [batch_size, num_features]
    mrc = sigma_w / sigma_p.unsqueeze(1)  # [batch_size, num_features]

    # Risk Contribution
    rc = weights * mrc  # [batch_size, num_features]
    rc = rc / rc.sum(dim=1, keepdim=True)  # Normalize

    # Target Risk Contribution (equal weight)
    target_rc = torch.full_like(rc, 1.0 / rc.shape[1])  # [batch_size, num_features]

    # Loss is the mean squared difference between actual and target risk contributions
    loss = ((rc - target_rc) ** 2).sum(dim=1).mean()
    return loss

def mean_variance(y_return, weights, risk_aversion=1.0):
    # y_return: [batch_size, time_steps, num_features]
    # weights: [batch_size, num_features]

    mean_return = y_return.mean(dim=1)  # [batch_size, num_features]
    covmat = batch_covariance(y_return)  # [batch_size, num_features, num_features]

    # Portfolio return
    port_return = (weights * mean_return).sum(dim=1)  # [batch_size]

    # Portfolio variance
    weights_unsqueezed = weights.unsqueeze(1)  # [batch_size, 1, num_features]
    port_variance = torch.bmm(weights_unsqueezed, torch.bmm(covmat, weights_unsqueezed.transpose(1, 2))).squeeze()  # [batch_size]

    # Objective function: maximize return - risk_aversion * variance
    objective = port_return - risk_aversion * port_variance  # [batch_size]

    # Loss is negative of the objective (since we minimize in training)
    loss = -objective.mean()
    return loss

def binary_prediction_loss(binary_true, binary_pred):
    return F.binary_cross_entropy(binary_pred, binary_true)

def combined_loss(y_true, y_pred, binary_true=None, binary_pred=None, beta=0.2, eps=1e-8):
    portfolio_loss = max_sharpe(y_true, y_pred)

    if binary_true is not None and binary_pred is not None:
        binary_pred = torch.clamp(binary_pred, min=eps, max=1 - eps)
        binary_true = binary_true.float()
        binary_loss = binary_prediction_loss(binary_true, binary_pred)
        total_loss = beta * portfolio_loss + (1 - beta) * binary_loss
    else:
        total_loss = portfolio_loss

    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.warning(f"Total loss is invalid: {total_loss}")
        total_loss = torch.tensor(1e6, device=total_loss.device, dtype=total_loss.dtype)
    
    return total_loss
