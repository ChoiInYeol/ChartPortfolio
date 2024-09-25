# loss.py
import torch
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

def batch_covariance(y_return):
    # y_return: [batch_size, time_steps, num_features]
    y_return = y_return - y_return.mean(dim=1, keepdim=True)  # Centering
    cov = torch.einsum('bti,btj->bij', y_return, y_return) / (y_return.shape[1] - 1)
    # cov: [batch_size, num_features, num_features]
    return cov

def max_sharpe(y_return, weights):
    # y_return: [batch_size, time_steps, num_features]
    # weights: [batch_size, num_features]
    
    mean_return = y_return.mean(dim=1)  # [batch_size, num_features]
    covmat = batch_covariance(y_return)  # [batch_size, num_features, num_features]

    # Portfolio return
    port_return = (weights * mean_return).sum(dim=1)  # [batch_size]

    # Portfolio volatility
    weights_unsqueezed = weights.unsqueeze(1)  # [batch_size, 1, num_features]
    port_vol = torch.bmm(weights_unsqueezed, torch.bmm(covmat, weights_unsqueezed.transpose(1, 2))).squeeze()  # [batch_size]
    port_vol = torch.sqrt(port_vol + 1e-8)  # [batch_size]

    # Adjust for annualization if needed
    scale = 12  # Assuming monthly returns
    port_return = port_return * scale
    port_vol = port_vol * np.sqrt(scale)

    # Sharpe Ratio
    risk_free_rate = 0.02
    sharpe_ratio = (port_return - risk_free_rate) / port_vol  # [batch_size]

    # Loss is negative Sharpe Ratio
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
