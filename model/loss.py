import torch
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

def batch_covariance(x):
    x = x - x.mean(dim=1, keepdim=True)
    cov = torch.bmm(x.transpose(1, 2), x) / (x.shape[1] - 1)
    return cov

def max_sharpe(y_return, weights):
    mean_return = y_return.mean(dim=1)
    covmat = batch_covariance(y_return)

    port_return = (weights * mean_return).sum(dim=1)
    weights_unsqueezed = weights.unsqueeze(1)
    port_vol = torch.bmm(weights_unsqueezed, torch.bmm(covmat, weights_unsqueezed.transpose(1, 2))).squeeze()

    risk_free_rate = 0.02
    scale = 12
    port_return = port_return * scale
    port_vol = torch.sqrt(port_vol * scale + 1e-8)

    sharpe_ratio = (port_return - risk_free_rate) / port_vol
    return -sharpe_ratio.mean()

def equal_risk_parity(y_return, weights):
    covmat = batch_covariance(y_return)
    weights_unsqueezed = weights.unsqueeze(1)

    port_var = torch.bmm(weights_unsqueezed, torch.bmm(covmat, weights_unsqueezed.transpose(1, 2))).squeeze()
    sigma_p = torch.sqrt(port_var + 1e-8)

    sigma_w = torch.bmm(covmat, weights_unsqueezed.transpose(1, 2)).squeeze()
    mrc = (1 / sigma_p.unsqueeze(1)) * sigma_w

    rc = weights * mrc
    rc = rc / rc.sum(dim=1, keepdim=True)

    target_rc = torch.ones_like(rc) / rc.shape[1]
    loss = ((rc - target_rc) ** 2).sum(dim=1).mean()
    return loss

def mean_variance(y_return, weights, risk_aversion=1.0):
    mean_return = y_return.mean(dim=1)
    covmat = batch_covariance(y_return)

    port_return = (weights * mean_return).sum(dim=1)
    weights_unsqueezed = weights.unsqueeze(1)
    port_variance = torch.bmm(weights_unsqueezed, torch.bmm(covmat, weights_unsqueezed.transpose(1, 2))).squeeze()

    objective = port_return - risk_aversion * port_variance
    return -objective.mean()

def l2_regularization(weights, lambda_reg=0.1):
    l2_loss = lambda_reg * (weights ** 2).sum(dim=1).mean()
    return l2_loss

def binary_prediction_loss(binary_true, binary_pred):
    return F.binary_cross_entropy(binary_pred, binary_true)

def combined_loss(y_true, y_pred, binary_true=None, binary_pred=None, beta=0.5, eps=1e-8):
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
