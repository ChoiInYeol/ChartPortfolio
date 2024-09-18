import torch
import numpy as np
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

def max_sharpe(y_return, weights, labels, binary_pred, beta):
    weights = torch.unsqueeze(weights, 1)
    meanReturn = torch.unsqueeze(torch.mean(y_return, axis=1), 2)
    covmat = torch.Tensor([np.cov(batch.cpu().T, ddof=0) for batch in y_return]).to("cuda")
    portReturn = torch.matmul(weights, meanReturn)
    portVol = torch.matmul(
        weights, torch.matmul(covmat, torch.transpose(weights, 2, 1))
    )
    objective = (portReturn * 12 - 0.02) / (torch.sqrt(portVol * 12))
    return -objective.mean()


def equal_risk_parity(y_return, weights):
    B = y_return.shape[0]
    F = y_return.shape[2]
    weights = torch.unsqueeze(weights, 1).to("cuda")
    covmat = torch.Tensor(
        [np.cov(batch.cpu().T, ddof=0) for batch in y_return]
    )  # (batch, 50, 50)
    covmat = covmat.to("cuda")
    sigma = torch.sqrt(
        torch.matmul(weights, torch.matmul(covmat, torch.transpose(weights, 2, 1)))
    )
    mrc = (1 / sigma) * (covmat @ torch.transpose(weights, 2, 1))
    rc = weights.view(B, F) * mrc.view(B, F)
    target = (torch.ones((B, F)) * (1 / F)).to("cuda")
    risk_diffs = rc - target
    sum_risk_diffs_squared = torch.mean(torch.square(risk_diffs))
    return sum_risk_diffs_squared

def binary_prediction_loss(binary_true, binary_pred):
    return F.binary_cross_entropy(binary_pred, binary_true)

def combined_loss(y_true, y_pred, binary_true, binary_pred, beta=0.5, eps=1e-8):
    portfolio_loss = max_sharpe(y_true, y_pred, binary_true, binary_pred, beta)
    
    binary_pred = torch.clamp(binary_pred, min=eps, max=1-eps)
    binary_true = binary_true.float()
    binary_loss = binary_prediction_loss(binary_true, binary_pred)
    
    total_loss = beta * portfolio_loss + (1 - beta) * (binary_loss)
    
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.warning(f"Total loss is invalid: {total_loss}")
        total_loss = torch.where(torch.isnan(total_loss) | torch.isinf(total_loss),
                                 torch.tensor(1e6, device=total_loss.device, dtype=total_loss.dtype),
                                 total_loss)
    
    return total_loss

if __name__ == "__main__":
    pass
