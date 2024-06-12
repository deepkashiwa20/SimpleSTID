import numpy as np
import torch

def getMetric(y_pred, y_true):
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
        
    rmse = masked_rmse(y_pred, y_true)
    mae = masked_mae(y_pred, y_true)
    mape = masked_mape(y_pred, y_true)

    return rmse, mae, mape

def masked_mse(preds, labels, null_val=0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=0):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)



class myLoss:
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=1e-3):
        return masked_mae(preds, labels, null_val)