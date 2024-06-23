import numpy as np
import pandas as pd
import torch
import os


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
    if torch.isnan(torch.tensor(null_val)):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    # Avoid nan error.
    labels_safe = torch.where(labels == 0, torch.ones_like(labels), labels)
    loss = torch.abs(preds-labels)/labels_safe
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)



class myLoss:
    def __init__(self, name):
        if name.upper() not in ["MAE", "MSE", "RMSE", "MAPE", "HYBRID"]:
            raise NotImplementedError
        else:
            self.name = name.upper()
            
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=1e-3):
        if self.name == "MAE":
            return masked_mae(preds, labels, null_val)
        elif self.name == "MSE":
            return masked_mse(preds, labels, null_val)
        elif self.name == "RMSE":
            return masked_rmse(preds, labels, null_val)
        elif self.name == "MAPE":
            return masked_mape(preds, labels, null_val)
        elif self.name == "HYBRID":
            return 0.05 * masked_mae(preds, labels, null_val) + masked_mape(preds, labels, null_val)
        
        

def getMetric(y_pred, y_true, null_val=0):
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
        
    rmse = masked_rmse(y_pred, y_true, null_val)
    mae = masked_mae(y_pred, y_true, null_val)
    mape = masked_mape(y_pred, y_true, null_val)

    return rmse, mae, mape
    
