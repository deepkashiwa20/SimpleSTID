import argparse
import os
import sys
import logging
import gpustat
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torchinfo import summary
from utils import *
from dataProcess import *

import STID
# import pickle

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), f'{MODEL_NAME}_checkpoint.pt')
        self.val_loss_min = val_loss


@torch.no_grad()
def model_evaluate(model, dataLoader, criterion, debug=False):
    model.eval()
    
    batch_losses = []
    y_true, y_pred = [], []
    for x_batch, y_batch in dataLoader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch)
        batch_losses.append(loss.item())

        if not debug:
            continue
        
        y_batch = SCALER.inverse_transform(y_batch)
        if y_batch.is_cuda:
            y_true.append(y_batch.cpu().numpy())
        else:
            y_true.append(y_batch.numpy())

        if out_batch.is_cuda:
            y_pred.append(out_batch.cpu().numpy())
        else:
            y_pred.append(out_batch.numpy())
    
    if debug:
        return np.mean(batch_losses), y_true, y_pred
    else:
        return np.mean(batch_losses)


@torch.no_grad()
def model_predict(model, dataLoader):
    y_pred, y_true = [], []
    for x_batch, y_batch in dataLoader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)
        
        out_batch = SCALER.inverse_transform(out_batch)
        # y_batch = SCALER.inverse_transform(y_batch)
        
        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        
        y_true.append(y_batch)
        y_pred.append(out_batch)

    y_true = np.vstack(y_true).squeeze()
    y_pred = np.vstack(y_pred).squeeze()
    
    return y_pred, y_true


def oneStepForward(model, dataLoader, optimizer, scheduler, criterion):
    model.train()
    batch_losses = []

    for x_batch, y_batch in dataLoader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        loss = criterion(out_batch, y_batch)
        batch_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    epoch_loss = np.mean(batch_losses)
    scheduler.step()
    
    return epoch_loss
        
    

def model_train(model, trainLoader, valLoader, optimizer, scheduler, criterion,
                epochs=200, early_stopping=None, compile_model=True, verbose=1, logger=None, plot=False, save=False):

    if torch.__version__ >= "2.0.0" and compile_model:
        model = torch.compile(model)
    
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        train_loss = oneStepForward(model, trainLoader, optimizer, scheduler, criterion)
        train_losses.append(train_loss)

        val_loss = model_evaluate(model, valLoader, criterion)
        val_losses.append(val_loss)

        early_stopping(val_loss, model)

        if logger:
            logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if early_stopping.early_stop:
            print("Early stopping")
            model.load_state_dict(torch.load(f'{MODEL_NAME}_checkpoint.pt'))
            break
            
    return model

@torch.no_grad()
def model_test(model, dataLoader, logger=None):
    model.eval()

    if logger:
        logger.info("--------- Test ---------")

    start_time = time.time()
    y_pred, y_true = model_predict(model, dataLoader)
    end_time = time.time()

    rmse_all, mae_all, mape_all = getMetric(y_pred, y_true)
    if logger:
        logger.info(f"All steps RMSE = {rmse_all:.5f}, MAE = {mae_all:.5f}, MAPE = {mape_all:.5f}\n")

    for i in range(y_pred.shape[1]):
        rmse, mae, mape = getMetric(y_pred[:, i, :], y_true[:, i, :])
        if logger:
            logger.info(f"--- Step {i+1} RMSE = {rmse:.5f}, MAE = {mae:.5f}, MAPE = {mape:.5f}\n")

    logger.info(f"Inference time:{(end_time - start_time):.2f}s.")
    

def model_save(model):
    pass


def metric_plot():
    pass



def select_gpu():
    mem = []
    gpus = list(set(range(torch.cuda.device_count()))) # list(set(X)) is done to shuffle the array
    for i in gpus:
        gpu_stats = gpustat.GPUStatCollection.new_query()
        mem.append(gpu_stats.jsonify()["gpus"][i]["memory.used"])
    return str(gpus[np.argmin(mem)])


if __name__ == "__main__":
    """
    INITIALIZATION
    """

    # Parameter setting
    print("Model Initialization...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="STID", help='Model name to save output in file')
    parser.add_argument('-s', '--step', default=12, type=int, help='Step to predict (*5 minute).')
    parser.add_argument('-n', '--node', type=int, default=207, help='Number of node in the dataset.')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Number of batches to train in each epoch.')
    parser.add_argument('-d','--dataset', type=str, default='METRLA')
    parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. -1(default) means GPU with most free memory will be chosen.')
    parser.add_argument('--verbose', default=1, type=int, help='Default is 1, if you donn\'t want any log, please / set verbose to 0.')
    

    args = parser.parse_args()
    MODEL_NAME = args.model
    print(MODEL_NAME)
    STEP = args.step
    print(args.dataset)
    DATASET = args.dataset
    BATCH_SIZE = args.batch_size
    
    current_time = datetime.now()
    formatted_time = current_time.strftime("%y%m%d%H%M")
    log_file = f"./logs/{formatted_time}_{MODEL_NAME}_{DATASET}_training.log"
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    
    GPU_ID = select_gpu() if args.gpu == -1 else args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

    
    print(f"Loading {DATASET} dataset...")
    #Data Loading
    trainLoader, valLoader, testLoader, SCALER = generate_train_val_test('./data/METRLA/metr-la.h5', batch_size=BATCH_SIZE, step=STEP)
    
    
    
    # Model initializaiton
    model = STID.STID(207, input_len=STEP, output_len=STEP).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0003,
        eps=1e-8,
    )

    print(f"{MODEL_NAME} initialization is finished.")

    criterion = myLoss()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=2e-3,
        weight_decay=3e-4,
        eps=1e-8,
    )
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[20,30],
        gamma=0.1,
        verbose=False,
    )
    
    early_stopping = EarlyStopping()
    
    # with open('data.pkl', 'rb') as f:
    #     trainLoader, valLoader, testLoader, SCALER = pickle.load(f)
    
    """
    MODEL TRANING
    """
    
    model = model_train(
        model, trainLoader, valLoader, optimizer, scheduler, criterion, early_stopping=early_stopping, logger=logger
    )
    
    print('Tranning is over!')
    
    model_test(model, testLoader, logger=logger)
    
    
    
