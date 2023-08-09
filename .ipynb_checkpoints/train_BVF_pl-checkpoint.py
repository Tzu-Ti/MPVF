from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from multiprocessing import cpu_count
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies import DDPStrategy

import numpy as np
import argparse
import random
from tqdm import trange, tqdm
import os

from dataset import pyramid_dataset
from models import BVF
from criterion import *
from utils import Unormalize, backwardwarping3D, flow_visualize, MeasureMetric

def parse():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # dataset information
    parser.add_argument('--train_folder', default='data/training_processed')
    parser.add_argument('--test_folder', default='data/testing_4')
    parser.add_argument('--t', default=0.5, type=float)
    # training hyper-parameter
    parser.add_argument('--model_name', required=True)
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('-l', '--lr', default=1e-3,  type=float)
    parser.add_argument('-e', '--epochs', default=2000, type=int)
    # model hyper-parameter
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--ckpt_path')
    
    return parser.parse_args()

class Model_factory(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Initialize Model
        print(">>Initialize {} Model...".format(self.args.model_name), end='')
        self.model = BVF.Model()
        print("Done")
        
        # Training setting
        self.charbonnier = Charbonnier_loss()

        self.metrics = ['PSNR', 'SSIM', 'MSE']
        self.MM0 = MeasureMetric(self.metrics)
        self.MM1 = MeasureMetric(self.metrics)
        
    def warp(self, flow, I0, I1):
        # backward warping
        flowt0, flowt1 = torch.chunk(flow, 2, dim=1)
        It_0 = backwardwarping3D(I0, flowt0)
        It_1 = backwardwarping3D(I1, flowt1)
        return flowt0, flowt1, It_0, It_1
    
    def eliminate_border(self, x, xo):
        B, C, D, H, W = x.shape
        x[:, :, :,  0, :] = xo[:, :, :,  0, :]
        x[:, :, :, -1, :] = xo[:, :, :, -1, :]
        x[:, :, :,  :, 0] = xo[:, :, :,  :, 0]
        x[:, :, :, :, -1] = xo[:, :, :, :, -1]
        return x

    def visualize(self, x, is_flow=False):
        if is_flow:
            x = flow_visualize(x)
        else:
            # convert 5D to one batch and flatten D to batch channel
            x = Unormalize(x)
            x = x[0, :, 1:-1, :, :]
            x = x.permute([1, 0, 2, 3])
        return x
        
    def training_step(self, batch, batch_idx):
        I0, I1, It, t = batch

        I0_1_2 = F.interpolate(I0, scale_factor=0.5, mode='trilinear')
        I1_1_2 = F.interpolate(I1, scale_factor=0.5, mode='trilinear')
        It_1_2 = F.interpolate(It, scale_factor=0.5, mode='trilinear')

        I0_1_4 = F.interpolate(I0_1_2, scale_factor=0.5, mode='trilinear')
        I1_1_4 = F.interpolate(I1_1_2, scale_factor=0.5, mode='trilinear')
        It_1_4 = F.interpolate(It_1_2, scale_factor=0.5, mode='trilinear')

        # predict multi-flow
        flow, flow_1_2, flow_1_4 = self.model(I0, I1, t)

        flowt0, flowt1, It_0, It_1 = self.warp(flow, I0, I1)
        flowt0_1_2, flowt1_1_2, It_0_1_2, It_1_1_2 = self.warp(flow_1_2, I0_1_2, I1_1_2)
        flowt0_1_4, flowt1_1_4, It_0_1_4, It_1_1_4 = self.warp(flow_1_4, I0_1_4, I1_1_4)

        recon_err_0 = self.charbonnier(It_0    , It) +\
                      self.charbonnier(It_0_1_2, It_1_2) +\
                      self.charbonnier(It_0_1_4, It_1_4)
        recon_err_1 = self.charbonnier(It_1    , It) +\
                      self.charbonnier(It_1_1_2, It_1_2) +\
                      self.charbonnier(It_1_1_4, It_1_4)

        loss = recon_err_0 + recon_err_1
        ################################################
        # Log
        self.log_dict({
            'Loss': loss
        }, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.logger.experiment.add_images("Train I0", self.visualize(I0), batch_idx)
        self.logger.experiment.add_images("Train I1", self.visualize(I1), batch_idx)
        self.logger.experiment.add_images("Train It", self.visualize(It), batch_idx)
        self.logger.experiment.add_images("Train It0", self.visualize(self.eliminate_border(It_0, I0)), batch_idx)
        self.logger.experiment.add_images("Train It1", self.visualize(self.eliminate_border(It_1, I1)), batch_idx)
        self.logger.experiment.add_images("Train Flowt0", self.visualize(flowt0, is_flow=True), batch_idx)
        self.logger.experiment.add_images("Train Flowt1", self.visualize(flowt1, is_flow=True), batch_idx)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        I0, I1, It, t = batch

        # predict flow
        flow, flow_1_2, flow_1_4 = self.model(I0, I1, t)
        flowt0, flowt1, It_0, It_1 = self.warp(flow, I0, I1)

        # Metric
        M_It_0 = self.eliminate_border(It_0, I0)[:, :, 1:-1, :, :] # only take D(1~-1)
        M_It_1 = self.eliminate_border(It_1, I1)[:, :, 1:-1, :, :]
        M_It = It[:, :, 1:-1, :, :]

        self.MM0.update(M_It_0, M_It, unormalize=True)
        self.MM1.update(M_It_1, M_It, unormalize=True)

    def on_validation_epoch_end(self):
        results0 = self.MM0.compute()
        results1 = self.MM1.compute()

        results = {}
        for m in self.metrics:
            results['{}-0'.format(m)] = results0[m]
            results['{}-1'.format(m)] = results1[m]
        self.log_dict(results, on_epoch=True, sync_dist=True)
        
    def test_step(self, batch, batch_idx):
        I0, I1, It, t = batch

        # predict flow
        flow, flow_1_2, flow_1_4 = self.model(I0, I1, t)
        flowt0, flowt1, It_0, It_1 = self.warp(flow, I0, I1)

        # Metric
        M_It_0 = self.eliminate_border(It_0, I0)[:, :, 1:-1, :, :] # only take D(1~-1)
        M_It_1 = self.eliminate_border(It_1, I1)[:, :, 1:-1, :, :]
        M_It = It[:, :, 1:-1, :, :]

        self.MM0.update(M_It_0, M_It, unormalize=True)
        self.MM1.update(M_It_1, M_It, unormalize=True)
        
    def on_test_end(self):
        results0 = self.MM0.compute()
        results1 = self.MM1.compute()
        print("Results0:")
        print(results0)
        print("Results1:")
        print(results1)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        return [optimizer], [scheduler]

    
def main():
    ### Main
    args = parse()

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.getcwd(),
        version=args.model_name,
        name='lightning_logs'
    )
    
    # Data loader
    trainDataset = pyramid_dataset.dataset(args.train_folder, 'random', train=True)
    trainDataloader = DataLoader(dataset=trainDataset,
                              batch_size=args.batch_size,
                              shuffle=True, 
                              num_workers=cpu_count())
    
    
    valDataset = pyramid_dataset.dataset(args.test_folder, 0.5, train=False)
    valDataloader = DataLoader(dataset=valDataset,
                               batch_size=1,
                               shuffle=True, 
                               num_workers=cpu_count())
    
    testDataset = pyramid_dataset.dataset(args.test_folder, args.t, train=False)
    testDataloader = DataLoader(dataset=testDataset,
                                batch_size=1,
                                shuffle=True, 
                                num_workers=cpu_count())

    # Initialize model
    Model = Model_factory(args)

    # trainer = pl.Trainer(fast_dev_run=True, logger=tb_logger)
    trainer = pl.Trainer(max_epochs=args.epochs, check_val_every_n_epoch=5,
                         logger=tb_logger, log_every_n_steps=5,
                         strategy=DDPStrategy(find_unused_parameters=True))
    if args.train:
        trainer.fit(model=Model, train_dataloaders=trainDataloader, val_dataloaders=valDataloader)
        # trainer.fit(model=Model, train_dataloaders=trainDataloader)
    elif args.resume:
        trainer.fit(model=Model, train_dataloaders=trainDataloader, val_dataloaders=valDataloader, ckpt_path=args.ckpt_path)
    elif args.test:
        test_trainer = pl.Trainer(devices=1, logger=tb_logger)
        test_trainer.test(model=Model, dataloaders=testDataloader, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()