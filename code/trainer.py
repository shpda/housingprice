
# trainer.py
# helper function to train on epochs

import torch
import torch.nn.functional as func
from utils import saveModel, loadModel, Logger, tryRestore
import time
import sys
from tqdm import tqdm
import csv

import numpy as np
from sklearn.decomposition import IncrementalPCA

class Trainer():
    def __init__(self, mode, model, loader, dev_loader, optimizer, device, exp_path, 
                 log_interval=5, eval_interval=5, save_interval=100):
        self.model = model
        self.loader = loader
        self.dev_loader = dev_loader
        self.optimizer = optimizer
        self.device = device
        self.exp_path = exp_path
        self.log_interval = log_interval 
        self.eval_interval = eval_interval 
        self.save_interval = save_interval 
        self.iteration = 0

        #self.loss_fn = torch.nn.MSELoss(reduction='sum')
        #self.loss_fn = torch.nn.MSELoss(reduction='elementwise_mean')
        self.loss_fn = torch.nn.MSELoss(size_average=True)
        self.loss_fn2 = torch.nn.MSELoss(size_average=False)

        cpFile = self.exp_path + '/hp-best.pth'
        if tryRestore(mode, cpFile, model, optimizer): # append
            self.train_logger = Logger(exp_path, 'train', 'a')
            self.dev_logger = Logger(exp_path, 'dev', 'a')
        else: # create new
            self.train_logger = Logger(exp_path, 'train', 'w')
            self.dev_logger = Logger(exp_path, 'dev', 'w')

    def train(self, epoch=5):
        self.model.train()  # set training mode
        self.iteration = 0 
        best_dev_loss = self.eval()
        for ep in range(epoch):
            epoch_tic = time.time()
            for batch_idx, (data, target) in enumerate(self.loader):
                if self.device != None:
                    data, target = data.cuda(self.device), target.cuda(self.device)
                self.optimizer.zero_grad()

                # forward pass
                output = self.model(data)
                loss = self.loss_fn(torch.squeeze(output), target.float())

                # backward pass
                loss.backward()

                # weight update
                self.optimizer.step()

                if self.iteration % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(ep, 
                        batch_idx * len(data), len(self.loader.dataset), 
                        100. * batch_idx / len(self.loader), loss.item()))
                    self.train_logger.writeLoss(self.iteration, loss.item())

                if self.iteration != 0 and self.iteration % self.eval_interval == 0:
                    dev_loss = self.eval()
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        saveModel('%s/lm-best.pth' % self.exp_path, 
                                  self.model, self.optimizer)
                #if self.iteration % self.save_interval == 0:
                #    self.saveModel()

                self.iteration += 1
            epoch_toc = time.time()
            print('End of epoch %i. Seconds took: %.2f s.' % (ep, epoch_toc - epoch_tic))
            #dev_loss = self.eval()
            #if dev_loss < best_dev_loss:
            #    best_dev_loss = dev_loss
            #    saveModel('%s/lm-best.pth' % self.exp_path, self.model, self.optimizer)

    def eval(self, loader=None, name=''):
        if loader == None:
            loader = self.dev_loader
            name = 'dev'
        self.model.eval()  # set evaluation mode
        loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in loader:
                if self.device != None:
                    data, target = data.to(self.device), target.to(self.device)

                # calculate accumulated loss
                output = self.model(data)
                loss += self.loss_fn2(torch.squeeze(output), target.float()) # sum up batch loss

        loss /= len(loader.dataset)
        print('{} set: Average loss: {:.4f}'.format(name, loss.item()))
        if name == 'dev':
            self.dev_logger.writeLoss(self.iteration, loss.item())

        return loss

    def calc(self, loader, idx2label):
        self.model.eval()  # set evaluation mode
        softmax = torch.nn.Softmax(dim=1).cuda()
        label2res = {}
        with torch.no_grad():
            for data, ids in tqdm(loader):
                if self.device != None:
                    data = data.to(self.device)

                output = self.model(data)
                confidence = softmax(output)
                maxConf = confidence.max(1)
                conf = maxConf[0].cpu().numpy()
                pred = maxConf[1].cpu().numpy()
                for i in range(len(pred)):
                    tmp = '%d %.6f' % (idx2label[pred[i]], conf[i])
                    label2res[ids[i]] = tmp
        return label2res
