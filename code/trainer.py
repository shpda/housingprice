
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
                 log_interval=1, eval_interval=5, save_interval=100):
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

        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        #self.loss_fn = torch.nn.MSELoss(reduction='elementwise_mean')
        #self.loss_fn = torch.nn.MSELoss(size_average=True)
        #self.loss_fn2 = torch.nn.MSELoss(size_average=False)

        cpFile = self.exp_path + '/hp-best.pth'
        if tryRestore(mode, cpFile, model, optimizer): # append
            self.train_logger = Logger(exp_path, 'train', 'a')
            self.dev_logger = Logger(exp_path, 'dev', 'a')
        else: # create new
            self.train_logger = Logger(exp_path, 'train', 'w')
            self.dev_logger = Logger(exp_path, 'dev', 'w')

    def train(self, epoch=5):
        self.iteration = 0 
        best_dev_loss = self.eval()

        for ep in range(epoch):
            epoch_tic = time.time()
            for batch_idx, (data, target, imageName) in enumerate(self.loader):
                self.model.train()  # set training mode
                self.optimizer.zero_grad()
                #self.model.zero_grad()
                self.model.hidden = self.model.init_hidden(data[0].shape[0])

                if self.device != None:
                    data = (data[0].cuda(self.device), data[1].cuda(self.device), data[2].cuda(self.device))
                    target = target.cuda(self.device)
                    self.model.hidden = (self.model.hidden[0].cuda(self.device), self.model.hidden[1].cuda(self.device))

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
                        saveModel('%s/hp-best.pth' % self.exp_path, 
                                  self.model, self.optimizer)
                #if self.iteration % self.save_interval == 0:
                #    self.saveModel()

                self.iteration += 1
            epoch_toc = time.time()
            print('End of epoch %i. Seconds took: %.2f s.' % (ep, epoch_toc - epoch_tic))
            #dev_loss = self.eval()
            #if dev_loss < best_dev_loss:
            #    best_dev_loss = dev_loss
            #    saveModel('%s/hp-best.pth' % self.exp_path, self.model, self.optimizer)

    def eval(self, loader=None, name=''):
        if loader == None:
            loader = self.dev_loader
            name = 'dev'
        self.model.eval()  # set evaluation mode
        loss = 0
        epoch = 0
        with torch.no_grad():
            maxDiff = 0
            worstImage = None
            realPrice = None
            predictedPrice = None

            minDiff = sys.maxsize
            bestImage = None
            brealPrice = None
            bpredictedPrice = None
            for data, target, imageName in loader:
                # calculate accumulated loss
                self.model.hidden = self.model.init_hidden(data[0].shape[0])

                if self.device != None:
                    data = (data[0].cuda(self.device), data[1].cuda(self.device), data[2].cuda(self.device))
                    target = target.cuda(self.device)
                    self.model.hidden = (self.model.hidden[0].cuda(self.device), self.model.hidden[1].cuda(self.device))
                    #data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                curLoss = self.loss_fn(torch.squeeze(output), target.float())

                # for error analysis
                '''
                for out, tgt, image in zip(torch.squeeze(output), target.float(), imageName):
                    #print((abs(out.item() - tgt.item()), maxDiff, image))
                    if abs(out.item() - tgt.item()) > maxDiff:
                        maxDiff = abs(out.item() - tgt.item())
                        predictedPrice = np.exp(out.item()) - 1
                        realPrice = np.exp(tgt.item()) - 1
                        worstImage = image

                    if abs(out.item() - tgt.item()) < minDiff:
                        minDiff = abs(out.item() - tgt.item())
                        bpredictedPrice = np.exp(out.item()) - 1
                        brealPrice = np.exp(tgt.item()) - 1
                        bestImage = image
                '''

                #print(curLoss.item())
                loss += curLoss
                epoch += 1
                #print(loss.item())

        loss /= epoch
        # for error analysis
        '''
        print('worstImage: %s' % worstImage)
        print('predicted price: %s' % predictedPrice)
        print('real price: %s' % realPrice)
        print('bestImage: %s' % bestImage)
        print('predicted price2: %s' % bpredictedPrice)
        print('real price2: %s' % brealPrice)
        print('{} set: Average loss: {:.4f}'.format(name, loss.item()))
        '''
        if name == 'dev':
            self.dev_logger.writeLoss(self.iteration, loss.item())

        return loss
