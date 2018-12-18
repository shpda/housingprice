
# main.py
# main entry point

from utils import *
from hp_model import getModel
from batcher import Batcher, readCSV
from trainer import Trainer
from vocab import get_glove

import os
import torch.optim as optim
import numpy as np
import time

parser = getArgParser()

def main():
    print('Housing Price Prediction Project')

    args = parser.parse_args()
    printArgs(args)

    #root = '/home/gangwu/cs224n/housingprice'
    root = '/home/ooo/projects/housingprice'
    exp_path = root + '/experiment/' + args.experiment_name
    os.system('mkdir -p ' + exp_path)
    print('experiment path: %s' % exp_path)

    #input_size = 128
    input_size = 224 # after crop
    testCSVfile = root + '/csvFiles/clean.csv'
    imageDir = root + '/images'

    glove_path = '../data/glove/glove.6B.50d.txt'
    hidden_dim = 50
    embedding_size = 50

    emb_matrix, word2id, id2word = get_glove(glove_path, embedding_size)

    dataset = readCSV(testCSVfile, imageDir, word2id)
    num_train, num_dev = splitTrainDevSet(dataset, 0.7)

    # percentage of data to load
    pct = 1.0 
    batch_size = 128
    #pct = 0.005 

    device = getDevice()
    model = getModel(args.mode, device, input_size, hidden_dim, emb_matrix)

    if args.mode == 'train':
        # resnet50 batch size: train = 100, dev = 256
	# p100: 64
        trainBatcher = Batcher(dataset, percent=pct, preload=False, batchSize=batch_size, num_train=num_train, tgtSet='train')
        loader = trainBatcher.loader
    
        devBatcher = Batcher(dataset, percent=pct, preload=False, batchSize=batch_size, num_train=num_train, tgtSet='dev')
        dev_loader = devBatcher.loader

        #optimizer = optim.SGD(model.getParameters(), lr=0.001, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

        trainer = Trainer(args.mode, model, loader, dev_loader, optimizer, device, exp_path)
        print('Start training...')
        trainer.train(epoch=60)

        '''
        elif args.mode == 'test':
            testBatcher = Batcher(percent=pct, preload=False, batchSize=512, targetSet='test')
            test_loader = testBatcher.loader
    
            trainer = Trainer(model, None, None, None, device, exp_path)
            print('Start evaluation on test set...')
            trainer.eval(test_loader, 'test')
        '''
    else:
        raise Exception('Unknown mode %s. Exiting...' % args.mode)

    print('Done!')

if __name__ == "__main__":
    main()

