
import torch
import argparse
import csv
import os.path

def getArgParser():
    parser = argparse.ArgumentParser(description='Housing Price Prediction Project')
    parser.add_argument('--experiment_name', metavar='EXP_NAME', default='unknown', 
                        help='name for the experiment')
    parser.add_argument('--mode', metavar='M', default='train', 
                        help='select mode')
    return parser

def printArgs(args):
    print('experiment_name = %s' % args.experiment_name)
    print('mode = %s' % args.mode)

def getDevice():
    use_cuda = torch.cuda.is_available()
    #use_cuda = None
    if not use_cuda:
        print('device = cpu')
        return None
    device = torch.cuda.device(0) # GPU 0
    print('device = %s' % device)
    return 0

def splitTrainDevSet(imageList, ratio):
    num_train = 0
    num_dev = 0
    if imageList != None and len(imageList) > 0:
        num_train   = int(len(imageList[0]) * ratio)
        num_dev     = len(imageList[0]) - num_train
        print('%d train pictures' % num_train)
        print('%d dev pictures' % num_dev)
    return num_train, num_dev

def saveModel(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def loadModel(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    if model != None:
        model.load_state_dict(state['state_dict'])
    else:
        print('model does not exist')
    if optimizer != None:
        optimizer.load_state_dict(state['optimizer'])
    else:
        print('optimizer does not exist')
    print('model loaded from %s' % checkpoint_path)

#load feature extraction model
def loadExtModel(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    states_to_load = {}
    for name, param in state['state_dict'].items():
        if name.startswith('conv') or name.startswith('mixed'):
            states_to_load[name] = param
    if model != None:
        model_state = model.state_dict()
        model_state.update(states_to_load)
        model.load_state_dict(model_state)
    else:
        print('model does not exist')
    print('model loaded from %s' % checkpoint_path)

def tryRestore(mode, fname, model, optimizer):
    if os.path.isfile(fname):
        print('Restoring best checkpoint')
        if mode != 'extract':
            loadModel(fname, model, optimizer)
        else:
            loadExtModel(fname, model)
        return True
    return False

class Logger():
    def __init__(self, exp_path, name, writeType):
        fileName = exp_path + '/' + name + '.csv'
        self.logFile = open(fileName, writeType, 1) # line buffering
        self.writer = csv.writer(self.logFile)
    def writeLoss(self, itr, loss):
        self.writer.writerow((itr, loss))
    def __del__(self):
        self.logFile.close()

def saveLabel2Idx(fileName, label2idx):
    with open(fileName, 'w') as csvFile:
        writter = csv.writer(csvFile)
        for label, idx in label2idx.items():
            writter.writerow((idx, label))

def loadLabel2Idx(fileName):
    label2idx = {}
    idx2label = {}
    with open(fileName, 'r') as csvFile:
        CSVreader = csv.reader(csvFile, skipinitialspace=True, delimiter=',')
        for row in CSVreader:
            idx = int(row[0])
            label = int(row[1])
            label2idx[label] = idx
            idx2label[idx] = label

    return label2idx, idx2label

def genResultFile(mode, testCSVfile, resultCSVfile, label2result):
    outputFile = open(resultCSVfile, 'w')
    CSVwriter = csv.writer(outputFile)
    if mode == 'submit0':
        CSVwriter.writerow(('id', 'landmarks'))
    elif mode == 'submit1':
        CSVwriter.writerow(('id', 'images'))
    with open(testCSVfile, 'r') as csvFile:
        CSVreader = csv.reader(csvFile, skipinitialspace=True, delimiter=',')
        for row in CSVreader:
            label = row[0]
            if label in label2result.keys():
                CSVwriter.writerow((label, label2result[label]))
            else:
                if mode == 'submit0':
                    CSVwriter.writerow((label, '0 0.0'))
                elif mode == 'submit1':
                    CSVwriter.writerow((label, ' '))
    outputFile.close()
    print('generated result file at %s' % resultCSVfile)

