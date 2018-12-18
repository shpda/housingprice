
import torch
import argparse
import csv
import os.path

from vocab import PAD_ID, UNK_ID
import re

def split_by_whitespace(sentence):                               
    words = []                                                             
    for space_separated_fragment in sentence.strip().split():       
        words.extend(re.split(" ", space_separated_fragment))              
    return [w for w in words if w] 

def cleanup_tokens(tokens):
    words = []
    for tk in tokens:
        tmp = ''
        for c in tk:
            if not c.isalnum():
                continue
            tmp += c.lower()
        if tmp:
            words.append(tmp)
    return words

def sentence_to_token_ids(sentence, word2id):                              
    tokens = split_by_whitespace(sentence) # list of strings
    clean_tokens = cleanup_tokens(tokens)
    ids = [word2id.get(w, UNK_ID) for w in clean_tokens]                         
    return tokens, clean_tokens, ids

def padded(idsList, tgtLength):
    if len(idsList) > tgtLength:
        return idsList[:tgtLength]
    return idsList + [PAD_ID]*(tgtLength - len(idsList))

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
    #print('model saved to %s' % checkpoint_path)

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

