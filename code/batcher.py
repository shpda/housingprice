
# batcher.py
# read in data and separate into batches

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import numpy as np
from PIL import Image

from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import time

from utils import saveLabel2Idx, loadLabel2Idx

# Resolve 'Set changed size during iteration'
#tqdm.monitor_interval = 0

def readCSVhelper(csvFileName, imageDir, readLabel=True):
    with open(csvFileName, 'r') as csvFile:
        CSVreader = csv.reader(csvFile, skipinitialspace=True, delimiter=',')
        fileIds = []
        fileNames = []
        labels = []
        missingFiles = 0
        print('Reading file %s' % csvFileName)
        next(CSVreader) # skip header
        for row in tqdm(CSVreader):
            fId = row[0]
            baseName = row[2]
            fName = imageDir + baseName
            if readLabel:
                label = row[6]
            fileIds.append(fId)
            fileNames.append(fName)
            if readLabel:
                labels.append(label)
        print('Got %d picture ids' % (len(fileNames)))
        print('Got %d picture filenames' % (len(fileNames)))

        return fileIds, fileNames, labels

def readCSV(csvFile, imageDir, readLabel=True):
    tic = time.time()
    # read filenames
    fileids, filenames, labels = readCSVhelper(csvFile, imageDir, readLabel)
    toc = time.time()
    print("Read filenames took %.2f s" % (toc-tic))
    return (fileids, filenames, labels)

def getImageList(csvFile, imageDir):
    return readCSV(csvFile, imageDir)

    '''
    if mode == 'train':
        return readCSV(rec_train_csv, checkMissingFile=True)
    elif mode == 'train-pruned':
        return readCSV(pruned_train_csv, checkMissingFile=True)
    return None
    '''

class HousingPriceData(Dataset):
    """
    Data loader for landmarks data.
    """
    def __init__(self,
                 imageList,
                 percent=1.0,
                 transform=None,
                 num_train=0,
                 tgtSet='train',
                 preload=False):

        self.images = None

        if tgtSet=='train':
            self.filenames = imageList[1][:num_train]
            self.labels = imageList[2][:num_train]
        elif tgtSet=='dev':
            self.filenames = imageList[1][num_train:]
            self.labels = imageList[2][num_train:]
        else:
            print('ERROR: unknow tgtSet')

        self.transform = transform

        fullLen = len(self.filenames)
        shorterLen = int(fullLen * percent)
        print('Percentage to load = {}/{} ({:.0f}%)'.format(shorterLen, fullLen, 100. * percent))
        self.filenames = self.filenames[:shorterLen]
        self.labels = self.labels[:shorterLen]

        # if preload dataset into memory
        if preload:
            self._preload()
            
        self.len = len(self.filenames)
                              
    def _preload(self):
        """
        Preload dataset to memory
        """
        #self.labels = []
        self.images = []
        print('Preloading...')
        tic = time.time()
        #for image_fn in self.filenames:
        for image_fn in tqdm(self.filenames):            
            # load images
            image = Image.open(image_fn)
            # avoid too many opened files bug
            self.images.append(image.copy())
            image.close()
        toc = time.time()
        print("Preload took %.2f s" % (toc-tic))

    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if self.images is not None:
            # If dataset is preloaded
            image = self.images[index]
            if self.labels:
                label = float(self.labels[index])
        else:
            # If on-demand data loading
            image_fn = self.filenames[index]
            image = Image.open(image_fn)
            if self.labels:
                label = float(self.labels[index])
            
        # May use transform function to transform samples
        # e.g., random crop, whitening
        if self.transform is not None:
            image = self.transform(image)
        # return image and label
        llabel = np.log(label + 1)
        return image, llabel

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

class Batcher(object):
    """
    Get preprocessed data batches
    """
    def __init__(self,
                 imageList=None,
                 percent=1.0, # load a subset of data
                 preload=False,
                 batchSize=64,
                 num_train=0,
                 tgtSet='train'):

        # preprocessing stuff
        #myTrans = transforms.Compose([transforms.Resize((32, 32)),
        #                              transforms.ToTensor()])
        #myTrans = transforms.Compose([transforms.CenterCrop(256),
        #                              transforms.ToTensor()])
        myTrans = transforms.Compose([
            #transforms.Resize(256),
            #transforms.Resize(299),
            transforms.Resize(224),
            #transforms.RandomCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                  std = [ 0.229, 0.224, 0.225 ])])

        dataset = HousingPriceData(imageList, percent=percent, preload=preload, 
                                   transform=myTrans, num_train=num_train, tgtSet=tgtSet)
        self.loader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=0)
        self.dataiter = iter(self.loader)
        #print(len(trainset))
        #print(len(testset))

# functions to show an image
def imshow(img):
    npimg = img.numpy()
    plt.figure(figsize = (50, 50))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def showDataInClass(classId):
    path = '/home/ooo/landmarks-data/landmarks-data'
    first = True
    myTrans = transforms.Compose([transforms.Resize((128, 128)),
                                  transforms.ToTensor()])
    #myTrans = transforms.Compose([transforms.ToTensor()])
    with open('/home/ooo/projects/landmarks/data/train.csv') as csvfile:
        CSVreader = csv.reader(csvfile, delimiter=',')
        fileNames = []
        images = []
        #for row in CSVreader:
        for row in tqdm(CSVreader):
            if first:
                first = False
                continue
            baseName = row[0]
            label = row[2]
            if int(label) != classId:
                continue
            fName = path + '/train/' + baseName + '.jpg'
            if osp.isfile(fName):
                fileNames.append(fName)
                image = Image.open(fName)
                images.append(myTrans(image.copy()))
                image.close()
            if len(fileNames) > 63:
                break
        print('Got %d train picture files with class %d' % (len(fileNames), classId))
        imshow(torchvision.utils.make_grid(images))

def showDataInClass2(th):
    ret_index_csv = '/home/ooo/projects/landmarks/csvFiles/new_ret_index-256.csv'
    idxImageList = readCSV(ret_index_csv, checkMissingFile=True, readLabel=False)
    idxFileIds = idxImageList[0]
    idxFileNames = idxImageList[1]
    idx2file = {}
    for i in range(len(idxFileIds)):
      idx2file[idxFileIds[i]] = idxFileNames[i]

    ret_test_csv = '/home/ooo/projects/landmarks/csvFiles/new_ret_test-256.csv'
    idxImageList = readCSV(ret_test_csv, checkMissingFile=True, readLabel=False)
    idxFileIds = idxImageList[0]
    idxFileNames = idxImageList[1]
    for i in range(len(idxFileIds)):
      idx2file[idxFileIds[i]] = idxFileNames[i]

    with open('/home/ooo/projects/landmarks/experiment/landmarks-full-seresnet101/ret_results.csv') as csvfile:
        CSVreader = csv.reader(csvfile, delimiter=',')
        first = True
        myTrans = transforms.Compose([transforms.ToTensor()])
        images = []
        
        itr = 0
        for row in CSVreader:
            if first:
                first = False
                continue
            itr += 1
            #if itr < th:
                #continue
            testImg = row[0]
            #if (testImg != '0034fcc8b622df6d') and (testImg != '08b28abd7a6f7b63') and (testImg != '160ae73128ba366f') and (testImg != '7650fb4cd97aa7e6') :
            if (testImg != '2de5a4123fcd1283') and (testImg != '17190cb57ec5217a') and (testImg != '04836ef755dfb229') and (testImg != '000506dc6ab3a40e') :
               continue
            print(testImg)
            if testImg not in idx2file.keys():
                continue
            fName = idx2file[testImg]
            if osp.isfile(fName):
                image = Image.open(fName)
                images.append(myTrans(image.copy()))
                image.close()
            indexImgs = row[1]
            for imgId in indexImgs.split()[:7]:
                fName = idx2file[imgId]
                if osp.isfile(fName):
                    image = Image.open(fName)
                    images.append(myTrans(image.copy()))
                    image.close()
            #if len(images) > 63:
            if len(images) > 31:
                break
        imshow(torchvision.utils.make_grid(images))

def run_test():
    #path = '/home/ooo/small-landmarks-data'
    path = '/home/ooo/projects/landmarks/data/tiny-landmarks'
    batcher = Batcher(path, preload=True)
    imagesIter = batcher.dataiter
    images, labels = imagesIter.next()

    # visualize the dataset
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % labels[j] for j in range(16)))

if __name__ == "__main__":
    run_test()

