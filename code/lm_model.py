
# lm_model.py
# landmarks model to be trained

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as func

import pretrainedmodels

def getModel(mode, device, input_size):

    model = HousingPriceModel(input_size)
    if device != None:
        model = model.cuda(device)

    return model

class HousingPriceModel(nn.Module):
    def __init__(self, input_size):
        super(HousingPriceModel, self).__init__()
        self.modelName = 'resnet'
        #self.nnet = torchvision.models.resnet101(pretrained=True)
        self.nnet = torchvision.models.resnet50(pretrained=True)
        #self.modelName = 'densenet'
        #self.nnet = torchvision.models.densenet161(pretrained=True)
        #self.modelName = 'inception'
        #self.nnet = torchvision.models.inception_v3(pretrained=True)
        #self.modelName = 'se_resnet'
        #self.nnet = pretrainedmodels.se_resnet101(pretrained='imagenet')
        self.nnet.avgpool = nn.AvgPool2d(input_size // 32, stride=1)

        self.features = nn.Sequential(*list(self.nnet.children())[:-1])
        self.classifier = nn.Sequential(
                            #nn.Linear(self.nnet.fc.in_features, 1)
                            nn.Linear(10240, 1)
                          )

        #self.nnet.fc = nn.Linear(self.nnet.fc.in_features, 1)   # for resnet, num_classes = 1
        #self.nnet.classifier = nn.Linear(self.nnet.classifier.in_features, num_classes)   # for densenet
        #self.nnet.last_linear = nn.Linear(self.nnet.last_linear.in_features, num_classes)   # for se_resnet

        '''
        self.features = nn.Sequential(*list(nnet.children())[:-1])
        self.classifier = nn.Sequential(
                            #nn.Linear(nnet.fc.in_features, num_classes)          # for resnet
                            #nn.Linear(nnet.classifier.in_features, num_classes)   # for densenet
                            nn.Linear(nnet.last_linear.in_features, num_classes)   # for se_resnet
                          )
        '''

        #for p in self.features.parameters():
        #    p.requires_grad = False


    def forward(self, x):
        #x = self.nnet(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def getParameters(self):
        #return filter(lambda p: p.requires_grad, self.parameters())
        return self.nnet.parameters()

def run_test():
    print('Test housing price module')

if __name__ == "__main__":
    run_test()

