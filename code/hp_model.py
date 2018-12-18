
# hp_model.py
# house price model to be trained

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as func

import pretrainedmodels
import time

def getModel(mode, device, input_size, hidden_dim, emb_matrix):

    print("Start building model ...")
    tic = time.time()
    model = HousingPriceModel(input_size, hidden_dim, emb_matrix)
    if device != None:
        model = model.cuda(device)
    toc = time.time()
    print("Build model took %.2f s" % (toc-tic))

    return model

class HousingPriceModel(nn.Module):
    def __init__(self, input_size, hidden_dim, emb_matrix):
        super(HousingPriceModel, self).__init__()
        self.modelName = 'prediction_model'

        # for image feature extraction
        #self.resnet = torchvision.models.resnet101(pretrained=True)
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnetFeatures = nn.Sequential(*list(self.resnet.children())[:-4]) # conv3_x
        self.mac = nn.MaxPool2d(input_size // 8, stride=1)

        # for text feature extraction
        vocab_size = len(emb_matrix)
        self.embedding_dim = len(emb_matrix[0])
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(emb_matrix))
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim)
        self.hidden = None
        
        # linear layer
        # 512 + 50 + 7 = 569
        #self.classifier = nn.Linear(512, 1)
        self.classifier = nn.Linear(569, 1)

        #for p in self.features.parameters():
        #    p.requires_grad = False


    def forward(self, x):
        image, sentence, feature = x

        # for image feature extraction
        #print('image: ' + str(image.shape))
        resnet_out = self.resnetFeatures(image)
        #print('resnet_out: ' + str(resnet_out.shape))
        mac_resnet_out = self.mac(resnet_out)
        #print('mac_resnet_out: ' + str(mac_resnet_out.shape))
        squeezed_mac_resnet_out = torch.squeeze(mac_resnet_out) 
        #print('squeezed_mac_resnet_out: ' + str(squeezed_mac_resnet_out.shape))

        # for text feature extraction
        #print('sentence: ' + str(sentence.shape))
        embeds = self.word_embeddings(sentence)
        #print('embeds: ' + str(embeds.shape))
        #print('embeds: ' + str(embeds.view(sentence.shape[1], self.batch_size, -1).shape))
        #print('hidden0: ' + str(self.hidden[0].shape))
        #print('hidden1: ' + str(self.hidden[1].shape))
        lstm_out, self.hidden = self.lstm(embeds.view(sentence.shape[1], -1, self.embedding_dim), self.hidden)
        #print('lstm_out: ' + str(lstm_out.shape))
        mean_lstm_out = torch.mean(lstm_out, 0)
        #print('mean_lstm_out: ' + str(mean_lstm_out.shape))

        fv = torch.cat((squeezed_mac_resnet_out, mean_lstm_out, feature), 1)
        #print('fv: ' + str(fv.shape))
        #fv = fv.view(fv.size(0), -1)
        fv = self.classifier(fv)
        return fv

    '''
    def getParameters(self):
        #return filter(lambda p: p.requires_grad, self.parameters())
        return self.nnet.parameters()
    '''

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))

def run_test():
    print('Test housing price module')

if __name__ == "__main__":
    run_test()

