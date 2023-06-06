import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from mlp import gMLPClassification
class nnModel(nn.Module):
    def __init__(self, args):
        super(nnModel, self).__init__()
        self.cuda_condition = args.cuda_condition
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.input_size = args.item_size
        self.attribute_size = args.attribute_size
        self.num_layers = args.num_hidden_layers
        self.seq_len = args.max_seq_length
        self.criterion = nn.BCELoss(reduction='none')
        self.linear1 = nn.Linear(1,self.input_size)
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.attribute_embeddings = nn.Embedding(args.attribute_size, args.hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers,batch_first=False)
        self.fc = nn.Linear(self.hidden_size, args.batch_size)
        self.mlp = gMLPClassification(patch_width=1, seq_len=20, num_classes=args.num_classes, dim=32, depth=2)

    def finetune(self, x):
        h_0 = Variable(torch.randn(self.num_layers, self.seq_len, self.hidden_size).float())
        c_0 = Variable(torch.randn(self.num_layers, self.seq_len, self.hidden_size).float())
        if self.cuda_condition:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        # print(x.dtype)
        x = torch.LongTensor(x.numpy()).float()
        x = x.unsqueeze(-1)
        # print(x.dtype)

        x = self.linear1(x)
        x, (h_out, y) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.seq_len, self.hidden_size)

        h_out = self.fc(h_out)
        h_out = h_out.view(self.batch_size, self.seq_len*self.num_layers)
        # print("h_out.shape",h_out.shape)
        h_out = self.mlp(h_out)

        return h_out
