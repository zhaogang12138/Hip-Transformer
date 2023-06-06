import numpy as np
import math
import random
import os
import json
import pickle
from scipy.sparse import csr_matrix
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')

def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0, loss=False, acc=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.loss = loss
        self.acc = acc

    def compare(self, loss):
        for i in range(len(loss)):
            # 有一个指标增加了就认为是还在涨False继续True就stop
            # loss就是这里的score如果不降了就要 earlystop
            # accuracy提升了就保存，不升了就 earlystop
            if self.acc==True:
                accuracy = loss

                if accuracy[i]  > self.best_score[i]+self.delta:
                    return False
                return True
            if self.loss==True:
                if loss[i] < self.best_score[i] + self.delta:
                    return False
                return True
        return False
    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            # ({self.score_min:.6f} --> {score:.6f}) # 这里如果是一个值的话输出才不会有问题
            print(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)

def avg_pooling(x, dim):
    return x.sum(dim=dim)/x.size(dim)


def get_activity_seqs(data_file):
    lines = open(data_file).readlines()
    activity_seq = []
    long_seq = []
    activity_set = set()
    for line in lines:
        id, activities = line.strip().split(' ', 1)
        activities = activities.split(' ')
        activities = [int(activity) for activity in activities]
        activity_seq.append(activities)
        long_seq.extend(activities)
        activity_set = activity_set | set(activities)

    max_activity = max(activity_set)

    num_cases = len(lines)
    num_activities = int(max_activity) + 2#一个活动是0一个活动是补长用的数字
    # print(activity_seq)
    return activity_seq, max_activity, long_seq, num_activities, num_cases


def get_attributes_seqs(attributes_file):
    lines = open(attributes_file).readlines()
    attributes_seq = []
    attribute_size = 0
    for line in lines:
        case, attributes = line.strip().split(' ', 1)
        attributes = attributes.split(" ")
        # attributes = attributes[0:-1]

        attributes = [float(attribute) for attribute in attributes]
        attribute_size = max(attribute_size, len(attributes))

        attributes_seq.append(attributes)


    return attributes_seq, attribute_size

def get_time_attributes_seqs(attributes_file):
    lines = open(attributes_file).readlines()
    attributes_seq = []
    attribute_size = 0
    for line in lines:
        case, attributes = line.strip().split(' ', 1)
        attributes = attributes.split(" ")
        # attributes = attributes[0:-1]
        attributes = [float(attribute) for attribute in attributes]
        attribute_size = max(attribute_size, len(attributes))

        attributes_seq.append(attributes)


    return attributes_seq, attribute_size


def apk(actual, predicted, k=3):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(val):
    return (val, val) if not isinstance(val, tuple) else val

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
