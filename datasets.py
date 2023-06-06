import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
from utils import neg_sample

class HiDataset(Dataset):

    def __init__(self, args, activity_seq, attributes_seq, time_attributes_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length
        self.max_attr_len = args.attribute_size
        self.max_time_attr_len = args.max_time_attr_len
        # print(time_attributes_seq)

        if data_type == 'train':

            self.activity_seq = activity_seq[0:int(len(activity_seq)*0.8)]
            self.attribute_seq = attributes_seq[0:int(len(activity_seq)*0.8)]
            self.time_attributes_seq = time_attributes_seq[0:int(len(activity_seq)*0.8)]
        elif data_type == 'valid':
            self.activity_seq = activity_seq[int(len(activity_seq) * 0.8):int(len(activity_seq) * 0.9)]
            self.attribute_seq = attributes_seq[int(len(activity_seq) * 0.8):int(len(activity_seq) * 0.9)]
            self.time_attributes_seq = time_attributes_seq[int(len(activity_seq) * 0.8):int(len(activity_seq) * 0.9)]
        else:
            self.activity_seq = activity_seq[int(len(activity_seq) * 0.9):]
            self.attribute_seq = attributes_seq[int(len(activity_seq) * 0.9):]
            self.time_attributes_seq = time_attributes_seq[int(len(activity_seq) * 0.9):]

    def __getitem__(self, index):

        case_id = index
        # print("userid ",user_id)
        activities = self.activity_seq[index]
        attributes = self.attribute_seq[index]
        time_attributes = self.time_attributes_seq[index]
        # print(len(self.user_seq))
        assert self.data_type in {"train", "valid", "test"}

        if self.data_type == "train":

            input_ids = activities[:-1]
            answer = [activities[-1]]
            next_attributes = [attributes[-1]]  # 要预测的属性特征
            attributes = attributes[:-1]
            next_time_attributes = [time_attributes[-1]]   # remaining time
            time_attributes = time_attributes[:-1]


        elif self.data_type == 'valid':

            input_ids = activities[:-1]
            answer = [activities[-1]]
            next_attributes = [attributes[-1]]
            attributes = attributes[:-1]
            next_time_attributes = [time_attributes[-1]]
            time_attributes = time_attributes[:-1]
        else:

            input_ids = activities[:-1]
            answer = [activities[-1]]
            next_attributes = [attributes[-1]]
            attributes = attributes[:-1]
            next_time_attributes = [time_attributes[-1]]
            time_attributes = time_attributes[:-1]

        seq_set = set(activities)

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        input_ids = input_ids[-self.max_len:]
        assert len(input_ids) == self.max_len

        pad_attr_len = self.max_attr_len-len(attributes)
        attributes = [0] * pad_attr_len + attributes
        attributes = attributes[-self.max_attr_len:]
        assert len(attributes) == self.max_attr_len

        pad_time_attr_len = self.max_time_attr_len - len(time_attributes)
        time_attributes = [0] * pad_time_attr_len + attributes
        time_attributes = time_attributes[-self.max_time_attr_len:]
        assert len(time_attributes) == self.max_time_attr_len


        cur_tensors = [
            torch.tensor(case_id, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
            torch.tensor(attributes, dtype=torch.long),
            torch.tensor(next_attributes, dtype=torch.float32),
            torch.tensor(time_attributes, dtype=torch.float32),
            torch.tensor(next_time_attributes, dtype=torch.float32),
        ]
        return cur_tensors

    def __len__(self):
        return len(self.activity_seq)