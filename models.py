import numpy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from modules import Encoder, LayerNorm
from mlp import gMLPClassification
from mlp import gMLP
from sklearn import preprocessing
from Transformer import Transformer, DownConv, Time_DownConv, DeformConv
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HiModel(nn.Module):
    def __init__(self, args):
        super(HiModel, self).__init__()
        self.activity_embeddings = nn.Embedding(args.activity_size, args.hidden_size, padding_idx=0)
        self.attribute_embedding = nn.Linear(args.attribute_size, args.hidden_size*args.max_seq_length)
        self.attribute_embeddings = nn.Linear(1, args.embed_dim)
        self.time_attributes_embedding = nn.Linear(3, args.embed_dim)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.SubLayerNorm = LayerNorm(args.embed_dim, eps=1e-12)
        self.AttNorm = nn.LayerNorm(args.attribute_size)
        self.AttNorm2 = nn.LayerNorm(args.attribute_size)
        self.seq_len = args.max_seq_length
        self.num_layers = args.num_hidden_layers
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.batch_size = args.batch_size
        self.args = args
        self.global_pooling = nn.AdaptiveAvgPool1d(args.embed_dim)
        self.attr_linear = nn.Linear(args.attribute_size, args.max_seq_length)
        self.sub_pooling = nn.AdaptiveAvgPool1d(int(args.embed_dim / 2))
        self.Att_LayerNorm = nn.LayerNorm(args.attribute_size)
        self.criterion = nn.BCELoss(reduction='none')
        self.loss_func = nn.KLDivLoss(reduction='batchmean')
        self.linear = nn.Linear(args.max_seq_length, args.max_seq_length * args.num_hidden_layers)
        self.linear2 = nn.Linear(args.hidden_size, args.attribute_size)
        self.activity_trans = nn.Linear(20, args.activity_size)
        self.merge_trans = nn.Linear(args.activity_size, args.activity_size)
        self.transformer = Transformer(20, input_dim=args.max_seq_length, attn_dropout=0.1, ff_dropout=0.1, args=args)
        self.attr_transformer = Transformer(20,input_dim=args.hidden_size, attn_dropout=0.1, ff_dropout=0.1, args=args)
        self.transformer2 = Transformer(20, input_dim=args.embed_dim, attn_dropout=0.1, ff_dropout=0.1, args=args)
        self.sub_transformer = Transformer(20, input_dim=int(args.embed_dim / 2), attn_dropout=0.1, ff_dropout=0.1,
                                           args=args)
        self.gmlp = gMLPClassification(patch_width=1, seq_len=args.embed_dim, num_classes=args.num_classes, dim=36, depth=2)
        self.down_conv = DownConv(args.embed_dim)
        self.deform_conv = DeformConv(batch=args.batch_size, channel=1, height=2*args.hidden_size,width=args.embed_dim)
        self.time_down_conv = Time_DownConv(args.embed_dim)
        self.mlp = gMLP(seq_len=args.hidden_size, dim=args.embed_dim, depth=2, num_classes=args.num_classes)
        self.merge_mlp = gMLP(seq_len=2*args.hidden_size, dim=args.embed_dim, depth=2, num_classes=args.num_classes)
        self.mlp_time = gMLP(seq_len=args.hidden_size+1, dim=args.embed_dim, depth=2, num_classes=args.num_classes)
        self.sigmoid = nn.Sigmoid()
        self.linear3 = nn.Linear(args.embed_dim, args.num_classes)
        self.linear4 = nn.Linear(args.num_classes, args.num_classes)
        self.out_norm = nn.LayerNorm(args.num_classes)
        self.out_layer = nn.Linear(args.hidden_size, args.num_classes)
        self.relu = nn.ReLU()
        self.relu_layer = nn.Tanh()
        self.apply(self.init_weights)
        self.num_classes = args.num_classes

    def attributes_embedding(self, attributes):

        attributes = self.AttNorm(attributes)
        attributes = self.attribute_embedding(attributes.float())
        # attributes = attributes.unsqueeze(-1)
        attributes = self.relu(attributes)
        return attributes

    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.activity_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # Fine tune
    # same as SASRec
    def finetune_Hi_Att(self, input_ids, attributes):
        # print(attributes.shape)
        # print(input_ids.shape)
        sequence_emb = self.add_position_embedding(input_ids)  # batch*max_len->batch*max_len*hidden
        # sequence_emb = F.normalize(sequence_emb.float(), dim=1)
        attributes = F.normalize(attributes.float(), dim=1)
        attributes = self.attribute_embedding(attributes)
        # attributes_emb = self.add_position_embedding(attributes)
        # print("attributes.shape", attributes.shape)
        # attributes_emb = self.attr_transformer(attributes)
        # print("attributes.shape", attributes.shape)
        activity_emb = sequence_emb.transpose(1, 2)
        attributes_emb = attributes.reshape(activity_emb.size(0), activity_emb.size(1), -1)
        input = torch.cat((activity_emb, attributes_emb),1)
        # print(input.shape) #batch*2hidden*len
        # attributes_emb = attributes_emb.transpose(1, 2)
        # print("activity_emb", activity_emb)
        # print("attributes", attributes)
        output = self.transformer(input)  # 64*64*128 batch*hidden*max_len->batch*hidden*embed_dim
        # print("output.shape",output.shape)
        # sublayer = self.down_conv(output)  # batch*hidden*(embed_dim)
        sublayer = self.deform_conv(output)
        # print("output after.shape", sublayer.shape)
        sub_output = self.SubLayerNorm(sublayer)
        # output2 = self.transformer2(output + sub_output)
        # print("seq_out.shape--------------", output.shape)
        sequence_output = self.merge_mlp(output+sub_output)  # batch*hidden*(embed_dim)-?batch*classes

        return sequence_output

    def finetune_Hi_Att_Time(self, input_ids, attributes, time_attributes):
        # print("time_attributes:", time_attributes.shape)
        sequence_emb = self.add_position_embedding(input_ids)  # batch*max_len->batch*max_len*hidden

        time_attributes = self.time_attributes_embedding(time_attributes)  # batch*3->batch*36
        time_attributes = self.relu(time_attributes)
        time_attributes = time_attributes.unsqueeze(1)
        # print("attributes.shape", attributes.shape)
        activity_emb = sequence_emb.transpose(1, 2)
        # print("activity_emb", activity_emb)
        # print("attributes", attributes.shape)
        output = self.transformer(activity_emb)  # 64*64*128 batch*hidden*max_len->batch*hidden*embed_dim
        sublayer = self.down_conv(output)  # batch*hidden*(embed_dim)

        sub_output = self.SubLayerNorm(sublayer)
        input2 = torch.cat((output, time_attributes), axis=1)
        # print("input2.shape ", input2.shape)
        # output2 = self.transformer2(input2)
        # print("output2.shape", output2.shape)
        # print("seq_out.shape--------------", output.shape)
        # out = output2[:, :, -1]
        out = self.linear3(input2)
        out = out[:, -1, :]
        out = self.out_norm(out)
        out = self.relu(out)
        out = self.linear4(out)
        # out = self.out_layer(out)
        # sequence_output = self.mlp(output + sub_output)  # batch*hidden*(embed_dim)->batch*classes
        sequence_output = self.mlp_time(input2)  # batch*hidden*(embed_dim)->batch*classes

        return out

    def finetune_Hi2(self, input_ids, attributes):
        # print(attributes.shape)
        sequence_emb = self.add_position_embedding(input_ids)  # batch*max_len->batch*max_len*hidden

        # attributes = self.attributes_embedding(attributes)
        # print("attributes.shape", attributes.shape)
        activity_emb = sequence_emb.transpose(1, 2)
        # print("activity_emb", activity_emb)
        # print("attributes", attributes)
        output = self.transformer(activity_emb)  # 64*64*128 batch*hidden*max_len->batch*hidden*embed_dim
        sublayer = self.down_conv(output)  # batch*hidden*(embed_dim)

        sub_output = self.SubLayerNorm(sublayer)
        output2 = self.transformer2(output + sub_output)
        # print("seq_out.shape--------------", output.shape)
        sequence_output = self.mlp(output + sub_output)  # batch*hidden*(embed_dim)-?batch*classes

        return sequence_output

    def finetune_Hi_Att_Time2(self, input_ids, attributes, time_attributes):
        # print("time_attributes:", time_attributes.shape)
        sequence_emb = self.add_position_embedding(input_ids)  # batch*max_len->batch*max_len*hidden

        time_attributes = self.time_attributes_embedding(time_attributes)  # batch*3->batch*36
        time_attributes = self.relu(time_attributes)
        time_attributes = time_attributes.unsqueeze(1)
        # print("attributes.shape", attributes.shape)
        activity_emb = sequence_emb.transpose(1, 2)
        # print("activity_emb", activity_emb)
        # print("attributes", attributes.shape)
        output = self.transformer(activity_emb)  # 64*64*128 batch*hidden*max_len->batch*hidden*embed_dim
        # print("output.shape", output.shape)
        sublayer = self.time_down_conv(output)  # batch*hidden*(embed_dim)

        sub_output = self.SubLayerNorm(sublayer)
        # input2 = torch.cat((output, time_attributes), axis=1)
        # print("input2.shape ", input2.shape)
        input2 = sub_output + output
        input2 = torch.cat((input2, time_attributes), axis=1)
        # output2 = self.transformer2(input2)
        # print("output2.shape", output2.shape)
        # print("seq_out.shape--------------", output.shape)
        # out = output2[:, :, -1]
        out = self.linear3(input2)
        out = out[:, -1, :]
        out = self.out_norm(out)
        out = self.relu(out)
        out = self.linear4(out)
        # out = self.out_layer(out)
        # sequence_output = self.mlp(output + sub_output)  # batch*hidden*(embed_dim)->batch*classes
        # sequence_output = self.mlp_time(input2)  # batch*hidden*(embed_dim)->batch*classes

        return out

    def finetune(self, input_ids, attributes):

        # print(attributes.shape)
        sequence_emb = self.add_position_embedding(input_ids)  # batch*max_len->batch*max_len*hidden

        # attributes = self.attributes_embedding(attributes)
        # print("attributes.shape", attributes.shape)
        activity_emb = sequence_emb.transpose(1, 2)
        # print("activity_emb", activity_emb)
        # print("attributes", attributes)
        output = self.transformer(activity_emb)  # 64*64*128 batch*hidden*max_len->batch*hidden*embed_dim

        # print("seq_out.shape--------------", output.shape)
        sequence_output = self.mlp(output)  # batch*hidden*(embed_dim)-?batch*classes

        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            # module.weight.data.normal_(mean=0.0, std=
            # print("isinstance1")
            module.weight.data.normal_(mean=3.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            # print("isinstance2")
            module.bias.data.zero_()
            module.weight.data.fill_(3.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            # print("isinstance3")
            module.bias.data.zero_()
