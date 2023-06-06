import os
import numpy as np
import random
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from datasets import HiDataset
from trainers_Sample import FinetuneTrainer
from models import HiModel
from LSTM import nnModel
import warnings
warnings.filterwarnings("ignore")

from utils import EarlyStopping, get_activity_seqs, get_attributes_seqs, check_path, set_seed, get_time_attributes_seqs

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='helpdesk', type=str)
    parser.add_argument('--do_eval', action="store_true")
    parser.add_argument('--ckp', default=30, type=int, help="epochs 10, 20参数+全连接, 30参数+mlp, 35参数+pool, 50参数+Hi, 60参数+pool+att...")
    # model args
    parser.add_argument("--model_name", default='Finetune_sample', type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=4, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=1)
    parser.add_argument('--max_seq_length', default=100, type=int)
    parser.add_argument('--num_classes', default=20, type=int)
    parser.add_argument('--embed_dim', default=36, type=int)
    parser.add_argument('--d_model', default=64, type=int)
    # train args

    parser.add_argument("--lr", type=float, default=0.01, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=12, help="number of batch_size")  # 64
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=2023, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    args = parser.parse_args()
    set_seed(args.seed)
    check_path(args.output_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # 输入应该包含三部分，一个是
    args.data_file = args.data_dir + args.data_name + '.txt'
    args.attribute_file = args.data_dir + args.data_name + '_attributes.txt'  # 离散特征
    args.time_attributes_file = args.data_dir + args.data_name + '_time_attributes.txt'  # 连续特征

    activity_seq, max_activity, long_sequence, num_activities, num_cases = get_activity_seqs(args.data_file)
    attributes_seq, attributes_size = get_attributes_seqs(args.attribute_file)
    time_attributes_seq, time_attributes_size, time_scaler = get_time_attributes_seqs(args.time_attributes_file)

    args.num_cases = num_cases
    # args.activity_seq = activity_seq
    args.activity_size = 100  # 后面用0来padding用max+1来sample
    # args.max_seq_length = 100
    args.mask_id = num_activities + 1
    args.attribute_size = 100
    args.max_time_attr_len = 3
    args.device = torch.device("cuda:0" if args.cuda_condition else "cpu")
    # print(args.device)
    # save model args
    args_str = f'{args.model_name}-{args.data_name}-{args.ckp} '
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    # print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    train_dataset = HiDataset(args, activity_seq, attributes_seq, time_attributes_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=False)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = HiDataset(args, activity_seq, attributes_seq, time_attributes_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, drop_last=True, shuffle=False)
    # eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = HiDataset(args, activity_seq, attributes_seq, time_attributes_seq, data_type='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True, shuffle=False)

    # model = S3RecModel(args=args)
    model = HiModel(args = args)
    model = model.to(args.device)
    # model = nnModel(args=args)
    trainer = FinetuneTrainer(model, train_dataloader, eval_dataloader,test_dataloader, args)
    predict_test_loss = "None"
    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True, acc=True, loss=False)
    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        predict_test_loss, accuracies, fscores, precisions, recalls = trainer.test(10, full_sort=True)
        print("测试loss:", predict_test_loss)
        print("accuracies %s, fscores %s, precisions %s, recalls %s " % (accuracies, fscores, precisions, recalls))
    else:

        for epoch in range(args.epochs):
            trainer.train(epoch)

            predict_valid_loss, accuracies, fscores, precisions, recalls = trainer.valid(epoch)
            print("验证loss：", predict_valid_loss)
            print("验证准确率：", accuracies)
            # print("accuracies %s, fscores %s, precisions %s, recalls %s " % (accuracies, fscores, precisions, recalls))
            early_stopping(np.array(accuracies), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('---------------Change to test_rating_matrix!-------------------')
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        predict_test_loss, accuracies, fscores, precisions, recalls = trainer.test(10, full_sort=True)
        print("测试loss:", predict_test_loss)
        print("测试准确率：", accuracies)
        print("测试F1：", fscores)
    print(args_str)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(str(predict_test_loss) + '\n')
main()