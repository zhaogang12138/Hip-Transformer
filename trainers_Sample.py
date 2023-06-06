import numpy as np
import tqdm
import random
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics, preprocessing
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=5, gamma=0.5)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.L1Loss = nn.L1Loss()
        self.L2Loss = nn.MSELoss()
        self.L3Loss = nn.BCEWithLogitsLoss()
    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError



    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
            self.model.load_state_dict(torch.load(file_name, map_location='cpu'))



    def one_hot_encoding(batch_size, y):

        '''
        batch : the batch size
        no_events : the number of events
        y_truth : the ground truth labels'''

        z = torch.zeros(batch_size)
        for i in range(z.size()[0]):
            z[i, y[i].long()] = 1

        # print(z)
        return z.view(batch_size, 1, -1)

class FinetuneTrainer(Trainer):

    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args):
        super(FinetuneTrainer, self).__init__(model, train_dataloader, eval_dataloader, test_dataloader, args
        )

    def iteration(self, epoch, dataloader, full_sort=True, train=True):

        str_code = "train" if train else "test"
        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(enumerate(dataloader), desc="Prediction_Activity_%s:%d" % (str_code, epoch),
                                  total=len(dataloader), bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            # print("len(dataloader) ",len(dataloader))
            for i, batch in rec_data_iter:
                self.optim.zero_grad()
                batch = tuple(t.to(self.device) for t in batch)
                case_id, input_ids, answers, attributes, next_attribute, time_attributes, next_time_attribute = batch
                # input_ids, attributes,time_attributes, answers = input_ids.to(device), attributes.to(device), time_attributes.to(device), answers.to(device)
                # print(input_ids.shape)
                # print(attributes.shape)
                # print(time_attributes.shape)

                # sequence_output = self.model.finetune_Hi2(input_ids, attributes)
                sequence_output = self.model.finetune_Hi_Att(input_ids, attributes)
                # y_pred = torch.argmax(sequence_output, dim=1).unsqueeze(-1)
                # 将要预测的类别进行one hot
                # pred = np.argmax(sequence_output.detach().numpy(), axis=0)
                # target = torch.zeros_like(sequence_output).scatter_(1, answers, torch.ones_like(answers, dtype=torch.float32))
                # print(y_pred.grad)
                # print(answers)
                # print(answers.squeeze(-1).shape)
                # gt_onehot = F.one_hot(answers.squeeze(-1), num_classes=30)
                # gt_onehot = torch.FloatTensor(gt_onehot.float())
                # print(gt_onehot.dtype)
                # sequence_output = F.softmax(sequence_output, dim=1)
                # print(y_pred)
                # print(answers.shape)
                loss = self.cross_entropy(sequence_output, answers.squeeze(-1))
                # print(loss)
                # loss = criterion(sequence_output, target)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
            # rec_avg_loss = rec_avg_loss / len(rec_data_iter)
            self.scheduler.step()
            # writer = SummaryWriter(log_dir='log/new_trans_train/')
            # writer.add_scalar('Trans_Loss_Train', rec_avg_loss, epoch)
            # # writer.add_scalars(main_tag='Rec_avg_Loss', tag_scalar_dict={'train_loss':rec_avg_loss}, global_step=epoch)
            # writer.close()
            # writer.add_scalars(main_tag='Metrics', tag_scalar_dict={'ValLoss': val_loss,
            #                                                         'RMSE': rmse}, global_step=epoch)
            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
            }

            if (epoch + 10) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:

            self.model.eval()

            rec_avg_loss = 0.0
            # print("len(dataloader) ", len(dataloader))
            accuracies, fscores, precisions, recalls = [], [], [], []
            for i, batch in rec_data_iter:
                case_id, input_ids, answers, attributes, next_attribute, time_attributes, next_time_attribute = batch
                input_ids, attributes, time_attributes, answers = input_ids.to(device), attributes.to(device), time_attributes.to(device), answers.to(device)
                # print("input_ids.shape ",input_ids.shape)
                # sequence_output = self.model.finetune_Hi2(input_ids, attributes)
                sequence_output = self.model.finetune_Hi_Att(input_ids, attributes)
                # print(sequence_output)

                y_pred = np.argmax(sequence_output.cpu().detach().numpy(), axis=1)
                # print(y_pred)
                # print(answers)
                accuracy = metrics.accuracy_score(answers.cpu(), y_pred)
                precision, recall, fscore, _ = metrics.precision_recall_fscore_support(answers.cpu(), y_pred, average="weighted")
                # print(sequence_output.shape)
                accuracies.append(accuracy)
                fscores.append(fscore)
                precisions.append(precision)
                recalls.append(recall)
                # target = torch.zeros_like(sequence_output).scatter_(1, answers, torch.ones_like(answers))

                # loss = self.cross_entropy(answers, torch.tensor(y_pred).unsqueeze(-1))
                # loss = self.cross_entropy(recommend_output, target_pos, target_neg)
                loss = self.cross_entropy(sequence_output, answers.squeeze(-1))
                rec_avg_loss += loss.item()
            rec_avg_loss = rec_avg_loss / len(rec_data_iter)
            # writer = SummaryWriter(log_dir='log/new_trans_test/')
            # writer.add_scalar('Trans_Loss_Test', rec_avg_loss, epoch)
            # # writer.add_scalars(main_tag='Rec_avg_Loss', tag_scalar_dict={'train_loss':rec_avg_loss}, global_step=epoch)
            # writer.close()
            with open(self.args.log_file, 'a') as f:
                f.write("valid_loss: " + str(rec_avg_loss) + ' ')
                f.write("accuracies: " + str(np.mean(accuracies)) + ' ')
                f.write("fscores: " + str(np.mean(fscores)) + ' ')
                f.write("precisions: " + str(np.mean(precisions)) + ' ')
                f.write("recalls: " + str(np.mean(recalls)) + '\n')
            return [rec_avg_loss], [np.mean(accuracies)], np.mean(fscores), np.mean(precisions), np.mean(recalls)

class FinetuneTrainer_Time(Trainer):

    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args):
        super(FinetuneTrainer_Time, self).__init__(model, train_dataloader, eval_dataloader, test_dataloader, args
        )

    def iteration(self, epoch, dataloader, full_sort=True, train=True):

        str_code = "train" if train else "test"
        # Setting the tqdm progress bar
        # time_scaler = preprocessing.MinMaxScaler()
        rec_data_iter = tqdm.tqdm(enumerate(dataloader), desc="Prediction_Time_%s:%d" % (str_code, epoch),
                                  total=len(dataloader), bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            # print("len(dataloader) ",len(dataloader))
            for i, batch in rec_data_iter:
                self.optim.zero_grad()
                batch = tuple(t.to(device) for t in batch)
                case_id, input_ids, answers, attributes, next_attribute, time_attributes, remaining_time_attribute = batch
                # 对剩余时间训练时归一化和验证时反归一化
                # print(remaining_time_attribute)
                # print(self.args.time_scaler.inverse_transform(np.array(remaining_time_attribute.cpu()).reshape(-1, 1)).astype(np.float32))

                # remaining_time_attribute = time_scaler.fit_transform(np.array(remaining_time_attribute.cpu()).reshape(-1, 1)).astype(np.float32)
                # print(input_ids.shape)
                # print(attributes.shape)
                # print(time_attributes.shape)
                # sequence_output = self.model.finetune_Hi_Att_Time(input_ids, attributes, time_attributes)
                sequence_output = self.model.finetune_Hi_Att_Time2(input_ids, attributes, time_attributes)
                # print("remaining_time_attribute", remaining_time_attribute)
                # print("sequence_output", sequence_output)
                loss = self.L1Loss(sequence_output, torch.tensor(remaining_time_attribute).to(device))
                # print(loss)
                # loss = criterion(sequence_output, target)
                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.scheduler.step()

            # rec_avg_loss = rec_avg_loss / len(rec_data_iter)

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
            }

            if (epoch + 10) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:

            self.model.eval()

            rec_avg_loss = 0.0
            for i, batch in rec_data_iter:
                batch = tuple(t.to(device) for t in batch)
                case_id, input_ids, answers, attributes, next_attribute, time_attributes, remaining_time_attribute = batch
                # sequence_output = self.model.finetune_Hi_Att_Time(input_ids, attributes, time_attributes)
                sequence_output = self.model.finetune_Hi_Att_Time2(input_ids, attributes, time_attributes)
                # print("sequence_output1", sequence_output)
                remaining_time_attribute= self.args.time_scaler.inverse_transform(np.array(remaining_time_attribute.cpu()).reshape(-1, 1)).astype(np.float32)
                sequence_output = self.args.time_scaler.inverse_transform(np.array(sequence_output.detach().cpu()).reshape(-1, 1)).astype(np.float32)
                # print("remaining_time_attribute", remaining_time_attribute)
                # print("sequence_output2", sequence_output)
                loss = self.L1Loss(torch.tensor(sequence_output).to(device), torch.tensor(remaining_time_attribute).to(device))
                rec_avg_loss += loss.item()
            rec_avg_loss = rec_avg_loss / len(rec_data_iter)
            # writer = SummaryWriter(log_dir='log/new_trans_test/')
            # writer.add_scalar('Trans_Loss_Test', rec_avg_loss, epoch)
            # # writer.add_scalars(main_tag='Rec_avg_Loss', tag_scalar_dict={'train_loss':rec_avg_loss}, global_step=epoch)
            # writer.close()
            with open(self.args.log_file, 'a') as f:
                f.write("valid_loss: " + str(rec_avg_loss) + ' ')

            return [rec_avg_loss]


