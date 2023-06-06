import numpy as np
import tqdm
import random
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from utils import recall_at_k, ndcg_k, get_metric
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

        rec_data_iter = tqdm.tqdm(enumerate(dataloader), desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader), bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            # print("len(dataloader) ",len(dataloader))
            for i, batch in rec_data_iter:
                self.optim.zero_grad()
                # 0. batch_data will be sent into the device(GPU or CPU)
                # batch = tuple(t.to(self.device) for t in batch)
                case_id, input_ids, target_pos, target_neg, answers, attributes = batch
                input_ids, attributes, answers = input_ids.to(device), attributes.to(device), answers.to(device)
                # print(input_ids.shape)
                # print(attributes.shape)
                sequence_output = self.model.finetune_Hi_remaining(input_ids, attributes)
                # answers_scaler = preprocessing.StandardScaler()
                # y = answers_scaler.fit_transform(answers.cpu()).astype(np.float32)
                # y = torch.tensor(y, dtype=torch.float32, device=device)
                loss = self.L1Loss(sequence_output, answers)
                # print(loss)
                # loss = criterion(sequence_output, target)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
            # rec_avg_loss = rec_avg_loss / len(rec_data_iter)
            self.scheduler.step()
            # writer = SummaryWriter(log_dir='log/time_trans_train/')
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
            maes, mses, rmses = [], [], []
            for i, batch in rec_data_iter:

                case_id, input_ids, target_pos, target_neg, answers, attributes = batch
                input_ids, attributes, answers = input_ids.to(device), attributes.to(device), answers.to(device)
                # print("input_ids.shape ",input_ids.shape)
                sequence_output = self.model.finetune_Hi_remaining(input_ids, attributes)
                print(sequence_output)
                # print(y_pred)
                print(answers)
                # answers_scaler = preprocessing.StandardScaler()
                # y = answers_scaler.fit_transform(answers.cpu()).astype(np.float32)
                # y = torch.tensor(y, dtype=torch.float32, device=device)
                loss = self.L1Loss(sequence_output, answers)
                # loss = self.L1Loss(sequence_output, answers.squeeze(-1))
                rec_avg_loss += loss.item()
                maes.append(metrics.mean_absolute_error(sequence_output.cpu().detach().numpy(), answers.cpu()))
                mses.append(metrics.mean_squared_error(sequence_output.cpu().detach().numpy(), answers.cpu()))
                rmses.append(np.sqrt(metrics.mean_squared_error(sequence_output.cpu().detach().numpy(), answers.cpu())))

            rec_avg_loss = rec_avg_loss / len(rec_data_iter)
            maes.append(np.mean(maes))
            mses.append(np.mean(mses))
            rmses.append(np.mean(rmses))
            print('Average MAE across all prefixes:', np.mean(maes))
            print('Average MSE across all prefixes:', np.mean(mses))
            print('Average RMSE across all prefixes:', np.mean(rmses))
            # writer = SummaryWriter(log_dir='log/time_trans_test/')
            # writer.add_scalar('Trans_Loss_Test', rec_avg_loss, epoch)
            # # # writer.add_scalars(main_tag='Rec_avg_Loss', tag_scalar_dict={'train_loss':rec_avg_loss}, global_step=epoch)
            # writer.close()
            with open(self.args.log_file, 'a') as f:
                f.write("valid_loss: " + str(rec_avg_loss) + ' ')
                f.write("maes: " + str(maes) + ' ')
                f.write("mses: " + str(mses) + ' ')
                f.write("rmses: " + str(rmses) + ' ')

            return [rec_avg_loss]




