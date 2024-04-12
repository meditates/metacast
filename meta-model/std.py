import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
#from torchmetrics.functional import mean_squared_log_error
#from torchmetrics.functional import mean_absolute_error
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import Counter,defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import median_absolute_error as mae
import random
from statistics import mean

device = ('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(1)
#writer = SummaryWriter("./metalog")
class RegressionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][:, :-1], dtype=torch.float32, device = device)
        y = torch.tensor(self.data[idx][:, [-1]], dtype=torch.float32, device = device)
        return x, y

class Meta_Wave(nn.Module):
    def __init__(self, units1,units2):       # input feature:18     output:1
        super(Meta_Wave, self).__init__()
        self.hidden1 = nn.Linear(18, units1)
        self.hidden2 = nn.Linear(units1,units2)
        self.out = nn.Linear(units2, 1)

    def forward(self,x):
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        output = self.out(x)
        return output

class RNN(nn.Module):
    def __init__(self, input_size=18, hidden_size=64, num_layers=2):
        super(RNN, self).__init__()

        # Define the RNN layers
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        # Input x should have shape (batch_size, sequence_length, input_size)

        # Pass input through the RNN layers
        out, _ = self.rnn(x)
        # Take the output from the last time step
        out = out[:, -1, :]

        # Pass the RNN output through the fully connected layer
        out = self.fc(out)

        return out

class CNN(nn.Module):
    def __init__(self, input_size=18, num_filters=16, kernel_size=3):
        super(CNN, self).__init__()

        # Define the CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # Fully connected layer for the final output
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        # Input x should have shape (batch_size, input_size, sequence_length)

        # Pass input through the CNN layers
        x = self.cnn(x)
        # Flatten the output
        x = x.view(x.size(0), -1)

        # Pass the flattened output through the fully connected layer
        x = self.fc(x)

        return x

out = []
origin = ["perlbench_r", "mcf_r", "xalancbmk_r",\
            "deepsjeng_r", "leela_r", "exchange2_r",\
            "xz_r", "bwaves_r", "cactuBSSN_r", "namd_r",\
            "lbm_r", "blender_r", "cam4_r", "nab_r",\
            "fotonik3d_r", "roms_r",
            "xalancbmk_s", "leela_s", "exchange2_s",\
            "povray_r","wrf_r","imagick_r"]
for index_test in range(22):
    random.shuffle(origin)
    print(origin)
    fake_app = origin[:-5]

    # Load the data
    data=[]
    #normalization  if new data : X_test_norm = scaler.transform(X_test)
    scaler = StandardScaler()
    fake = [app+"confs"+str(i)+".csv" for i in range(20) for app in fake_app]

    for name in fake:
        temp = pd.read_csv("fake/"+name, index_col=0, header=0) # 0-17 columns for features, the 18 is second
        #temp = temp.head(100)
        temp['mem-type'] = pd.factorize(temp['mem-type'])[0]
        x=temp.values[:,:-1]
        if not data:
            x_norm = scaler.fit_transform(x)
        else:
            x_norm = scaler.transform(x)
        task_value = np.concatenate((x_norm, temp.values[:, -1:]), axis=1)
        data.append(task_value) # here append is np

    print(len(data))
    for name in origin:
        temp = pd.read_csv("csv/"+name+'confs.csv', index_col=0, header=0) # 0-17 columns for features, the 18 is second
        #temp = temp.head(100)
        temp['mem-type'] = pd.factorize(temp['mem-type'])[0]
        x=temp.values[:,:-1]
        x_norm = scaler.transform(x)
        task_value = np.concatenate((x_norm, temp.values[:, -1:]), axis=1)
        data.append(task_value) # here append is np
    print(len(data))

    dataset = RegressionDataset(data[:-10])
    class Meta_Learning:
        def __init__(self, model): #writer is for tensorboard
            self.model = model.to(device)
            #self.writer = writer

        def train_maml(self, k, iterations, outer_step_size, inner_step_size,
            inner_gradient_steps, tasks=10):
            # loss = 0
            # batches = 0
            myloss=[]

            for iteration in tqdm(range(iterations)):
                init_weights = deepcopy(self.model.state_dict())
                meta_params = {}
                loader = DataLoader(dataset, batch_size=tasks, shuffle=True)
                inputs,labels = next(iter(loader))
                loss=0
                for task in range(tasks):

                    # sample for meta-update
                    index = np.random.randint(140,size=2*k) # k points for each task,other k for the test
                    x,y = inputs[task][index[:k]], labels[task][index[:k]]
                    x_test,y_test = inputs[task][index[k:]], labels[task][index[k:]]

                    for grad_step in range(inner_gradient_steps):
                        loss_base = self.train_loss(x,y)
                        loss_base.backward()
                        for param in self.model.parameters():
                            param.data -= inner_step_size * param.grad.data
                    loss_meta = self.train_loss(x_test, y_test)
                    loss_meta.backward()
                    for name,param in self.model.named_parameters():
                        if(task == 0):
                            meta_params[name] =  param.grad.data
                        else:
                            meta_params[name] += param.grad.data
                    loss+= loss_meta.cpu().data.numpy()
                    #every task uses the init weights
                    self.model.load_state_dict(init_weights)

                learning_rate = outer_step_size * (1 - iteration/iterations)
                self.model.load_state_dict({name: init_weights[name] -
                    learning_rate/tasks * meta_params[name] for name in init_weights})

                #self.writer.add_scalar('MAML/Training/Loss/', loss/batches, iteration)
                myloss.append(loss)
            #print(myloss)
            # plt.plot(range(iterations),myloss)
            # plt.title("maml train loss")
            # plt.show()
            # torch.save(self.model.state_dict(),"maml.pth")


        def train_reptile(self, k, iterations, outer_step_size, inner_step_size,
            inner_gradient_steps,tasks=10):
            myloss=[]

            for iteration in tqdm(range(iterations)):
                init_weights = deepcopy(self.model.state_dict())
                meta_params = {}
                loader = DataLoader(dataset, batch_size=tasks, shuffle=True)
                inputs,labels = next(iter(loader))
                loss=0
                for task in range(tasks):

                    # sample for meta-update
                    index = np.random.randint(140,size=2*k) # k points for each task,other k for the test
                    x,y = inputs[task][index[:k]], labels[task][index[:k]]
                    x_test,y_test = inputs[task][index[k:]], labels[task][index[k:]]

                    for grad_step in range(inner_gradient_steps):
                        loss_base = self.train_loss(x,y)
                        loss_base.backward()
                        for param in self.model.parameters():
                            param.data -= inner_step_size * param.grad.data
                    loss_meta = self.train_loss(x_test, y_test)
                    loss_meta.backward()
                    loss+= loss_meta.cpu().data.numpy()

                    curr_weights = self.model.state_dict()
                    for name in curr_weights:
                        if(task == 0):
                            meta_params[name] =  (curr_weights[name]-init_weights[name])/inner_gradient_steps
                        else:
                            meta_params[name] += (curr_weights[name]-init_weights[name])/inner_gradient_steps

                    self.model.load_state_dict(init_weights)

                learning_rate = outer_step_size * (1 - iteration/iterations)

                self.model.load_state_dict({name: (init_weights[name] - learning_rate *
                    meta_params[name]/tasks) for name in init_weights})

                #self.writer.add_scalar('Reptile/Training/Loss/', loss/batches, iteration)
                myloss.append(loss/tasks)


        def train_loss(self, x, y):
            self.model.zero_grad()
            out = self.model(x)
            loss = (out - y).pow(2).mean()
            #loss = mean_absolute_error(out,y)
            return loss

        #app_num could be
        def eval(self, k, app_num, gradient_steps=10, inner_step_size=0.02,graph=True,testnum=30,myindex=140):
            x_eval = torch.tensor(data[app_num][:, :-1], dtype=torch.float32, device = device)
            y_eval = torch.tensor(data[app_num][:, [-1]], dtype=torch.float32, device = device)
            index = np.random.randint(140,size=2*k)  # k points for each task,other k for the test
            x_p,y_p = x_eval[index[:k]], y_eval[index[:k]]
            pred = [self.predict(x_p)]
            meta_weights = deepcopy(self.model.state_dict())
            for i in range(gradient_steps):
                loss_base = self.train_loss(x_p,y_p)
                loss_base.backward()
                for param in self.model.parameters():
                    param.data -= inner_step_size * param.grad.data
                pred.append(self.predict(x_p))

            # loss = np.power(pred[-1] - y_p.cpu().numpy(), 2).mean()
            # pecent_loss = mape(pred[-1],y_p.cpu().numpy())
            # print("prediction",pred[-1])
            # print("ground truth",y_p.cpu().numpy())

            test_index=index[k:] #
            x_test,y_test = x_eval[test_index], y_eval[test_index]

            test_pred = self.predict(x_test)
            loss_eval = np.power(test_pred - y_test.cpu().numpy(), 2).mean()
            # print("prediction",test_pred)
            # print("ground truth",y_test.cpu().numpy())
            if graph==True:
                plt.scatter(y_test.cpu().numpy(), test_pred, color='blue')
                plt.plot([0, 2400], [0, 2400], linestyle='--', color='gray')
                plt.xlim(min(y_test.cpu()) - 10, max(y_test.cpu()) + 10)
                plt.ylim(min(test_pred) - 10, max(test_pred) + 10)
                plt.title('Ground Truth vs. Predictions')
                plt.xlabel('Ground Truth (seconds)')
                plt.ylabel('Predictions (seconds)')

                plt.show()
            percent_loss = mape(y_test.cpu().numpy(),test_pred)
            print("mape ",percent_loss)
            self.model.load_state_dict(meta_weights)
            #return {"pred": pred, "sampled_points":(x_test, y_test)},loss_eval,percent_loss
            return loss_eval,percent_loss

        def predict(self, x):
            return self.model(x).cpu().data.numpy()


    parameters={'n_shot': 9, 'tasks': 18, 'outer_step_size': 0.001, 'inner_step_size': 0.009263018012657349, 'inner_grad_steps': 6, 'eval_grad_steps': 18, 'first_level': 40, 'second_level': 31}
    n_shot = parameters["n_shot"]
    tasks = parameters["tasks"]
    outer_step_size = parameters["outer_step_size"]
    inner_step_size = parameters["inner_step_size"]
    inner_grad_steps = parameters["inner_grad_steps"]
    eval_grad_steps = parameters["eval_grad_steps"]
    first_level = parameters["first_level"]
    second_level = parameters["second_level"]

    iterations = 7000

    model = Meta_Wave(first_level,second_level)
    meta = Meta_Learning(model)
    meta.train_maml(n_shot, iterations, outer_step_size, inner_step_size,
            inner_grad_steps,tasks)
    #torch.save(meta.model.state_dict(),"maml_7000_321_"+str(index_test)+".pth")
    loss = []
    for _ in range(10):
      for index in range(340,362):
          print("app index is ", index)
          loss_one = meta.eval(10, index, eval_grad_steps, inner_step_size,graph=False)
          print("loss is ",loss_one)
          loss.append(loss_one[1])
    out.append(mean(loss))
print("out",out)
