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

from statistics import mean
class RegressionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][:, :-1], dtype=torch.float32, device = device)
        y = torch.tensor(self.data[idx][:, [-1]], dtype=torch.float32, device = device)
        return x, y

class Transfer:
    def __init__(self, model): 
        self.model = model.to(device)

    def train(self, k, iterations, lr,steps):

        for iteration in tqdm(range(iterations)):

            loader = DataLoader(dataset, batch_size=1, shuffle=True)
            loss=0
            for i in range(len(data[:-10])): #len(data[:-10])
                inputs,labels = next(iter(loader))
                # sample for meta-update
                index = np.random.choice(140,2*k,replace=False) # k points for each task,other k for the test
                x,y = inputs[0][index[:k]], labels[0][index[:k]]
                x_test,y_test = inputs[0][index[k:]], labels[0][index[k:]]

                for grad_step in range(steps):
                    loss_base = self.train_loss(x,y)
                    loss_base.backward()
                    for param in self.model.parameters():
                        param.data -= lr * param.grad.data


                loss_test = self.train_loss(x_test, y_test)
                loss_test.backward()
                loss+= loss_test.cpu().data.numpy()

    def train_loss(self, x, y):
        self.model.zero_grad()
        out = self.model(x)
        loss = (out - y).pow(2).mean()
        #loss = mean_absolute_error(out,y)
        return loss

    #app_num could be 378-382
    def eval(self, k, app_num, gradient_steps=10, lr=0.02,graph=True):
        x_eval = torch.tensor(data[app_num][:, :-1], dtype=torch.float32, device = device)
        y_eval = torch.tensor(data[app_num][:, [-1]], dtype=torch.float32, device = device)

        index = np.random.choice(140,2*k,replace=False) #30 is test numbers


        x_p,y_p = x_eval[index[:k]], y_eval[index[:k]]
        pred = [self.predict(x_p)]
        meta_weights = deepcopy(self.model.state_dict())
        for i in range(gradient_steps):
            loss_base = self.train_loss(x_p,y_p)
            loss_base.backward()
            for param in self.model.parameters():
                param.data -= lr * param.grad.data
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
            x1 = range(30)
            plt.scatter(x1, test_pred,label='prediction')
            plt.scatter(x1, y_test.cpu().numpy(),label='ground truth')
            plt.legend()
            plt.show()

        pecent_loss = mape(test_pred,y_test.cpu().numpy())
        print("mape ",pecent_loss)
        self.model.load_state_dict(meta_weights)
        #return {"pred": pred, "sampled_points":(x_test, y_test)},loss_eval,pecent_loss
        return loss_eval,pecent_loss

    def predict(self, x):
        return self.model(x).cpu().data.numpy()

class Wave(nn.Module):
    def __init__(self, units1,units2):       # input feature:18     output:1
        super(Wave, self).__init__()
        self.hidden1 = nn.Linear(18, units1)
        self.hidden2 = nn.Linear(units1,units2)
        self.out = nn.Linear(units2, 1)

    def forward(self,x):
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        output = self.out(x)
        return output

n_shot = 9#parameters["n_shot"]
steps = 18#parameters["outer_step_size"]
lr = 0.03#parameters["inner_step_size"]
#eval_grad_steps = 10#parameters["eval_grad_steps"]
first_level = 40#parameters["first_level"]
second_level = 30#parameters["second_level"]


iterations = 20

tra = []
origin = ["perlbench_r", "mcf_r", "xalancbmk_r",\
            "deepsjeng_r", "leela_r", "exchange2_r",\
            "xz_r", "bwaves_r", "cactuBSSN_r", "namd_r",\
            "lbm_r", "blender_r", "cam4_r", "nab_r",\
            "fotonik3d_r", "roms_r",
            "xalancbmk_s", "leela_s", "exchange2_s",\
            "povray_r","wrf_r","imagick_r"]
for index_test in range(20):###
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

    trans_model = Wave(first_level,second_level)
    trans = Transfer(trans_model)
    trans.train(n_shot, iterations, lr,steps)
    loss=[]
    for index in range(352,362):
        print("app index is ", index)
        loss_one= trans.eval(10, index,graph=False)
        print("loss is ",loss_one)
        loss.append(loss_one[1])
    tra.append(mean(loss))

#random
lr = 0.05
ran=[]
for index_test in range(20):###
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

    trans_model = Wave(first_level,second_level)
    rand = Transfer(trans_model)
    #trans.train(n_shot, iterations, lr,steps)
    loss=[]
    for index in range(352,362):
        print("app index is ", index)
        loss_one= rand.eval(10, index,graph=False)
        print("loss is ",loss_one)
        loss.append(loss_one[1])
    ran.append(mean(loss))
print("tra",tra)
print("ran",ran)
