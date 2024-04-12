import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
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

origin = ['roms_r', 'cactuBSSN_r', 'exchange2_r', 'cam4_r', 'blender_r', 'namd_r', 'nab_r', 'leela_s', 'xz_r', 'deepsjeng_r', 'xalancbmk_s', 'xalancbmk_r', 'mcf_r', 'lbm_r', 'exchange2_s', 'povray_r', 'bwaves_r', 'imagick_r', 'fotonik3d_r', 'perlbench_r', 'wrf_r', 'leela_r']

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
                index = np.random.randint(100,size=2*k) # k points for each task,other k for the test
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
                index = np.random.randint(100,size=2*k) # k points for each task,other k for the test
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
                        meta_params[name] =  curr_weights[name]-init_weights[name]
                    else:
                        meta_params[name] += curr_weights[name]-init_weights[name]

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

    def eval(self, k, app_num, gradient_steps=10, inner_step_size=0.02):
        x_eval = torch.tensor(data[app_num][:, :-1], dtype=torch.float32, device = device)
        y_eval = torch.tensor(data[app_num][:, [-1]], dtype=torch.float32, device = device)

        index = np.random.randint(100,size=k+30) #30 is test numbers


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
        x1 = range(30)
        plt.plot(x1, test_pred,label='prediction')
        plt.plot(x1, y_test.cpu().numpy(),label='ground truth')
        plt.legend()
        plt.show()

        pecent_loss = mape(test_pred,y_test.cpu().numpy())
        print("mape ",pecent_loss)
        self.model.load_state_dict(meta_weights)
        #return {"pred": pred, "sampled_points":(x_test, y_test)},loss_eval,pecent_loss
        return loss_eval

    def predict(self, x):
        return self.model(x).cpu().data.numpy()

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

# n_shot = 10   #How many samples points to regress on while training
# tasks = 10
# iterations = 1000
# outer_step_size = 0.1
# inner_step_size = 0.01
# inner_grad_steps = 5 #changed
# eval_grad_steps = 10
# eval_iters = 4 #How many testing samples of k different shots you want to run
# first_level = 32
# second_level = 16

np.random.seed(1)

def booth(parameters):
    n_shot = parameters["n_shot"]
    tasks = parameters["tasks"]
    outer_step_size = parameters["outer_step_size"]
    inner_step_size = parameters["inner_step_size"]
    inner_grad_steps = 1#parameters["inner_grad_steps"]
    eval_grad_steps = parameters["eval_grad_steps"]
    first_level = parameters["first_level"]
    second_level = parameters["second_level"]
    #decay = parameters["decay"]

    model = Meta_Wave(first_level,second_level)
    meta = Meta_Learning(model, writer)
    meta.train_maml(n_shot, iterations, outer_step_size, inner_step_size,
            inner_grad_steps,tasks)

    loss=0
    for i in range(357,362): 
        loss+= meta.eval(10, i, eval_grad_steps, inner_step_size)
    return loss#/5

# n_shot = 10   #How many samples points to regress on while training
# tasks = 10
iterations = 1000
# outer_step_size = 0.1
# inner_step_size = 0.01
# inner_grad_steps = 5 #changed
# eval_grad_steps = 10

# first_level = 32
# second_level = 16

from ax.service.ax_client import AxClient
ax = AxClient(enforce_sequential_optimization=False)
ax.create_experiment(
 name="booth_experiment",
 parameters=[
    {"name": "n_shot",
  "type": "range",
  "value_type": "int",
  "bounds": [5, 20],},
  {"name": "tasks",
  "type": "range",
  "value_type": "int",
  "bounds": [10, 30],},
  {"name": "outer_step_size",
  "type": "range",
  "bounds": [0.001, 0.13],
  "log_scale": True},
  {"name": "inner_step_size",
  "type": "range",
  "bounds": [0.001, 0.03], #0.05
   "log_scale": True},
#  {"name": "inner_grad_steps",
#   "type": "range",
#   "bounds": [1,8],
#   "value_type": "int",},
 {"name": "eval_grad_steps",
  "type": "range",
  "bounds": [5,20],
  "value_type": "int",},
  {"name": "first_level",
  "type": "range",
  "value_type": "int",
  "bounds": [32, 64],},
  {"name": "second_level",
  "type": "range",
  "value_type": "int",
  "bounds": [10, 32],},
#  {"name": "decay",
#   "type": "range",
#   "bounds": [0.5,0.99],},
  ],
    objective_name="booth",
    minimize=True,
)
for _ in range(30):
 next_parameters, trial_index = ax.get_next_trial()
 ax.complete_trial(trial_index=trial_index, raw_data=booth(next_parameters))
best_parameters, metrics = ax.get_best_parameters()
print(best_parameters, metrics )
