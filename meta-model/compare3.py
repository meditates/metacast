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
import matplotlib
matplotlib.use('Agg')
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
import re



file_path = 'log'  # Replace 'your_file.txt' with the actual file path
with open(file_path, 'r') as file:
    for line in file:
        if line.startswith('out '):
            # Split the line by spaces to get the array after "out"
            parts = line.strip().split()
            array_str = ' '.join(parts[1:])
            array = re.findall(r'[-+]?\d*\.\d+|\d+', array_str)
            out = [float(x) for x in array]
file_path = 'log_tra'  # Replace 'your_file.txt' with the actual file path
with open(file_path, 'r') as file:
    for line in file:
        if line.startswith('tra '):
            # Split the line by spaces to get the array after "tra"
            parts = line.strip().split()
            array_str = ' '.join(parts[1:])
            array = re.findall(r'[-+]?\d*\.\d+|\d+', array_str)
            tra = [float(x) for x in array]
        elif line.startswith('ran '):
            # Split the line by spaces to get the array after "ran"
            parts = line.strip().split()
            array_str = ' '.join(parts[1:])
            array = re.findall(r'[-+]?\d*\.\d+|\d+', array_str)
            ran = [float(x) for x in array]
#out=[0.150846409201622, 0.16569091081619263, 0.35643693804740906, 0.58947386145591736, 0.16520620584487915, 0.15053632259368896, 0.18350391705830893, 0.18917444348335266, 0.22297938664754233, 0.33858413636684418, 0.1633978521823883, 0.175554370880127, 0.1925318670272827, 0.15328829109668732, 0.17643693736396873,0.1713636584487386,0.1928271976418058,0.1993680920153632,0.18243642151355743, 0.1519764018058777, 0.44589646657308, 0.170920429611206, 0.2131582796573639, 0.1814279854297638, 0.16937412321567535, 0.18358197808265686]
#tra=[0.45679566, 1.0572833, 8.265229, 1.5017613, 1.0412745, 0.46621513, 1.4516146, 0.7097329, 0.6471089, 1.8108257, 0.6817465, 1.7739993, 3.9928856, 0.5593009, 1.536317, 1.3806574, 1.3714627, 0.5987247, 1.1030364, 2.310776]
#ran = [4.0196915, 6.8091083, 4.9590774, 2.9654834, 2.5544827, 2.8198411, 5.274068, 3.6314406, 2.7423983, 5.8006616, 4.6376066, 5.375067, 11.31024, 2.6513686, 2.1620903, 5.9738746, 7.106769, 4.089224, 6.015715, 10.300849]

data = [out, tra, ran]
arr = np.array(data, dtype=object)
plt.figure(figsize=(12, 9))
a=plt.boxplot(arr, labels=['Meta Learning', 'Transfer Learning', 'random initialization'],
              showmeans=True, meanline=True, patch_artist=True,boxprops=dict(facecolor='white'))
# Customize the mean line appearance
mean_line = a['means'][0]
mean_line.set_linestyle('--')  # Dashed line style
mean_line.set_color('green')    # Line color
# Customize the median line appearance
median_line = a['medians'][0]
median_line.set_linestyle('-')  # Solid line style
median_line.set_color('orange')    # Line color

plt.ylim(0, 6.2)#
plt.yticks(np.arange(0, 6.5, 0.5))
#plt.xlabel('three methods')
plt.ylabel('average MAPE')
plt.title('Box Plot of MAPE for three methods')
plt.legend([mean_line, median_line], ['Mean', 'Median'], loc='best')

plt.savefig('compare3.png')

