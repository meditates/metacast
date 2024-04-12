import os
import sys
import time
import numpy as np
import pandas as pd
from cal_stats import *

def getStat(csv_dir,num,app):
    df = pd.read_csv(csv_dir, index_col=0, header=0)
    for i in range(140): #140
        row = df.iloc[i].tolist()

        row_string = ' '.join(map(str, row))

        command = "./reload.sh "+num+" "+ app+ " "+ str(i)+" "+ row_string
        
        #this is for speed control
        command0 = 'pgrep gem5.fast | wc -l'
        output0 = os.popen(command0).read()
        num_process = int(output0.strip())
        while num_process>40:
            print("wait for less process running! Now i is:",i)
            time.sleep(600)
            command0 = 'pgrep gem5.fast | wc -l'
            output0 = os.popen(command0).read()
            num_process = int(output0.strip())
        os.system(command)

if __name__ == '__main__':
    num = "429"
    app = "mcf"
    getStat("test.csv",num,app)
