import numpy as np
import os
import argparse
import pandas as pd
import random

num_list=['0','1','2','3','4','5','6','7','8','9','.']

def isdig(a): 
    if a in num_list:
        return True
    return False

def fin(para_sym,line):
    flag=False
    para=0
    if para_sym in line:
        line=line.split(para_sym)[1]
        temp=""
        for element in line:
            if isdig(element):
                temp+=element
            if element=="#":
                break
        para=float(temp)
        flag=True
    return flag,para

def read_weight_file(weight_route,sim_route):
    f_w=open(weight_route)
    f_s=open(sim_route)
    weight=f_w.readlines()
    sim=f_s.readlines()
    length=len(weight)
    wei_sim=np.zeros((length,2))
    for i in range(length):
        temp=""
        for num in weight[i]:
            if isdig(num):
                temp+=num
                continue
            break
        wei_sim[i,0]=float(temp)
        temp=""
        for num in sim[i]:
            if isdig(num):
                temp+=num
                continue
            break
        wei_sim[i,1]=float(temp)
    sorted_wei_sim=sorted(wei_sim,key=lambda x:x[1])
    res=[]
    for item in sorted_wei_sim:
        res.append(item[0])
    return res

def cal_sim(sim_name,index,fakeweight=None):
    route="/users/meditate/RELOAD/"+sim_name+str(index)
    weight_route="/users/meditate/result_"+sim_name+"/weight_file"
    sim_route="/users/meditate/result_"+sim_name+"/simpoint_file"

    dir=[]
    res=[]
    stats_name='stats.txt'
    start_sym='---------- Begin Simulation Statistics ----------'
    seconds_sym='sim_seconds'

    if fakeweight:
        weight=fakeweight
    else:
        weight=read_weight_file(weight_route,sim_route)
    #print(weight) 
    for root,subdir,file in os.walk(route):
        if root == route:
            for i in subdir:
                dir.append(i)
        for subfile in file:
            #print(subfile)
            if subfile == stats_name:
                fp=open(root+'/'+subfile)
                res.append(fp.readlines())
    
    num=0
    second=0
    for stats in res:
        tot=-1
        for line in stats:
            if start_sym in line:
                tot+=1
 
            if tot == 1:                
                flag,para=fin(seconds_sym,line)
                if flag == True:
                    #print("index",int(dir[num]),para)
                    second+=para*weight[int(dir[num])-1]
        num+=1

    command = 'gunzip -c '+"/users/meditate/result_"+sim_name+'/simpoint.bb.gz | wc -l'
    output = os.popen(command).read()

    num_lines = int(output.strip())
    second*=num_lines
    print("Simpoint results are:")
    print("sim_second =",second)

    
    return second

if __name__=="__main__":    
    #print(read_weight_file(weight_route,sim_route))
    for app in ["perlbench","bzip2","gcc","mcf",
    "gobmk","hmmer","h264ref","omnetpp"
    "gamess","milc","soplex","tonto","lbm"]:
        """
        df = pd.read_csv('test.csv', index_col=0, header=0)
        df["sim_second"]=0.0
        for i in range(1):
            print("this is app :",app,"index", i)
            df.at[i,"sim_second"]=cal_sim(app,i)
        df.to_csv(app+"confs.csv")
        """
        weight_route="/users/meditate/result_"+app+"/weight_file"
        f_w=open(weight_route)
        weight0=f_w.readlines()
        length=len(weight0)
        numbers = [random.random() for _ in range(length-1)]
        # Sort the numbers in ascending order
        numbers.sort()
        # Add a 0 to the beginning and a 1 to the end of the list
        numbers = [0] + numbers + [1]
        # Calculate the differences between each adjacent pair of numbers
        weight = [numbers[i+1] - numbers[i] for i in range(length)]
        for k in range(20):
            df = pd.read_csv('test.csv', index_col=0, header=0)
            df["sim_second"]=0.0      


            for i in range(100):
                print("this is app :",app,"index", i)
                df.at[i,"sim_second"]=cal_sim(app,i,fakeweight=weight)
            df.to_csv("fake/"+app+"confs"+str(k)+".csv")


