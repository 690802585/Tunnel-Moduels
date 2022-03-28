import numpy as np
from numpy import *
from Network import Network

def TunnelBranch():

    inp=zeros((3*nk-1,6))
    # inp第一列
    for i in range(0,3*nk-1):
        inp[i,0]=i+1 


    #inp第二 三列
    for i in range(0,nk):
        inp[i,1]=i+1
        inp[i,2]=i+2

    for i in range(0,nk-1):
        inp[nk+i,1]=i+2
        inp[nk+i,2]=i+nk+2

    inp[2*nk-1,1]=1
    inp[2*nk-1,2]=nk+2
    for i in range(0,nk-2):
        inp[2*nk+i,1]=nk+i+2
        inp[2*nk+i,2]=nk+i+3

    inp[3*nk-2,1]=2*nk
    inp[3*nk-2,2]=nk+1
 

    #inp第四列
    inp[0,3]=ZFZ1
    for i in range(1,nk-1):
        inp[i,3]=ZFZ2

    inp[nk-1,3]=ZFZ3

    for i in range(0,nk-1):
        inp[nk+i,3]=ZFZ4


    #inp第五列
    for i in range (0,nk):
        inp[i,4]=1
   
    Network()
    




