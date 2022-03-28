from AirVolume import *
import numpy as np
from numpy import *


Po=zeros((int((n+1)/3),9))#污染物浓度矩阵赋初值
for i in range(0,int((n+1)/3)):
    Po[i,0]=i+1
    Po[i,1]=inp[i,6]
    Po[i,2]=inp[i+int((n+1)/3),6]
    Po[i,3]=DCO

Po[0,4]=Po[0,3]

if Po[0,2]>0:
    Po[0,5]=Po[0,4]/Po[0,1]*Po[0,2]
else:
    Po[0,5]=0

for i in range(1,int((n+1)/3)):
    Po[i,4]=Po[i,3]+Po[i-1,4]-Po[i-1,5]
    if Po[i,2]>0:
        Po[i,5]=Po[i,4]/Po[i,1]*Po[i,2]
    else:
        Po[i,5]=0


for i in range(0,int((n+1)/3)):
    Po[i,6]=abs(Po[i,4])/Po[i,1]*10**6
    Po[i,7]=Po[i,6]*NO2
    Po[i,8]=Po[i,1]/Ar

Po[int((n+1)/3)-1,2]=nan
Po[int((n+1)/3)-1,5]=nan
print(Po)
