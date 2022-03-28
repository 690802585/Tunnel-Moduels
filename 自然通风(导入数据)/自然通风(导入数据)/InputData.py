import numpy as np
from numpy import *
from function import *
inpor = np.loadtxt("in.txt")
np.set_printoptions(linewidth=400)

L=1440#隧道长度
vt=60#行车速度
N1=3220#风速同向的设计高峰小时交通量
nk=int((len(inpor)+1)/3)#隧道分段数

acs=2.13#小型车正面投影面积
Acl=5.37#大型车正面投影面积
rl=0#大型车比例
nc1=0.3525#小型车空气阻力系数
nc2=0.3562#大型车空气阻力系数
Am=(1-rl)*acs*nc1+rl*Acl*nc2#汽车等效阻抗面积
B=12#计算宽度
H=5.5#计算高度
Ar=B*H#隧道净空断面积
rou=1.2#通风计算点空气密度
h=0#通风计算点海拔高程
x1=acs/Ar#小型车在隧道行车空间的占积率
x2=Acl/Ar#大型车在隧道行车空间的占积率
T=298.5#通风计算点夏季气温
Vn=3#自然风速


vt1=vt/3.6#与风速同向的各工况车速（单位m/s）
n1=zeros(3*nk-1)
for i in range(0,nk):
    n1[i]=N1*inpor[i][3]/(3600*vt1)#隧道内与风速同向车辆数
Qr=0#隧道设计风量

tflcs=Am/Ar*rou/2*n1
Acs=tflcs*(1/Ar)**2
Bcs=tflcs*(-2)*vt1/Ar
Ccs=tflcs*vt1**2#交通通风力参数

#隧道阻力系数 
ycfz=0.02#沿程风阻系数
Cr=2*(B+H)#隧道断面周长
Dr=4*Ar/Cr#隧道断面当量直径
ycz=zeros(nk)
for i in range(0,nk):
    ycz[i]=ycfz*inpor[i][3]/Dr*rou/2*1/Ar**2#隧道沿程风阻系数
ne1=0.5
ne2=0.4
ne3=1.4
yczrk=ne1*rou/2*1/Ar**2
Rne2=ne2*rou/2*1/Ar**2
yczck=ne3*rou/2*1/Ar**2
ZFZ1=yczrk+ycz[0]
ZFZ2=zeros(nk-2)
for i in range(0,nk-2):
    ZFZ2[i]=Rne2+ycz[i+1]
ZFZ3=yczck+ycz[nk-1]#总风阻系数

#通风井阻力系数
B0=10#风井长度
H0=4#风井宽度
ycfz0=0.02#沿程风阻参数
Ar0=B0*H0
Cr0=2*(B0+H0)
Dr0=4*Ar0/Cr0
ycs=zeros(nk-1)
for i in range(0,nk-1):
    ycs[i]=ycfz0*inpor[i+nk][3]/Dr0*rou/2*1/Ar0**2#风井沿程风阻系数
ne4=2.1
Rne4=ne4*rou/2*1/Ar0**2
ZFZ4=zeros(nk-1)
for i in range(0,nk-1):
    ZFZ4[i]=Rne4+ycs[i]#总风阻系数

CO=2.46E-03#CO排放量
DCO=zeros(nk)
for i in range(0,nk):
    DCO[i]=CO/L*inpor[i][3]#每段CO排放量
NOX=2.04E-04#NOX排放量
DNOX=zeros(nk)
NO2=zeros(nk)
for i in range(0,nk):
    DNOX[i]=NOX/L*inpor[i][3]#每段NOX排放量
    NO2[i]=DNOX[i]/DCO[i]


PW=zeros(3*nk-1)
for i in range(0,nk):
    PW[i]=(yczrk+inpor[i][3]/Dr*ycz[i])*rou/2*Vn**2#自然风阻