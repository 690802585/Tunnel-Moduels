import numpy as np
from numpy import *
from function import *
import matplotlib.pyplot as plt
from pylab import mpl  
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus']=False
np.set_printoptions(linewidth=400)

inpor = np.loadtxt("in.txt")

L=1440#隧道长度
vt=10#行车速度
N1=1610#风速同向的设计高峰小时交通量
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
  

#组建inp矩阵


inp=zeros((3*nk-1,11))
# inp第一列(分支编号)
for i in range(0,3*nk-1):
    inp[i,0]=i+1 


#inp第二 三列(节点编号)
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
 

#inp第四列(总阻力系数)
inp[0,3]=ZFZ1
for i in range(1,nk-1):
    inp[i,3]=ZFZ2[i-1]

inp[nk-1,3]=ZFZ3

for i in range(0,nk-1):
    inp[nk+i,3]=ZFZ4[i]


#inp第五列(交通通风力标识)
for i in range (0,nk):
    inp[i,4]=1

#inp第六列 温升压力(暂不考虑)

#inp第七、八、九列(交通通风系数)
inp[:,6]=Ccs
inp[:,7]=Bcs
inp[:,8]=Acs

#inp第十列(自然风压)
inp[:,9]=PW

#inp第十一列(沿程阻力系数)
for i in range(0,nk):
    inp[i,10]=ycz[i]
for i in range(0,nk-1):
    inp[i+nk,10]=ycs[i]


V1=set(inp[:,1])
V2=set(inp[:,2])
V=V1.union(V2) #节点集合：始节点和末节点的并集
m=0
for i in V:
    m+=1     #节点数：节点集合的行数  M个节点       
#print(V)
#print(m)
n=np.size(inp,0)  #分支数：in矩阵的行数  N条分支
#print(n)
#print(inp)


#计算关联矩阵
B=zeros((m,n))#生成m行n列0矩阵
for i in range(1,n+1):#对所有分支循环
    B[int(inp[i-1,1])-1,i-1]=1#in的第2列始节点
    B[int(inp[i-1,2])-1,i-1]=-1#in的第3列末节点
#print(B)

#计算最小生成树（prim算法）
s=np.array([1])#节点集合,加入第1个节点
tree=[]
tree=np.array(tree) #树枝集合

for k in range(0,m-1):#循环m-1次，每次加入1树枝
    s1=np.array(s-1, dtype = np.uint32)
    [a,b]= np.nonzero(B[s1,:])#寻找s的所有关联分支
    # print(a)
    # print(b)  
    Rmin=10000;#最小风阻赋初值

    bsize=size(b)
    #print(bsize)
    for i in range(0,bsize):#从1到关联分支数（b的行数）
        #print(b[i])
        e=np.where(s==inp[b[i],1])
        f=np.where(s==inp[b[i],2])
        #print(e)
        #print(f)
        if(np.size(e) and np.size(f)):
            continue #跳过该分支
        if(inp[b[i],3]<Rmin):#in的第四列为风阻
            Rmin=inp[b[i],3]#更新最小风阻
            #print(Rmin)
            select1=b[i]#更新选择的树枝
            #print(select1)
            #print()

    tree=np.hstack([tree,select1])#树中加入树枝
    #print(tree)
    g=np.where(s==inp[select1,1])
    h=np.where(s==inp[select1,2])
    if(np.size(g)>0):#始节点在节点集合中
        s=np.hstack([s,inp[select1,2]])#加入末节点
    elif (np.size(h)>0):#末节点在节点集合中
        s=np.hstack([s,inp[select1,1]])#加入始节点
    #print()


#tree=np.array([0,5,6,7,1])


#计算余树
E=inp[:,0]
E=array(E-1, dtype = np.uint32)
#print(E)
tree=array(tree, dtype = np.uint32)
E=np.delete(E,tree)
#print(E)
E=array(E, dtype = np.uint32)
cotree=E



#计算基本关联矩阵
temp=0#参考节点
B=np.delete(B,temp,0)#删除第temp行
#print(B)

#计算独立回路矩阵
B11=B[:,cotree]#余树所在列
#print(B11)
B12=B[:,tree]#树所在列
#print(B12)
B=np.hstack([B11,B12])
C11=eye(n-m+1)#n-m+1行n-m+1列单位矩阵
#print(B12)
C12=-(B11).T@(np.linalg.inv(B12)).T
#print(C12)
C=np.hstack([C11,C12])#独立回路矩阵
#print(C)
#将in矩阵按余树在前，树在后的方法排序

in1=inp[cotree,:]#余树所在行
#print(in1)
in2=inp[tree,:]#树所在行
#print(in2)
inp=np.vstack([in1,in2])
#print(inp)


Hf=zeros(n)#风机风压列向量(n行1列0矩阵)
#风量赋初值
Q=zeros(n)
Q[0:n]=0
#print(C)
#print(inp)


################################################################################

#开始迭代计算
for k in range(0,30000):#控制最大迭代次数


    maxdq=0
    maxf=0

    for i in range(0,n-m+1):#对所有回路循环
        f=0;#累加前赋初值
        df=1;#累加前赋初值
        for j in range(0,n):#对所有分支循环
            fanindex=int(inp[j,4])#inp矩阵的第5列为交通通风系数
            if(fanindex==1):
               Hf[j]=inp[j,6]+inp[j,7]*Q[j]+inp[j,8]*(Q[j]**2)#计算交通通风力
            elif(fanindex==2):
               Hf[j]=inp[j,6]+inp[j,7]*Q[j]+inp[j,8]*(Q[j]**2)#计算交通通风力
            elif(fanindex==3):
               Hf[j]=inp[j,6]+inp[j,7]*Q[j]+inp[j,8]*(Q[j]**2)#计算交通通风力
            elif(fanindex==4):
               Hf[j]=inp[j,6]+inp[j,7]*Q[j]+inp[j,8]*(Q[j]**2)#计算交通通风力
            elif(fanindex==5):
               Hf[j]=inp[j,6]+inp[j,7]*Q[j]+inp[j,8]*(Q[j]**2)#计算交通通风力
            else:
                Hf[j]=0
            f=f+C[i,j]*(inp[j,3]*Q[j]*abs(Q[j])+PW[j]-Hf[j]-inp[j,5])#inp矩阵第4列为风阻，第6列为温升压力(暂不考虑)
            df=df+(abs(C[i,j])*2*inp[j,3]*abs(Q[j])-inp[j,7]-2*inp[j,8]*abs(Q[j]))
        dq=-f/df#计算回路风量修正值
        for j in range(0,n):
            Q[j]=Q[j]+C[i,j]*dq#修正回路风量
        f=0.0#累加前赋初值
        for j in range(0,n):
            f=f+C[i,j]*(inp[j,3]*Q[j]*abs(Q[j])+PW[j]-Hf[j]-inp[j,5])#重新计算回路不平衡风压

        if(abs(dq)>maxdq):
            maxdq=abs(dq)#计算最大回路风量修正值
        if(abs(f)>maxf):
            maxf=abs(f)#计算最大回路不平衡风压

    if(abs(maxdq)<1e-06 and abs(maxf)<1e-06):
        break


    #隧道入口
    pp=(n+1)/3+1
    vv1=np.where(inp[:,0]==pp)#Q3
    vv2=np.where(inp[:,0]==pp-(n+1)/3)#Q1
    vv3=np.where(inp[:,0]==pp)#Q3
    vv4=np.where(inp[:,0]==pp-((n+1)/3-1))#Q2

    if (Q[vv1]>0):
        a1=abs(Q[vv1]/Q[vv2])#Q3/Q1
        a2=abs(Q[vv4]/Q[vv2])#Q2/Q1
        a11=f1(a1)+inp[vv1,10];
        a22=f2(a2)+yczrk+inp[vv2,10];
        inp[vv1,3]=a11;
        inp[vv2,3]=a22;
    
    else:
        a3=abs(Q[vv3]/Q[vv4])#Q3/Q2
        a4=abs(Q[vv2]/Q[vv4])#Q1/Q2
        a33=f3(a3)+inp[vv3,10];
        a44=f4(a4)+inp[vv4,10];
        inp[vv3,3]=a33;
        inp[vv4,3]=a44;

    #隧道出口
    pp=(n+1)/3*2-1 
    vv1=np.where(inp[:,0]==pp)#Q3
    vv2=np.where(inp[:,0]==pp-(n+1)/3)#Q1
    vv3=np.where(inp[:,0]==pp)#Q3
    vv4=np.where(inp[:,0]==pp-((n+1)/3-1))#Q2
    
    if (Q[vv1]>0):
        a1=abs(Q[vv1]/Q[vv2])#Q3/Q1
        a2=abs(Q[vv4]/Q[vv2])#Q2/Q1
        a11=f1(a1)+inp[vv1,10];
        a22=f2(a2)+inp[vv2,10];
        inp[vv1,3]=a11;
        inp[vv2,3]=a22;
    else:
        a3=abs(Q[vv3]/Q[vv4])#Q3/Q2
        a4=abs(Q[vv2]/Q[vv4])#Q1/Q2
        a33=f3(a3)+inp[vv3,10];
        a44=f4(a4)+yczck+inp[vv4,10];
        inp[vv3,3]=a33;
        inp[vv4,3]=a44;

    for pp in range(int((n+1)/3+2),int((n+1)/3*2-1)):
        
        vv1=np.where(inp[:,0]==pp)#Q3
        vv2=np.where(inp[:,0]==pp-(n+1)/3)#Q1
        vv3=np.where(inp[:,0]==pp)#Q3
        vv4=np.where(inp[:,0]==pp-((n+1)/3-1))#Q2
        
        if (Q[vv1]>0):
            a1=abs(Q[vv1]/Q[vv2])#Q3/Q1
            a2=abs(Q[vv4]/Q[vv2])#Q2/Q1
            a11=f1(a1)+inp[vv1,10];
            a22=f2(a2)+inp[vv2,10];
            inp[vv1,3]=a11;
            inp[vv2,3]=a22;
        else:
            a3=abs(Q[vv3]/Q[vv4])#Q3/Q2
            a4=abs(Q[vv2]/Q[vv4])#Q1/Q2
            a33=f3(a3)+inp[vv3,10];
            a44=f4(a4)+inp[vv4,10];
            inp[vv3,3]=a33;
            inp[vv4,3]=a44;


HR=inp[:,3]*abs(Q)*Q
H=HR-Hf-inp[:,5]

Q=np.array(Q).reshape((-1,1))
H=np.array(H).reshape((-1,1))
HR=np.array(HR).reshape((-1,1))
Hf=np.array(Hf).reshape((-1,1))

inp=np.hstack([inp,Q])
inp=np.hstack([inp,H])
inp=np.hstack([inp,HR])
inp=np.hstack([inp,Hf])

inp=inp[np.lexsort(inp[:,::-1].T)]
print('      分支编号   始节点    末节点    风机索引      风量        风阻系数          风压          阻力        交通通风力      风井热压差     自然通风力\n\n')
for i in range(0,n):
           print('%10d%10d%10d%10d%15.4f%15.8f%15.4f%15.4f%15.4f%15.4f%15.4f' %(inp[i,0],inp[i,1],inp[i,2],inp[i,4],inp[i,11],inp[i,3],inp[i,12],inp[i,13],inp[i,14],inp[i,5],PW[i]))



'''---------------------------------------------------------------------------------------------------------------------------------------------------'''

Po=zeros((int((n+1)/3),9))#污染物浓度矩阵赋初值

for i in range(0,int((n+1)/3)):
    Po[i,0]=i+1
    Po[i,1]=inp[i,11]
    Po[i,2]=inp[i+int((n+1)/3),11]
    Po[i,3]=DCO[i]

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
    Po[i,7]=Po[i,6]*NO2[i]
    Po[i,8]=Po[i,1]/Ar

Po[int((n+1)/3)-1,2]=nan
Po[int((n+1)/3)-1,5]=nan

print('     分支编号         隧道风量          通风井风量         每段CO生成值     分支内污染物      风井内污染物         CO浓度            NO2浓度           风速\n\n');
for i in range(0,int((n+1)/3)):
    print('%10d     %15.8f    %15.8f      %15.8f  %15.8f   %15.8f   %15.8f    %15.8f   %15.8f' %(Po[i,0],Po[i,1],Po[i,2],Po[i,3],Po[i,4],Po[i,5],Po[i,6],Po[i,7],Po[i,8]));

'''------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
plt.figure(figsize=(8,6))
plt.plot(inp[int((n+1)/3):int((n+1)/3*2)-1,0],inp[int((n+1)/3):int((n+1)/3*2)-1,11],'r*:')
plt.title("通风井风量")
plt.xlabel("分支编号")
plt.ylabel("风量")
plt.show()

plt.figure(figsize=(16,6))
plt.suptitle("污染物浓度")
ax1=plt.subplot(1,2,1)
ax2=plt.subplot(1,2,2)
plt.sca(ax1)
plt.plot(Po[0:int((n+1)/3),0],Po[0:int((n+1)/3),6],'bo-')
plt.xlabel("分支编号")
plt.ylabel("CO浓度")
plt.sca(ax2)
plt.plot(Po[0:int((n+1)/3),0],Po[0:int((n+1)/3),7],'ro-')
plt.xlabel("分支编号")
plt.ylabel("NO2浓度")
plt.show()