import numpy as np
from variables import *
from numpy import *
from mapminmax import *
from mapminmax2 import *
from mymorlet import Mymorlet
from d_mymorlet import D_mymorlet
import matplotlib.pyplot as plt
from pylab import mpl  
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus']=False
'''----------------------------------------------------网络参数配置-------------------------------------------------'''	

input = np.loadtxt('input.txt', dtype=np.int32, delimiter=' ')
#print(input)
output = np.loadtxt('output.txt', dtype=np.int32, delimiter=' ')
#print(output)
input_test = np.loadtxt('input_test.txt', dtype=np.int32, delimiter=' ')
#print(input_test)
output_test = np.loadtxt('output_test.txt', dtype=np.int32, delimiter=' ')
#print(output_test)

M=len(input[0])#输入节点个数
N=1#输出节点个数
n=6#隐形节点个数
lr1=0.01#学习概率
lr2=0.001#学习概率
maxgen=1000#迭代次数


#权值初始化
Wjk=random.randn(n,M)
Wjk_1=Wjk
Wjk_2=Wjk_1
Wij=random.randn(N,n)
Wij_1=Wij
Wij_2=Wij_1
a=random.randn(n)
a_1=a
a_2=a_1
b=random.randn(n)
b_1=b
b_2=b_1

#节点初始化
y=zeros(N)
net=zeros(n)
net_ab=zeros(n)

#权值学习增量初始化
d_Wjk=zeros((n,M))
d_Wij=zeros((N,n))
d_a=zeros(n)
d_b=zeros(n)

'''------------------------------------------------输入输出数据归一化--------------------------------------------'''

inputn,inputps=mapminmax(input.T)
outputn,outputps=mapminmax2(output.T)
inputn=inputn.T
outputn=outputn.T
#print(inputn)
#print(outputn)

'''--------------------------------------------------网络训练-----------------------------------------------------'''
#print(len(input))
error1=zeros(maxgen)
net=zeros(n)
net_ab=zeros(n)
#print("程序运算中...")
for i in range(0,10):
    #误差累计
    error1[i]=0
    #循环训练
    for kk in range(0,len(input)):
        x=inputn[kk]
        #print(x)
        yqw=outputn[0][kk]
        #print(yqw)
        for j in range(0,n):
            for k in range(0,M):
                net[j]=net[j]+Wjk[j][k]*x[k]
                #print(net)
                net_ab[j]=(net[j]-b[j])/a[j]
                #print(net_ab[j])
            temp=Mymorlet(net_ab[j])
            for k in range(0,N):
                y=y+Wij[k][j]*temp#小波函数
   #计算误差和
        error1[i]=error1[i]+sum(abs(yqw-y))

   #权值调整
        for j in range(0,n):
            #计算d_Wij
            temp=Mymorlet(net_ab[j])
            for k in range(0,N):
                d_Wij[k][j]=d_Wij[k][j]-(yqw-y)*temp
            #计算d_Wjk
            temp=D_mymorlet(net_ab[j])
            for k in range(0,M):
                for l in range(0,N):
                    d_Wjk[j][k]=d_Wjk[j][k]+(yqw-y)*Wij[l][j]
                d_Wjk[j][k]=-d_Wjk[j][k]*temp*x[k]/a[j]
            #计算d_b
            for k in range(0,N):
                d_b[j]=d_b[j]+(yqw-y)*Wij[k][j]
            d_b[j]=d_b[j]*temp/a[j]
            #计算d_a
            for k in range(0,N):
                d_a[j]=d_a[j]+(yqw-y)*Wij[k][j]
            d_a[j]=d_a[j]*temp*((net[j]-b[j])/b[j])/a[j]

   #权值参数更新
        Wij=Wij-lr1*d_Wij
        Wjk=Wjk-lr1*d_Wjk
        b=b-lr2*d_b
        a=a-lr2*d_a
    
        d_Wjk=zeros((n,M))
        d_Wij=zeros((N,n))
        d_a=zeros(n)
        d_b=zeros(n)

        y=zeros(N)
        net=zeros(n)
        net_ab=zeros(n)
        
        Wjk_1=Wjk
        Wjk_2=Wjk_1
        Wij_1=Wij
        Wij_2=Wij_1
        a_1=a
        a_2=a_1
        b_1=b
        b_2=b_1


'''----------------------------------------------------网络预测--------------------------------------------------'''
print("程序运算中.......")

#预测输入归一化
x=inputps(input_test.T)
x=x.T
#print(x)

#网络预测
yuce=zeros(len(input_test))
for i in range(0,len(input_test)):
    x_test=x[i]
    for j in range(0,n):
        for  k in range(0,M):
            net[j]=net[j]+Wjk[j][k]*x_test[k]
            net_ab[j]=(net[j]-b[j])/a[j]
        temp=Mymorlet(net_ab[j])
        for k in range(0,N):
            y[k]=y[k]+Wij[k][j]*temp

    yuce[i]=y[k]
    y=zeros(N)
    net=zeros(n)
    net_ab=zeros(n)
#print(yuce)
#预测输出反归一化
print("预测交通量为:")
ynn=outputps.reverse(yuce)
print(ynn)

'''-------------------------------------------预测交通量写入文本文档---------------------------------------------'''
#ynn1=[]
#for i in ynn[0]:
    #ynn1.append(i)  
#f1=open('预测交通量.txt','w')
#for i in range(0,len(input_test)):
    #f1.write(str(ynn1[i])+'\n')
#f1.close()
#print("已成功写入预测交通量")


'''-----------------------------------------------需风量计算-----------------------------------------------------'''
Nup=ynn*shangpo#上坡0.45，下坡0.55,后面会乘上系数
Ndown=ynn*xiapo#ynn 预测车流量
Nm1up=Nup*cy#柴油车比例
Nm2up=Nup*xkc#小客车比例
Nm3up=Nup*qhc#轻型货车比例
Nm4up=Nup*zhc#中型货车比例
Nm5up=Nup*dhc#大型货车比例
NNup=Nm1up*1+Nm2up*1+Nm3up*2+Nm4up*5+Nm5up*7
NNup=NNup.T#上坡车型交通量
Nm1down=Ndown*0.35
Nm2down=Ndown*0.45
Nm3down=Ndown*0.15
Nm4down=Ndown*0.03
Nm5down=Ndown*0.02
NNdown=Nm1down*1+Nm2down*1+Nm3down*2+Nm4down*5+Nm5down*7#下坡车型交通量
NNdown=NNdown.T
Qcoup=(qco*0.545*NNup)*fa*(fd*fivup)*fh*L1/3.6/1000000#上坡CO排放量
Qcodown=(qco*0.55*NNdown)*fa*(fd*fivdown)*fh*L2/3.6/1000000#下坡CO排放量
Qcoup=Qcoup*300*1000000/taoup/273#上坡CO需风量
Qcodown=Qcodown*300*1000000/taodown/273#下坡CO需风量
Nm1up=Nup*qx#轻型车
Nm2up=Nup*zx1#中型车
Nm3up=Nup*zx2#重型车
Nm4up=Nup*tg#拖挂车
NNup=Nm1up*0.4+Nm2up*1+Nm3up*1.5+Nm4up*3#上坡交通量
NNup=NNup.T
Nm1down=Ndown*qx
Nm2down=Ndown*zx1
Nm3down=Ndown*zx2
Nm4down=Ndown*tg
NNdown=Nm1down*0.4+Nm2down*1+Nm3down*1.5+Nm4down*3#下坡交通量
NNdown=NNdown.T
Qvup=(qv*0.45*NNup)*fa*(fd*fivup)*fh*L1/3.6/1000000#上坡烟尘
Qvdown=(qv*0.55*NNdown)*fa*(fd*fivdown)*fh*L2/3.6/1000000#下坡烟尘
Qvup=Qvup/K#上坡烟尘需风量
Qvdown=Qvdown/K#下坡烟尘需风量
Qrequp=L1*A1*3/3600
Qreqdown=L2*A1*3/3600
temp=1.5*A1;
if (temp>Qrequp):
    Qrequp=temp
if (temp>Qrequp):
    Qreqdown=temp
#Q01=max([Qcoup Qvup Qrequp]);
Q01=max(Qcoup)
Q02=max(Qvup)
Q03=max(Qcodown)
Q04=max(Qvdown)
QQ=max(Q01,Q02,Q03,Q04)
print("需风量为"+str(QQ))


'''-----------------------------------------------风机台数计算---------------------------------------------'''

pm=(1+zte+ldr*L/Dr)*p/2*vn**2
ztcs=0.0768*(1-rl)+0.35#空气阻力系数
ztcl=0.0768*rl+0.35
Am=(1-rl)*Acs*ztcs+rl*Acl*ztcl#汽车等效阻抗面积
#a代表同向 an代表反向

Na=ynn/2#设计高峰小时交通量
Nan=ynn/2
na=Na*L/3600/vta#车辆数
nan=Nan*L/3600/vtan
vr=QQ/A1#设计风速
pt=Am/A1*p/2*na*(vta-vr)**2-Am/A1*p/2*nan*(vtan+vr**2);
pld=(ldr*L/Dr)*p/2*vr**2#沿程摩擦阻力
Epzt=zte*p/2*vr**2#局部阻力
pr=pld+Epzt
pj=p*vj**2*Aj/A1*(1-vr/vj)*et#每台射流风机升压力
fj=(pr+pm-pt)*pj;
fjts=-fj/86400#风机台数
print("需要风机台数为"+str(fjts))

'''----------------------------------------------------绘图------------------------------------------------'''
ynn2=[]
for i in ynn[0]:
    ynn2.append(i)  
X=np.linspace(1,len(ynn2),len(ynn2))
plt.figure(figsize=(12,10))
plt.plot(X,ynn2,'r*:')
plt.plot(X,output_test,'bo--')
plt.title("预测交通流量")
plt.xlabel("时间点")
plt.ylabel("交通流量")
plt.legend(["预测交通流量","实际交通流量"],frameon=False,loc='upper right')
plt.show()


