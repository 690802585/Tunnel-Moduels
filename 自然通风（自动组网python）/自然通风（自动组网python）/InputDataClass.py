import numpy as np
from numpy import *
from function import *
import matplotlib.pyplot as plt
from pylab import mpl  
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus']=False

class InputDataClass():
    def __init__(self, L,vt,N1,nk):
        self.L=L#隧道长度
        self.vt=vt#行车速度
        self.N1=N1#风速同向的设计高峰小时交通量
        self.nk=nk#隧道分段数

        self.Acs=2.13#小型车正面投影面积
        self.Acl=5.37#大型车正面投影面积
        self.rl=0#大型车比例
        self.nc1=0.3525#小型车空气阻力系数
        self.nc2=0.3562#大型车空气阻力系数
        self.Am=(1-self.rl)*self.Acs*self.nc1+self.rl*self.Acl*self.nc2#汽车等效阻抗面积
        self.B=12#计算宽度
        self.H=5.5#计算高度
        self.Ar=self.B*self.H#隧道净空断面积
        self.rou=1.2#通风计算点空气密度
        self.h=0#通风计算点海拔高程
        self.x1=self.Acs/self.Ar#小型车在隧道行车空间的占积率
        self.x2=self.Acl/self.Ar#大型车在隧道行车空间的占积率
        self.T=298.5#通风计算点夏季气温
        self.Vn=3#自然风速
        self.L1=self.L/self.nk#分段长度
        self.vt1=self.vt/3.6#与风速同向的各工况车速（单位m/s）
        self.n1=self.N1*self.L1/(3600*self.vt1)#隧道内与风速同向车辆数
        self.Qr=0#隧道设计风量
        self.tflcs=self.Am/self.Ar*self.rou/2*self.n1
        self.Av=self.tflcs*(1/self.Ar)**2
        self.Bv=self.tflcs*(-2)*self.vt1/self.Ar
        self.Cv=self.tflcs*self.vt1**2#交通通风力参数
        
        #隧道阻力系数 
        self.ycfz=0.02#沿程风阻系数
        self.Cr=2*(self.B+self.H)#隧道断面周长
        self.Dr=4*self.Ar/self.Cr#隧道断面当量直径
        self.ycz=self.ycfz*self.L1/self.Dr*self.rou/2*1/self.Ar**2#隧道沿程风阻系数
        self.ne1=0.5
        self.ne2=0.4
        self.ne3=1.4
        self.yczrk=self.ne1*self.rou/2*1/self.Ar**2
        self.Rne2=self.ne2*self.rou/2*1/self.Ar**2
        self.yczck=self.ne3*self.rou/2*1/self.Ar**2
        self.ZFZ1=self.yczrk+self.ycz
        self.ZFZ2=self.Rne2+self.ycz
        self.ZFZ3=self.yczck+self.ycz#总风阻系数
        
        #通风井阻力系数
        self.B0=10#风井长度
        self.H0=4#风井宽度
        self.ycfz0=0.02#沿程风阻系数
        self.Z=6#风井高度
        self.Ar0=self.B0*self.H0
        self.Cr0=2*(self.B0+self.H0)
        self.Dr0=4*self.Ar0/self.Cr0
        self.ycs=self.ycfz0*self.Z/self.Dr0*self.rou/2*1/self.Ar0**2#风井沿程风阻系数
        self.ne4=2.1
        self.Rne4=self.ne4*self.rou/2*1/self.Ar0**2
        self.ZFZ4=self.Rne4+self.ycs#总风阻系数
        self.CO=2.46E-03#CO排放量
        self.DCO=self.CO/self.nk#每段CO排放量
        self.NOX=2.04E-04#NOX排放量
        self.DNOX=self.NOX/self.nk#每段NOX排放量
        self.NO2=self.DNOX/self.DCO
        self.PW=zeros(3*self.nk-1)
        for i in range(0,self.nk):
            self.PW[i]=(self.yczrk+self.L/self.Dr*self.ycz/self.nk)*self.rou/2*self.Vn**2#自然风阻
        self.inp=zeros((3*self.nk-1,6))
        # Trow=inp(1:n,7)
        # Tl=zeros(n,1)
        # Ts=293#隧址气温
        # T0=273#标准气温，取273K
        # Z=6#通风井高度
        # rou=1.2#空气密度
        # g=9.8#重力加速度
        # p0=101.325#标准大气压
        # p=98.825#隧址大气压
        # pr=98.125#地下道路内大气压

        self.m=0#生成树节点数
        self.n=0#生成树分支数
       
    def TunnelBranch(self):
        
        # inp第一列
        for i in range(0,3*self.nk-1):
            self.inp[i,0]=i+1 


        #inp第二 三列
        for i in range(0,self.nk):
            self.inp[i,1]=i+1
            self.inp[i,2]=i+2

        for i in range(0,self.nk-1):
            self.inp[self.nk+i,1]=i+2
            self.inp[self.nk+i,2]=i+self.nk+2

        self.inp[2*self.nk-1,1]=1
        self.inp[2*self.nk-1,2]=self.nk+2
        for i in range(0,self.nk-2):
            self.inp[2*self.nk+i,1]=self.nk+i+2
            self.inp[2*self.nk+i,2]=self.nk+i+3

        self.inp[3*self.nk-2,1]=2*self.nk
        self.inp[3*self.nk-2,2]=self.nk+1
 

        #inp第四列
        self.inp[0,3]=self.ZFZ1
        for i in range(1,self.nk-1):
            self.inp[i,3]=self.ZFZ2

        self.inp[self.nk-1,3]=self.ZFZ3

        for i in range(0,self.nk-1):
            self.inp[self.nk+i,3]=self.ZFZ4


        #inp第五列
        for i in range (0,self.nk):
            self.inp[i,4]=1

    def Network(self):
        V1=set(self.inp[:,1])
        V2=set(self.inp[:,2])
        V=V1.union(V2) #节点集合：始节点和末节点的并集
        self.m=0
        for i in V:
           self.m+=1     #节点数：节点集合的行数  M个节点       
        #print(V)
        #print(m)
        self.n=np.size(self.inp,0)  #分支数：in矩阵的行数  N条分支
        #print(n)
        #print(inp)


        #计算关联矩阵
        B=zeros((self.m,self.n))#生成m行n列0矩阵
        for i in range(1,self.n+1):#对所有分支循环
            B[int(self.inp[i-1,1])-1,i-1]=1#in的第2列始节点
            B[int(self.inp[i-1,2])-1,i-1]=-1#in的第3列末节点
        #print(B)

        #计算最小生成树（prim算法）
        s=np.array([1])#节点集合,加入第1个节点
        tree=[]
        tree=np.array(tree) #树枝集合

        for k in range(0,self.m-1):#循环m-1次，每次加入1树枝
           s1=np.array(s-1, dtype = np.uint32)
           [a,b]= np.nonzero(B[s1,:])#寻找s的所有关联分支
          # print(a)
          # print(b)  
           Rmin=10000;#最小风阻赋初值

           bsize=size(b)
           #print(bsize)
           for i in range(0,bsize):#从1到关联分支数（b的行数）
               #print(b[i])
               e=np.where(s==self.inp[b[i],1])
               f=np.where(s==self.inp[b[i],2])
               #print(e)
               #print(f)
               if(np.size(e) and np.size(f)):
                   continue #跳过该分支
               if(self.inp[b[i],3]<Rmin):#in的第四列为风阻
                   Rmin=self.inp[b[i],3]#更新最小风阻
                   #print(Rmin)
                   select1=b[i]#更新选择的树枝
                   #print(select1)
                   #print()

           tree=np.hstack([tree,select1])#树中加入树枝
           #print(tree)
           g=np.where(s==self.inp[select1,1])
           h=np.where(s==self.inp[select1,2])
           if(np.size(g)>0):#始节点在节点集合中
               s=np.hstack([s,self.inp[select1,2]])#加入末节点
           elif (np.size(h)>0):#末节点在节点集合中
               s=np.hstack([s,self.inp[select1,1]])#加入始节点
           #print()


        #tree=np.array([0,5,6,7,1])


         #计算余树
        E=self.inp[:,0]
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
        C11=eye(self.n-self.m+1)#n-m+1行n-m+1列单位矩阵
        #print(B12)
        C12=-(B11).T@(np.linalg.inv(B12)).T
        #print(C12)
        C=np.hstack([C11,C12])#独立回路矩阵
        #print(C)
        #将in矩阵按余树在前，树在后的方法排序

        in1=self.inp[cotree,:]#余树所在行
        #print(in1)
        in2=self.inp[tree,:]#树所在行
        #print(in2)
        self.inp=np.vstack([in1,in2])
        #print(inp)
        return C #返回独立回路矩阵

    def AirVolume(self,C):
        Hf=zeros(self.n)#风机风压列向量(n行1列0矩阵)
        #风量赋初值
        Q=zeros(self.n)
        Q[0:self.n]=1e-6
        #print(C)
        #print(self.inp)
        #开始迭代计算
        for k in range(0,30000):#控制最大迭代次数


            maxdq=0
            maxf=0

            for i in range(0,self.n-self.m+1):#对所有回路循环
                f=0;#累加前赋初值
                df=1;#累加前赋初值
                for j in range(0,self.n):#对所有分支循环
                    fanindex=int(self.inp[j,4])#self.inp矩阵的第5列为交通通风系数
                    if(fanindex==1):
                        Hf[j]=self.Cv+self.Bv*Q[j]+self.Av*(Q[j]**2)#计算交通通风力
                    elif(fanindex==2):
                        Hf[j]=self.Cv+self.Bv*Q[j]+self.Av*(Q[j]**2)#计算交通通风力
                    elif(fanindex==3):
                        Hf[j]=self.Cv+self.Bv*Q[j]+self.Av*(Q[j]**2)#计算交通通风力
                    elif(fanindex==4):
                        Hf[j]=self.Cv+self.Bv*Q[j]+self.Av*(Q[j]**2)#计算交通通风力
                    elif(fanindex==5):
                        Hf[j]=self.Cv+self.Bv*Q[j]+self.Av*(Q[j]**2)#计算交通通风力
                    else:
                        Hf[j]=0
                    f=f+C[i,j]*(self.inp[j,3]*Q[j]*abs(Q[j])+self.PW[j]-Hf[j]-self.inp[j,5])#self.inp矩阵第4列为风阻，第6列为温升压力(暂不考虑)
                    df=df+(abs(C[i,j])*2*self.inp[j,3]*abs(Q[j]))
                dq=-f/df#计算回路风量修正值
                for j in range(0,self.n):
                    Q[j]=Q[j]+C[i,j]*dq#修正回路风量
                f=0.0#累加前赋初值
                for j in range(0,self.n):
                    f=f+C[i,j]*(self.inp[j,3]*Q[j]*abs(Q[j])+self.PW[j]-Hf[j]-self.inp[j,5])#重新计算回路不平衡风压

                if(abs(dq)>maxdq):
                    maxdq=abs(dq)#计算最大回路风量修正值
                if(abs(f)>maxf):
                    maxf=abs(f)#计算最大回路不平衡风压

            if(abs(maxdq)<1e-3 and abs(maxf)<1e-3):
                break


            #隧道入口
            pp=(self.n+1)/3+1
            vv1=np.where(self.inp[:,0]==pp)#Q3
            vv2=np.where(self.inp[:,0]==pp-(self.n+1)/3)#Q1
            vv3=np.where(self.inp[:,0]==pp)#Q3
            vv4=np.where(self.inp[:,0]==pp-((self.n+1)/3-1))#Q2

            if (Q[vv1]>0):
                a1=abs(Q[vv1]/Q[vv2])#Q3/Q1
                a2=abs(Q[vv4]/Q[vv2])#Q2/Q1
                a11=f1(a1)+self.ycs;
                a22=f2(a2)+self.yczrk+self.ycz;
                self.inp[vv1,3]=a11;
                self.inp[vv2,3]=a22;
    
            else:
                a3=abs(Q[vv3]/Q[vv4])#Q3/Q2
                a4=abs(Q[vv2]/Q[vv4])#Q1/Q2
                a33=f3(a3)+self.ycs;
                a44=f4(a4)+self.ycz;
                self.inp[vv3,3]=a33;
                self.inp[vv4,3]=a44;

            #隧道出口
            pp=(self.n+1)/3*2-1 
            vv1=np.where(self.inp[:,0]==pp)#Q3
            vv2=np.where(self.inp[:,0]==pp-(self.n+1)/3)#Q1
            vv3=np.where(self.inp[:,0]==pp)#Q3
            vv4=np.where(self.inp[:,0]==pp-((self.n+1)/3-1))#Q2
    
            if (Q[vv1]>0):
                a1=abs(Q[vv1]/Q[vv2])#Q3/Q1
                a2=abs(Q[vv4]/Q[vv2])#Q2/Q1
                a11=f1(a1)+self.ycs;
                a22=f2(a2)+self.ycz;
                self.inp[vv1,3]=a11;
                self.inp[vv2,3]=a22;
            else:
                a3=abs(Q[vv3]/Q[vv4])#Q3/Q2
                a4=abs(Q[vv2]/Q[vv4])#Q1/Q2
                a33=f3(a3)+self.ycs;
                a44=f4(a4)+self.yczck+self.ycz;
                self.inp[vv3,3]=a33;
                self.inp[vv4,3]=a44;

            for pp in range(int((self.n+1)/3+2),int((self.n+1)/3*2-1)):
        
                vv1=np.where(self.inp[:,0]==pp)#Q3
                vv2=np.where(self.inp[:,0]==pp-(self.n+1)/3)#Q1
                vv3=np.where(self.inp[:,0]==pp)#Q3
                vv4=np.where(self.inp[:,0]==pp-((self.n+1)/3-1))#Q2
        
                if (Q[vv1]>0):
                    a1=abs(Q[vv1]/Q[vv2])#Q3/Q1
                    a2=abs(Q[vv4]/Q[vv2])#Q2/Q1
                    a11=f1(a1)+self.ycs;
                    a22=f2(a2)+self.ycz;
                    self.inp[vv1,3]=a11;
                    self.inp[vv2,3]=a22;
                else:
                    a3=abs(Q[vv3]/Q[vv4])#Q3/Q2
                    a4=abs(Q[vv2]/Q[vv4])#Q1/Q2
                    a33=f3(a3)+self.ycs;
                    a44=f4(a4)+self.ycz;
                    self.inp[vv3,3]=a33;
                    self.inp[vv4,3]=a44;


        HR=self.inp[:,3]*abs(Q)*Q
        H=HR-Hf-self.inp[:,5]

        Q=np.array(Q).reshape((-1,1))
        H=np.array(H).reshape((-1,1))
        HR=np.array(HR).reshape((-1,1))
        Hf=np.array(Hf).reshape((-1,1))

        self.inp=np.hstack([self.inp,Q])
        self.inp=np.hstack([self.inp,H])
        self.inp=np.hstack([self.inp,HR])
        self.inp=np.hstack([self.inp,Hf])

        self.inp=self.inp[np.lexsort(self.inp[:,::-1].T)]
        print('      分支编号   始节点    末节点    风机索引      风量        风阻系数          风压          阻力        交通通风力      风井热压差     自然通风力\n\n')
        for i in range(0,self.n):
           
            print('%10d%10d%10d%10d%15.4f%15.8f%15.4f%15.4f%15.4f%15.4f%15.4f' %(self.inp[i,0],self.inp[i,1],self.inp[i,2],self.inp[i,4],self.inp[i,6],self.inp[i,3],self.inp[i,7],self.inp[i,8],self.inp[i,9],self.inp[i,5],self.PW[i]))

    
    def Pollutant(self):
        Po=zeros((int((self.n+1)/3),9))#污染物浓度矩阵赋初值
        for i in range(0,int((self.n+1)/3)):
            Po[i,0]=i+1
            Po[i,1]=self.inp[i,6]
            Po[i,2]=self.inp[i+int((self.n+1)/3),6]
            Po[i,3]=self.DCO

        Po[0,4]=Po[0,3]

        if Po[0,2]>0:
            Po[0,5]=Po[0,4]/Po[0,1]*Po[0,2]
        else:
            Po[0,5]=0

        for i in range(1,int((self.n+1)/3)):
            Po[i,4]=Po[i,3]+Po[i-1,4]-Po[i-1,5]
            if Po[i,2]>0:
                Po[i,5]=Po[i,4]/Po[i,1]*Po[i,2]
            else:
                Po[i,5]=0


        for i in range(0,int((self.n+1)/3)):
            Po[i,6]=abs(Po[i,4])/Po[i,1]*10**6
            Po[i,7]=Po[i,6]*self.NO2
            Po[i,8]=Po[i,1]/self.Ar

        Po[int((self.n+1)/3)-1,2]=nan
        Po[int((self.n+1)/3)-1,5]=nan
        print('     分支编号         隧道风量          通风井风量         每段CO生成值     分支内污染物      风井内污染物         CO浓度            NO2浓度           风速\n\n');
        for i in range(0,int((self.n+1)/3)):
            print('%10d     %15.8f    %15.8f      %15.8f  %15.8f   %15.8f   %15.8f    %15.8f   %15.8f' %(Po[i,0],Po[i,1],Po[i,2],Po[i,3],Po[i,4],Po[i,5],Po[i,6],Po[i,7],Po[i,8]));
        return Po#返回污染物浓度矩阵


    def PlotAirlume(self):
        plt.figure(figsize=(8,6))
        plt.plot(self.inp[int((self.n+1)/3):int((self.n+1)/3*2)-1,0],self.inp[int((self.n+1)/3):int((self.n+1)/3*2)-1,6],'r*:')
        plt.title("通风井风量")
        plt.xlabel("分支编号")
        plt.ylabel("风量")
        plt.show()

    def PlotPollutant(self,Po1):
        
        plt.figure(figsize=(16,6))
        plt.suptitle("污染物浓度")
        ax1=plt.subplot(1,2,1)
        ax2=plt.subplot(1,2,2)
        plt.sca(ax1)
        plt.plot(Po1[0:int((self.n+1)/3),0],Po1[0:int((self.n+1)/3),6],'bo-')
        plt.xlabel("分支编号")
        plt.ylabel("CO浓度")
        plt.sca(ax2)
        plt.plot(Po1[0:int((self.n+1)/3),0],Po1[0:int((self.n+1)/3),7],'ro-')
        plt.xlabel("分支编号")
        plt.ylabel("NO2浓度")
        plt.show()