def InputData(L0,vt0,N10,nk0):

    L=L0
    vt=vt0
    N1=N10
    nk=nk0

    Acs=2.13#小型车正面投影面积
    Acl=5.37#大型车正面投影面积
    rl=0#大型车比例
    nc1=0.3525#小型车空气阻力系数
    nc2=0.3562#大型车空气阻力系数
    Am=(1-rl)*Acs*nc1+rl*Acl*nc2#汽车等效阻抗面积
    B=12#计算宽度
    H=5.5#计算高度
    Ar=B*H#隧道净空断面积
    rou=1.2#通风计算点空气密度
    h=0#通风计算点海拔高程
    x1=Acs/Ar#小型车在隧道行车空间的占积率
    x2=Acl/Ar#大型车在隧道行车空间的占积率
    T=298.5#通风计算点夏季气温
    Vn=3#自然风速


    L1=L/nk#分段长度
    vt1=vt/3.6#与风速同向的各工况车速（单位m/s）
    n1=N1*L1/(3600*vt1)#隧道内与风速同向车辆数
    Qr=0#隧道设计风量

    tflcs=Am/Ar*rou/2*n1
    Acs=tflcs*(1/Ar)**2
    Bcs=tflcs*(-2)*vt1/Ar
    Ccs=tflcs*vt1**2#交通通风力参数

    #隧道阻力系数 
    ycfz=0.02#沿程风阻系数
    Cr=2*(B+H)#隧道断面周长
    Dr=4*Ar/Cr#隧道断面当量直径
    ycz=ycfz*L1/Dr*rou/2*1/Ar**2#隧道沿程风阻系数
    ne1=0.5
    ne2=0.4
    ne3=1.4
    yczrk=ne1*rou/2*1/Ar**2
    Rne2=ne2*rou/2*1/Ar**2
    yczck=ne3*rou/2*1/Ar**2
    ZFZ1=yczrk+ycz
    ZFZ2=Rne2+ycz
    ZFZ3=yczck+ycz#总风阻系数

    #通风井阻力系数
    B0=10#风井长度
    H0=4#风井宽度
    ycfz0=0.02#沿程风阻系数
    Z=6#风井高度
    Ar0=B0*H0
    Cr0=2*(B0+H0)
    Dr0=4*Ar0/Cr0
    ycs=ycfz0*Z/Dr0*rou/2*1/Ar0**2#风井沿程风阻系数
    ne4=2.1
    Rne4=ne4*rou/2*1/Ar0**2
    ZFZ4=Rne4+ycs#总风阻系数


    CO=2.46E-03#CO排放量
    DCO=CO/nk#每段CO排放量
    NOX=2.04E-04#NOX排放量
    DNOX=NOX/nk#每段NOX排放量
    NO2=DNOX/DCO

    PW=(yczrk+L/Dr*ycz/nk)*rou/2*Vn**2#自然风阻

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
                    Hf[j]=Ccs+Bcs*Q[j]+Acs*(Q[j]**2)#计算交通通风力
                elif(fanindex==2):
                    Hf[j]=Ccs+Bcs*Q[j]+Acs*(Q[j]**2)#计算交通通风力
                elif(fanindex==3):
                    Hf[j]=Ccs+Bcs*Q[j]+Acs*(Q[j]**2)#计算交通通风力
                elif(fanindex==4):
                    Hf[j]=Ccs+Bcs*Q[j]+Acs*(Q[j]**2)#计算交通通风力
                elif(fanindex==5):
                    Hf[j]=Ccs+Bcs*Q[j]+Acs*(Q[j]**2)#计算交通通风力
                else:
                    Hf[j]=0
                f=f+C[i,j]*(inp[j,3]*Q[j]*abs(Q[j])+PW-Hf[j]-inp[j,5])#inp矩阵第4列为风阻，第6列为温升压力(暂不考虑)
                df=df+(abs(C[i,j])*2*inp[j,3]*abs(Q[j]))
            dq=-f/df#计算回路风量修正值
            for j in range(0,n):
                Q[j]=Q[j]+C[i,j]*dq#修正回路风量
            f=0.0#累加前赋初值
            for j in range(0,n):
                f=f+C[i,j]*(inp[j,3]*Q[j]*abs(Q[j])+PW-Hf[j]-inp[j,5])#重新计算回路不平衡风压

            if(abs(dq)>maxdq):
                maxdq=abs(dq)#计算最大回路风量修正值
            if(abs(f)>maxf):
                maxf=abs(f)#计算最大回路不平衡风压

        if(abs(maxdq)<1e-3 and abs(maxf)<1e-3):
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
            a11=f1(a1)+ycs;
            a22=f2(a2)+yczrk;
            inp[vv1,3]=a11;
            inp[vv2,3]=a22;
    
        else:
            a3=abs(Q[vv3]/Q[vv4])#Q3/Q2
            a4=abs(Q[vv2]/Q[vv4])#Q1/Q2
            a33=f3(a3)+ycs;
            a44=f4(a4)+ycz;
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
            a11=f1(a1)+ycs;
            a22=f2(a2)+ycz;
            inp[vv1,3]=a11;
            inp[vv2,3]=a22;
        else:
            a3=abs(Q[vv3]/Q[vv4])#Q3/Q2
            a4=abs(Q[vv2]/Q[vv4])#Q1/Q2
            a33=f3(a3)+ycs;
            a44=f4(a4)+yczck;
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
                a11=f1(a1)+ycs;
                a22=f2(a2)+ycz;
                inp[vv1,3]=a11;
                inp[vv2,3]=a22;
            else:
                a3=abs(Q[vv3]/Q[vv4])#Q3/Q2
                a4=abs(Q[vv2]/Q[vv4])#Q1/Q2
                a33=f3(a3)+ycs;
                a44=f4(a4)+ycz;
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
    print(inp)
