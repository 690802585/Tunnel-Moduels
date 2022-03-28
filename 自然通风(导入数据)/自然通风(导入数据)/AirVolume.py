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
            df=df+(abs(C[i,j])*2*inp[j,3]*abs(Q[j])-inp[j,7]-2*inp[j,8]*Q[j])
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
print(inp[:,3])