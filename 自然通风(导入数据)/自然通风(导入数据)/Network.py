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
Q[0:n]=1e-6
#print(C)
#print(inp)