# fire warning
import pandas as pd
import numpy as np
import xlrd
firedata = xlrd.open_workbook(r'test1.xls')
# 获取excel第1个sheet数据，其中第1列数据记录时间，2~n列记录温度；第1行记录位置，2~n行记录温度
table = firedata.sheets()[0]

# 注意：第1行所记录的位置编号顺序应与隧道内行车方向相同，若相反，则互换j-1列和j+1列温度差阈值--已经修改数据
# 注意：5s温度差判定准则为：沿行车或通风或交通风方向，异常点位上游阈值为0.3°c，下游为0.1°c
# 注意：输入的温度数据应保证时间上的连续性，即每秒一个温度数据
# 注意：保存温度数据时应至少保留两位小数

rownum = table.nrows  # 获取数据行数
colnum = table.ncols  # 获取数据列数
for i in range(1, rownum-9):  # 从第2行开始循环至倒数第9行
    firesignal = np.zeros((1, colnum-1))  # 1行colnum列 计数器
    for m in range(0, 5):  # 计算本次及后续4s内火灾信号次数
        rowa = table.row_values(i+m)
        rowb = table.row_values(i+m+5)
        dif = [b-a for b, a in zip(rowb, rowa)]  # 求第i+5行和第i行的5s温度差
        for j in range(2, colnum-1):
            # if dif[j-1]>=0.09999 and dif[j]>=0.19999 and dif[j+1]>=0.29999:
            if dif[j+1] >= 0.09999 and dif[j] >= 0.19999 and dif[j-1] >= 0.29999:
                firesignal[0, j] = firesignal[0, j]+1  # 统计j列（即感温探测器位置）的火灾信号次数
    for k in range(1, colnum-1):  # 判断rowzero计数器第k列>=5与否，若是则为火灾
        if firesignal[0, k] >= 5:
            # 输出火灾预警，i+9为行数，代表输出火灾预警的时刻，k为输出火灾预警的位置
            print(table.cell(i+9, 0), table.cell(0, k), 'fire')
