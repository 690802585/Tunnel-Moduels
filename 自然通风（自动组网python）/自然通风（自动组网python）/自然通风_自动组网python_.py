import wx
import Mainwind
from InputDataClass import InputDataClass
import numpy as np
from numpy import *
from function import *
np.set_printoptions(linewidth=400)


class CalcFrame(Mainwind.Mainwind): 
    def __init__(self,parent): 
    	Mainwind.Mainwind.__init__(self,parent)  


    #按键事件触发函数	
    def Button_Click1(self,event):
        L=eval(self.m_textCtrl1.GetValue())
        vt=eval(self.m_textCtrl2.GetValue())
        N1=eval(self.m_textCtrl3.GetValue())
        nk=eval(self.m_textCtrl4.GetValue())
        print("隧道长度:"+str(L))
        print("行车速度:"+str(vt))
        print("风速同向的设计高峰小时交通量:"+str(N1))
        print("隧道分段数:"+str(nk))
        print("程序运算中....")

        Volume=InputDataClass(L,vt,N1,nk)
        Volume.TunnelBranch()
        C=Volume.Network()
        Volume.AirVolume(C)        
        Po=Volume.Pollutant()
        Volume.PlotAirlume()
        Volume.PlotPollutant(Po)
    def Button_Click2(self,event):
        self.Destroy()
    
def main():        
    app = wx.App(False) 
    frame = CalcFrame(None)
    frame.Show(True) 
    #start the applications 
    app.MainLoop() 
    
if __name__ == '__main__':
    main()




