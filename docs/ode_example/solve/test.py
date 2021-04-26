import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
# 解决中文乱码问题
myfont = font_manager.FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc"
, size=14)
N = 1000
plt.close() # 关闭打开的图形窗口
def anni():
    fig = plt.figure()
    plt.ion() # 打开交互式绘图interactive
    for i in range(N):
        plt.cla()           # 清除原有图像
        plt.xlim(-0.2,20.4) # 设置x轴坐标范围
        plt.ylim(-1.2,1.2)  # 设置y轴坐标范围
        # 每当i增加的时候，增加自变量x的区间长度，可以理解为不断叠加绘图，所以每次循环之前都使用plt.cla()命令清除原有图像
        x = np.linspace(0,i+1,1000)
        y = np.sin(x)
        plt.plot(x,y)
        plt.pause(0.1)
    # plt.ioff() #关闭交互式绘图
    plt.show()

anni()
