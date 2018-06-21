import sys
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
from numpy import arange, sin, pi
import pylab as pl
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure



def test1():
    root = tk.Tk()
    root.title("matplotlib in TK")
    # 设置图形尺寸与质量
    f = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.show()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # 绘制图形
    a = f.add_subplot(111)
    t = arange(0.0, 3, 0.01)
    s = sin(2 * pi * t)
    a.cla()
    a.plot(t, s)

    # 把matplotlib绘制图形的导航工具栏显示到tkinter窗口上
    toolbar = NavigationToolbar2TkAgg(canvas, root)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # # 添加事件绑定
    # def on_key_event(event):
    #     print('you pressed %s' % event.key)
    #     key_press_handler(event, canvas, toolbar)
    #
    # canvas.mpl_connect('key_press_event', on_key_event)

    # # 添加退出按钮
    # def _quit():
    #     root.quit()
    #     root.destroy()
    #
    # button = tk.Button(master=root, text='Quit', command=_quit)
    # button.pack(side=tk.BOTTOM)

    tk.mainloop()


def plot_figure(fig_handle):
    t = arange(0.0, 3, 0.01)
    s = sin(2 * pi * t)
    fig_handle.plot(t, s)


def test2():
    root = tk.Tk()
    root.title("Playing testing")

    # ！！！用其它两个会报错
    # figUV = pl.figure(figsize=(15, 8))
    # figUV = plt.figure(figsize=(15, 8))
    figUV = Figure(figsize=(15, 8))
    canvas = FigureCanvasTkAgg(figUV, master=root)
    canvas.show()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    UVPlot = figUV.add_subplot(111, aspect='equal', facecolor=(0.4, 0.4, 0.4))
    plot_figure(UVPlot)

    root.mainloop()


if __name__ == '__main__':
    test1()
