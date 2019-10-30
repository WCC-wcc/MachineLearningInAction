from numpy import * 
from tkinter import * 																				#导入Tkinter库   创建图形化用户界面  GUI
from Compare import *
import matplotlib 																					#导入matplotlib文件  
matplotlib.use('TkAgg')																				#设定后端为TkAgg  Agg是一个C++的库，可以从图像创建光栅图
#下面两个import声明把TkAgg和Matplotlib连接起来
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg										#导入创建画布需要的库
from matplotlib.figure import Figure 																#导入画图常用的库

# 怎么添加其他目录下的python文件，主要就是把目录的路径添加到系统中，sys.path.append(o_path)，其中目录可以自己指定，可以用程序取去读取 
# import os 
# o_path = os.getcwd() # 返回当前工作目录
# o_path = 'F:\\MachineLearning\\Tree_Regression\\Compare';
# import sys
# sys.path.append(o_path) # 添加自己指定的搜索路径

#绘图函数
def reDraw(tolS,tolN):
	reDraw.f.clf()																							#首先清空已经存在的图

	reDraw.a = reDraw.f.add_subplot(111)																	#创建一个新的画布
	if chkBtnVar.get(): 																					#复选旋钮选中构建模型树 调整叶节点生成函数  误差计算函数
		if tolN < 2:
			tolN = 2 																						#设定最小的样本切分数为2  
		myTree = createTree(reDraw.rawDat,modelLeaf,modelErr,(tolS,tolN)) 									#调用createTree()函数 生成模型树 
		print(myTree)
		yHat = createForeCast(myTree,reDraw.testDat,modelTreeEval) 											#调用createTreeForeCast()函数  生成预测树
	
	else: 																									#复选旋钮未选中构建回归树  叶节点生成函数  误差计算函数默认
		myTree = createTree(reDraw.rawDat,ops = (tolS,tolN))	 											#调用createTree()函数 生成回归树
		print(myTree)
		yHat = createForeCast(myTree,reDraw.testDat)														#调用createTreeForeCast()函数  生成预测树
	reDraw.a.scatter(reDraw.rawDat[:,0].tolist(),reDraw.rawDat[:,1].tolist(),s = 5)							#绘制样本散点图 scatter和plot方法属于Matplotlib构建程序的前端
	reDraw.a.plot(reDraw.testDat,yHat,linewidth = 2.0)														#绘制测试样本点和预测值之间的关系  几个线段就是几个叶节点
	reDraw.canvas.draw() 																					#显示绘制出的图片

#获取输入数据
def getInputs():
	try: 																									#处理tolN的输入
		tolN = int(tolNentry.get()) 																		#tolNentry.get()文本框读取输入值
	except: 																								#异常处理
		tolN = 10 
		print('enter Integer for tolN')
		tolNentry.delete(0,END) 																			#清空输入框
		tolNentry.insert(0,'10')																			#恢复默认值

	try:																									#同上
		tolS = float(tolSentry.get())
	except:
		tolS = 1.0
		print('enter Float for tolS')
		tolSentry.delete(0,END) 																			#entry.delete(a,b)删除从a索引到b索引的内容
		tolSentry.insert(0,'1.0')

	return tolN,tolS 																						#返回tolN,tolS值

#重新绘制树
def drawNewTree():
	tolN,tolS = getInputs()
	reDraw(tolS,tolN)



root = Tk()						#为了初始化Tkinter，我们需要先创建一个root控件。他是一个普通的窗口，包含一个标题栏和其他有窗口管理器提供的装饰。
# Label(root,text = 'Plot Place Holder').grid(row = 0,columnspan = 3)  占位符   不需要
reDraw.f = Figure(figsize = (5,4),dpi = 100) 											#设置图形尺寸与质量
reDraw.canvas = FigureCanvasTkAgg(reDraw.f,master = root)								#创建画布   把绘制的画布显示到tkinter窗口上
reDraw.canvas.draw() 																	#绘图
reDraw.canvas.get_tk_widget().grid(row = 0, columnspan = 3) 							#显示画布  规定画布放置位置


Label(root,text = 'tolN').grid(row = 1,column = 0) 										#添加tolN标签 规定位置
tolNentry = Entry(root) 																#添加文本框，输入tolN的数值
tolNentry.grid(row = 1,column = 1) 														#规定文本框位置
tolNentry.insert(0,'10') 																#插入默认值  entry.insert(a,b)在第a个索引位置  插入数据

Label(root,text = 'tolS').grid(row = 2,column = 0) 										#同上
tolSentry = Entry(root)
tolSentry.grid(row = 2,column = 1)
tolSentry.insert(0,'1.0')

Button(root,text = 'ReDraw',command = drawNewTree).grid(row = 2,column = 2,rowspan = 3) #添加ReDraw按钮  点击触发drawNewTree()函数  同时规定按钮位置
																						#rowspan  columnspan  是否  跨列  或者 跨行

chkBtnVar = IntVar() 																	#IntVar()为复选框是否被选中的标识
chkBtn = Checkbutton(root,text = 'Model Tree',variable = chkBtnVar)						#添加Model Tree复选框   button.get()获得0,1值，表示选中与否
chkBtn.grid(row = 3,column = 0,columnspan = 2)											#规定复选框位置，大小

Button(root,text = 'Quit',fg = 'black',command = root.quit).grid(row = 1,column = 2)	#创建退出按钮  执行退出GUI命令

reDraw.rawDat = mat(loadDataSet('sine.txt'))											#读取数据，样本集  200个样本
reDraw.testDat = arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)			#arange(a,b,c)函数的作用是选取(b-a)/c个点   也就是从a开始，每次增加c，直到b停止

# print(max(reDraw.rawDat[:,0]))     [[0.996606]]                                       样本最大值为0.99    本例中生成测试样本点为99个
# print(min(reDraw.rawDat[:,0]))	   [[0.008507]] 									样本最小值为0.00
# print(reDraw.testDat)

reDraw(1.0,10)
root.mainloop()															#启动事件循环  使得窗口在众多事件中可以响应鼠标点击、按键和重绘等动作