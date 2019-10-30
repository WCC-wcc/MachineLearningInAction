import matplotlib.pyplot as plt 

decisionNode = dict(boxstyle = "sawtooth",fc = "0.8")		#定义文本框样式 boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细
leafNode = dict(boxstyle = "round4",fc = "0.8")				#定义文本框样式 boxstyle为文本框的类型，round4是四个角是圆形的矩形，fc是边框线粗细
arrow_args = dict(arrowstyle = "<-")						#定义箭头样式   arrowstyle为箭头的类型，"<-"指向文本框


def plotNode(nodeTxt,centerPt,parentPt,nodeType):           #输入分别为文本内容，起始点坐标，文本中心点坐标，节点类型
	createPlot.ax1.annotate(nodeTxt,xy = parentPt,xycoords = "axes fraction",xytext = centerPt,textcoords = "axes fraction",
							va = "center",ha = "center",bbox = nodeType,arrowprops = arrow_args)
#xy是箭头尖的坐标，xytext设置注释内容显示的中心位置
#xycoords和textcoords是坐标xy与xytext的说明（按轴坐标），若textcoords=None，则默认textcoords与xycoords相同，若都未设置，默认为data
#va/ha设置节点框中文字的位置，va为纵向取值为(u'top', u'bottom', u'center', u'baseline')，ha为横向取值为(u'center', u'right', u'left')


def createPlot():
	fig = plt.figure(1,facecolor = "white")					 #创建一个画布，背景为白色
	fig.clf()												 #画布清空
	createPlot.ax1 = plt.subplot(111,frameon = False)		 #frameon表示是否绘制坐标轴矩形 111表示figure中的图有1行1列，即1个，最后的1代表第一个图 
	plotNode('decesion node',(0.5,0.1),(0.1,0.5),decisionNode)
	plotNode('leaf node',(0.8,0.1),(0.3,0.8),leafNode)
	plt.show()

createPlot()

