# ID3算法的核心是在决策树各个结点上对应信息增益准则选择特征，递归地构建决策树。具体方法是：
# 从根结点(root node)开始，对结点计算所有可能的特征的信息增益，选择信息增益最大的特征作为结点的特征，
# 由该特征的不同取值建立子节点；再对子结点递归地调用以上方法，构建决策树；直到所有特征的信息增益均很小或没有特征可以选择为止。
# 最后得到一个决策树。ID3相当于用极大似然法进行概率模型的选择。



from math import log
import operator
import matplotlib.pyplot as plt

#香农熵越高，混合数据(也就是分类，类标签)越多
#符号x1的信息定义为  			L(x1) = -log(p(x1),2)    								p(x) 是选择该分类的概率 
#符号x1的期望定义为  			G(x1) = [p(x1) * L(x1)]  
#香农熵定义为所有信息的期望值 	H(x) = G(x1) + G(x2) + ... + G(xn)
#计算给定数据集的香农熵     	综合公式：	H -= { p(xi) * log(p(xi),2) }  i从1到n   	不同的标签<---->不同的类别
def calcShannonEnt(dataSet):
	numEntries = len(dataSet)											    #获取数据集记录数
	labelCounts = {}														#定义标签记录数 
	for featVec in dataSet :
		currentLabel = featVec[-1]											#获取每一条记录的标签
		labelCounts[currentLabel] = labelCounts.get(currentLabel,0) + 1   	#统计标签出现次数，标签不存在的，初始化为0，然后加1  标签存在的不在初始化，计数即可
	shannonEnt = 0.0
	eachLabelShannonEnt = 0.0												#定义香农熵初始值
	for key in labelCounts :										
		prob = float(labelCounts[key]) / numEntries						    #计算各个标签对应的出现概率     	概率 = 标签数 / 总记录数
		eachLabelShannonEnt = (-prob * log(prob,2))							#计算各个标签对应的信息期望值		log(prob,2)	以2为底的对数函数
		shannonEnt += eachLabelShannonEnt									#计算香农熵   对所有类别(标签)的信息期望值求和  
	return shannonEnt

# myDat = [[1,1,'maybe'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
# a = calcShannonEnt(myDat)
# print(a)

#划分数据集
def splitDataSet(dataSet,axis,value):					#输入参数分别是  待划分数据集，划分数据集特征(选择特征矩阵第几个特征划分)，该特征对应值，相符合就划分
	retDataSet = []										#声明新的列表对象   不影响原列表
	for featVec in dataSet :
		if featVec[axis] == value :						#如果特征矩阵第(axis + 1) 个特征的值 = value  
			
														#在总的特征矩阵中抽取符合要求的特征，形成新的特征集，注意要去掉dataSet[:,axis]的特征，因为他是一个参照
			reducedFeatVec = featVec[:axis]				#将特征矩阵各个特征值，从第一个开始到第(axis + 1) 个值赋值给reducedFeatVec
			reducedFeatVec.extend(featVec[axis + 1 :])	#将特征矩阵各个特征值，从第(axis + 2)个到最后一个值赋值给reducedFeatVec
														#extend()函数是扩展矩阵，把前后两个矩阵合二为一，[1,2,3].extend[4,5,6] = [1,2,3,4,5,6]

			retDataSet.append(reducedFeatVec)			#append()函数是添加元素，a.append(b)在矩阵a中添加新元素(b),[1,2,3].append[4,5,6] = [1,2,3,[4,5,6]]
	return retDataSet

# myDat = [[1,1,'maybe'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
# a = splitDataSet(myDat,1,1)
# print(a)

#通过calcShannonEnt(dataSet)函数和splitDataSet(dataSet,axis,value)函数选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
	numFeature = len(dataSet[0]) - 1							#获取原数据集中特征总个数
	baseEntropy = calcShannonEnt(dataSet)						#计算原数据集的香农熵
	bestInfoGain = 0.0 											#初始化最大信息增益
	bestFeature = -1											#初始化最佳划分特征索引
	

	for i in range(numFeature) :								#外循环，依次处理每一个特征值的情况
		featList = [example[i] for example in dataSet]    	 	#先遍历整个数据集，获取特征矩阵中每一种特征的所有值，赋值给一个链表featList
		uniqueVals = set(featList)								#创建set{}集合，元素不可重复，处理统一特征不同特征值情况即可    从列表中创建无重复元素集合是最快的
		newEntropy = 0.0 										#初始化某一特征划分下的香农熵


		for value in uniqueVals :								#内循环，处理同一特征不同值情况下的划分，计算每种划分下的香农熵，并对总的香农熵的值求和
			subDataSet = splitDataSet(dataSet,i,value)			#subDataSet是划分后的子集
			prob = len(subDataSet) / float(len(dataSet))		#按照特征值的一个可能值划分    计算对应的概率
			

													# 香农熵是一个期望值，计算对某特征值进行划分时的香农熵，就要对该特征值不同取值情况下划分的香农熵进行一个求和
													# 香农熵					H1	H2
													# 特征值划分出子集概率	p1	p2
													# E(H) = H1 * p1 + H2 * p2

			newEntropy += prob * calcShannonEnt(subDataSet)		#按照某一特征所有可能值的划分  其全部的香农熵

		infoGain = baseEntropy - newEntropy						#计算新的信息增益，信息增益使熵的减少或者数据无序度的减少，信息增益越大好

		if (infoGain > bestInfoGain) :							#哨兵法
			bestInfoGain = infoGain 							#选择最大信息增益
			bestFeature = i 									#确定最佳划分的特征选取
	return bestFeature
   

# myDat = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
# a = chooseBestFeatureToSplit(myDat)
# print(a)

#定义多数表决方法 解决  全部属性处理完后类标签不唯一问题
def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		classCount[vote] = classCount.get(vote,0) + 1          #对类标签中各个标签进行计数
	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)  #对标签次数排序 将classcount分为元组列表  按照第二个值排序 降序
	return sortedClassCount[0][0]                              #返回次数最多的标签，作为多数表决的结果


#创建树的函数代码
def createTree(dataSet,labels):
	classList = [example[-1] for example in dataSet]   		   	#取出数据集最后一列所有的标签
	if classList.count(classList[0]) == len(classList):			#特征未遍历完，标签完全相同，返回该标签，这是迭代构造决策树的第一个停止条件，	
		return classList[0]										
	if len(dataSet[0]) == 1:									#遍历完所有特征，标签仍不唯一，返回出现次数最多的标签，这是迭代构造决策树的第二个停止条件，	
		return majorityCnt(classList)							
	bestFeat = chooseBestFeatureToSplit(dataSet)				#找到数据集最佳划分点，返回该划分特征的下标
	bestFeatLabel = labels[bestFeat]							#获取最佳划分特征
	myTree = {bestFeatLabel:{}}									#构造树的信息，字典结构，具有映射关系
	del(labels[bestFeat])										#将已选维度标签从标签列表中删除
	featValues = [example[bestFeat] for example in dataSet]		#获取数据集中，该标签（特征）每一个可能的取值
	uniqueVals = set(featValues)								#创建set{}集合，元素不可重复，处理统一特征不同特征值情况即可    从列表中创建无重复元素集合是最快的
	for value in uniqueVals :									#对最优标签（特征）的每个可能的取值进行操作，一个值对应一个分支
		subLabels = labels[:]									#复制标签列表，保证每次递归调用createTree时不改变原始列表内容，labels中已删除当前最优标签
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
																#递归对每个分支建立子树
	return myTree 
# myDat = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
# labels = ['no surfacing','flippers']
# myTree = createTree(myDat,labels)
# print(myTree)

#定义样式
decisionNode = dict(boxstyle = "sawtooth",fc = "0.8")		#定义文本框样式 boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细
leafNode = dict(boxstyle = "round4",fc = "0.8")				#定义文本框样式 boxstyle为文本框的类型，round4是四个角是圆形的矩形，fc是边框线粗细
arrow_args = dict(arrowstyle = "<-")						#定义箭头样式   arrowstyle为箭头的类型，"<-"指向文本框

#绘制箭头线和节点注解   绘制的是一个箭头和一个节点
def plotNode(nodeTxt,centerPt,parentPt,nodeType):           #输入分别为文本内容，起始点坐标，文本中心点坐标，节点类型
	createPlot.ax1.annotate(nodeTxt,xy = parentPt,xycoords = "axes fraction",xytext = centerPt,textcoords = "axes fraction",
							va = "center",ha = "center",bbox = nodeType,arrowprops = arrow_args)
#xy是箭头尖的坐标，xytext设置注释内容显示的中心位置
#xycoords和textcoords是坐标xy与xytext的说明（按轴坐标），若textcoords=None，则默认textcoords与xycoords相同，若都未设置，默认为data
#va/ha设置节点框中文字的位置，va为纵向取值为(u'top', u'bottom', u'center', u'baseline')，ha为横向取值为(u'center', u'right', u'left')

#获取叶子节点数，也就是x轴长度
def getNumLeafs(myTree) :									#输入字典集合myTree
	numLeafs = 0
	firstStr = list(myTree.keys())[0]						#list(myTree.keys())[0]返回第一个键，python3.x中要先将字典转换成集合list才能使用索引
	secondDict = myTree[firstStr]							#获取树形结构首节点的各个子节点内容
	
	for key in secondDict.keys() :							#字典中是键值对组成，secondDict.keys()返回所有的键
		if type(secondDict[key]).__name__ == 'dict' :		#如果该节点为字典，此节点为判断节点，需要向下递归调用，如果不是字典，则说明该节点是叶子节点
			numLeafs += getNumLeafs(secondDict[key])
		else: numLeafs +=1
	return numLeafs

#获取树的层数，也就是y轴长度
def getTreeDepth(myTree) :									#输入字典集合myTree
	maxDepth = 0
	firstStr = list(myTree.keys())[0]						#list(myTree.keys())[0]返回第一个键，python3.x中要先将字典转换成集合list才能使用索引
	secondDict = myTree[firstStr]							#获取树形结构首节点的各个子节点内容
	for key in secondDict.keys() :							#在子节点可能的键的取值循环
		if type(secondDict[key]).__name__ == 'dict' :		#如果该节点为字典，此节点为判断节点，需要向下递归调用，如果不是字典，则说明该节点是叶子节点
			thisDepth = 1+ getTreeDepth(secondDict[key])
		else: thisDepth = 1
		if thisDepth > maxDepth : maxDepth = thisDepth
	return maxDepth


#在父子节点间填充文本信息
def plotMidText(cntrPt,parentPt,textString):					#输入子节点坐标，父节点坐标，文本描述信息
	xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]			#求值得到文本信息的x轴坐标，位于父子节点中间的位置
	yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]			#求值得到文本信息的y轴坐标，位于父子节点中间的位置
	createPlot.ax1.text(xMid,yMid,textString)					#绘制文本

#(1)绘制自身　
#(2)判断子节点非叶子节点，递归
#(3)判断子节点为叶子节点，绘制
#重点，递归绘图，决定整个树的绘制，逻辑绘制。    通过计算当前所有叶子节点个数，划分图形宽度，决定当前节点的位置  
def plotTree(myTree,parentPt,nodeText):							
	numLeafs = getNumLeafs(myTree)								#计算当前的叶子节点数，
	depth = getTreeDepth(myTree)								#计算当前的树的高度，
	firstStr = list(myTree.keys())[0]							#获取树第一个节点
	cntrPt = (plotTree.xOff + (1.0 + float(numLeafs) ) / 2.0 / plotTree.totalW,plotTree.yOff) #按照比例绘图  方便缩放
	#利用整棵树的叶子节点数作为份数将整个x轴的长度进行平均切分，利用树的深度作为份数将y轴长度进行平均切分，
	#plotTree.xOff是最近绘制的一个叶子节点的x坐标
	#plotTree.totalW  x轴总长度
	#float(numLeafs) ) / plotTree.totalW 当前分支叶子节点个数 / 总的x轴距离    得到的是当前分支叶子节点数所占的总长度
	#float(numLeafs) ) / 2 / plotTree.totalW 再除以2就得到这个分支初始节点的位置，当前节点的位置即为其所有叶子节点所占距离的中间
	#plotTree.xOff初始值为-0.5/plotTree.totalW,加上偏移量

	#plotTree.yOff是最近绘制的一个叶子节点的y坐标
	plotMidText(cntrPt,parentPt,nodeText)							#绘制中间文本
	plotNode(firstStr,cntrPt,parentPt,decisionNode)					#绘制箭头和节点

	secondDict = myTree[firstStr]
	plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD			#按比例减少全局变量plotTree.yOff
	for key in secondDict.keys() :									#判断子节点是不是字典类型，若是，继续迭代，若不是，就把这个节点画出来
		if type(secondDict[key]).__name__ == 'dict' :	
			plotTree(secondDict[key],cntrPt,str(key))
		else :
			plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW   				#x坐标增加(1/叶子个数)长度
			plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)	#绘制箭头线和节点注解
			plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))				#绘制中间文本
	plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD			                #有可能树的一个子节点下沿2层，这里加上y的坐标是为了递归结束时，y坐标正常				

#主函数，绘制树形图
def createPlot(inTree):
	fig = plt.figure(1,facecolor = 'white')
	fig.clf()
	axprops = dict (xticks = [],yticks = [])						#为x,y轴的主刻度和次刻度设置颜色、大小、方向，以及标签大小。均为[],画出的图像中就不含xy轴坐标。
	createPlot.ax1 = plt.subplot(111,frameon = 'False',**axprops)

	plotTree.totalW = float(getNumLeafs(inTree))					#全局变量plotTree.totalW  存储树的宽度  叶子节点个数
	plotTree.totalD = float(getTreeDepth(inTree))					#全局变量plotTree.totalD  存储树的深度  树的深度
																	#通过上面两个变量，可以计算节点摆放位置，把树绘制在水平和竖直位置的中心处
																	
	plotTree.xOff = -0.5 / plotTree.totalW							#全局变量plotTree.xOff追踪上一个绘制节点的x坐标，初始化有偏移量，方便计算x坐标
	plotTree.yOff = 1.0												#全局变量plotTree.yOff追踪上一个绘制节点的y坐标，初始化为1

	plotTree(inTree,(0.5,1.0),'')								#树的初始节点必然在整个图像的中间，所以位置设置成（0.5,1）
	plt.show()
																#利用这样的逐渐增加x的坐标，以及逐渐降低y的坐标能能够很好的将树的叶子节点数和深度考虑进去，
																#因此图的逻辑比例就很好的确定了，这样不用去关心输出图形的大小，一旦图形发生变化，函数会重新绘制，
																#但是假如利用像素为单位来绘制图形，这样缩放图形就比较有难度了

# def retrieveTree(i):
# 	listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
# 				   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head':{0:'no', 1: 'yes'}},1:'no'}}}}]
# 	return listOfTrees[i]
# myTree = retrieveTree(0)
# createPlot(myTree) 

#使用决策树的分类函数
#输入：决策树，分类标签，测试数据          输出：决策结果           描述：跑决策树
def classify(inputTree,featLabels,testVec):										
	firstStr = list(inputTree.keys())[0]										#决策树是经过计算信息熵得到的树形结构
	secondDict = inputTree[firstStr]											#分类标签是对应的数据的特征集合
																				#测试数据实际上是该数据对应的各个特征的取值

	featIndex = featLabels.index(firstStr)										#这个索引值对应的是标签(也就是特征位置)

	for key in secondDict.keys() :
		if testVec[featIndex] == key :											#比较特征值，决策树是根据特征的值划分的，看进入哪个取值分支中
			if type(secondDict[key]).__name__ == 'dict' :						#比较是否到达叶结点
				classLabel = classify(secondDict[key],featLabels,testVec)		#递归调用
			else :
				classLabel = secondDict[key]									#到达叶子节点时返回决策结果
	return classLabel



# 序列化过程将文本信息转变为二进制数据流。这样就信息就容易存储在硬盘之中，当需要读取文件的时候，从硬盘中读取数据，然后再将其反序列化便可以得到原始的数据。
# 在Python程序运行中得到了一些字符串、列表、字典等数据，想要长久的保存下来，方便以后使用，而不是简单的放入内存中关机断电就丢失数据。python模块大全中的Pickle模块就派上用场了，
# 它可以将对象转换为一种可以传输或存储的格式。loads()函数执行和load() 函数一样的反序列化。取代接受一个流对象并去文件读取序列化后的数据，它接受包含序列化后的数据的str对象,
# 直接返回的对象。

#存储决策树
def storeTree(inputTree,filename):												#输入决策树，文件名
	import pickle																
	fw = open(filename,'wb+')													#参数wb+是指用二进制方式打开文件，准备写入
	pickle.dump(inputTree,fw)													#以二进制形式写入，序列化对象
	fw.close()

#读取决策树
def grabTree(filename):
	import pickle
	fr = open(filename,'rb+')													#参数wb+是指用二进制方式打开文件，准备读取
	return pickle.load(fr)														#以二进制形式读出，反序列化对象

myDat = [[1,1,'maybe'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
labels = ['no surfacing','flippers']
myTree = createTree(myDat,labels)
print(myTree)
storeTree(myTree,'StoreTree.txt')
grabTree('StoreTree.txt')
print(grabTree('StoreTree.txt'))
createPlot(myTree)