# y=f(x)=ax+b				线性回归模型中，输出一般是连续的
# y=f(x)=a0x0+a1x1+a2x2  	逻辑回归也被称为广义线性回归模型，它与线性回归模型的形式基本上相同，都具有 ax+b，其中a和b是待求参数，
# 即y=a1x1+a2x2+b			其区别在于他们的因变量不同，多重线性回归直接将ax+b作为因变量，即y = ax+b，而logistic回归则通过函数S将ax+b对应到一个隐状态p，
# 为了处理常数项				p = S(ax+b)，然后根据p与1-p的大小决定因变量的值。这里的函数S就是Sigmoid函数
#sigmod函数：						S = 1/(1+exp(-z))     
#获得所有特征的最佳拟合曲线的参数		z = w0x0+w1x1+w2x2+...wnxn+b
#根据sigmod函数进行分类				hw(x) =  1/(1 + exp[-(w0x0+w1x1+w2x2+...wnxn+b)]) = 1/(1+exp(-wx))
#损失函数          					cost(hw(x),y) = hw(x)^y + (1-hw(x)^(1-y))         
#对数似然函数							L(hw(x),y) = y*log(hw(x)) + (1-y)*log(1-hw(x)) 
#采用梯度上升算法求最优ｗ值时，只需要求得对数似然函数的导数，即可获得梯度上升的梯度，然后就知道迭代公式
#L对w求导得		 					L'     = y*(1/hw(x))*hw(x)' + (1-y)*(1/(1-hw(x)))*(1-hw(x))'
#hw(x)对w求导						hw(x)' = 1/(1+exp(-wx))^2 * exp(-wx) * x
#hw(x)'带入 L'得 				L'= y*(1+exp(-wx))*1/(1+exp(-wx))^2 * exp(-wx) * x + (1-y) * (1+exp(-wx)/exp(-wx)) * (-1/(1+exp(-wx))^2 * exp(-wx) * x)
#化简							L'= y*1/(1+exp(-wx))* exp(-wx) * x - (1-y) * (x/(1+exp(-wx)))
#化简							L'= x*y*exp(-wx)/(1+exp(-wx)) - x*(1-y)/(1+exp(-wx))
#化简							L'= x*[y - hw(x)]
#所以，梯度上升迭代公式为         		wi+1 = wi + α[y-hw(x)]*x 	即代码：weights = weights + alpha * dataMatrix.transpose() * error

from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():														
	dataMat = []
	labelMat = []
	fr = open('testSet.txt')
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])			#对于dataMat[]后两个数据是样本两个特征，开始加入的1.0就是X0，对应参数W0
		labelMat.append(int(lineArr[2]))
	return dataMat,labelMat

def sigmoid(inX):															#sigmoid函数计算
	return 1.0/(1+exp(-inX))

#梯度上升算法的实现
def gradAscent(dataMatIn,classLabels):										#输入参数dataMatIn[]是一个三维数组，n个样本，n*3，classLabels是1*n的行向量
	dataMatrix = mat(dataMatIn)												#转换成numpy矩阵b
	labelMat = mat(classLabels).transpose()									#为了让计算方便，把行向量矩阵classLabels转换成列向量，做法是矩阵转置  a.transpose()
	m,n = shape(dataMatrix)													#求得矩阵的行列数100*3    m->数据量，样本数     n->特征数
	alpha = 0.001															#像目标移动的步长
	maxCycles = 500															#最大迭代次数
	weights = ones((n,1))													#初始化回归系数，weights回归系数[[1],[1],[1]]
	for k in range(maxCycles):
		h = sigmoid(dataMatrix * weights)									#h是一个列向量，也就是100组样本数据在初始的回归系数条件下，判断其属于哪一个类别
		error = (labelMat - h)												#真实类别 与 与预测类别的差值
		weights = weights + alpha * dataMatrix.transpose() * error			#修改回归系数dataMatrix.transpose() * error 3*100的矩阵和100*1的矩阵乘积是3*1矩阵
																			#系数变化量就是步长alpha乘以损失函数对于系数的求导    关键之处在于对损失函数的求解
																			#https://blog.csdn.net/cxjoker/article/details/83002197参考推导过程
	return weights

#随机梯度上升算法
def stocGradAscent0(dataMatrix,classLabels):
	m,n = shape(dataMatrix)
	alpha = 0.01
	weights = ones(n)
	for i in range(m):
		h = sigmoid(sum(dataMatrix[i]*weights))								#h是数值，不是向量
		error = classLabels[i] - h
		weights = weights + alpha * error * dataMatrix[i]
	return weights

#改进的随机梯度上升算法
def stocGradAscent1(dataMatrix,classLabels,numIter = 150):
	m,n = shape(dataMatrix)
	weights = ones(n)
	for j in range(numIter):
		dataIndex = list(range(m))
		for i in range(m):
			alpha = 4 / (1.0+j+i) + 0.01					#alpha每次迭代时会调整，缓解回归系数波动情况，但不会减小到0，因为需要常数项存在让新数据有影响
			randIndex = int(random.uniform(0,len(dataIndex))) 				#随机选取样本更新回归系数   随机从列表中选取一个值，单后在列表中删除这个值
			h = sigmoid(sum(dataMatrix[randIndex]*weights))							#h是数值，不是向量
			error = classLabels[randIndex] - h
			weights = weights + error * alpha * dataMatrix[randIndex]
			del dataIndex[randIndex]
	return weights


#画出数据集和Logistic回归最佳拟合直线
def plotBestFit(weights):
	dataMat,labelMat = loadDataSet()										#获取数据
	dataArr = array(dataMat)
	n = shape(dataArr)[0]													#获取样本数
	xcord1 = []																#初始化样本点
	ycord1 = []
	xcord2 = []
	ycord2 = []
	for i in range(n):														#获取样本点，不同类别的样本点存入不同的数组中
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i,1])										#读取特征矩阵100*3中的[i,1]的坐标，即x1
			ycord1.append(dataArr[i,2])  									#读取特征矩阵100*3中的[i,2]的坐标，即x2
		else:
			xcord2.append(dataArr[i,1])
			ycord2.append(dataArr[i,2])  
	fig = plt.figure()
	ax 	= fig.add_subplot(111)												#add_subplot()函数在一张figure里面生成多张子图参数111，表示1行1列第1个位置
	ax.scatter(xcord1,ycord1,s = 30,c = 'red',marker = 's')					#绘制散点图  x.scater(x,y,s,c,marker),x,y表示坐标，s表示尺寸大小，c表示颜色，
	ax.scatter(xcord2,ycord2,s = 30,c = 'green')							#marker表示点的类型   正方形点，圆点等  默认为圆点
	x = arange(-3.0,3.0,0.1)												#arange(-3.0,3.0,0.1)其中三个参数意思为：(start, end, step)
	y = (-weights[0]-weights[1]*x)/weights[2]								#0是两个分类的分界处，设定 0 = w0x0+w1x1+w2x2   解出x1和x2的关系
	ax.plot(x,y)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()

dataArr,LabelMat = loadDataSet()
weights = gradAscent(dataArr,LabelMat)
plotBestFit(weights.getA())													#矩阵通过这个getA()这个方法可以将自身返回成一个n维数组对象,画图需要

dataArr,LabelMat = loadDataSet()
weights = stocGradAscent0(array(dataArr),LabelMat)
plotBestFit(weights)

dataArr,LabelMat = loadDataSet()
weights = stocGradAscent1(array(dataArr),LabelMat)
plotBestFit(weights)