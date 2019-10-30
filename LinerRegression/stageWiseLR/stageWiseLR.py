from numpy import * 
import matplotlib.pyplot as plt

#读样本数据
def loadDataSet(fileName) :
	numFeat = len(open(fileName).readline().split('\t')) - 1
	dataMat = []
	labelMat = []
	fr = open(fileName)
	for line in fr.readlines() :
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat) :
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat,labelMat

#计算误差  均方误差和
def rssError(yArr,yHatArr):
	return ((yArr - yHatArr) ** 2).sum()

#对数据进行标准化处理    
def regularize(xMat): #regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0) #计算平均数，然后减去它
    inVar = var(inMat,0)
    inMat = (inMat-inMeans)/inVar
    return inMat 

#前向逐步回归
def stageWise(xArr,yArr,eps = 0.01,numIt = 100):								#参数eps(需要调整的步长)设置为0.01
	xMat = mat(xArr)
	yMat = mat(yArr).T

	#数据标准化处理
	yMean = mean(yMat,0)														#numpy.mean(a, axis, dtype, out，keepdims)
	yMat = yMat - yMean															#mean()函数功能：求取均值 经常操作的参数为axis，以m * n矩阵举例：
																				#axis 不设置值，对 m*n 个数求均值，返回一个实数
																				#axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
																				#axis = 1：压缩列，对各行求均值，返回 m *1 矩阵

	xMat = regularize(xMat)														#对xMat进行标准化处理，标准化处理函数为regularize()  使得均值为0 方差为1
	m,n = shape(xMat)
	returnMat = zeros((numIt,n))												#返回所有迭代中ws的变化情况，初始化为(numIt,n)维矩阵


	ws = zeros((n,1))															#回归系数ws初始化为(n,1)维零数组
	wsTest = ws.copy()															#为了实现贪心算法 建立ws的两个副本
	wsMax = ws.copy()

	#迭代numIt次，每次迭代，循环n*2次(每个特征有增大和减小两个方向)，找出令rssError最小的方向(哪个特征，对应增大还是减小),保存ws,下次迭代在ws基础上做更新
	for i in range(numIt):
		print(ws.T)																#迭代numIt次  每次迭代都要打印出 回归系数向量  用于分析算法执行的过程和效果
		lowestError = inf
		for j in range(n):														#在每一个特征上运行两次for循环，分别计算增加或减少该特征对误差的影响
			for sign in [-1,1]:
				wsTest = ws.copy()
				wsTest[j] += eps * sign											#在单个特征值上进行 增加或减少步长的操作
				yTest = xMat * wsTest											#生成预测结果
				rssE = rssError(yMat.A,yTest.A)									#误差采用均方误差
				if rssE < lowestError:											#更新误差，选取小误差情况
					lowestError = rssE
					wsMax = wsTest
		ws = wsMax.copy()														#更新ws
		returnMat[i,:] = ws.T 													#记录每一次迭代的ws
	return returnMat 

xArr,yArr = loadDataSet('abalone.txt')											#训练模型，在前100条数据上训练得到w
Weights = stageWise(xArr,yArr,0.01,200)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Weights)														#这里的plot()绘图是把所有的权重连起来画的，只有8条曲线，最左得到的权重与LR一致，最右系数缩减成0
plt.show()

#偏差和方差的权衡
#缩减系数的方法  比如 岭回归  前向逐步回归 lasso等  模型增加了偏差  但是与此同时模型的方差减少了
# 缩减系数  减少模型复杂度 降低预测误差   方差偏差折中
#方差  模型之间的差异
#偏差  预测值和数据之间的差异
#高偏差  --->  欠拟合    预测值 和  真实值偏离较大
#高方差  --->  过拟合		预测值 和  真实值偏离较小
#模型越复杂，偏差就越小，而模型越简单，偏差就越


#均方误差 = 偏差 + 测量误差 + 随机噪声