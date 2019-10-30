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

#计算回归系数
def ridgeRegres(xMat,yMat,lam = 0.2):
	xTx = xMat.T * xMat
	denom = xTx + eye(shape(xMat)[1]) * lam 										#eye()生成单位矩阵
	if linalg.det(denom) == 0.0 :													#判断矩阵是狗属于非奇异矩阵
		print('This matrix is singular, cannot do inverse')
		return 
	ws = denom.I * (xMat.T * yMat)
	return ws 																		#返回回归系数

#在一组λ上测试结果
def ridgeTest(xArr,yArr):
	xMat = mat(xArr)
	yMat = mat(yArr).T



	#下面做数据标准化处理	  对各个特征值进行处理	（所有特征 - 各自均值）/ 方差						
	yMean = mean(yMat,0)															#numpy.mean(a, axis, dtype, out，keepdims )
	yMat = yMat - yMean																#mean()函数功能：求取均值 经常操作的参数为axis，以m * n矩阵举例：
																					# axis 不设置值，对 m*n 个数求均值，返回一个实数
																					# axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
																					# axis = 1：压缩列，对各行求均值，返回 m *1 矩阵
	xMeans = mean(xMat,0)
	xVar = var(xMat,0)																#var()函数功能 : 求方差 经常操作的参数为axis，以m * n矩阵举例：
																					# axis 不设置值，对 m*n 个数求均值，返回一个实数
																					# axis = 0：压缩行，对各列求方差，返回 1* n 矩阵
																					# axis = 1：压缩列，对各行求方差，返回 m *1 矩阵
	xMat = (xMat - xMeans) / xVar


	numTestPts = 30
	wMat = zeros((numTestPts,shape(xMat)[1]))
	for i in range(numTestPts) :													#在30个不同的lambda下调用ridgeRegres	
		ws = ridgeRegres(xMat,yMat,exp(i-10))										#lambda以指数级变化
		wMat[i,:] = ws.T
	return wMat																		#将所有的回归系数输出到一个矩阵


abX,abY = loadDataSet('abalone.txt')												#训练模型，在前100条数据上训练得到w
ridgeWeights = ridgeTest(abX,abY)
print(ridgeWeights.shape)
print(ridgeWeights)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeights)																#这里的plot()绘图是把所有的权重连起来画的，只有8条曲线，最左的权重与LR一致，最右系数缩减成0
plt.show()