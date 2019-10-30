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

def lwlr(testPoint,xArr,yArr,k = 1.0) :										# w = (X^T * W * X)^-1 * X^T * W * Y ,在lwlr中，给待预测点附近每个点赋予一定的权重
	xMat = mat(xArr)
	yMat = mat(yArr).T
	m = shape(xMat)[0]														#获取样本数目
	weights = mat(eye((m)))													#为每一个样本点初始化一个权重，生成对角阵,即单位矩阵
	for j in range(m):														#权重值大小以指数级衰减
		# print(testPoint)
		# print(xMat[j,:])
		diffMat = testPoint - xMat[j,:]										#这里计算的是 xi - x
		# print(diffMat)
		weights[j,j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))			#使用高斯核赋予权重   w(i,i) = exp(|xi - x| / (-2 * k^2))
	xTx = xMat.T * (weights * xMat)
	if linalg.det(xTx) == 0.0:												#判断(X^T * W * X) 行列式是否为0 为0的话  没有逆矩阵  退出
		print('This matrix is singular, cannot do inverse')
		return 
	ws = xTx.I * (xMat.T * (weights * yMat))								#计算权重 w = (X^T * W * X)^-1 * X^T * W * Y ,矩阵.I 作用是求矩阵的逆矩阵
	return testPoint * ws 													#返回预测值  

def lwlrTest(testArr,xArr,yArr,k = 1.0) :
	m = shape(testArr)[0]
	yHat = zeros(m)
	for i in range(m) :
		yHat[i] = lwlr(testArr[i],xArr,yArr,k)
	return yHat


def rssError(yArr,yHatArr):
	return ((yArr - yHatArr) ** 2).sum()

abX,abY = loadDataSet('abalone.txt')											#训练模型，在前100条数据上训练得到w
yHat01 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)							#计算不同K值下的预测结果
yHat1 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
yHat10 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)

print(rssError(abY[0:99],yHat01.T))												#输出预测误差   采用均方误差
print(rssError(abY[0:99],yHat1.T))
print(rssError(abY[0:99],yHat10.T))

print('\n\n')

yHat01 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)							#在后100条数据进行测试，可以发现较小的核有可能造成过拟合，效果并不是特别好
yHat1 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
yHat10 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)

print(rssError(abY[100:199],yHat01.T))
print(rssError(abY[100:199],yHat1.T))
print(rssError(abY[100:199],yHat10.T))