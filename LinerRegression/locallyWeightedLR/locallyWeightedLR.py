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

xArr,yArr = loadDataSet('ex0.txt')
# print(lwlr(xArr[0],xArr,yArr,1.0))
# print(lwlr(xArr[0],xArr,yArr,0.001))
yHat = lwlrTest(xArr,xArr,yArr,0.01)
# print(yHat)


xMat = mat(xArr)
srtInd = xMat[:,1].argsort(0) 										#返回排序后的索引
xSort = xMat[srtInd][:,0,:]											#按照从小到大的顺序重新排序输入向量,这里的xMat[srtInd]是一个三维数组

print(srtInd[:,])																	
																	
print(xMat[srtInd][:])											#xMat[srtInd][:]是全部的三维数组
print(xMat[srtInd][:,0])											#xMat[srtInd][:,0]去掉一个括号，变成二维数组
print(xMat[srtInd][:,0,:])										#xMat[srtInd][:,0,：]去掉一个括号，变成一维数组

fig = plt.figure()
ax  = fig.add_subplot(111)
ax.plot(xSort[:,1],yHat[srtInd])
ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s = 2, c = 'red')
																	#flatten()方法能将matrix的元素变成一维的，.A能使matrix变成array  .A[0]能少一个[]
 																	#flatten用于array和mat对象，flatten是深拷贝，copy是浅拷贝
 																	#xMat[:,1].flatten()==>matrix[[]]         
 																	#xMat[:,1].flatten().A==>array[[]]           
 																	#xMat[:,1].flatten().A[0]==>array[]
plt.show()


