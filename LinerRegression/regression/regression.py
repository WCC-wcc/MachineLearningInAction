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

#计算最佳拟合直线
def standRegress(xArr,yArr) :										#矩阵相乘要把结构改成matrix,否则array的*只能对元素进项相乘
	xMat = mat(xArr)												#读取x
	yMat = mat(yArr).T 												#读取y
	xTx  = xMat.T * xMat 											#计算x.T * x     w = (X^T * X)^-1 * X^T * Y  求矩阵逆  先求X^T * X 的行列式
	if linalg.det(xTx) == 0.0 :										#linalg是一个线性代数函数库  linalg.det()计算行列式  判断xTx是否为0 为0的话矩阵没有逆 要去除
		print('This matrix is singular,cannot do inverse')
		return 
	ws = xTx.I * (xMat.T * yMat)									#计算得到 w     矩阵.I 作用是求矩阵的逆矩阵
	return ws


xArr,yArr = loadDataSet('ex0.txt')
ws = standRegress(xArr,yArr)
xMat = mat(xArr)
yMat = mat(yArr)


fig = plt.figure()
ax  = fig.add_subplot(111)

ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])		#flatten()方法能将matrix的元素变成一维的，.A能使matrix变成array  .A[0]能少一个[]
																	#flatten用于array和mat对象，flatten是深拷贝，copy是浅拷贝
																	#xMat[:,1].flatten()==>matrix[[]]         
																	#xMat[:,1].flatten().A==>array[[]]           
																	#xMat[:,1].flatten().A[0]==>array[]

xCopy = xMat.copy()
yCopy = yMat.copy()
yHat  = xCopy * ws

#计算相关系数
ysum = yHat.sum()													#计算总值
yaverage = ysum / (float(len(yHat)))								#计算平均值
yerr = yHat - yaverage 												#用x - xaverage
D = (multiply(yerr,yerr).sum() / (float(len(yHat)))) ** 0.5			#标准差 sigma(x)= 根号下[方差 / n] = 根号下[(x - xaverage)^2 / n]

ysum1 = yMat.T.sum()
yaverage1 = ysum1 / (float(len(yMat.T)))
yerr1 = yMat.T - yaverage1
D1 = (multiply(yerr1,yerr1).sum() / (float(len(yHat))))** 0.5

cov = multiply(yerr,yerr1).sum() / (float(len(yHat)))				#协方差 cov(x,y) = (x - xaverage)(y - yaverage) / n

print(cov/(D*D1))													#相关系数  p(x,y) = cov(x,y) / [sigma(x) * sigma(y)] 
																	#	= (x - xaverage)(y - yaverage) / 根号下[(x - xaverage)^2 / n * [(y - yaverage)^2 / n]]


print(corrcoef(yHat.T,yMat))										#调用函数  计算预测值和真实值的相关系数


ax.plot(xCopy[:,1],yHat)
plt.show()


