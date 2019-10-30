#通过遍历，改变不同的阈值，计算最终的分类误差，找到分类误差最小的分类方式，即为我们要找的最佳单层决策树。
#这里lt表示less than，表示分类方式，对于小于阈值的样本点赋值为-1，gt表示greater than，也是表示分类方式，对于大于阈值的样本点赋值为-1。
#经过遍历，我们找到，训练好的最佳单层决策树的最小分类误差为0.2，就是对于该数据集，无论用什么样的单层决策树，分类误差最小就是0.2。这就是我们训练好的弱分类器

from numpy import *
import matplotlib.pyplot as plt

def loadSimpData():
	dataMat = matrix([[1.,2.1],
					  [2.,1.1],
					  [1.3,1.],
					  [1.,1.],
					  [2.,1.]])
	classLabels = [1.0,1.0,-1.0,-1.0,1.0]
	return dataMat,classLabels

#单层决策树的阈值过滤函数，也称决策树桩，它是一种简单的决策树，通过给定的阈值，进行分类
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):							#dimen - 第dimen列，也就是第几个特征    threshVal - 阈值   threshIneq - 标志
	retArray = ones((shape(dataMatrix)[0],1))										#初始化返回数组  n个样本  就是 n行1列的矩阵 值为1
	if threshIneq == 'lt':															#遍历大于小于的情况     lt : less than 
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0							#将小于某一阈值的特征归类为-1
	else :																			#遍历大于小于的情况     gt : greater than
		retArray[dataMatrix[:,dimen] > threshVal] = -1.0							#将大于某一阈值的特征归类为-1
	return retArray																	#返回分类结果


#建立弱分类器，保存样本权重 弱分类器使用单层决策树（decision stump）
def buildStump(dataArr,classLabels,D):												# D - 样本权重
	dataMatrix = mat(dataArr)														#将数据集和标签列表转为矩阵形式
	labelMat = mat(classLabels).T
	m,n = shape(dataMatrix)
	numSteps = 10.0																	#步长或区间总数 最优决策树信息 最优单层决策树预测结果
	bestStump = {}
	bestClassEst = mat(zeros((m,1)))
	minError = inf																	#最小错误率初始化为+∞
	for i in range(n) :																#遍历每一列的特征值
		rangeMin = dataMatrix[:,i].min()											#找出列中特征值的最小值和最大值
		rangeMax = dataMatrix[:,i].max()
		stepSize = (rangeMax - rangeMin) / numSteps									#求取步长大小或者说区间间隔
		for j in range(-1,int(numSteps) + 1):										#遍历各个步长区间  (从-1 到 10  共12个区间)     这一步是为了取得特征值所有可能的取值
			for inequal in ['lt','gt'] :											#遍历大于小于的情况     lt : less than  gt : greater than
				threshVal = (rangeMin + float(j) * stepSize)						#阈值计算公式：最小值+j(-1<=j<=numSteps+1)*步长
				predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)		#选定阈值后，调用阈值过滤函数分类预测
				errArr = mat(ones((m,1)))
				errArr[predictedVals == labelMat] = 0								#将错误向量中分类正确项置0
				weightedError = D.T * errArr										#计算"加权"的错误率

				print('split: dim  %d, thresh %.2f, thresh ineqal  %s, the weighted error is %.3f' % (i,threshVal,inequal,weightedError))

				if weightedError < minError:
					minError = weightedError
					bestClassEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	print(bestStump)
	return bestStump,minError,bestClassEst											#返回最佳单层决策树相关信息的字典，最小错误率，决策树预测输出结果

# D = mat(ones((5,1))/5)																#D初始化为0.2
# dataMat,classLabels = loadSimpData()
# buildStump(dataMat,classLabels,D)

#完整AdaBoost算法的实现    返回决策数组
def adaBoostTrainDs(dataArr,classLabels,numIt = 40):            					#numIt 是迭代次数 adaBoostTrainDs 尾部的DS表示单层决策树 是最流行的弱分类器
	weakClassArr = []																#存储每一个生成的弱分类器 ，也就是分类所需要的所有信息
	m = shape(dataArr)[0]
	D = mat(ones((m,1))/m)															#D表示每个样本的权重，初始化为 n行1列，值为1/n 其中n为样本个数
	aggClassEst = mat(zeros((m,1)))													#列向量aggClassEst记录每个数据点的类别估计累计值
	for i in range(numIt) :															#开始进行迭代  如果训练的弱分类器个数达到要求或者错误率为0时 退出循环
		bestStump,error,ClassEst = buildStump(dataArr,classLabels,D)				#构建单层决策树   
		print('D : ' , D.T)															#输出样本权重

		alpha = float(0.5 * log((1.0-error) / max(error,1e-16)))					#计算alpha  alpha = 0.5 * ln[(1 - e) / e] 使error不等于0,因为分母不能为0
		bestStump['alpha'] = alpha													#存储弱学习算法权重  即分类器权重alpha
		weakClassArr.append(bestStump)												#存储单层决策树完整信息
		print('alpha : ' , alpha)

		print('ClassEst : ' , ClassEst.T)											#输出预测类别值
																					#Dnew = Dold * exp(+-alpha)/sum(D)     样本正确 取-alpha 样本错误 取+alpha
		expon = multiply(-alpha * mat(classLabels).T,ClassEst)						#计算e的指数项 由于类别为+1或-1 若预测正确 二者乘积为1 也就是取 +alpha 反之取-alpha 
		D = multiply(D,exp(expon))													#按照公式计算
		D = D / D.sum()																#更新D

		aggClassEst += alpha * ClassEst 											#每个数据点的类别估计累计值  ClassEst 是单层决策树的预测结果 
		print('aggClassEst : ' , aggClassEst.T)										#输出几个分类器的加权结果求和    累加变成强分类器
		aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))	#利用符号函数获得最终的预测结果，和真实值比较 ,累计错误个数
						   #(sign(aggClassEst) != mat(classLabels).T)   若正确返回TRUE 做乘法时看做1 否则返回FALSE,做乘法时看做0
		errorRate = aggErrors.sum() / m
		print('total error : ' , errorRate)
		if errorRate == 0.0 :														#误差为0 退出循环
			break
	return weakClassArr

# dataMat,classLabels = loadSimpData()
# classifierArray = adaBoostTrainDs(dataMat,classLabels,9)
# print(classifierArray)


def adaClassify(datToClass,classifierArr) :											#输入参数，一个待划分类别的数据集，一个是经过训练的多个弱分类器
	dataMatrix = mat(datToClass)
	m = shape(dataMatrix)[0]
	aggClassEst = mat(zeros((m,1)))
	for i in range(len(classifierArr)) :
		ClassEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
		aggClassEst += classifierArr[i]['alpha'] * ClassEst
		print(aggClassEst)
	return sign(aggClassEst)

dataMat,classLabels = loadSimpData()
classifierArray = adaBoostTrainDs(dataMat,classLabels,9)
adaClassify([[0,0],[5,5]],classifierArray)