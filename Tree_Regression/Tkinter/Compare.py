from numpy import *
import matplotlib.pyplot as plt

#创建树结构
class treeNode():
	def __init__(self,feat,val,right,left):
		featureToSplitOn = feat
		valueOfSplit = val
		rightBranch = right
		leftBranch = left

#读数据
def loadDataSet(fileName):																#读取数据，把每行数据数据转换成一组浮点数
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		#fltLine = map(float,curLine)													#使用了一个map方法来对从文本文件中读取的数据进行映射处理，
																						#也就是把读取到的string转换为float。这个类型转换按照书上实现的方法在Python 2中
																						#不会报错。  fltLine = map(float,curLine)	
																					
																						#但是在Python 3中，map方法返回的是一个map对象，因此对于这个错误，解决办法很简单
		fltLine = list(map(float,curLine))												#方法1，将map方法返回的map对象再转换为list对象就行了。 
		#fltLine = [float(item) for item in curLine]									#方法2，使用列表推导式做一个处理。 
		dataMat.append(fltLine)
	return dataMat

#数据集划分  二元划分
def binSplitDataSet(dataSet,feature,value):												#切分数据集，输入参数为原始数据集，待切分特征，特征值
	mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]							#以数组过滤方式得到两个字数据集
																						#dataSet[:,feature] > vlaue  返回的是TRUE FALSE
																						#nonzero(a)返回数组a中值不为零的元素的下标，返回值是一个长度为a.ndim(数组a的轴数)的元组，
	mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]							#对于特征feature，先过滤出特征feature的值大于（或小于等于）value的。
																						#这里使用nonzero()方法返回的是一个元素均为boolean类型的list，
																						#而这个list[0]的值就是对应过滤出的元素下标，换句话说就是过滤出的值在原数组中的位置。
																						#最后一步是一个Python切片操作，通过dataSet[index, :]把对应的向量提取出来。 
	
	# print(dataSet[:,feature] > value)													#返回的是真假值  判断是否满足条件							[[False][False][True]]
	# print(nonzero(dataSet[:,feature] > value))										#返回两个数组,第一个数组的存放行的值,第二个数组存放列的值 	(array([1, 2], dtype=int64), array([0, 0], dtype=int64))
	# print(nonzero(dataSet[:,feature] > value)[0])										#返回下标 												[1 2]
	# print(dataSet[nonzero(dataSet[:,feature] > value)[0],:])							#利用下标划分原始数据集
	#当使用布尔数组直接作为下标对象或者元组下标对象中有布尔数组时，都相当于用nonzero()将布尔数组转换成一组整数数组，然后使用整数数组进行下标运算
	return mat0,mat1

#生成叶节点   使用叶子节点对应的y值的平均值作为预测值
def regLeaf(dataSet):
	return mean(dataSet[:,-1])															#求数据集最后一列均值  （特征均值）

#平方误差：  各测量值与均值差值的平方和 (xi - x)^2
#均方误差：	各测量值与均值差值的平方和的平均值 (xi - x)^2 / n      也就是平方差的均值

#误差计算     这里用平方误差的总和作为误差函数：   可以先计算平均值 然后计算每个差值再平方  这里调用均方差函数更方便 
def regErr(dataSet):
	return var(dataSet[:,-1]) * shape(dataSet)[0]										#需要总的方差   所以用均方差乘以样本数


#本函数中三个提前终止条件  就是预剪枝技术
#选择最佳划分   遍历所有特征及其可能的取值来找到使误差最小化的切分阈值 			通过误差计算 可以得到数据集上的最佳二元切分
def chooseBestSplit(dataSet,leafType = regLeaf, errType = regErr,ops = (1,4)): 			#用户指定参数 控制函数停止时机
	tolS = ops[0]																		#容许的最小误差下降值  小于这个数值就不用切分了 效果影响不大
	tolN = ops[1]																		#切分得最少样本数

																						#叶子结点   只剩下一个节点时的情况 （也即是只剩一个样本）
	if len(set(dataSet[:,-1].T.tolist()[0])) == 1:										#具体操作   取样本集最后一列  转置  转换成列表 取第一行值 变成集合
		return None,leafType(dataSet) 													#找不到一个“好”的二元切分，返回None并同时调用leafType来生成叶节点
	m,n = shape(dataSet)
	S = errType(dataSet)   																#计算初始数据集的误差  作为初始误差值
	bestS = inf 																		#定义最小化误差
	bestIndex = 0 																		#最佳切分位置
	bestValue = 0 																		#最佳切分特征对应的具体取值
	

	for featIndex in range(n-1): 														#遍历所有特征值
		for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]): 					#遍历所有特征值可能的取值情况
			mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal) 					#通过数组过滤的方式  实现二元切分
			if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN) : 					#切分出来的两个数据集  样本数少于最低切分样本数时 放弃本次切分
				continue
			newS = errType(mat0) + errType(mat1) 										#计算切分出来的两个数据集的误差和
			if newS < bestS: 															#切分后的误差小的情况下 更新切分位置
				bestIndex = featIndex
				bestValue = splitVal
				bestS = newS

																						#遍历所有特征及其可能取值情况后找到最佳切分
	if (S - bestS) < tolS: 																#如果误差减少不大则退出
		return None,leafType(dataSet)
	mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)

																						#再加一次判断
	if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): 								#如果切分出的数据集很小则退出
		return None,leafType(dataSet)
	return bestIndex,bestValue 															#返回最佳切分特征  特征对应取值


#树构建函数 leafType 参数 给出建立叶节点的函数   errType代表误差计算函数  ops是一个包含树构建所需其他参数的元祖
def createTree(dataSet,leafType = regLeaf,errType = regErr,ops = (1,4)):				
	feat,val = chooseBestSplit(dataSet,leafType,errType,ops)							#找到最佳切分特征位置  特征值
	if feat == None: return val 														#满足停止条件时  返回叶节点值（回归树返回常数）（模型树返回线性方程）
	retTree = {}																		#创建树字典结构
	retTree['spInd'] = feat
	retTree['spVal'] = val
	lSet,rSet = binSplitDataSet(dataSet,feat,val)										#递归的构建左子树  右子数
	retTree['left'] = createTree(lSet,leafType,errType,ops)
	retTree['right'] = createTree(rSet,leafType,errType,ops)
	return retTree

#判断树结构
def isTree(obj):
	return (type(obj).__name__ == 'dict')


#对数据集格式化成目标变量Y和自变量X
def linearSolve(dataSet):
	m,n = shape(dataSet)
	X = mat(ones((m,n))) 																			#初始化自变量X
	Y = mat(ones((m,1))) 																			#初始化自变量Y
	X[:,1:n] = dataSet[:,0:n-1] 																	#格式化数据集 把各个记录的的各个特征值赋值给X X[:,1:n] 第2列到第n-1列
	Y = dataSet[:,-1] 																				#按照线性规划的函数计算回归系数
	xTx = X.T * X 														
	if linalg.det(xTx) == 0.0:
		raise NameError('This matrix is singular, cannot do inverse,\n try increasing the second value of ops')
	ws = xTx.I * (X.T * Y)
	return ws,X,Y

#生成叶子节点
def modelLeaf(dataSet):
	ws,X,Y = linearSolve(dataSet)
	return ws 																						#返回回归系数ws

#误差计算函数
def modelErr(dataSet):
	ws,X,Y = linearSolve(dataSet)
	yHat = X * ws 																					#计算预测值
	return sum(power(yHat - Y,2)) 																	#计算误差   预测值和实际值的差值的平方和

#回归树的叶节点为float型常量
def regTreeEval(model,inDat):
	return float(model)																				#返回回归树叶子节点预测值

#模型树的叶节点为浮点型参数的线性方程
def modelTreeEval(model,inDat):
	n = shape(inDat)[1] 																			#获取输入数据的列数
	X = mat(ones((1,n+1)))																			#构建n+1维的单列矩阵
	X[:,1:n+1] = inDat 																				#第一列设置为1，线性方程偏置项b
	# print('-------------------------------------')
	# print(X)
	# print(model)
	# print( X * model )
	# print('+++++++++++++++++++++++++++++++++++++')
	return float(X * model)																			#返回浮点型的回归 数据集 * 系数向量  即模型树预测的y值

#@tree；树回归模型
#@inData：输入数据
#@modelEval：叶节点生成类型，需指定，默认回归树类型
def treeForeCast(tree,inData,modelEval = regTreeEval):
	if not isTree(tree): 																			#如果当前树为叶节点，生成叶节点  这两行代码好像没用
	 	return modelEval(tree,inData)																#对叶子节点进行计算   返回预测值

	#非叶节点，按照该子树对应的切分点对输入数据进行切分    自上而下进行递归
	if inData[tree['spInd']] > tree['spVal']:
		if isTree(tree['left']): 																	#该树的左分支为非叶节点类型 每次只处理一个样本点 inData不用再做切分
			return treeForeCast(tree['left'],inData,modelEval)										#递归调用treeForeCast函数继续树预测过程，直至找到叶节点 
		else:
			return modelEval(tree['left'],inData) 													#左分支为叶节点，生成叶节点	返回预测值
	else:
		if isTree(tree['right']): 																
			return treeForeCast(tree['right'],inData,modelEval)										#继续递归treeForeCast函数寻找叶节点	
		else:
			# print(tree['right'])
			# print(inData)
			return modelEval(tree['right'],inData) 													#叶节点，生成叶节点类型 	返回预测值

#创建预测树 
def createForeCast(tree,testData,modelEval = regTreeEval):
	m = len(testData) 																				#测试集样本数
	yHat = mat(zeros((m,1))) 																		#初始化行向量各维度值为1
	for i in range(m): 																				#遍历每个样本    每次只处理一个样本点
		yHat[i,0] = treeForeCast(tree,mat(testData[i]),modelEval) 									#利用树预测函数对测试集进行树构建过程，并计算模型预测值
		# print(mat(testData[i]))
	return yHat 																					#返回预测值


# trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
# testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))

# #构建回归树
# myTreeReg = createTree(trainMat,ops=(1,20))
# yHatReg = createForeCast(myTreeReg,testMat[:,0])
# corrReg = corrcoef(yHatReg,testMat[:,1],rowvar=0)[0,1]
# print(corrReg) 

# #构建模型树
# myTreeModel = createTree(trainMat,modelLeaf,modelErr,(1,20))
# # print(myTreeModel)
# yHatModel = createForeCast(myTreeModel,testMat[:,0],modelTreeEval)
# corrModel = corrcoef(yHatModel,testMat[:,1],rowvar=0)[0,1]
# print(corrModel)

# #构建标准回归树
# ws,X,Y = linearSolve(trainMat)
# # print(ws)
# yHat = mat(ones((shape(trainMat)[0],1)))
# for i in range(shape(testMat)[0]):
# 	yHat[i] = testMat[i,0] * ws[1,0] + ws[0,0]
# corr = corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
# print(corr)

# #绘制样本点
# myMat3 = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
# plt.plot(myMat3[:,0],myMat3[:,1],'ro') 
# plt.show()