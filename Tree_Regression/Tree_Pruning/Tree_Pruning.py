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

# testMat = mat(eye(4)) 																	#单位矩阵函数eye()
# print(testMat)
# mat0,mat1 = binSplitDataSet(testMat,1,0.5)
# print('\n\n\n')
# print(mat0)
# print('\n\n\n')
# print(mat1)


########################################################################预剪枝##################################################################################
myDat2 = loadDataSet('ex2.txt')
myMat2 = mat(myDat2)																		
rTree2 = createTree(myMat2,ops=(0,1))                                                   #函数对于ops(a,b)停止条件敏感，这也是预剪枝的关键
print(rTree2)																			#通过修改停止条件来得到合理结果不好  太随机性  太慢 
plt.plot(myMat2[:,0],myMat2[:,1],'ro') 
plt.show()

print('\n\n\n\n')
########################################################################预剪枝##################################################################################






#使用后剪枝方法需要将数据集分成测试集和训练集。首先指定参数，使得构建出的树足够大、足够复杂，便于剪枝。
#接下来从上而下找到叶结点，用测试集来判断这些叶结点合并是否能降低测试集误差。如果是的话就合并。
#后剪枝可能不如预剪枝有效。一般地，为了寻求最佳模型可以同时使用两种剪枝技术

########################################################################后剪枝##################################################################################

#判断对象是否是一棵树   判断当前节点是否属于叶节点
def isTree(obj):
	return (type(obj).__name__ == 'dict')

#递归函数 从上往下遍历树直到叶节点为止 找到两个叶节点就计算均值
def getMean(tree):
	if isTree(tree['right']) : tree['right'] = getMean(tree['right'])			#递归的计算出每一个节点的值
	if isTree(tree['left']) : tree['left'] = getMean(tree['left'])
	return (tree['left'] + tree['right']) / 2.0



#基于已存在的树切分测试数据 ：
#    如果存在任意一子集是一棵树，就在该子集递归剪枝过程
#    计算当前两个叶节点合并后的误差
#    计算不合并的误差
#    如果合并后降低误差，就合并


#后剪枝	输入参数    待剪枝的树      剪枝所需的测试数据
def prune(tree,testData):
	if shape(testData)[0] == 0: return getMean(tree)                               #没有测试数据时，对树进行塌陷处理


	if isTree(tree['right']) or isTree(tree['left']):							  #假设发生过拟合，采用测试数据对树进行剪枝  左右子树非空
		lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])		  #按照CART切分测试数据


	if isTree(tree['left']) : tree['left'] = prune(tree['left'],lSet)			  #递归的对左子树进行剪枝   更新测试数据集  为  母数据集  在CART下划分的左子树
	if isTree(tree['right']) : tree['right'] = prune(tree['right'],rSet)	 	  #递归的对右子树进行剪枝	  更新测试数据集  为  母数据集  在CART下划分的右子树


	# 上面的一系列操作本质上就是将测试数据集按照训练完成的树拆分好，对应的值放到对应的节点



	if not isTree(tree['left']) and not isTree(tree['right']): 					  #两边都是叶子
		lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])		  #切分数据
																												#偏差平方和  （实际y值 - 预测值）^ 2
		errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) + sum(power(rSet[:,-1] - tree['right'],2))       #对两边叶子合并前计算 误差  误差仍然是左右子树平方和
		
		##合并叶子节点的误差  合并后  数据集不再进行划分  也就没有左右子树 统一进行处理  本来左右子树和左右子树的均值比较  现在是 全部数据集 和 当前节点的均值比较
		treeMean = (tree['left'] + tree['right']) / 2.0 														#合并后的叶子均值 
		errorMerge = sum(power(testData[:,-1] - treeMean,2))													#合并后的误差
		
		if errorMerge < errorNoMerge : 																			#合并后的误差小于合并前的误差
			print('merging')																					#说明合并后的树误差更小
			return treeMean 																					#返回两个叶子的均值作为合并后的叶子节点
																												#如果可以合并，原来的dict就变为了 数值
		else :
			return tree
	else:
		return tree


def pruneTest(): 
  MyDat = loadDataSet('ex2.txt')  
  MyMat = mat(MyDat) 
  MyTree = createTree(MyMat,ops=(0,1))  												#为了得到最大的树  误差设置为0 个数设置为1 即不进行预剪枝 
  MyDatTest = loadDataSet('ex2test.txt') 
  MyMatTest = mat(MyDatTest) 
  print(prune(MyTree,MyMatTest)) 


pruneTest()



########################################################################后剪枝##################################################################################