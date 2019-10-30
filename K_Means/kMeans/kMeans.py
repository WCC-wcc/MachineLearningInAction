from numpy import *
import matplotlib.pyplot as plt
#读取数据集
def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = list(map(float,curLine))
		dataMat.append(fltLine)
	return dataMat

#计算各个向量到簇心的距离
def distEclud(vecA,vecB):
	return sqrt(sum(power(vecA - vecB,2)))										#距离计算方式  选择欧氏距离  

#初始化簇心  随机生成k个簇
def randCent(dataSet,k):
	n = shape(dataSet)[1]
	centroids = mat(zeros((k,n)))												#初始化n个簇心向量
	for j in range(n):
		minJ = min(dataSet[:,j]) 												#找到每一列特征最小值
		rangeJ = float(max(dataSet[:,j]) - minJ) 								#增量设为  每一列特征最大值 - 最小值
		centroids[:,j] = minJ + rangeJ * random.rand(k,1) 						#簇心的特征初始化为特征最小值加上一个  增量 * (0,1)之间随机值
															#random.rand(m,n)生成m行n列的数组，并使用来自[0,1]的均匀分布的随机样本填充它。
																				#random.rand()没有给出参数，则返回单个python浮点数
	return centroids 															#返回簇心   

# dataMat = mat(loadDataSet('testSet.txt'))
# print(min(dataMat[:,0]))
# print(max(dataMat[:,0]))
# print(max(dataMat[:,0]) - min(dataMat[:,0]))
# print(random.rand(5,2))
# print(dataMat)

# rendCent = randCent(dataMat,2)
# print(rendCent)

#k均值聚类算法
def kMeans(dataSet,k,distMeas = distEclud,createCent = randCent):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m,2)))  										#记录每一个样本的分配情况 第一列代表分配到的簇 第二列代表距离 初始化为0
	centroids = createCent(dataSet,k) 											#初始化每一个簇的质心
	clusterChanged = True 														#记录簇的质心是否变化
	while clusterChanged: 														#只要质心发生变化  就重新分配样本点
		clusterChanged = False 
		for i in range(m):														#循环遍历每一个样本
			minDist = inf 														#初始化距离
			minIndex = -1														#初始化分配的簇的位置
			for j in range(k):
				distJI = distMeas(centroids[j,:],dataSet[i,:]) 					#计算每个样本和每个簇质心之间的距离
				if distJI < minDist: 											#若样本和某个簇质心距离更近
					minDist = distJI 											#更新最近距离
					minIndex = j 												#更新分配的簇的位置
			if clusterAssment[i,0] != minIndex: 								#只要样本分配的位置发生变化 ，就将参数 clusterChanged = True  以便继续循环
				clusterChanged = True
			clusterAssment[i,:] = minIndex,minDist ** 2 						#修改样本点分配情况  重新记录分配簇的位置  距离的平方
		print(centroids)
		for cent in range(k): 													#循环遍历每一个簇  更新质心
			ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]     #数组过滤   得到给定簇的所有点  
			centroids[cent,:] = mean(ptsInClust,axis = 0) 						#计算所有点的均值  更新质心的值
	return centroids,clusterAssment 											#返回所有簇的质心  样本点的分配结果


#绘图
def showCluster(dataSet, k, centroids, clusterAssment):
	numSamples, dim = shape(dataSet)
	if dim != 2:
		print("Sorry! I can not draw because the dimension of your data is not 2!")
		return 1
 
	mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']         #https://blog.csdn.net/roguesir/article/details/77932526 绘图参数设置参考
	if k > len(mark):
		print("Sorry! Your k is too large! please contact Zouxy")
		return 1
 
	# draw all samples
	for i in range(numSamples): 												#绘制出所有样本点
		markIndex = int(clusterAssment[i, 0])
		plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])					#plt.plot(x,y,format_string)	(x,y)坐标  format_string控制曲线的格式字串
 
	mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb'] 		#https://blog.csdn.net/roguesir/article/details/77932526 绘图参数设置参考
	# draw the centroids
	for i in range(k): 															#绘制所有的簇的质心
		plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)
 
	plt.show()

dataMat = mat(loadDataSet('testSet.txt'))
centroids,clusterAssment = kMeans(dataMat,4)
showCluster(dataMat,4,centroids,clusterAssment)