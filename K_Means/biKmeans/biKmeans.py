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
		# print(centroids)
		for cent in range(k): 													#循环遍历每一个簇  更新质心
			ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]     #数组过滤   得到给定簇的所有点  
			centroids[cent,:] = mean(ptsInClust,axis = 0) 						#计算所有点的均值  更新质心的值
	return centroids,clusterAssment 											#返回所有簇的质心  样本点的分配结果



#二分k均值聚类算法
def biKmeans(dataSet,k,distMeas = distEclud):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m,2)))  										#记录每一个样本的分配情况 第一列代表分配到的簇 第二列代表距离 初始化为0
	centroid0 = mean(dataSet,axis = 0).tolist()[0]								#初始化簇的质心  整个数据集看成一个簇  慢慢增加簇直到k
	# print(mean(dataSet,axis = 0).tolist()) 									[[-0.15772275000000002, 1.2253301166666664]]
	# print(centroid0)															[-0.15772275000000002, 1.2253301166666664]
	centList = [centroid0] 														#将该质心放入簇质心列表
	for j in range(m):															#对每一个样本点进行访问
		clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j,:]) ** 2		#计算所有点与初始簇质心的距离
	while(len(centList) < k): 													#当簇数目达到设定的k值，退出循环
		lowestSSE = inf 														#将最小损失值先设为∞
		for i in range(len(centList)): 											#依次访问每一个簇质心
			ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]#将当前簇质心内的所有样本点存入ptsInCurrCluster 通过数组过滤筛选出属于第i类的数据集合
			centroidMat,splitClustAss = kMeans(ptsInCurrCluster,2,distMeas)		#对该大簇进行普通K-Means，分为两小簇  返回簇的质心 和 样本点分配结果

			sseSplit = sum(splitClustAss[:,1]) 									#计算该大簇内样本点的距离之和
			sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1]) 	#计算该大簇外样本点的距离之和
			print("i,sseSplit, and notSplit",i,sseSplit,sseNotSplit)

			if(sseSplit + sseNotSplit) < lowestSSE: 							#如果大簇内外样本点距离和（总SSE）小于当前最小  选择使得SSE增幅最小的簇划分
				bestCentToSplit = i 											#将该大簇设为最佳分割簇 			 #第i类作为本次划分类
				bestNewCents = centroidMat 										#保存此时新生成的两个小簇的簇质心   #第i类划分后得到的两个质心向量
				bestClustAss = splitClustAss.copy() 							#保存新形成两小簇的分配结果		 #复制第i类中数据点的聚类结果即误差值
				lowestSSE = sseSplit + sseNotSplit 								#计算新的总SSE 					 #将划分第i类后的总误差作为当前最小误差
	
		
		bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) 	#给两小簇之一（编号1）内的样本点赋予新编号（最后一个簇）
		bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit	#给两小簇之二（编号0）内的样本点赋予原大簇编号

		print("the bestCentToSplit is : ",bestCentToSplit)
		print("the len of bestClustAss is :",len(bestClustAss))
		centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] 				#用两小簇中的一个替代簇质心列表中的原大簇
		centList.append(bestNewCents[1,:].tolist()[0]) 							#将两小簇中的另一个接在簇质心列表的最后
		clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss #用大簇的切割结果代替总ClusterAssment中该簇所含样本点的原结果
	return mat(centList), clusterAssment


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
		# print(clusterAssment[i,0])
		plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])					#plt.plot(x,y,format_string)	(x,y)坐标  format_string控制曲线的格式字串
 
	mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb'] 		#https://blog.csdn.net/roguesir/article/details/77932526 绘图参数设置参考
	# draw the centroids
	for i in range(k): 															#绘制所有的簇的质心
		plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)
 
	plt.show()

dataMat = mat(loadDataSet('testSet2.txt'))
centList,newClusterAssment = biKmeans(dataMat,3)
print(centList)
showCluster(dataMat,3,centList,newClusterAssment)