from numpy import *
import matplotlib
import matplotlib.pyplot as plt

#读取数据集
def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():													#								0	  1      2	      			  3    4
		curLine = line.strip().split('\t') 										#该数据集有5列  前三列为初始数据  酒吧名 门牌号 城市 后两列是处理得到的 维度 经度
		dataMat.append([float(curLine[4]),float(curLine[3])])					#先读 经度  再度 纬度
	return dataMat 																#返回数据集只包括       经度纬度

#计算各个向量到簇心的距离  欧氏距离
def distEclud(vecA,vecB):
	return sqrt(sum(power(vecA - vecB,2)))										#距离计算方式  选择欧氏距离  

#计算球面距离   选择球面余弦定理来计算
#设所求点A ，纬度角β1 ，经度角α1 ；点B ，纬度角β2 ，经度角α2。则距离S=R·arc cos[cosβ1cosβ2cos（α1-α2）+sinβ1sinβ2]，其中R为球体半径
def distSLC(vecA,vecB):
	a = sin(vecA[0,1] * pi / 180) * sin(vecB[0,1] * pi / 180)
	b = cos(vecA[0,1] * pi / 180) * cos(vecB[0,1] * pi / 180) * cos((vecB[0,0] - vecA[0,0]) * pi / 180)
	return arccos(a + b) * 6371.0 


#初始化簇心  随机生成k个簇
def randCent(dataSet,k):
	n = shape(dataSet)[1]
	# print(dataSet)
	# print(len(dataSet))
	# print('-----------------------------------------------------------------------------------------------------')
	centroids = mat(zeros((k,n)))												#初始化n个簇心向量
	for j in range(n):
		minJ = min(dataSet[:,j])												#找到每一列特征最小值
		rangeJ = float(max(dataSet[:,j]) - minJ) 								#增量设为  每一列特征最大值 - 最小值
		centroids[:,j] = minJ + rangeJ * random.rand(k,1) 						#簇心的特征初始化为特征最小值加上一个  增量 * (0,1)之间随机值
																				#random.rand(m,n)生成m行n列的数组，并使用来自[0,1]的均匀分布的随机样本填充它。
																				#random.rand()没有给出参数，则返回单个python浮点数
	return centroids 															#返回簇心   

#k均值聚类算法
def kMeans(dataSet,k,distMeas = distEclud,createCent = randCent):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m,2)))  										#记录每一个样本的分配情况 第一列代表分配到的簇 第二列代表距离 初始化为0
	centroids = createCent(dataSet,k)											#初始化每一个簇的质心
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
			print("i,sseSplit, and notSplit",i,sseSplit,sseNotSplit,len(splitClustAss))

			if(sseSplit + sseNotSplit) < lowestSSE: 							#如果大簇内外样本点距离和（总SSE）小于当前最小  选择使得SSE增幅最小的簇划分
				bestCentToSplit = i 											#将该大簇设为最佳分割簇 			 #第i类作为本次划分类
				bestNewCents = centroidMat 										#保存此时新生成的两个小簇的簇质心   #第i类划分后得到的两个质心向量
				bestClustAss = splitClustAss.copy() 							#保存新形成两小簇的分配结果		 #复制第i类中数据点的聚类结果即误差值
				lowestSSE = sseSplit + sseNotSplit 								#计算新的总SSE 					 #将划分第i类后的总误差作为当前最小误差
	
		#下面两句执行顺序很重要不能颠倒   为什么呢  原因在于  bestCentToSplit有可能等于1  若顺序颠倒  可能会把要划分大簇的数据全部分给两小簇中的第2簇   
		#										 而 		  len(centList)  是新加的一列  不会出现数据划分出错的问题
		#分配矩阵[i,j]   											下面两行代码进行数组过滤时  [0,j] -> [(0,1,2~k),j] 
		#i代表分配的簇 j代表距离 [0,j] [1,j]是两个簇中的点										  [1,j] ->[len(centlist),j]
		bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) 	#给两小簇之一（编号1）内的样本点赋予新编号（最后一个簇）
		bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit	#给两小簇之二（编号0）内的样本点赋予原大簇编号
		print("the bestCentToSplit is : ",bestCentToSplit)
		print("the len of bestClustAss is :",len(bestClustAss))

		centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] 				#用两小簇中的一个替代簇质心列表中的原大簇
		centList.append(bestNewCents[1,:].tolist()[0]) 							#将两小簇中的另一个接在簇质心列表的最后
		clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss #用大簇的切割结果代替总ClusterAssment中该簇所含样本点的原结果
	return mat(centList), clusterAssment


#绘图
def clusterClubs(numClust = 5):
	datList = loadDataSet('places.txt') 										#修改了读取数据的函数
	datMat = mat(datList) 
	myCentroids,clustAssing = biKmeans(datMat,numClust,distMeas = distSLC)		#调用biKmeans()函数 生成簇的质心列表  样本点分配矩阵


	fig = plt.figure() 															#创建一个图
	rect = [0.1,0.1,0.8,0.8] 													#创建矩形
	scatterMarkers = ['s','o','^','8','d','p','v','h','>','<'] 					#标记类型
	axprops = dict(xticks = [],yticks = [])
	ax0 = fig.add_axes(rect,label = 'ax0',**axprops)
	imgP = plt.imread('Portland.png') 											#imread()函数基于一副图像来创建矩阵
	ax0.imshow(imgP) 															#绘制该图像
	ax1 = fig.add_axes(rect,label = 'ax1',frameon = False)						#叠加图层时frameon必须设置成False，不然会覆盖下面的图层
	for i in range(numClust): 													#遍历每一个簇
		ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A == i)[0],:]  		#找到每一个簇中的点
		markerStytle = scatterMarkers[i] 										#选择点的样式
		ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0],ptsInCurrCluster[:,1].flatten().A[0],marker = markerStytle,s = 90)#绘制散点图
	ax1.scatter(myCentroids[:,0].flatten().A[0],myCentroids[:,1].flatten().A[0],marker = '+',s = 300) #绘制簇的质心
	plt.show()

clusterClubs()
