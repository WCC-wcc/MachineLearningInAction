from numpy import *
import matplotlib.pyplot as plt

# https://blog.csdn.net/luoluonuoyasuolong/article/details/90711318   推倒公式参考


#数据导入
def loadDataSet(fileName,delim = '\t'): 
	fr = open(fileName)
	stringArr = [line.strip().split(delim) for line in fr.readlines()]
	datArr = [list(map(float,line)) for line in stringArr]
	return mat(datArr)

#PCA算法流程  输入参数为数据集  和  N个特征
def pca(dataMat,topNfeat = 9999999):
	# print(dataMat)
	meanVals = mean(dataMat,axis = 0)								#按列求取数据的平均值，也就是计算各个特征的均值
	# print(meanVals)
	meanRemoved = dataMat - meanVals 								#去除平均值
	# print(meanRemoved)
	covMat = cov(meanRemoved,rowvar = 0) 							#计算协方差矩阵，除数n-1是为了得到协方差的无偏估计
																    #cov(X,0) = cov(X) 除数是n-1(n为样本个数) 防止数据样本大小对数据离散程度的影响
																    #cov(X,1) 除数是n 	简单来说，得到全部样本除以n  得到部分样本除以n-1
	# print(covMat)
	eigVals,eigVects = linalg.eig(mat(covMat)) 						#计算协方差矩阵的特征值，特征向量
	# print(eigVals)
	# print(eigVects)
	eigValInd = argsort(eigVals) 									#将特征值从小到大排序,返回的是排序后的下标
																	#sort():对特征值矩阵排序(由小到大)
    																#argsort():对特征值矩阵进行由小到大排序，返回对应排序后的索引
	# print(eigValInd)
	eigValInd = eigValInd[:-(topNfeat + 1):-1] 						#从排序后的矩阵最后一个开始自下而上选取最大的N个特征值，返回其对应的索引
																	#（python里面，list[a:b:c]代表从下标a开始到b，步长为c。list[::-1]可以看作是列表逆序）
	# print(eigValInd)
	# print(eigVects)
	redEigVects = eigVects[:,eigValInd] 						    #将特征值最大的N个特征值对应索引的特征向量提取出来，组成压缩矩阵  这里取的是列值
	# print(redEigVects)
	lowDDataMat = meanRemoved * redEigVects 						#将去除均值后的数据矩阵*压缩矩阵，转换到新的空间，使维度降低为N
	# print(meanRemoved)
	# print(redEigVects)
	# print(lowDDataMat)
	# print(redEigVects.T)
	reconMat = (lowDDataMat * redEigVects.T) + meanVals 			#利用降维后的矩阵反构出原数据矩阵(用作测试，可跟未压缩的原矩阵比对)
	return lowDDataMat,reconMat 									#返回压缩后的矩阵和由该矩阵重构出来的原矩阵
 
dataMat = loadDataSet('testSet.txt')
lowDMat,reconMat = pca(dataMat,1)
# print(shape(lowDMat)) 

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker = '^',s = 90)
ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker = 'o',s = 50,c = 'red')
plt.show()
