from numpy import *
import matplotlib.pyplot as plt

#数据导入
def loadDataSet(fileName,delim = '\t'): 
	fr = open(fileName)
	stringArr = [line.strip().split(delim) for line in fr.readlines()]
	datArr = [list(map(float,line)) for line in stringArr]
	return mat(datArr)

#数据预处理   把NaN值转换成均值
def replaceNaNWithMean():
	datMat = loadDataSet('secom.data',' ')
	numFeat = shape(datMat)[1]
	for i in range(numFeat):
		# print(~isnan(datMat[:,i].A))											#判断列值是否是NaN，返回true，false
		# print(nonzero(~isnan(datMat[:,i].A)))									#返回array数组(array([0,1,..., 1564, 1565, 1566], dtype=int64), array([0, 0, 0, ..., 0, 0, 0], dtype=int64))
		# print(nonzero(~isnan(datMat[:,i].A))[0])								#取数组第一列(指哪一行)，结合每一列的i值，得到具体位置  数组过滤
		meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) 			#计算非NaN值的平均值
		datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal 					#将所有的NaN设置为平均值
	return datMat

dataMat = replaceNaNWithMean()
meanVals = mean(dataMat,axis = 0)
meanRemoved = dataMat - meanVals
covMat = cov(meanRemoved,rowvar = 0)
eigVals,eigVects = linalg.eig(mat(covMat))
# print(eigVals[0])
# print(eigVects)
eigValInd = argsort(eigVals)      
# print(eigValInd)      
eigValInd = eigValInd[::-1]  # list反转，从大到小
sortedEigVals = eigVals[eigValInd]
# print(len(sortedEigVals))
total = sum(sortedEigVals)
# print(total)
varPercentage = sortedEigVals/total*100  # 方差的百分比
# print(len(varPercentage))
# print(sortedEigVals[0]/total)
# print(sum(varPercentage))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1, 21), varPercentage[:20], marker='^')
plt.xlabel('Principal Component Number')  # 主成分数目
plt.ylabel('Percentage of Variance')  # 方差的百分比
plt.show()  # 观察图可以发现，只保留6个特征即可

# 因此，调用pca(dataMat,6)可以将590维的特征降成6维
