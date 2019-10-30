from numpy import *
from matplotlib import *
import operator

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group,labels

def classify0(inX,dataSet,labels,k) :
	dataSetSize = dataSet.shape[0]  #返回数据集矩阵行数
	diffMat = tile(inX,(dataSetSize,1)) - dataSet  #向量相减，tile函数 将目标矩阵扩成和数据集矩阵行列数相同的矩阵(也就是n行1列) 计算欧氏距离  相减
	sqDiffMat = diffMat ** 2   #计算欧氏距离  平方
	sqDistances = sqDiffMat.sum(axis=1)   #计算欧氏距离  平方和  axis=1  按行相加   axis=0  按列相加
	distances = sqDistances ** 0.5
	sortedDistIndices = distances.argsort()#  按距离从小到大排序，返回值是下标 如 ： [4,1,2] -->[1,2,0]
	classCount = {}
	for i in range(k) :
		votelabel = labels[sortedDistIndices[i]]   #获取前K个对应标签
		classCount[votelabel] = classCount.get(votelabel,0) + 1   #统计标签出现次数
	sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)#对标签次数排序 将classcount分为元组列表   按照第二个值排序             按照从大到小

	return sortedClassCount[0][0]

group,labels = createDataSet()
print(classify0([1.2,1.1],group,labels,3))
print(classify0([0,0],group,labels,3))

