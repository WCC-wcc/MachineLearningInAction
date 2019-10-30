from numpy import *
from os import listdir
import operator

#KNN算法
def classify0(inX,dataSet,labels,k) :
	dataSetSize = dataSet.shape[0]  #返回数据集矩阵行数
	diffMat = tile(inX,(dataSetSize,1)) - dataSet  #向量相减，tile函数 将目标矩阵扩成和数据集矩阵行列数相同的矩阵(也就是n行1列) 计算欧氏距离  相减
	sqDiffMat = diffMat ** 2   #计算欧氏距离  平方
	sqDistances = sqDiffMat.sum(axis=1)   #计算欧氏距离  平方和  axis=1  按行相加   axis=0  按列相加
	distances = sqDistances ** 0.5
	sortedDistIndices = distances.argsort()#  按距离从小到大排序，返回值是索引值 如 ： [4,1,2] -->[1,2,0]  
	classCount = {}
	for i in range(k) :
		votelabel = labels[sortedDistIndices[i]]   #获取前K个对应标签
		classCount[votelabel] = classCount.get(votelabel,0) + 1   #统计标签出现次数
	sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
															#对标签次数排序 将classcount分为元组列表   按照第二个值排序   按照从大到小
	return sortedClassCount[0][0]

#准备数据  将图像转换成测试向量
def img2vector(filename) :   								#定义函数，通过文件名，获取文件中的文本记录，返回特征矩阵
	returnVect = zeros((1,1024))							#创造一个0填充矩阵  1*1024列  
	fr = open(filename)       								#打开文件
	for i in range(32) :									#每一个字迹图像都转换成32 * 32的矩阵，
		lineStr = fr.readline()								#按行读取文件
		for j in range(32) :
			returnVect[0,32 * i + j] = int(lineStr[j])		#把每一行的32个特征都赋值给returnVect[0,1024]
	return returnVect										#返回特征矩阵

#手写数字识别系统测试代码
def handwritingClassTest():							
															#先处理训练集
	hwLabels = []											#创建标签矩阵
	trainingFileList = listdir('trainingDigits')			#获取训练集的目录
	m = len(trainingFileList)								#得到训练集记录数m
	trainingMat = zeros((m,1024))							#创建m*1024的特征矩阵，存入m个记录的1024个特征
	for i in range(m) :
		fileNameStr = trainingFileList[i]					#获取每一条记录的文件名 格式为0_0.txt
		fileStr = fileNameStr.split('.')[0]					#截取文件名，删去.txt 格式为0_0
		classNumStr = int(fileStr.split('_')[0])			#截取该记录的对应数字，即为标签
		hwLabels.append(classNumStr)						#按顺序添加每一个文件对应的标签到标签矩阵
		trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)  #按顺序读取每一个文件中的1024个特征，添加到特征矩阵中


															#处理测试集
	testFileList = listdir('testDigits')					#获取测试集的目录
	errorCount = 0											#设定错误记录
	mTest = len(testFileList)								#得到测试集记录数m
	for i in range (mTest) :
		fileNameStr = testFileList[i]						#获取每一条记录的文件名 格式为0_0.txt
		fileStr = fileNameStr.split('.')[0]					#截取文件名，删去.txt  格式为0_0
		classNumStr = int(fileStr.split('_')[0])			#截取该记录的对应数字，即为标签
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr) #调用img2vector()函数，获取该记录特征矩阵
		classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3) #调用classify0()函数，实现KNN算法，得到标签
		print('the classifier came back with : %d , the real answer is %d' % (classifierResult,classNumStr)) #输出计算得到的标签值  和真实的标签
		if (classifierResult != classNumStr) : errorCount += 1.0											 #统计错误数
	print('\nthe total number of errors is : %d' % errorCount)							#输出错误数
	print('\nthe total error rate is : %f ' %(errorCount/float(mTest)))					#输出错误率


handwritingClassTest()