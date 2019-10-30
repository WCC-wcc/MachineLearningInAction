from numpy import *
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import operator

#KNN算法
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
	sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
															#对标签次数排序 将classcount分为元组列表   按照第二个值排序   按照从大到小
	return sortedClassCount[0][0]

#准备数据  从文本文件中解析数据
def file2matrix(filename) :   								#定义函数，通过文件名，获取文件中的文本记录，返回矩阵列表以及标签列表
	fr = open(filename)       								#打开文件
	arrayOLines = fr.readlines()							#按行读取文件
	numberOfLines = len(arrayOLines) 						#获取文件行数
	returnMat = zeros((numberOfLines,3))					#创造一个填充矩阵   numberOfLines行3列
	classLabelVector = []									#创建标签矩阵
	index = 0	
	for line in arrayOLines:								#按行读取文件文本数据
		line = line.strip()									#去掉每一行开头结尾处的空白
		listFromLine = line.split('\t')						#字符串划分，生成listFromLine向量，有四个数据，前三个为特征，最后一个是标签
		returnMat[index,:] = listFromLine[0:3]				#赋值，结果是把listFromLine中0,1,2元素赋值给returnMat第index行中的三个列项
		if listFromLine[-1] == 'didntLike' :
			classLabelVector.append(1)
		elif listFromLine[-1] == 'smallDoses' : 
			classLabelVector.append(2)
		else : 
			classLabelVector.append(3)
		index += 1
	return returnMat,classLabelVector						#返回标签和特征矩阵


#归一化数据                            newValue = (oldValue - min) / (max - min)  （当前值 - 最小值 ）/ 取值范围
def autoNorm(dataSet):
	minVals = dataSet.min(0)                                #dataSet.min()函数是numpy函数的一个方法，返回每一列最小值，即每一个特征最小值，1*n的矩阵
	maxVals = dataSet.max(0)								#dataSet.max()函数是numpy函数的一个方法，返回每一列最小值，即每一个特征最大值，1*n的矩阵
	ranges = maxVals - minVals								#确定取值范围
	normDataSet = zeros(shape(dataSet))						#初始化  归一化特征数据集，zeros()	函数表示用0填充，shape(dataSet)函数获取矩阵或数组的维度
	m = dataSet.shape[0]									#获取样本个数  矩阵第一维度，类似于数组  a[][]  a.length返回几行 a[0].lenght返回第一行有几个元素
	normDataSet = dataSet - tile(minVals,(m,1))				#tile(minVals,(m,1))函数将每一个特征的最小值minVals，扩成m行*1列，以进行(oldValue - min)操作
	normDataSet = normDataSet / tile(ranges,(m,1))			
	return normDataSet,ranges,minVals						#返回归一化矩阵，各个特征最大值最小值之差，各个特征最小值

#测试分类器效果
def datingClassTest():
	hoRatio = 0.1											#选择总数的hoRatio作为测试样本，(1 - hoRatio)作为训练样本
	datingDataMat,datingLabels = file2matrix("DateSet.txt") #调用file2matrix函数，从文本中解析数据，返回样本集和标签集
	normMat,ranges,minVals = autoNorm(datingDataMat)		#调用autoNorm函数，对样本集中数据进行归一化处理，返回归一化矩阵，各个特征最大值最小值之差，各个特征最小值
	m = normMat.shape[0]									#获取样本总数，normMat.shape[0]返回矩阵第一维度的个数
	numTestVecs = int(m * hoRatio)							#赋值，计算得到，测试样本的大小
	errorCount = 0.0										#定义错误数
	for i in range(numTestVecs) :							#测试样本中的样本数据
		classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
															#输入数据为归一化数据中前(numTestVecs)个记录，训练样本为(numTestVecs-m)个记录
															#标签集为(numTestVecs-m)个记录，K为3，即最接近的前三个记录
															#返回测试样本中各个记录的种类划分结果
		print ("the classifier came back with : %d,the real answer is : %d" % (classifierResult,datingLabels[i]))
		if (classifierResult != datingLabels[i]) : errorCount += 1.0   #测试结果和标签不同的，即为出错
	print ("the total error rate is %f" % (errorCount/float(numTestVecs)) )  #出错率 = 出错个数 / 总共的测试样本数


#测试算法：分类器对约会网站的测试代码
datingClassTest()



#使用算法：构建完整可用系统
def classifyPerson():
	resultList = ['not at all','in small doses','in large doses']

	percentTats = float(input("percentage of time spent playing video games:  "))
	ffMiles = float(input("frequent flier miles earned per year:  "))
	iceCream = float(input("liters of ice cream consumed per year:  "))

	datingDataMat,datingLabels = file2matrix("DateSet.txt")	   
	normMat,ranges,minVals = autoNorm(datingDataMat)	
	inArr = array([ffMiles,percentTats,iceCream])
	classifierResult = classify0((inArr - minVals) / ranges,normMat,datingLabels,3)
	print("You will probably like this person : %s" % resultList[classifierResult - 1])

#使用算法：构建完整可用系统
classifyPerson()

#分析数据  用matplotlib绘制散点图   
datingDataMat,datingLabels = file2matrix("DateSet.txt")	

# 将三类数据分别取出来
# x轴代表飞行的里程数
# y轴代表玩视频游戏的百分比
type1_x = []
type1_y = []
type2_x = []
type2_y = []
type3_x = []
type3_y = []

for i in range(len(datingLabels)):
    if datingLabels[i] == 1:  # 不喜欢
        type1_x.append(datingDataMat[i][0])
        type1_y.append(datingDataMat[i][1])
 
    if datingLabels[i] == 2:  # 魅力一般
        type2_x.append(datingDataMat[i][0])
        type2_y.append(datingDataMat[i][1])
 
    if datingLabels[i] == 3:  # 极具魅力
        type3_x.append(datingDataMat[i][0])
        type3_y.append(datingDataMat[i][1])
 
plt.figure(figsize=(8, 5), dpi=80)
axes = plt.subplot(111) 										#画布分割函数  111 指 画布分为 1 行 1 列  图像画在第一块
type1 = axes.scatter(type1_x, type1_y, s=20, c='red')
type2 = axes.scatter(type2_x, type2_y, s=40, c='green')
type3 = axes.scatter(type3_x, type3_y, s=50, c='blue')

# plt.scatter(datingDataMat[:,0],datingDataMat[:,1],			#scatter（x,y,s=1,c="g",marker="s",linewidths=0）x,y是x轴，y轴数据
#  	15.0 * array(datingLabels),15.0 * array(datingLabels))	#s:散列点的大小,c:散列点的颜色，marker：形状，linewidths：边框宽度	

font = FontProperties(fname="c:/windows/fonts/simsun.ttc", size=14)
plt.xlabel('每年获取的飞行里程数', fontproperties=font)
plt.ylabel('玩视频游戏所消耗的时间百分比', fontproperties=font)
axes.legend((type1, type2, type3), ('不喜欢', '魅力一般', '极具魅力'), loc=2, prop=font)

plt.show()


