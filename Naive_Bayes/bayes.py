#贝叶斯公式： P(ci | w) = p(w ci)/p(w) = p(w | ci)p(ci)/p(w)
#问题：已知文档中出现某些词条，求是侮辱性评论的概率
#转化：利用侮辱性文档在总文档中出现的概率，侮辱性文档中词条出现的概率，求分类情况
#tip1：假设各词条相互独立，那么p(w | ci) = p(w0 | ci)*p(w1 | ci)*p(w2 | ci)...p(wn | ci)
#tip2：判断是或者不是某一类别，在比较p0和p1时，由于除以相同的p(w)，因此可以省去这一步，直接比较分子上的大小，即p(w | c0)与 p(w | c1)即可

from numpy import *

def loadDataSet():													#创建实验样本。第一个变量是词条切分后的文档，第二个变量是类别标签的集合
	postingList = [
					['my','dog','has','flea','problems','help','please'],
					['maybe','not','take','him','to','dog','park','stupid'],
					['my','dalmation','is','so','cute','I','love','him'],
					['stop','posting','stupid','worthless','garbage'],
					['mr','licks','ate','my','steak','how','to','stop','him'],
					['quit','buying','worthless','dog','food','stupid']
				  ]
	classVec = [0,1,0,1,0,1]
	return postingList,classVec

def createVocabList(dataSet):										#创建词汇表，包含所有文档中出现的不重复的词
	vocabSet = set([])												#set()作为集合，不包含重复的数据
	for document in dataSet :
		vocabSet = vocabSet | set(document)							#集合的与运算用 | 实现
	return list(vocabSet)
																	#list()中元素可重复，可为空，有序，输出顺序就是插入顺序
																	#set()中元素不可重复，只能有一个null，无序


def setOfWords2Vec(vocabList,inputSet):								#输入参数为词汇表和文档，输出的是文档向量，向量元素为0或1，表示文档中单词是否在词汇表中出现
	returnVec = [0] * len(vocabList)								#词集模型，词集中每个单词只出现一次
	for word in inputSet :
		if word in vocabList :
			returnVec[vocabList.index(word)] = 1
		else :
			print("the word : %s is not in my Vocabulary!" % word)
	return returnVec

# listOPosts,listClasses = loadDataSet()
# myVocabList = createVocabList(listOPosts)
# print(myVocabList)
# print(setOfWords2Vec(myVocabList,listOPosts[0]))

#朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):								#输入参数为文档矩阵和文档类标签构成的向量
	numTrainDocs = len(trainMatrix)										#统计所有文档数	trainMatrix是一个向量，由setOfWords2Vec(vocabList,inputSet)生成
	numWords = len(trainMatrix[0])										#统计所有单词数	trainMatrix是一个向量，由setOfWords2Vec(vocabList,inputSet)生成
	pAbusive = sum(trainCategory)/float(numTrainDocs)					#二分类问题，0表示否，1表示是，这里计算是某一分类的概率
	# p0Num = zeros(numWords)												初始化分母变量  概率计算需要分子除以分母  不是某一类的分母
	# p1Num = zeros(numWords)												初始化分母变量	概率计算需要分子除以分母	是某一类的分母
	p0Num = ones(numWords)												#因为要计算多个概率值乘积，有一个为0，整体为0，为了降低影响，将所有词条出现次数初始化为1
	p1Num = ones(numWords)												#因为要计算多个概率值乘积，有一个为0，整体为0，为了降低影响，将所有词条出现次数初始化为1
	# p0Denom = 0.0															#初始化分子变量	概率计算需要分子除以分母	不是某一类的分子
	# p1Denom = 0.0															#初始化分子变量	概率计算需要分子除以分母	是某一类的分子
	p0Denom = 2.0														#因为要计算多个概率值乘积，有一个为0，整体为0，为了降低影响，同时将分母初始化为2
	p1Denom = 2.0														#因为要计算多个概率值乘积，有一个为0，整体为0，为了降低影响，同时将分母初始化为2
	for i in range(numTrainDocs):										#循环处理每一篇文档
		if trainCategory[i] == 1:										#若文档类别属于某一分类进行以下处理
			p1Num += trainMatrix[i]										#分子按照向量相加,词条属于某一分类，该分类下，词条数目加一,每个单词出现在类别1的次数
			p1Denom += sum(trainMatrix[i])   							#分母统计出所有单词数目，总词条数目,类别1出现的所有句子的单词总和
		else:															#若文档类别不属于某一分类也进行相似处理
			p0Num += trainMatrix[i]										#每个单词出现在类别0的次数
			p0Denom += sum(trainMatrix[i])								#类别0出现的所有句子的单词总和
	# p0Vect = p0Num/p0Denom												#计算每一个词条分类的概率，条件概率=词条数目/总词条数目，结果仍然是向量
	# p1Vect = p1Num/p1Denom												#计算每一个词条分类的概率，条件概率=词条数目/总词条数目，结果仍然是向量
	p0Vect = log(p0Num/p0Denom)											#多个小数相乘，数值太小时，乘积四舍五入后得到0，出现下溢出错误，通过求对数避免这个错误
	p1Vect = log(p1Num/p1Denom)											#log(每个单词出现在类别1的次数/类别1出现的所有句子的单词总和)
	return p0Vect,p1Vect,pAbusive										#返回两个类别的概率向量，和属于某一分类的概率

# trainMat = []
# for postinDoc in listOPosts:											#使用词向量填充trainMat列表
# 	trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
# p0V,p1V,pAb = trainNB0(trainMat,listClasses)
# print(p0V)
# print(p1V)
# print(pAb)

#朴素贝叶斯分类器测试函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)					#vec2Classify代表句子中有哪些单词,所以vec2Classify*p0Vec就是 每一个单词在类0中出现的概率的log，sum来讲这些概率全部乘起来
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)                       #log(a*b)=log(a)+log(b),即计算p(w | c1)
	if p0 < p1 :
		return 1
	else:
		return 0

#便利函数，封装所有操作
def testingNB():
	listOPosts,listClasses = loadDataSet()                                 #生成测试文档和文档类别
	myVocabList = createVocabList(listOPosts)							   #建立词条库
	trainMat = []														   #填充trainMat列表
	for postinDoc in listOPosts:											
		trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
	p0V,p1V,pAb = trainNB0(trainMat,listClasses)						   #计算得到各个词条分属类别的概率以及所有文档中属于某一分类的概率
	testEntry = ['love','my','dalmation']							       #测试数据
	thisDoc = setOfWords2Vec(myVocabList,testEntry)						   #查看测试数据在词条库中出现的单词 
	print(testEntry ,' classified as: ' ,classifyNB(thisDoc,p0V,p1V,pAb))  #返回结果


	testEntry = ['stupid','garbage']									   #新的测试数据
	thisDoc = setOfWords2Vec(myVocabList,testEntry)
	print(testEntry ,' classified as: ' ,classifyNB(thisDoc,p0V,p1V,pAb))

testingNB()
