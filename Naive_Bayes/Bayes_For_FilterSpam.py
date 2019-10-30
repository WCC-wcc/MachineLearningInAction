#贝叶斯公式： P(ci | w) = p(w ci)/p(w) = p(w | ci)p(ci)/p(w)
#问题：已知文档中出现某些词条，求是侮辱性评论的概率
#转化：利用侮辱性文档在总文档中出现的概率，侮辱性文档中词条出现的概率，求分类情况
#tip1：假设各词条相互独立，那么p(w | ci) = p(w0 | ci)*p(w1 | ci)*p(w2 | ci)...p(wn | ci)
#tip2：判断是或者不是某一类别，在比较p0和p1时，由于除以相同的p(w)，因此可以省去这一步，直接比较分子上的大小，即p(w | c0)与 p(w | c1)即可

from numpy import *
import re


def createVocabList(dataSet):										#创建词汇表，包含所有文档中出现的不重复的词
	vocabSet = set([])												#set()作为集合，不包含重复的数据
	for document in dataSet :
		vocabSet = vocabSet | set(document)							#集合的与运算用 | 实现
	return list(vocabSet)
																	#list()中元素可重复，可为空，有序，输出顺序就是插入顺序
																	#set()中元素不可重复，只能有一个null，无序


def bagOfWords2VecMN(vocabList,inputSet):							#词袋模型，每个单词可以出现多次
	returnVec = [0] * len(vocabList)
	for word in inputSet :
		if word in vocabList :
			returnVec[vocabList.index(word)] += 1
		else :
			print("the word : %s is not in my Vocabulary!" % word)
	return returnVec

#朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):								#输入参数为文档矩阵和文档类标签构成的向量
	numTrainDocs = len(trainMatrix)										#统计所有文档数	trainMatrix是一个向量，由setOfWords2Vec(vocabList,inputSet)生成
	numWords = len(trainMatrix[0])										#统计所有单词数	trainMatrix是一个向量，由setOfWords2Vec(vocabList,inputSet)生成
	pAbusive = sum(trainCategory)/float(numTrainDocs)					#二分类问题，0表示否，1表示是，这里计算是某一分类的概率
	# p0Num = zeros(numWords)												初始化分母变量  概率计算需要分子除以分母  不是某一类的分母
	# p1Num = zeros(numWords)												初始化分母变量	概率计算需要分子除以分母	是某一类的分母
	p0Num = ones(numWords)												#因为要计算多个概率值乘积，有一个为0，整体为0，为了降低影响，将所有词条出现次数初始化为1
	p1Num = ones(numWords)												#因为要计算多个概率值乘积，有一个为0，整体为0，为了降低影响，将所有词条出现次数初始化为1
	p0Denom = 2.0														#因为要计算多个概率值乘积，有一个为0，整体为0，为了降低影响，同时将分母初始化为2
	p1Denom = 2.0														#因为要计算多个概率值乘积，有一个为0，整体为0，为了降低影响，同时将分母初始化为2
	for i in range(numTrainDocs):										#循环处理每一篇文档
		if trainCategory[i] == 1:										#若文档类别属于某一分类进行以下处理
			p1Num += trainMatrix[i]										#分子按照向量相加,词条属于某一分类，该分类下，词条数目加一,每个单词出现在类别1的次数
			p1Denom += sum(trainMatrix[i])   							#分母统计出所有单词数目，总词条数目,类别1出现的所有句子的单词总和
		else:															#若文档类别不属于某一分类也进行相似处理
			p0Num += trainMatrix[i]										#每个单词出现在类别0的次数
			p0Denom += sum(trainMatrix[i])								#类别0出现的所有句子的单词总和
	p0Vect = log(p0Num/p0Denom)											#多个小数相乘，数值太小时，乘积四舍五入后得到0，出现下溢出错误，通过求对数避免这个错误
	p1Vect = log(p1Num/p1Denom)											#log(每个单词出现在类别1的次数/类别1出现的所有句子的单词总和)
	return p0Vect,p1Vect,pAbusive										#返回两个类别的概率向量，和属于某一分类的概率


#朴素贝叶斯分类器测试函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)					#vec2Classify代表句子中有哪些单词,所以vec2Classify*p0Vec就是 每一个单词在类0中出现的概率的log，sum来讲这些概率全部乘起来
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)                       #log(a*b)=log(a)+log(b),即计算p(w | c1)
	if p0 < p1 :
		return 1
	else:
		return 0

#文本切分函数，从给定文本中构建词列表
def textParse(bigString):
	listOfTokens = re.split(r'\W',bigString)						#r'\W'模式表示原生字符串，不要转义‘\’字符  否则要匹配‘\w’需要写成‘\\w’
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]	#删除掉字符长度不到2的字符

#垃圾邮件测试函数
def spamTest():
	docList = []														#docList存放每一条记录，每一个邮件作为一条记录
	classList = []														#classList存放类别
	fullText = []														#fullText存放所有记录，一次录入每一个邮件的文本信息
	
																		#下面操作导入spam文件和ham的文本文件，将他们解析为词列表
	for i in range(1,26):												#对1-25个邮件，依次按行读取  open.read()按行读取文本文件
		wordList = textParse(open('email/spam/%d.txt' % i,"rb").read().decode('GBK','ignore') )  #decode('GBK','ignore'))参数是为了避免类似“�”等非法字符
		# print(wordList)
		docList.append(wordList)										#apend()函数是将后者作为一个对象加入到前者  [1,2,3].append[4,5] = [1,2,3,[4,5]]
		fullText.extend(wordList)										#extend()函数是将两个序列合并  [1,2,3].extend[4,5] = [1,2,3,4,5]
		classList.append(1)												#记录邮件类别									

		wordList = textParse(open('email/ham/%d.txt' % i,"rb").read().decode('GBK','ignore') )   #对垃圾邮件也进行相应处理
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)

	vocabList = createVocabList(docList)  								#创建词汇表
	trainingSet = list(range(50))										#设置训练数据集大小为50 range(50) = [0,1,2,3...49]
	testSet= []															#设置测试数据集

	for i in range(10):
		randIndex = int(random.uniform(0,len(trainingSet)))				#random模块用于生成随机数 random.uniform(a,b)用于生成制定范围内的随机浮点数
		testSet.append(trainingSet[randIndex]) 							#testSet[]存储随机生成的10个下标
		del  trainingSet[randIndex]										#从range(50) = [0,1,2,3...49]删除测试的数据下标,留下训练数据集的下标
	trainMat = []														#训练数据集整合在一起，
	trainClasses = []
	for docIndex in trainingSet:										#依次取出剩下的50条邮件数据的下标
		trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))	#利用词汇表，每个邮件的词条，生成矩阵trainMat，词条向量
		trainClasses.append(classList[docIndex])						#记录训练数据的类别
	p0V,p1V,pSpam = trainNB0(trainMat,trainClasses)						#返回两个类别的概率向量，和属于某一分类的概率
	errorcount = 0														#记录错误数
	for docIndex in testSet:
		wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
		if classifyNB(wordVector,p0V,p1V,pSpam) != classList[docIndex]:
			errorcount +=1
			print('classification error : ',docList[docIndex])
	print('the error rate is ',float(errorcount)/len(testSet))

spamTest()
