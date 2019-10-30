from numpy import *
import matplotlib.pyplot as plt
from os import listdir

#读取图像文件函数
def loadImages(dirName) :
	hwLabels = []											#定义标签矩阵
	trainingFileList = listdir(dirName)						#打开文件dirName  返回文件中的项目列表
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))							#定义数据向量矩阵
	for i in range(m) :										#依次处理文件dirName中的每一个文本数据
		fileNameStr = trainingFileList[i]					#获取每个样本文件的名称
		fileStr = fileNameStr.split('.')[0]					#划分名称，去掉.txt
		classNumStr = int(fileStr.split('_')[0])			#得到类别标签，去掉1_xx后面的_xx       _xx是个数统计
		if classNumStr == 9:
			hwLabels.append(-1)								#将9类别定义为-1
		else :
			hwLabels.append(1)								#将1类别定义为+1
		trainingMat[i,:] = img2vector('%s/%s' % (dirName,fileNameStr))  #利用img2vector()函数 转换向量 dirName/fileNameStr 等价于 trainingDigits/1_01.txt
	return trainingMat,hwLabels


#准备数据  将图像转换成测试向量
def img2vector(filename) :   								#定义函数，通过文件名，获取文件中的文本记录，返回特征矩阵
	returnVect = zeros((1,1024))							#创造一个0填充矩阵  1*1024列  
	fr = open(filename)       								#打开文件
	for i in range(32) :									#每一个字迹图像都转换成32 * 32的矩阵，
		lineStr = fr.readline()								#按行读取文件
		for j in range(32) :
			returnVect[0,32 * i + j] = int(lineStr[j])		#把每一行的32个特征都赋值给returnVect[0,1024]
	return returnVect										#返回特征矩阵

#核转换函数
def kernelTrans(X,A,kTup):													#输入参数为2个数值型变量和一个元组信息 元祖Ktup给出核函数信息，第一个字符变量是核函数类型，第二个变量是给定的sigma值
																			#X 表示所有数据集  A 表示数据集中的一行
	m,n = shape(X)
	K = mat(zeros((m,1)))													#初始化K值，列向量，记录的是 所有数据集 依次和 数据集中的每一行 的高斯函数值
	if kTup[0] == 'lin' :													#线性情况下  取得是向量内积
		K = X * A.T
	elif kTup[0] == 'rbf' :													#高斯径向基核函数的计算方式
		for j in range(m) :
			deltaRow = X[j,:] - A 											#首先计算（x - y）
			K[j] = deltaRow * deltaRow.T 									#计算(x - y) ^ 2   由于 （x-y)是向量， 令 deltaRow * deltaRow.T即可
		K = exp(K / (-kTup[1] ** 2))									#高斯径向基核函数表达式为 ： K(xy) = exp( - ( x - y ) ^ 2 / 2 * sigma ^2) 2作为常数可以忽略
	else :
		raise NameError('There is a problem about kernel.')
	return K

#辅助函数  在某个范围区间内随机选择一个整数	
def selectJrand(i,m):														#输入参数i是第一个alpha下标   m是所有alpha数目
	j = i 																	#这个函数是为了随机选择另一个alpha值
	while(j == i):
		j = int(random.uniform(0,m))
	return j

#辅助函数  调整小于L，大于H的alpha值											#调整alpha值，使得其在取值区间内
def clipAlpha(aj,L,H):
	if aj < L:
		aj = L
	if aj > H:
		aj = H
	return aj

#建造一个数据结构保存重要的值 __init__  前后是两个_			通过一个对象完成
class optStruct:															
	def __init__(self,dataMatIn,classLabels,C,toler,kTup) :
		self.X = dataMatIn													#记录数据样本
		self.labelMat = classLabels											#记录标签
		self.C = C 															#常数C
		self.tol = toler													#容错率
		self.m = shape(dataMatIn)[0]										#记录数据个数
		self.alphas = mat(zeros((self.m,1)))								#数据个数对应的alphas值
		self.b = 0															#纪录常数b
		self.eCache = mat(zeros((self.m,2)))								#增加一个 m * 2 的矩阵成员变量eCache  第一列是eCache是否有效的标志位 第二列是实际的E值
		self.K = mat(zeros((self.m,self.m)))								#初始化矩阵K 是一个100 * 100 的矩阵  每一行得
		for i in range(self.m) :											#依次计算总数据向量 和每一个数据向量 的高斯函数值
			self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup)			    #返回值是一列数值，赋值给对应的self.K[:,i]

#计算误差E值并返回  注意  E值  是动态变化的  E = wx + b - y 		w = alphai * yi * xi (i = 1,2...100 ) 
def calcEk(oS,k) :															
	fXk = float(multiply(oS.alphas,oS.labelMat).T * oS.K[:,k] + oS.b) #multiply(oS.alphas,oS.labelMat)内积，对应位置相乘
	Ek = fXk - float(oS.labelMat[k])
	return Ek

#选择内循环的alpha   保证每次优化中采用最大步长
def selectJ(i,oS,Ei) :														# i - 标号为i的数据的索引值 oS - 数据结构 Ei - 标号为i的数据误差
	maxK = -1
	maxDeltaE = 0
	Ej = 0
	oS.eCache[i] = [1,Ei]													#根据Ei更新误差缓存，将Ei在缓存中设置成有效的
	validEcacheList = nonzero(oS.eCache[:,0].A)[0]							#返回误差不为0的数据的索引值，是一个列表(也就是对计算过的Ej值进行操作)
																			#oS.eCache是一个 100 * 2 的矩阵
																			#oS.eCache[:,0]表示取出100行中第一列的数据
																			#oS.eCache[:,0].A表示将矩阵转换为array数组类型(100 * 1)
																			#nonzero(oS.eCache[:,0].A)表示找到非零的项  返回值和oS.eCache[:,0].A维数对应
																			#对应100 * 1 数组情况，第一个维数对应行数  第二个维数对应列数  
	if (len(validEcacheList)) > 1 :											#遍历,找到最大的Ek
		for k in validEcacheList :											#此时全局变量中的缓存值存在Ej信息
			if k == i :														#不计算i,浪费时间
				continue
			Ek = calcEk(oS,k)												#计算误差Ek
			deltaE = abs(Ei - Ek)											#计算步长
			if (deltaE > maxDeltaE) :										#取最大步长
				maxK = k
				maxDeltaE = deltaE 											#找到maxDeltaE
				Ej = Ek
		return maxK,Ej 														#maxK 标号为maxK的数据的索引值 Ej - 标号为j的数据误差
	else :																	#若此时Ej值没有计算过，就先随机选取j
		j = selectJrand(i,oS.m)												#随机选择alpha_j的索引值
		Ej = calcEk(oS,j)
	return j,Ej																#j标号为j的数据的索引值 Ej - 标号为j的数据误差

#计算误差值，更新缓存
def updateEk(oS,k) :
	Ek = calcEk(oS,k)
	oS.eCache[k] = [1,Ek]

#决策边界优化历程，优化的SMO算法
def innerL(i,oS) :															#输入参数索引值i  数据结构oS
	Ei = calcEk(oS,i)
	if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)) :#在这里判断KTT条件
																			#对正间隔和负间隔进行测试，同时判断alpha[i]是否超出0和C的界限，若在边界上就不能再减小或增大了
																			#也就不用在进行优化
																			#对三个KTT条件同时比较
																			#alpha > 0 yi * (wx + b) <= 1    这里判断的是违反情况
																			#alpha < C yi * (wx + b) >= 1
		j,Ej = selectJ(i,oS,Ei)												#第二个alpha的启发式选择
		alphaIold = oS.alphas[i].copy()										#python通过引用的方式传递所有列表，所以要明确分配内存，方便比较新旧alpha的值
		alphaJold = oS.alphas[j].copy()										#计算alphaIold,alphaJold;alphaIold,alphaJold为初始可行解
		if (oS.labelMat[i] != oS.labelMat[j]) :								#接下来计算L,H的值，用于将alphas[j]调整到0-C之间
																			#y1 y2 取值有四种情况 同号（1 1）（-1 -1） 异号（1 -1）（-1 1）
			L = max(0,oS.alphas[j] - oS.alphas[i])							#异号情况
			H = min(oS.C,oS.C + oS.alphas[j] - oS.alphas[i])
		else :																#同号情况
			L = max(0,oS.alphas[j] + oS.alphas[i] - oS.C)
			H = min(oS.C,oS.alphas[j] + oS.alphas[i])
		if L == H :
			# print('L == H')													#如果LH相等，退出这次循环
			return 0
		eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
																			#eta是alpha[j]的最优修改量eta=(K11+K22-2*K12)，这里取得是负数eta
		if eta >= 0:														#当eta < 0 不是二次函数，不能用求导方式对alpha求极值，跳过这次循环
			# print('eta >= 0')
			return 0
		oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta 					#alpha2_new = a2pha1_old + y2(E1-E2)/(k11+k22-2k12)
		oS.alphas[j] = clipAlpha(oS.alphas[j],L,H)							#调整alpha的值，下界L。上界H
		updateEk(oS,j)														#更新误差缓存
		if (abs(oS.alphas[j] - alphaJold) < 0.00001) :						#判断alpha调整的幅度，太小的话就跳过循环
			# print('j not moving enough')
			return 0
		oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
																			#alpha1_old * y1 + alpha2_old * y2 = alpha1_new * y1 + alpha2_new * y2
																			#alpha1_new = alpha1_old + y1 * y2 *(alpha2_old - alpha2_new) 	
		updateEk(oS,i)														#更新误差缓存
		b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i,j]
		b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j,j]
		#更新b1,b2
		#y1 =  wx1 + b1
		# w = alphai * yi * xi 																		(1 <= i <= m)
		#y1 = alphai * yi * ki1 +b1   																(1 <= i <= m)

		#b1_new = y1 - alphai * yi * ki1 - 	alpha1_new * k11 * y1 - alpha2_new *k21 *y2				(3 <= i <= m)

		#E1 = g(x1) - y1
		#g(xi) = wxi + bi																			(1 <= i <= m)
		#E1 = g(x1) - y1 = wx1 + b1 - y1 = alphai * yi * ki1 +b1 - y1					            (1 <= i <= m)

		#E1 = alphai * yi * ki1 + alpha1_old * k11 * y1 + alpha2_old *k21 *y2 + b_old - y1 			(3 <= i <= m)
				
		#b1_new = b_old - (-alpha_old1 + alpha1_new) * y1 * k11 -  (-alpha2_old + alpha2_new) * y2 * k12 -E1
		#b2_new = b_old - (-alpha_old1 + alpha1_new) * y1 * k12 -  (-alpha2_old + alpha2_new) * y2 * k22 -E2
		if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
 			oS.b = b1
		elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]) :
 			oS.b = b2
		else :
 			oS.b = (b1 + b2) / 2.0
		return 1
	else :
 		return 0

def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup = ('lin',0)) :									#新变量kTup，使用的是高斯核函数
	oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)								#导入数据结构
	iter = 0
	entireSet = True
	alphaPairsChanged = 0
	while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)) :  							#迭代次数大于最大迭代次数 上次遍历整个集合无alpha修改  退出循环	
		alphaPairsChanged = 0
		if entireSet :																				
			for i in range(oS.m) :																	#遍历所有的值，i是第一个选取的alpha
				alphaPairsChanged += innerL(i,oS)													#调用innerL函数，选取第二个alpha，判断是否有alpha改变，计数
				# print('fullSet, iter: %d i: %d,pairs changed %d' % (iter,i,alphaPairsChanged))
			iter += 1
		else :																					
			noBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]							#oS.alphas.A > 0 判断oS.alphas中各个数据是否大于0 返回值是true
																									#oS.alphas.A < C 判断oS.alphas中各个数据是否小于C 或false
																									#(oS.alphas.A > 0) * (oS.alphas.A < C)  列表相乘
																									#nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]取得下标值，是一个列表
			# print(noBoundIs)
			for i in noBoundIs :																	#遍历所有非边界的值
				alphaPairsChanged += innerL(i,oS)													#调用innerL函数，选取第二个alpha，判断是否有alpha改变，计数
				# print('non-bound, iter: %d i: %d,pairs changed %d' % (iter,i,alphaPairsChanged))
			iter += 1
		if entireSet : entireSet = False															#交替进行非边界循环和完整遍历循环,这次遍历全部集合，下次遍历非边界点
		# 因为随着多次子优化过程，边界变量倾向于留在边界，而非边界变量倾向于波动，这一步启发式的选择算法是基于节省时间考虑的，并且算法会一直在非边界变量集合上遍历，
		# 直到所有非边界变量都满足KKT条件（self-consistent）随后算法继续在整个集合上遍历寻找违反KKT条件的变量作为优化的第一个变量
		# 要注意的是，算法在整个集合上最多只连续遍历一次，但在非边界变量集合上可能连续遍历多次 
		elif (alphaPairsChanged == 0) :
			entireSet = True
		# print('iteration number : %d' % iter)
	return oS.b,oS.alphas


#利用核函数进行分类的径向基测试函数
def testDigits(kTup = ('rbf',100)):
	dataArr,labelArr = loadImages('trainingDigits') 								#打开文件，获取样本向量矩阵，样本标签矩阵
	b,alphas = smoP(dataArr,labelArr,200,0.0001,10000,kTup)							#调用smoP()函数，获取alphas，b
	dataMat = mat(dataArr)
	labelMat = mat(labelArr).transpose()

	svInd = nonzero(alphas.A > 0)[0]												#找到支持向量下标
	sVs = dataMat[svInd]															#保存支持向量对应x y
	labelSV = labelMat[svInd]														#支持向量的类别标签

	print('there are %d Support Vectors' % shape(sVs)[0])
	m,n = shape(dataMat)
	errorCount = 0
	for i in range(m) :
		kernelEval = kernelTrans(sVs,dataMat[i,:],kTup)						        #w = alpha * label * x  这里的计算步骤是先把 支持向量 * 每一个行向量
		predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b 				#label = w * x + b 	然后再把他们的乘积  * (alpha * label) 这里的alpha label 都是SV
		if (sign(predict) != sign(labelArr[i])) :									#sign(predict) 是符号函数 predict < 0 值为 -1 predict > 0 值为 1
			errorCount += 1															#预测错误 错误数加一下
	print('the training error rate is %f ' % (float(errorCount / m)))

 

	dataArr,labelArr = loadImages('testDigits')
	dataMat = mat(dataArr)
	labelMat = mat(labelArr).transpose()
	errorCount = 0
	m,n = shape(dataMat)
	for i in range(m) :
		kernelEval = kernelTrans(sVs,dataMat[i,:],kTup)
		predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
		if (sign(predict) != sign(labelArr[i])) :
			errorCount += 1
	print('the test error rate is %f ' % (float(errorCount / m)))

testDigits(('rbf',0.1))
testDigits(('rbf',5))
testDigits(('rbf',10))
testDigits(('rbf',50))
testDigits(('rbf',100))
testDigits(('lin',0))

# 138 143 149 192 202 210  行输出被注释掉了
