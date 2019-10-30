from numpy import *
import matplotlib.pyplot as plt

#读取文件函数
def loadDataSet(fileName):
	dataMat = []
	labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]),float(lineArr[1])])
		labelMat.append(float(lineArr[2]))
	return dataMat,labelMat

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
	def __init__(self,dataMatIn,classLabels,C,toler) :
		self.X = dataMatIn													#记录数据样本
		self.labelMat = classLabels											#记录标签
		self.C = C 															#常数C
		self.tol = toler													#容错率
		self.m = shape(dataMatIn)[0]										#记录数据个数
		self.alphas = mat(zeros((self.m,1)))								#数据个数对应的alphas值
		self.b = 0															#纪录常数b
		self.eCache = mat(zeros((self.m,2)))								#增加一个 m * 2 的矩阵成员变量eCache  第一列是eCache是否有效的标志位 第二列是实际的E值

#计算误差E值并返回  注意  E值  是动态变化的  E = wx + b - y 		w = alphai * yi * xi (i = 1,2...100 ) 
def calcEk(oS,k) :															
	fXk = float(multiply(oS.alphas,oS.labelMat).T * (oS.X * oS.X[k,:].T)) + oS.b #multiply(oS.alphas,oS.labelMat)内积，对应位置相乘
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
			print('L == H')													#如果LH相等，退出这次循环
			return 0
		eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:] * oS.X[i,:].T - oS.X[j,:] * oS.X[j,:].T  
																			#eta是alpha[j]的最优修改量eta=(K11+K22-2*K12)，这里取得是负数eta
		if eta >= 0:														#当eta < 0 不是二次函数，不能用求导方式对alpha求极值，跳过这次循环
			print('eta >= 0')
			return 0
		oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta 					#alpha2_new = a2pha1_old + y2(E1-E2)/(k11+k22-2k12)
		oS.alphas[j] = clipAlpha(oS.alphas[j],L,H)							#调整alpha的值，下界L。上界H
		updateEk(oS,j)														#更新误差缓存
		if (abs(oS.alphas[j] - alphaJold) < 0.00001) :						#判断alpha调整的幅度，太小的话就跳过循环
			print('j not moving enough')
			return 0
		oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
																			#alpha1_old * y1 + alpha2_old * y2 = alpha1_new * y1 + alpha2_new * y2
																			#alpha1_new = alpha1_old + y1 * y2 *(alpha2_old - alpha2_new) 	
		updateEk(oS,i)														#更新误差缓存
		b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i,:] * oS.X[i,:].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i,:] * oS.X[j,:].T
		b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i,:] * oS.X[j,:].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j,:] * oS.X[j,:].T
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
	oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)								#导入数据结构
	iter = 0
	entireSet = True
	alphaPairsChanged = 0
	while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)) :  							#迭代次数大于最大迭代次数 上次遍历整个集合无alpha修改  退出循环	
		alphaPairsChanged = 0
		if entireSet :																				
			for i in range(oS.m) :																	#遍历所有的值，i是第一个选取的alpha
				alphaPairsChanged += innerL(i,oS)													#调用innerL函数，选取第二个alpha，判断是否有alpha改变，计数
				print('fullSet, iter: %d i: %d,pairs changed %d' % (iter,i,alphaPairsChanged))
			iter += 1
		else :																					
			noBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]							#oS.alphas.A > 0 判断oS.alphas中各个数据是否大于0 返回值是true
																									#oS.alphas.A < C 判断oS.alphas中各个数据是否小于C 或false
																									#(oS.alphas.A > 0) * (oS.alphas.A < C)  列表相乘
																									#nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]取得下标值，是一个列表
			# print(noBoundIs)
			for i in noBoundIs :																	#遍历所有非边界的值
				alphaPairsChanged += innerL(i,oS)													#调用innerL函数，选取第二个alpha，判断是否有alpha改变，计数
				print('non-bound, iter: %d i: %d,pairs changed %d' % (iter,i,alphaPairsChanged))
			iter += 1
		if entireSet : entireSet = False															#交替进行非边界循环和完整遍历循环,这次遍历全部集合，下次遍历非边界点
		# 因为随着多次子优化过程，边界变量倾向于留在边界，而非边界变量倾向于波动，这一步启发式的选择算法是基于节省时间考虑的，并且算法会一直在非边界变量集合上遍历，
		# 直到所有非边界变量都满足KKT条件（self-consistent）随后算法继续在整个集合上遍历寻找违反KKT条件的变量作为优化的第一个变量
		# 要注意的是，算法在整个集合上最多只连续遍历一次，但在非边界变量集合上可能连续遍历多次 
		elif (alphaPairsChanged == 0) :
			entireSet = True
		print('iteration number : %d' % iter)
	return oS.b,oS.alphas

def calculateW(dataArr,labelArr,alphas):									#dataArr是100 * 2 的数组，下面生成的Mw是一个2 * 1的数组，初始化为[0,0]
	X = mat(dataArr)
	labelMat = mat(labelArr).transpose() 
	m,n = shape(X)
	w = zeros((n,1))
	for i in range(m):
		w += multiply(alphas[i] * labelMat[i],X[i,:].T)
	return w.tolist()
	# Mw = matrix(zeros(shape(dataArr)[1]))								#shape(dataArr)[0]表示数组dataArr是几维的  shape(dataArr)[1]表示数组第一条记录中有几个数据
	# for i in range (shape(alpha)[0]):
	# 	if alpha[i]>0:
	# 		Mw += multiply(labelArr[i]*alpha[i],dataArr[i])				#计算w的值  w = alphai * yi * xi     (1 <= i <= m) 当alphai . 0 时
	# w = Mw.T.tolist()													#将w转置再转换成列表  2*1
	# return w


#画出数据集和simpleSVM回归最佳拟合直线

def drawing(dataArr,labelArr,alphas,b): 

	n = shape(labelArr)[0] 													#取数，数据点分为四类   class1 class-1 和class1 SV class-1 SV
	xcord1 = []; ycord1 = []   
	xcord2 = []; ycord2 = []
	xcord3 = []; ycord3 = []
	xcord4 = []; ycord4 = []
	for i in range(n):
		if int(labelArr[i])== 1:
			if 0.6>alphas[i]>0:
				xcord3.append(dataArr[i][0]); ycord3.append(dataArr[i][1])
			else:
				xcord1.append(dataArr[i][0]); ycord1.append(dataArr[i][1])
		else:
			if 0.6>alphas[i]>0:
				xcord4.append(dataArr[i][0]); ycord4.append(dataArr[i][1])
			else:
				xcord2.append(dataArr[i][0]); ycord2.append(dataArr[i][1]) 
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=40, c='yellow', marker='s',label='class 1')			#开始绘图
	ax.scatter(xcord2, ycord2, s=40, c='green',label='class -1')
	ax.scatter(xcord3, ycord3, s=40, c='red',marker='s',label='SV 1')		
	ax.scatter(xcord4, ycord4, s=40, c='red',label='SV -1')	

	ax.legend(loc='best')

	x = arange(2.7, 6.6, 0.1)															#取出40个x进行绘图，这里处理的是分隔超平面
	y1 = (b+(calculateW(dataArr,labelArr,alphas)[0][0])*x)/(-calculateW(dataArr,labelArr,alphas)[1][0])	#w * x + b = 0 就是分割超平面，由于是二维的，计算y表达式
	y = mat(y1).T 			#w1x1 +w2x2 + b = 0    x2 = (b + w1x1)/(-w2)    w1 =w[0][0]  w2 =w[1][0]
	ax.plot(x, y,'-')		#画出x,y关系图像
	plt.xlabel('X1')
	plt.ylabel('X2');
	# plt.savefig('SMO.png',dpi=2000)
	plt.show()



if __name__ == '__main__':
	dataArr,labelArr = loadDataSet('testSet.txt')
	b,alphas = smoP(dataArr,labelArr,0.6,0.001,40)
	drawing(dataArr,labelArr,alphas,b)


# dataMat = mat(dataArr)
# a = dataMat[2] * calculateW(dataArr,labelArr,alphas) + b
# print(a)
# print(labelArr[2])