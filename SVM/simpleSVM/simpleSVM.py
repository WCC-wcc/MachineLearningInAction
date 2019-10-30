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


#simple_SMO
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):				#输入参数为数据集，类别标签，常数C，容错率，最大循环次数
	dataMatrix = mat(dataMatIn)										#转换成numpy矩阵，方便运算
	labelMat = mat(classLabels).transpose()							#类别标签转置，变成n*1列
	b = 0															#常数b初始化为0
	m,n = shape(dataMatrix)											
	alphas = mat(zeros((m,1)))										#初始化alpha向量，m*1的0向量
	iter = 0														#初始化迭代次数0
	while (iter < maxIter):											#外循环，迭代次数小于最大循环次数
		alphaPairsChanged = 0										#记录alpha是否已经优化
		for i in range(m):											#内循环，处理数据集中每个数据向量
			fXi = float(multiply(alphas,labelMat).T * (dataMatrix * dataMatrix[i,:].T)) + b        	#fXi就是计算出的类别 yi = wxi+b
																	#w = alphai * yi * xi(1 <= i <= m) 			 
																	#multiply(a,b) 数量积				

			Ei  = fXi - float(labelMat[i])							#实际结果与真实结果的误差，如果误差很大，那么就要对该数据实例所对应的alpha值进行优化
			if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
																	#对正间隔和负间隔进行测试，同时判断alpha[i]是否超出0和C的界限，若在边界上就不能再减小或增大了
																	#也就不用在进行优化
				j = selectJrand(i,m)								#随机选择第二个alpha
				fXj = float(multiply(alphas,labelMat).T * (dataMatrix * dataMatrix[j,:].T)) + b     #fXj就是计算出的类别 yj = wxj+b
				Ej = fXj - float(labelMat[j])						#实际结果与真实结果的误差，如果误差很大，那么就要对该数据实例所对应的alpha值进行优化
				alphaIold = alphas[i].copy()						#python通过引用的方式传递所有列表，所以要明确分配内存，方便比较新旧alpha的值
				alphaJold = alphas[j].copy()						#计算alphaIold,alphaJold;alphaIold,alphaJold为初始可行解
				if (labelMat[i] != labelMat[j]):					#接下来计算L,H的值，用于将alphas[j]调整到0-C之间
																	#y1 y2 取值有四种情况 同号（1 1）（-1 -1） 异号（1 -1）（-1 1）
					L = max(0,alphas[j] - alphas[i])				#异号情况
					H = min(C,C + alphas[j] - alphas[i])
				else:
					L = max(0,alphas[j] + alphas[i] - C)			#同号情况
					H = min(C,alphas[j] + alphas[i])
				if L == H :
					print('L == H')									#如果LH相等，退出这次循环
					continue
				eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T - dataMatrix[j,:] * dataMatrix[j,:].T
																	#eta是alpha[j]的最优修改量eta=(K11+K22-2*K12)，这里取得是负数eta
				if eta >= 0:										#当eta < 0 不是二次函数，不能用求导方式对alpha求极值，跳过这次循环
					print('eta >= 0')
					continue
				alphas[j] -= labelMat[j] * (Ei - Ej) / eta          #alpha2_new = a2pha1_old + y2(E1-E2)/(k11+k22-2k12)
				alphas[j] = clipAlpha(alphas[j],L,H) 				#调整alpha的值，下界L。上界H
				if (abs(alphas[j] - alphaJold) < 0.00001):			#判断alpha调整的幅度，太小的话就跳过循环
					print('j not moving enough')
					continue
				alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])  
				#alpha1_old * y1 + alpha2_old * y2 = alpha1_new * y1 + alpha2_new * y2
				#alpha1_new = alpha1_old + y1 * y2 *(alpha2_old - alpha2_new) 						alpha2_new 即 alphas[j]
				b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] *dataMatrix[j,:].T
				b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[j,:].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j,:] *dataMatrix[j,:].T
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
				#b1_new = b_old - (-alpha_old1 + alpha1_new) * y1 * k12 -  (-alpha2_old + alpha2_new) * y2 * k22 -E2
				if (0 < alphas[i]) and (C > alphas[i]):
					b = b1
				elif (0 < alphas[j]) and (C > alphas[j]):
					b = b2
				else:
					b = (b1 + b2) / 2.0
				alphaPairsChanged += 1									#程序到这里不执行continue语句，说明成功改变一对alpha值，alphaPairsChanged + 1
				print('iter : %d i : %d j : %d,pairs changed %d' % (iter,i,j,alphaPairsChanged))  #输出  执行到第几次循环 第一个alpha是谁 改变的alpha对数目
		if (alphaPairsChanged == 0):									#for循环之外 判断alpha是否更新，无更新就将iter + 1 															
 			iter += 1
		else :															#否则令 iter = 0  重新遍历   达到最大遍历次数时停止
 			iter = 0
		print('iteration number: %d' % iter)
	return b,alphas														

def calculateW(dataArr,labelArr,alpha):									#dataArr是100 * 2 的数组，下面生成的Mw是一个2 * 1的数组，初始化为[0,0]
	Mw = matrix(zeros(shape(dataArr)[1]))								#shape(dataArr)[0]表示数组dataArr是几维的  shape(dataArr)[1]表示数组第一条记录中有几个数据
	for i in range (shape(alpha)[0]):
		if alpha[i]>0:
			Mw += multiply(labelArr[i]*alpha[i],dataArr[i])				#计算w的值  w = alphai * yi * xi     (1 <= i <= m) 当alphai . 0 时
	w = Mw.T.tolist()													#将w转置再转换成列表  2*1
	return w


#画出数据集和simpleSVM回归最佳拟合直线

def drawing(dataArr,labelArr,alpha,b): 
	n = shape(labelArr)[0] 													#取数，数据点分为四类   class1 class-1 和class1 SV class-1 SV
	xcord1 = []; ycord1 = []   
	xcord2 = []; ycord2 = []
	xcord3 = []; ycord3 = []
	xcord4 = []; ycord4 = []
	for i in range(n):
		if int(labelArr[i])== 1:
			if alpha[i]>0:
				xcord3.append(dataArr[i][0]); ycord3.append(dataArr[i][1])
			else:
				xcord1.append(dataArr[i][0]); ycord1.append(dataArr[i][1])
		else:
			if alpha[i]>0:
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
	y1 = (b+(calculateW(dataArr,labelArr,alpha)[0][0])*x)/(-calculateW(dataArr,labelArr,alpha)[1][0])	#w * x + b = 0 就是分割超平面，由于是二维的，计算y表达式
	y = mat(y1).T 			#w1x1 +w2x2 + b =0    x2 = (b + w1x1)/(-w2)    w1 =w[0][0]  w2 =w[1][0]
	ax.plot(x, y,'-')		#画出x,y关系图像
	plt.xlabel('X1')
	plt.ylabel('X2');
	# plt.savefig('SMOSimple.png',dpi=2000)
	plt.show()

dataArr,labelArr = loadDataSet('testSet.txt')
b,alpha = smoSimple(dataArr,labelArr,0.6,0.001,40)
drawing(dataArr,labelArr,alpha,b)

