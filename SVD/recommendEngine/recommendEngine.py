#基于协同过滤的推荐引擎  将用户和其他用户的数据进行对比来实现推荐

from numpy import *
from numpy import linalg as la

def loadExData():
	return [[0,0,0,2,2],
			[0,0,0,3,3],
			[0,0,0,1,1],
			[1,1,1,0,0],
			[2,2,2,0,0],
			[5,5,5,0,0],
			[1,1,1,0,0]]

#欧氏距离
def euclidSim(inA,inB):
	return 1.0 / (1.0 + la.norm(inA - inB))										#la.norm()函数，是用来计算向量的二范数  1/(1+dist)希望相似度值在0~1之间

#皮尔逊相关系数
def pearsSim(inA,inB):
	if len(inA) < 3: 															#在numpy中可以用线性代数模块linalg中的corrcoef()来计算相关系数（correlation coefficient）。
		return 1.0 																#得出结果的取值范围是-1～1，可通过“0.5+0.5*corrcoef()”将其缩放到0～1之间。
	return 0.5 + 0.5 * corrcoef(inA,inB,rowvar = 0)[0][1] 						

#余弦相似度
def cosSim(inA,inB):
	num = float(inA.T * inB)													#cos(A,B) = (A * B) / (||A|| * ||B||)   A,B两个向量余弦相似度的计算方法
	denom = la.norm(inA) * la.norm(inB)
	return 0.5 + 0.5 * (num / denom) 											#余弦相似度也在-1~1之间，用0.5+0.5*cos是为了使相似度值在0~1之间

#所谓基于物品的相似度也就是看物品之间的相似程度，比如对于苹果，香蕉两个物品，10位用户分别给出评价，看他们评价的相关性就可以得到一个系数，这时候第11位用户只要对任意
#一种产品评价，就可以用这个评价乘以相关系数，得到他对另一个物品的评价

#对两种物品而言，比较他们的相似度(也就是所有用户对这两者的评价) 用这个相似度做一个权值 * 准备预测的用户对其中一个物品的评价，就可以得到他对剩下那个物品的评价
#在给定相似度计算方法的条件下，用来计算用户对物品的估计评分值   
def standEst(dataMat,user,simMeas,item):										#输入参数：数据矩阵，用户编号，相似度计算方法，物品编号
	n = shape(dataMat)[1] 														#对数据矩阵而言，行对应用户，列对应物品，这里得到数据集中物品数目
	simTotal = 0.0 																#初始化变量，用于统计总的相似度
	ratSimTotal = 0.0 															#初始化变量，用于表示贡献度(总的评价)
	for j in range(n): 															#遍历行中每一个物品
		userRating = dataMat[user,j] 											#取出用户对该物品的评价
		if userRating == 0: 													#如果用户没有评价过该物品，就跳过
			continue 
		overLap = nonzero(logical_and(dataMat[:,item].A > 0,dataMat[:,j].A > 0))[0] 	#找到所有用户对这两种物品都打过分的项的索引
								#dataMat[:,item].A转换为数组
						        #dataMat[:,item].A>0，数组中对应位置元素大于0（有评级），则为ture。反之为false
						        #logical_and：逻辑与（两者同为1，与结果为1）：同时对两个物品评级为1，否则为0
						        #nonzero（）：返回非零元素的索引值。返回两个矩阵：(array([0, 3, 4, 5, 6], dtype=int64), array([0, 0, 0, 0, 0], dtype=int64))
						        #表示相应维度上非零元素所在的行和列索引。
						        #nonzero()[0]表示非零元素所在行索引数组
		# print(overLap)
		if len(overLap) == 0: 													#若没有用户同时对这两件物品进行评级，则相似度为0
			similarity = 0
		else: 																	#根据所给定的相似度计算方法，计算同时对这两件物品评级的评级数据之间的相似度
			# print(dataMat[overLap,item])
			similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j])
		print('the %d and %d similarity is: %f' % (item,j,similarity)) 			#打印当前物品和欲评级物品之间的相似度
		simTotal += similarity 													#计算总的相似度
		# print(simTotal)
		ratSimTotal += similarity * userRating 									#不仅仅使用相似度，而是将  评分当权值 * 相似度 = 贡献度
		# print(ratSimTotal)
	if simTotal == 0: 															#若该推荐物品与所有列都未比较则评分为0
		return 0
	else:
		return ratSimTotal / simTotal 											#除以所有评分总和进行归一化  将评分值映射到[1,5]之间，返回该未打分项物品的评分

#推荐引擎  调用standEst(dataMat,user,simMeas,item)函数
def recommend(dataMat,user,N = 3,simMeas = cosSim,estMethod = standEst):
	unratedItems = nonzero(dataMat[user,:].A == 0)[1] 							#相似度累积，该用户对当前物品的评分*两个物品之间的相似度累积
	if len(unratedItems) == 0: 													#若所有物品均已评级
		return 'you rated everything'
	itemScores = [] 															#存放物品编号及对应的估计评级
	for item in unratedItems:  													#遍历未被评级的物品，依次调用估计方法，获得估计评级
		estimatedScore = estMethod(dataMat,user,simMeas,item)
		itemScores.append((item,estimatedScore))
	return sorted(itemScores,key = lambda jj : jj[1],reverse = True)[:N] 		#按照估计评级进行降序排列，并返回前N个物品编号，作为对该用户未尝过菜肴的推荐

myMat = mat(loadExData())
myMat[0,0] = myMat[0,1] = myMat[1,0] = myMat[2,0] = 4
myMat[3,3] = 2
# print(myMat)
print(recommend(myMat,2))
print()
print(recommend(myMat,2,simMeas = euclidSim))
print()
print(recommend(myMat,2,simMeas = pearsSim))