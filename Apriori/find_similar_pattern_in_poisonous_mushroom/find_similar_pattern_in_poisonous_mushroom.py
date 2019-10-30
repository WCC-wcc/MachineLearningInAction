from numpy import *

#对数据集中每一个项构建一个不变集合
def createC1(dataSet):
	C1 = []
	for transaction in dataSet: 											#读取数据集中每一行
		for item in transaction: 											#读取数据集中每一行的每一个列值
			if not [item] in C1: 											#如果C1中没有该数据
				C1.append([item]) 											#在C1中添加该数据
	C1.sort()
	return list(map(frozenset,C1)) 											#返回一个frozenset()

#计算所有项集支持度   C1->L1->C2->L2->C3->L3->C4 直到Lk为空  Li 是生成的频繁项集(包含i个元素) 用来生成Ci+1的候选项集(包含i+1个元素)   
def scanD(D,Ck,minSupport):													#输入参数为 原始数据集  候选项集列表   最小支持度
	ssCnt = {} 																#初始化一个字典     记录每一个候选项集出现次数
	for tid in D: 															#遍历原始数据集中每一行记录
		for can in Ck: 														#遍历每一个候选项集
			if can.issubset(tid): 											#如果候选项集在原始数据集中出现过
				if can not in ssCnt:                                  		#记录候选项集的出现次数
					ssCnt[can] = 1 											#没有记录时  初始化为1
				else:
					ssCnt[can] +=1 											#有记录时  出现次数加1
	numItems = float(len(D)) 												#数据集中全部记录数
	retList = []															#初始化结果列表
	supportData = {}														#初始化频繁项集支持度
	for key in ssCnt: 														#遍历每一个候选项集
		support = ssCnt[key] / numItems										#计算每一个候选项集的支持度
		if support >= minSupport: 											#支持度满足条件时
			retList.insert(0,key) 											#将该候选项集插入到结果列表中第一个位置处
		supportData[key] = support 											#保存每一个候选项集的支持度
	return retList,supportData 												#返回频繁项集(Li)  各个频繁项集的支持度

#由频繁项集列表(Lk)(k个元素)生成候选项集列表(Ck+1)(k+1个元素)      这里主要是做集合合并的操作
def aprioriGen(Lk,k): 														#Lk，频繁项集列表       k，项集元素的个数
	retList = []															#Lk相当于是二维的 Lk[i][j]	i选取哪一个规模(i+1个元素)的频繁项集   j选取该规模下的小集合
	lenLk = len(Lk)															#当前频繁项集列表有多少个集合不等于k  {{3}，{1}} lenLk = 2 k = 1
	for i in range(lenLk): 													#依次遍历频繁项集列表中每一个小集和
		for j in range(i+1,lenLk): 											#将频繁项集列表中每一个小集和 和 他身后的每个小集合进行合并
			L1 = list(Lk[i])[:k-2]											#对单个元素的取值比如([{1}])   [:k-2]   取出的是空集合  因此两个单元素集合会合并成一个二元素集合
			L2 = list(Lk[j])[:k-2]											#Python中使用下标0表示第一个元素，因此[:k-2]的实际作用为取列表的前k-1个元素。
			L1.sort()
			L2.sort() 														#对每个项集按元素排序，然后每次比较两个项集
			if L1 == L2: 													#只有在前k-1项相同、最后一项不相同的情况下合并才为所需要的新候选项集。
				retList.append(Lk[i] | Lk[j]) 								#合并集合
	return retList 															#返回生成的(Ck+1)候选项集

#apriori算法  输入参数为      初始数据集				最小支持度
def apriori(dataSet,minSupport = 0.5):
	C1 = createC1(dataSet) 													#用单个物品生成大集合(冰冻集合)
	D = list(map(set,dataSet)) 												#初始化数据集
	L1,supportData = scanD(D,C1,minSupport)									#生成L1(也就是单个物品的频繁项集)，以及支持度列表(单个物品)
	L = [L1] 																#将L1添加到L列表(各个规模的总的频繁项集列表)中
	k = 2
	while(len(L[k-2]) > 0):													#当频繁项集(m个元素)集合为空时  就不能再生成候选项集(m+1个元素)  退出算法
		Ck = aprioriGen(L[k-2],k) 											#由频繁项集列表(Lk)(k个元素)生成候选项集列表(Ck+1)(k+1个元素)
		Lk,supK = scanD(D,Ck,minSupport)									#生成新的频繁项集LK(其中元素个数加1)，以及对应的集合的支持度(K+1个元素)
		supportData.update(supK) 											#更新支持度列表(各个规模下不同集合的支持度)
		L.append(Lk)														#将新生成的频繁项集加入到频繁项集列表中去
		k += 1 																#更新k值  也就是元素个数
	return L,supportData 													#返回频繁项集列表  支持度列表

#找出毒蘑菇相似特征   只需要得到频繁项集  对关联规则不感兴趣   读取文件数据  得到各个规模下的频繁项集 
mushDat = [line.split() for line in open('mushroom.dat').readlines()] 		#打开文件  按行读取  对每一行按照‘/t’切分数据   每一列的最小值都比前一列最大值大1 方便确认特征位置
# print(mat(mushDat)) 														#每行23个特征  都是数值型数据  第一列表示是否有毒  1 没毒  2 有毒  其他列是一些其他特征
L,suppData = apriori(mushDat,minSupport = 0.3)
# print(L)
for item in L[1]:
	if item.intersection('2'): 												#包含有毒特征2的频繁2项集
		print(item)