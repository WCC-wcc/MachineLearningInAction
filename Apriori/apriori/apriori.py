#创建初始数据
def loadDataSet():
	return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

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

# dataSet = loadDataSet()
# print(dataSet) 															#[[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
# C1 = createC1(dataSet)
# print(C1) 																#[frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})]
# D = list(map(set,dataSet))
# print(D) 																	#[{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}]
# L1,supportData0 = scanD(D,C1,0.5)
# print(L1) 																#[frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})]
# print(supportData0) 														#{frozenset({1}): 0.5, frozenset({3}): 0.75, frozenset({4}): 0.25, frozenset({2}): 0.75, frozenset({5}): 0.75}

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

# dataSet = loadDataSet()
# # print(dataSet) 															#[[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
# L,suppData = apriori(dataSet)
# print(L[0]) 																#[frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})]
# print(L[1]) 																#[frozenset({2, 3}), frozenset({3, 5}), frozenset({2, 5}), frozenset({1, 3})]
# print(L[2]) 																#[frozenset({2, 3, 5})]
# print(L[3])																#[]

#由频繁项集生成关联规则列表 		 minConf为最小可信度阈值，supportDate里面存放了每一个频繁项集的对应的支持度 
def generateRules(L,supportData,minConf = 0.7):
	bigRuleList = []
	for i in range(1,len(L)): 												##只获取两个或者更多的频繁项集合,L0是频繁1项集，没关联规则 
		for freqSet in L[i]: 												#从规模为2的频繁项集开始遍历
			H1 = [frozenset([item]) for item in freqSet] 					#针对每个频繁项集  构建只包含单个元素集合的列表H1
			if (i > 1): 													##从频繁3项集开始，从置信度算出关联规则
				rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf) 
			else: 															##对频繁2项集，计算置信度 
				calcConf(freqSet,H1,supportData,bigRuleList,minConf)        #(a) --> (b) 指的是 support(a,b)/support(a) > minConf
	return bigRuleList 													    #生成一个包含可信度的规则列表    也就是下面两个函数中的的参数br1
 
#评估规则函数    计算可信度   该函数接收5个参数，分别是用于计算的频繁项集、此项集各个元素(可以出现在规则右部的元素列表H)、包含项集的支持度字典、关联规则数组、置信度阈值；
def calcConf(freqSet,H,supportData,br1,minConf = 0.7):
	prunedH = []
	for conseq in H: 														#用每个conseq作为后件
		conf = supportData[freqSet] / supportData[freqSet - conseq] 		#freqSet-conseq是集合减去集合。即使freqSet中的元素减去conseq中的元素，而不是数减数 
		if conf >= minConf: 												#满足可信度要求
			print(freqSet - conseq,'-->',conseq,' conf : ',conf)
			br1.append((freqSet - conseq,conseq,conf)) 						#元组中的三个元素：前件、后件、置信度
			prunedH.append(conseq)
	return prunedH 															#返回后件列表   以便进行后件的合并  生成新的关联规则  

#对规则后件进行合并，以此生成后件有两元素的规则，有三元素的规则....
#生成候选规则集    该函数接收5个参数，分别是用于计算的频繁项集、此项集各个元素(可以出现在规则右部的元素列表H)、包含项集的支持度字典、关联规则数组、置信度阈值；
def rulesFromConseq(freqSet,H,supportData,br1,minConf = 0.7): 				#这里的freqSet = (2,3,5)
	# print(freqSet) 														#frozenset({2, 3, 5})
	# print(H)																#[frozenset({2}), frozenset({3}), frozenset({5})]
	m = len(H[0]) 															#H[0] = 2   #获取后件元素的个数
	# print(m)
	if (len(freqSet) > (m + 1)): 											#如果频繁集元素个数大于规则后件个数  这里创造的关联规则  后间个数从2开始增加直到n-1
		Hmp1 = aprioriGen(H,m + 1) 											##由H，创建m+1候选项集   包含所有可能的规则(Hmp1是可能的规则后件)
		# print(Hmp1)
		Hmp1 = calcConf(freqSet,Hmp1,supportData,br1,minConf) 				#调用calcConf()函数  判断生成的候选关联规则  是否满足最低可信度要求 
		 																	#这个地方运用Apriori原理     对于满足可信度要求的关联规则进行进一步合并  
		# print(Hmp1) 														#[frozenset({2, 3}), frozenset({2, 5}), frozenset({3, 5})]
		if (len(Hmp1) > 1): 												#至少需要两列合并   后件多余两个的情况下生成新的关联规则
			rulesFromConseq(freqSet,Hmp1,supportData,br1,minConf)


dataSet = loadDataSet()
# print(dataSet) 															#[[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
L,suppData = apriori(dataSet,0.5)
rules = generateRules(L,suppData,minConf = 0.5)
# print(rules)