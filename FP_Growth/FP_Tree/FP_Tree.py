class treeNode: 																#FP树中节点的类定义
	def __init__(self,nameValue,numOccur,parentNode): 							
		self.name = nameValue 													#存放节点元素名字的变量
		self.count = numOccur 													#计数值  统计节点元素出现次数
		self.nodeLink = None  													#链接和本节点元素同类元素
		self.parent = parentNode 												#指向本节点的父亲节点
		self.children = {} 														#存放本节点的子节点

	def inc(self,numOccur): 													#对count(节点元素出现次数)变量增加给定值
		self.count += numOccur

	def disp(self,ind = 1): 													#将树以文本形式显示  方便调试
		print(' '*ind,self.name, ' ',self.count) 								#用文本方式输出
		for child in self.children.values():									#深度优先
			child.disp(ind + 1)

# rootNode = treeNode('pyramid',9,None)
# rootNode.children['eye'] = treeNode('eye',13,None)
# print(rootNode.disp())
# rootNode.children['eye'].children['phoenix'] = treeNode('phoenix',3,None)
# print()
# print(rootNode.disp())

#创建FP-Tree 输入参数为数据集和最小支持度   #由于用例中存在出现次数相同的项，如t和y所以每次排序结果可能会不同从而导致最终的FP树有所不同，但应该是等价的
def createTree(dataSet,minSup = 1):
	headerTable = {} 															#键值对集合，存放元素项和其出现次数    也就是头指针列表前身  用来链接同类元素
	for trans in dataSet: 														#trans:事务   第一次遍历 获取每个元素出现的次数
		for item in trans: 														#item:元素
			headerTable[item] = headerTable.get(item,0) + dataSet[trans] 		#元素出现总次数 = headerTable中已有的次数 + 事务中出现的1次
 																				#正常情况下  一个二维的矩阵A  A[i][j]代表i行n列位置上的数据 
 																				#							A[i]代表i行所有的数据       
 																				#这里输入的数据不是一个简单的二维矩阵，他对矩阵做过处理  使得A[i] = 1
 																				#数据处理参考函数createInitSet()函数
	# print(headerTable)
	for k in list(headerTable.keys()): 											#以下三行 移除不满足最小支持度的元素项，headerTable中的key即为具体元素
		if headerTable[k] < minSup:
			del(headerTable[k])
	# print(headerTable)
	
	freqItemSet = set(headerTable.keys()) 										#转换为集合得到频繁项集（去除重复元素项，只保留满足支持度要求的元素项）
	# print(freqItemSet)
	
	if len(freqItemSet) == 0 : 													#如果没有元素项满足要求， 则退出
		return None,None
	
	for k in headerTable: 														#遍历headerTable获得头指针表（元素：元素出现次数，相似元素指针） 初始化头指针列表
		headerTable[k] = [headerTable[k],None]									#headerTable原数据是{'r': 3, 'z': 5, 't': 3, 'x': 4, 's': 3, 'y': 3}
	# print(headerTable)														#此时更新为{'z': [5, None], 'r': [3, None], 'x': [4, None], 'y': [3, None], 't': [3, None], 's': [3, None]}
	
	retTree = treeNode('Null Set',1,None)										#树初始化：空集：1 无父节点

	for tranSet, count in dataSet.items(): 										#第二次遍历处理后的数据集字典
		localD = {} 															
		for item in tranSet:													#遍历每一项事务中每一项元素
			if item in freqItemSet: 											
				localD[item] = headerTable[item][0] 							#找出所有事务中的频繁项（元素：元素出现次数）集合  每次处理一行事务
		# print(localD) 														#{'r': 3, 'z': 5}
		if len(localD) > 0 : 													#根据全局频率对每个事务中的元素进行排序
			orderedItems = [v[0] for v in sorted(localD.items(),key = lambda p:p[1],reverse = True)]#事务中删除非频繁项后剩余的元素并逆向排序集合
									#sorted(排序对象，key，reverse),当待排序列表的元素由多字段构成时，
									#我们可以通过sorted(iterable，key，reverse)的参数key来制定我们根据哪个字段对列表元素进行排序
									#这里key=lambda p: p[1]指明要根据键对应的值，即根据频繁项的频数进行从大到小排序
									#v[0]取得是key值   
									#items()方法返回一个可迭代的dict_items类型，其元素是键值对组成的2-元组 
			# print(orderedItems)												#['z', 'r']										
			updateTree(orderedItems,retTree,headerTable,count) 					#使用排序后的频率项集对树进行填充  
	return retTree,headerTable

#更新树的代码  结合书上P227图理解
def updateTree(items,inTree,headerTable,count):
																				#首先更新树
	if items[0] in inTree.children: 											#判断项集第一个元素是否已经作为子节点已存在，存在就直接增加出现次数值
		inTree.children[items[0]].inc(count)
	else: 																		#否则向树增加一个分支	  使其父节点指向inTree
		inTree.children[items[0]] = treeNode(items[0],count,inTree) 			#创建一个新的树节点，并更新了父节点inTree，父节点是一个类对象，包含很多特性
		

		if headerTable[items[0]][1] == None: 									#其次更新头指针列表  如果头表目标节点为空
			headerTable[items[0]][1] = inTree.children[items[0]] 				#把指向每种类型第一个元素项放在头指针表里
		else:
			updateHeader(headerTable[items[0]][1],inTree.children[items[0]]) 	#更新生成链表，注意，链表也是每过一个样本，更一次链表，且链表更新都是从头指针表开始的

	if len(items) > 1:															#仍有未分配完的树，迭代，注意这时迭代的父亲节点就是该项集第一个元素，而不再是inTree
		updateTree(items[1::],inTree.children[items[0]],headerTable,count)		#由items[1::]可知，每次调用updateTree时都会去掉列表中第一个元素，递归

#函数说明：它确保节点链接指向树中该元素项的每一个实例。   用来更新头指针列表  链接同类型所有元素
def updateHeader(nodeToTest,targetNode):
	while(nodeToTest.nodeLink != None):											#不断更新  nodeToTest  一直到当前该类元素最后一个位置处
		nodeToTest = nodeToTest.nodeLink										#从头指针表的 nodeLink 开始,一直沿着nodeLink直到到达链表末尾，这就是一个链表
	nodeToTest.nodeLink = targetNode 											#链接相似的元素项  每次调用这个函数都是找到一个新的同类元素，连接到该类元素链表末尾

#创建数据
def loadDataSet():
	simDat = [['r','z','h','j','p'],
			  ['z','y','x','w','v','u','t','s'],
			  ['z'],
		      ['r','x','n','o','s'],
			  ['y','r','x','z','q','t','p'],
			  ['y','z','x','e','q','s','t','m']]
	return simDat

#数据预处理
def createInitSet(dataSet):
	retDict = {} 																#初始化一个字典，存放数据
	for trans in dataSet: 														#对数据集中的每一行
		retDict[frozenset(trans)] = 1 											#创建一系列键值对  键是原数据集中每一行的数据  这里处理成一个冰冻集合  值均为1
	return retDict																#{frozenset({'p', 'j', 'h', 'r', 'z'}): 1, frozenset({'s', 'u', 'v', 'y', 'w', 't', 'z', 'x'}): 1, frozenset({'z'}): 1, frozenset({'s', 'n', 'r', 'x', 'o'}): 1, frozenset({'p', 'y', 't', 'r', 'q', 'z', 'x'}): 1, frozenset({'s', 'm', 'y', 'e', 't', 'q', 'z', 'x'}): 1}

simDat = loadDataSet()
# print(simDat)
initSet = createInitSet(simDat)
# print(initSet)
myFPtree,myheaderTab = createTree(initSet,3)
myFPtree.disp()