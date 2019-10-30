def getNumLeafs(myTree) :									#输入字典集合myTree
	numLeafs = 0
	firstStr = list(myTree.keys())[0]						#list(myTree.keys())[0]返回第一个键，python3.x中要先将字典转换成集合list才能使用索引
	secondDict = myTree[firstStr]							#获取树形结构首节点的各个子节点内容
	
	for key in secondDict.keys() :							#字典中是键值对组成，secondDict.keys()返回所有的键
		if type(secondDict[key]).__name__ == 'dict' :		#如果该节点为字典，此节点为判断节点，需要向下递归调用，如果不是字典，则说明该节点是叶子节点
			numLeafs += getNumLeafs(secondDict[key])
		else: numLeafs +=1
	return numLeafs

def getTreeDepth(myTree) :									#输入字典集合myTree
	maxDepth = 0
	firstStr = list(myTree.keys())[0]						#list(myTree.keys())[0]返回第一个键，python3.x中要先将字典转换成集合list才能使用索引
	secondDict = myTree[firstStr]							#获取树形结构首节点的各个子节点内容
	for key in secondDict.keys() :							#在子节点可能的键的取值循环
		if type(secondDict[key]).__name__ == 'dict' :		#如果该节点为字典，此节点为判断节点，需要向下递归调用，如果不是字典，则说明该节点是叶子节点
			thisDepth = 1+ getTreeDepth(secondDict[key])
		else: thisDepth = 1
		if thisDepth > maxDepth : maxDepth = thisDepth
	return maxDepth

def retrieveTree(i):
	listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
				   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head':{0:'no', 1: 'yes'}},1:'no'}}}}]
	return listOfTrees[i]

myTree = retrieveTree(0)

print(getNumLeafs(myTree))
print(getTreeDepth(myTree))