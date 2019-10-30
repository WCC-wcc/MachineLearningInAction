from numpy import *
from numpy import linalg as la
# U,Sigma,VT = linalg.svd([[1,1],[7,7]]) 					#对输入矩阵进行SVD分解
# print(U)
# print(Sigma)
# print(VT)

def loadExData():
	return [[1,1,1,0,0],
			[2,2,2,0,0],
			[1,1,1,0,0],
			[5,5,5,0,0],
			[1,1,0,2,2],
			[0,0,0,3,3],
			[0,0,0,1,1]]


# Data = loadExData()
# U,Sigma,VT = linalg.svd(Data)
# # print(U)
# print(Sigma)
# sum1 = mat(Sigma) * mat(Sigma).T											#将奇异值矩阵中各个奇异值平方相加
# print(sum1) 																	
# sum2 = mat(Sigma[:2]) * mat(Sigma[:2]).T 									#将奇异值矩阵前n个奇异值平方相加，判断保留的奇异值个数是否满足要求
# print(sum2)																	#(确定保留的奇异值数目方法)保留矩阵90%的能量信息或者对于包含上万的奇异值时，保留前2000,3000个
# # print(VT)
# Sig3 = mat([[Sigma[0],0,0],													#重构矩阵  同原始矩阵近似
# 			[0,Sigma[1],0],
# 			[0,0,Sigma[2]]])
# reBuildRec = U[:,:3] * Sig3 * VT[:3,:]
# print(reBuildRec)

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

myMat = mat(loadExData())
print(euclidSim(myMat[:,0],myMat[:,4]))
print(euclidSim(myMat[:,0],myMat[:,0]))

print(cosSim(myMat[:,0],myMat[:,4]))
print(cosSim(myMat[:,0],myMat[:,0]))

print(pearsSim(myMat[:,0],myMat[:,4]))
print(pearsSim(myMat[:,0],myMat[:,0]))
