from numpy import *

import tkinter
tkinter._test()

a = [[1,2,3],
	[4,5,6],
	[7,8,9],]
b = matrix(a)
# print(b[:,1] > 4)
# print(nonzero(b[:,1] > 4))
# print(nonzero(b[:,1] > 4)[0])

print(var(b[:,-1]) * shape(b)[0])

print(b[:,-1])
print(len(set(b[:,-1].T.tolist()[0])))


m,n = shape(b)
X = mat(ones((m,n)))
print(n)
print(X)

X[:,0:1] = b[:,0:1]
 
print(X)

print(ones((1,5))[:,1:2])

def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = list(map(float,curline))
		fltLine = [float(item) for item in curLine]
		dataMat.append(fltLine)
	return dataMat


a = [[ 1.15]]
b = [[43.41251481],[ 6.37966738]]
a = [[1 ,4]]
b = [[ 68.87014372],[-11.78556471]]
c = [[21.72788489]]
print(mat(a) * mat(b))