from numpy import *
from numpy import linalg as la

#打印图像像素点
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1,end='')                                             #end默认是'\n'，所以如果不换行，将end设置为''或' ' 
            else: print(0,end='')
        print()

# @param numSV：奇异值数目，默认为3
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)
    U,Sigma,VT = la.svd(myMat)  
    SigRecon = mat(zeros((numSV, numSV)))                               
    for k in range(numSV):                                                  #构造对角阵，保留前numSV个奇异值
        SigRecon[k,k] = Sigma[k]                                            #                               A(m*n) = U(m*m) * Sigma(m*n) * VT(n*n)
    
    reconMat = U[:,:numSV] * SigRecon * VT[:numSV,:]                        #重构后的矩阵 (m,n)截断U、VT矩阵   A(m*n) ~= U(m*r) * Sigma(r*r) * VT(r*n)
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)

imgCompress(2) # 发现只需要保留两个奇异值，就可以得到近似的压缩图像