#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import *
import operator
from os import listdir


# In[2]:


def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels


# # 数据预处理

# In[3]:



def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    
    for line in arrayOLines:
        ##截取掉所有回车字符
        line = line.strip()
        ##以\t分割
        listFromLine = line.split('\t')
        ##存入矩阵第index行
        returnMat[index,:]=listFromLine[0:3]
        ##-1表示最后一列元素
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector
    


# # 特征值归一化

# In[4]:


def autoNorm(dataSet):
    #参数0代表从每列中选取最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = zeros(shape(dataSet))
    #获取行数
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals


# # 分类器针对约会网站测试

# In[5]:


def datingClassTest():
    hoRatio=0.1
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("分类器返回结果: %d, 真实结果: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print ("错误率: %f" % (errorCount/float(numTestVecs)))
    print (errorCount)
                                           
                                            


# # 算法实现

# In[6]:


def classify0(inX, dataSet, labels, k):
    #使用欧式距离
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# # 约会网站预测函数

# In[7]:


def classifyPerson():
    resultList = ['一点也不','有一点','非常']
    percentTats = float(input("玩视频游戏花费时间百分比？"))
    ffMiles = float(input("每年飞行距离？"))
    iceCream = float(input("每周吃多少公升冰淇淋？"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("你可能",resultList[classifierResult-1],"喜欢这个人")


# # 处理图像信息，将32 * 32矩阵转换为1 * 1024矩阵

# In[8]:


def imgvector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


# # 使用KNN识别手写数字

# In[9]:


def handwritingClassTest():
    #存放真实结果
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    #m行1024列，每行代表一个图像
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = imgvector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = imgvector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("识别结果为：%d，真实结果为：%d" %(classifierResult,classNumStr))
        if(classifierResult!=classNumStr):errorCount+=1.0
    print("\n总共错误数为：%d" % errorCount)
    print("\n错误率为：%f" % (errorCount/float(mTest)))
        


# In[10]:


try:
    get_ipython().system('jupyter nbconvert --to python kNN.ipynb')
except:
    pass


# In[ ]:





# In[ ]:




