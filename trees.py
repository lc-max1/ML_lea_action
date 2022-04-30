#!/usr/bin/env python
# coding: utf-8

# In[1]:


from math import log
import operator


# # 简单的测试数据集

# In[2]:


def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'%notebook']]
    labels = ['no surfacing','flippers']
    return dataSet,labels


# # 计算香农熵

# In[3]:


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        #提取标签
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt


# # 按照给定特征划分数据集

# In[4]:


def splitDataSet(dataSet,axis,value):
    #axis:选取第axis个特征来划分数据集，value:特征的值
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            #选取axis位置前的特征
            reducedFeatVec = featVec[:axis]
            #选取axis位置后的特征
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
        


# # 利用香农熵筛选出最好的划分特征

# In[5]:


def chooseBestFeatureToSplit(dataSet):
    #最后一列是标签，不是特征
    numFeatures = len(dataSet[0])-1
    #计算初始熵
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        #将dataSet中的数据先按行依次放入example中，然后取得example中的example[i]元素，放入列表featList中
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        #计算剔除i特征后的熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy-newEntropy
        #若剔除i特征后信息混乱度减小，则说明i特征有用
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
        


# # 若判断到叶子所有的标签还不统一，则选取最多数标签作为当前标签

# In[6]:


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# # 现在开始创建决策树

# In[7]:


def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    #只剩下一个类别
    if classList.count(classList[0])==len(classList):
        return classList[0]
    #叶子标签不统一
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    #以字典存储决策树
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    #得到当前标签的所有值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        #由于python默认列表作为参数时以引用传递，所以不选择传递原始列表
        subLabels = labels[:]
        #递归构建决策树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


# # 使用决策树的分类函数

# In[ ]:


def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in list(secondDict.keys()):
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else: 
                classLabel = secondDict[key]
    return classLabel
        


# # 使用pickle模块存储决策树

# In[9]:


def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close
    
def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)


# # 利用隐形眼镜数据集进行测试

# In[ ]:





# # 将本文件转为.py文件

# In[10]:


try:
    get_ipython().system('jupyter nbconvert --to python trees.ipynb')
except:
    pass


# In[ ]:





# In[ ]:




