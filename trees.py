'''
Created on July 8, 2019

@author: Terry
@email：terryluohello@qq.com
'''
from math import log
import operator
import pickle

def calcShannonEnt(dataset):
    """ 计算给定数集的香农熵

    INPUT：
        dataset：数据集，形式为[[,,],[,,],[,,]]
    OUPUT： 
        shannonEnt: 香农熵 
    """
    # 计算list的长度，表示计算参与训练的数据量
    numEntries = len(dataset)
    # 计算分类标签label出现的次数
    labelCounts = {}

    for featVec in dataset:
        # 数据集[[,,],[,,],[,,]]的最后一个元素为label
        currentLabel = featVec[-1]
        # 为所有可能的分类创建字典，如果键值不存在，则扩展字典
        # 每个键值记录当前类别出现的次数
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

        # 根据label标签的占比计算出label的香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        # 利用label的频率计算估计概率
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

def spiltDataSet(dataset,axis,value):
    """ 按照给定的特征划分数据集

    INPUT：
        dataset：待划分的数据集，形式为[[,,],[,,],[,,]]
        axis: 划分数据集的特征
        value: 特征的返回值
    OUPUT： 
        retDataSet: 划分后的数据集 
    """
    retDataSet = []
    for featVec in dataset:
        # 因为参考书中特征值只有1和0，因此判断是否等于
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            # 注意 extend 和 append 的区别
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
        return retDataSet

def chooseBestFeatureToSplit(dataset):
    """ 选择修好的数据集划分方式

    INPUT：
        dataset：待划分的数据集，形式为[[,,],[,,],[,,]]
    OUPUT： 
        bestFeature: 最优的特征列 
    """
    # 求第一行有多少列 Feature , 最后一列是label
    numFeatures = len(dataset[0]) - 1
    # 数据集的原始信息熵
    baseEntropy = calcShannonEnt(dataset)
    # 初始化最优的信息增益值和Feature编号
    bestInfoGain, bestFeature = 0.0, -1

    # 循环处理所有的特征
    for i in range(numFeatures):
        # 获取对应feature下的所有数据
        featList = [example[i] for example in dataset]
        # 使用set对list进行去重
        uniqueVals = set(featList)
        # 创建一个临时的信息熵
        newEntropy = 0.0
        # 遍历某一列value集合，计算该列的信息熵
        for value in uniqueVals:
            subDataSet = spiltDataSet(dataset, i ,value)
            # 由频率估计概率
            prob = len(subDataSet)/float(len(dataset))
            # 计算信息熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 比较所有特征的信息增益，返回最好的划分特征
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestFeature):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
                             key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def creatTree(dataset, labels):
    """ 递归创建结构树

    INPUT：
        dataset：待划分的数据集，形式为[[,,],[,,],[,,]]
        labels: 标签
    OUPUT： 
        myTree: 生成的决策树 
    """  
    classList = [example[-1] for example in dataset]

    # 如果数据集的最后一列的第一个值的出现次数==整个集合的数量
    # 表示只有一个类别，直接返回结果
    # 第一个停止条件：所有类的标签完全相同，则直接返回该类别标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 第二个停止条件：使用完了所有特征，仍不能将数据集划分为仅包含唯一类别的分组
    if len(dataset[0]) == 1:
        return majorityCnt(classList)

    # 选择最优的列，得到对应的label含义
    bestFeat = chooseBestFeatureToSplit(dataset)
    # 获取label名称
    bestFeatLabel = labels[bestFeat]
    # 初始化mytree
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # 取出最优列，然后他的branch分类
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的label
        sublabels = labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用creatTree()
        myTree[bestFeatLabel][value] = creatTree(spiltDataSet(dataset,bestFeat,value),sublabels)
    return myTree 

def classify(inputTree, featLabels, testVec):
    """ 使用决策树执行分类

    INPUT：
        inputTree：决策树模型
        featLabels：feature标签对应的名称
        testVec：测试的输入数据
    OUPUT： 
        classLabel: 分类的结果值，需要映射label才能知道名称 
    """ 
    # 获取tree的根节点对应的key值
    firstStr = inputTree.keys()[0]
    # 通过key得到根节点对应的value
    secondDict = inputTree[firstStr]
    # 判断根节点名称获取根节点在label的先后顺序，这样就知道注入的testVec怎么开始对照树开始分类
    featIndex = featLabels.index(firstStr)
    # 测试数据，找到根节点对应的label位置，知道输入数据从第几位开始分类
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]

    print("+++",firstStr,"xxx",secondDict,"---",key,">>>",valueOfFeat)
    # 判断分支是否结束：判断valueOfFeat是否是dict类型
    if isinstance(valueOfFeat,dict):
        classLabel = classify(valueOfFeat,featLabels,testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    """ 储存决策树模型

    INPUT：
        inputTree：决策树模型
        filename：保存的文件名字
    OUPUT： 
        无
    """ 
    fw = open(filename,'w')
    pickle.dump(inputTree, fw)
    fw.close

def grabTree(filename):
    """ 读取决策树模型

    INPUT：
        filename：保存的文件名字
    OUPUT： 
        pickle.load(fr):读取的决策树模型
    """ 
    fr = open(filename)
    return pickle.load(fr)