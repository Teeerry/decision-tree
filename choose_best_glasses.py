'''
Created on July 8, 2019

@author: Terry
@email：terryluohello@qq.com
'''
import trees
import treePlotter

def chooseBestGlass():
    fr = open('./data/lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age','presrcipt','astigmatic','tearRate']
    lensesTree = trees.creatTree(lenses,lensesLabels)
    treePlotter.createPlot(lensesTree)


if __name__ == "__main__":
    print(__doc__)
    chooseBestGlass()
