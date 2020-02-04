import monkdata as m
import dtree
import drawtree_qt5 as draw
import pandas as pd

# Assignment1 - Entropy
'''
The file dtree.py defines a function entropy which
calculates the entropy of a dataset. 
Import this file along with the monks datasets 
and use it to calculate the entropy of the training datasets.
'''

entropy_mock1 = dtree.entropy(m.monk1)
entropy_monk1test = dtree.entropy(m.monk1test)
entropy_mock2 = dtree.entropy(m.monk2)
entropy_monk2test = dtree.entropy(m.monk2test)
entropy_mock3 = dtree.entropy(m.monk3)
entropy_monk3test = dtree.entropy(m.monk3test)
labels = ['entropy']
entropy_dict = {
    'Monk1': [entropy_mock1],
    # 'Monk1test': [entropy_monk1test],
    'Monk2': [entropy_mock2],
    # 'Monk2test': [entropy_monk2test],
    'Monk3': [entropy_mock3],
    # 'Monk3test': [entropy_monk3test],
}
entropy_df = pd.DataFrame(entropy_dict, index=labels).T
print(entropy_df)

# Assignment3 - Information Gain
'''
Use the function averageGain (defined in dtree.py) to calculate 
the expected information gain corresponding to each of the six attributes. 
Note that the attributes are represented as instances of the class Attribute (defined in monkdata.py) 
which you can access via m.attributes[0], ..., m.attributes[5]. 
Based on the results, which attribute should be used for splitting the examples at the root node?
'''
averageGain_mock1 = []
averageGain_monk1test = []
averageGain_mock2 = []
averageGain_monk2test = []
averageGain_mock3 = []
averageGain_monk3test = []

for i in range(6):
    averageGain_mock1.append(dtree.averageGain(m.monk1, m.attributes[i]))
    averageGain_monk1test.append(dtree.averageGain(m.monk1test, m.attributes[i]))
    averageGain_mock2.append(dtree.averageGain(m.monk2, m.attributes[i]))
    averageGain_monk2test.append(dtree.averageGain(m.monk2test, m.attributes[i]))
    averageGain_mock3.append(dtree.averageGain(m.monk3, m.attributes[i]))
    averageGain_monk3test.append(dtree.averageGain(m.monk3test, m.attributes[i]))

labels = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
averageGain_dict = {
    'Monk1': averageGain_mock1,
    # 'Monk1test': averageGain_monk1test,
    'Monk2': averageGain_mock2,
    # 'Monk2test': averageGain_monk2test,
    'Monk3': averageGain_mock3,
    # 'Monk3test': averageGain_monk3test,
}
averageGain_df = pd.DataFrame(averageGain_dict, index=labels).T
print(averageGain_df)

# Assignment4 - Information Gain with choose one attribute
entropy_mock1 = []
entropy_mock2 = []
entropy_mock3 = []
for v in m.attributes[4].values:
    monk1_subset = dtree.select(m.monk1, m.attributes[4], v)
    entropy_mock1.append(dtree.entropy(monk1_subset))
for v in m.attributes[4].values:
    monk2_subset = dtree.select(m.monk2, m.attributes[4], v)
    entropy_mock2.append(dtree.entropy(monk2_subset))
for v in m.attributes[1].values:
    monk3_subset = dtree.select(m.monk3, m.attributes[1], v)
    entropy_mock3.append(dtree.entropy(monk3_subset))

labels = ['1', '2', '3', '4']
entropy_dict = {
    'Monk1': entropy_mock1,
    'Monk2': entropy_mock2,
}
entropy_df = pd.DataFrame(entropy_dict, index=labels).T
print(entropy_df)

labels = ['1', '2', '3']
entropy_dict = {
    'Monk3': entropy_mock3,
}
entropy_df = pd.DataFrame(entropy_dict, index=labels).T
print(entropy_df)

# Assignment5 - Building Decision Trees
'''
Build the full decision trees for all three Monk datasets using buildTree. 
Then, use the function check to measure the performance of the decision tree 
on both the training and test datasets
'''

def twoLevelTree(dataset, averageGain):
    select_attribute = []
    max_averageGain = 0
    max_index = 0
    for i in range(6):
        if max_averageGain < averageGain[i]:
            max_averageGain = averageGain[i]
            max_index = i
    select_attribute.append(max_index)
    for v in m.attributes[max_index].values:
        subset = dtree.select(dataset, m.attributes[max_index], v)
        if dtree.allPositive(subset):
            select_attribute.append('T')
        elif dtree.allNegative(subset):
            select_attribute.append('F')
        else:
            max_averageGain = 0
            new_averageGain = []
            second_max_index = 0
            for i in range(6):
                new_averageGain.append(dtree.averageGain(subset, m.attributes[i]))
            for i in range(6):
                if max_averageGain < new_averageGain[i]:
                    max_averageGain = new_averageGain[i]
                    second_max_index = i
            select_attribute.append(second_max_index)
            for v2 in m.attributes[second_max_index].values:
                nextSubset = dtree.select(subset, m.attributes[second_max_index], v2)
                if dtree.mostCommon(nextSubset):
                    select_attribute.append('T')
                else:
                    select_attribute.append('F')
    return select_attribute

def printTwoLevelAttribute(attribute):
    for i in range(len(attribute)):
        if type(attribute[i]) == int:
            attribute[i] = 'A' + str(attribute[i] + 1)
    print(attribute)
    
select_attribute = twoLevelTree(m.monk1, averageGain_mock1)
printTwoLevelAttribute(select_attribute)
select_attribute = twoLevelTree(m.monk2, averageGain_mock2)
printTwoLevelAttribute(select_attribute)
select_attribute = twoLevelTree(m.monk3, averageGain_mock3)
printTwoLevelAttribute(select_attribute)

# ID3 decision tree
monk1_t = dtree.buildTree(m.monk1, m.attributes)
monk2_t = dtree.buildTree(m.monk2, m.attributes)
monk3_t = dtree.buildTree(m.monk3, m.attributes)

labels = ['Train', 'Test']
accuracy_dict = {
    'Monk1': [dtree.check(monk1_t, m.monk1), dtree.check(monk1_t, m.monk1test)],
    'Monk2': [dtree.check(monk2_t, m.monk2), dtree.check(monk2_t, m.monk2test)],
    'Monk3': [dtree.check(monk3_t, m.monk3), dtree.check(monk3_t, m.monk3test)],
}
accuracy_df = pd.DataFrame(accuracy_dict, index=labels).T
print(accuracy_df)

# draw.drawTree(monk1_t)
# draw.drawTree(monk2_t)
# draw.drawTree(monk3_t)