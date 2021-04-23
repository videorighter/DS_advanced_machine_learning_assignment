# -*- coding: utf-8 -*-

# DO NOT CHANGE
import pandas as pd
import numpy as np

def get_order(structure):
    # structure: dictionary of structure
    #            key is variable and value is parents of a variable
    # return list of learning order of variables 
    # ex) ['A', 'R', 'E', 'S', 'T', 'O']
    structure_copy = structure.copy()
    var_order = [x for x in structure if structure[x]==[]]
    for i in var_order:
        structure_copy.pop(i)
    for n, i in enumerate(structure):
        for j in structure_copy:
            if set(structure_copy[j]).issubset(set(var_order)):
                if j not in var_order:
                    var_order.append(j)
    return var_order

def learn_parms(data,structure,var_order):
    # data: training data
    # structure: dictionary of structure
    # var_order: list of learning order of variables
    # return dictionary of trained parameters (key=variable, value=learned parameters)
    parms = {}
    tmp = {}
    new_structure = {}

    for var in var_order:
        new_structure[var] = structure[var]

    for var in var_order:
        if new_structure[var] == []:
            tmp[var] = [] # parents가 없는 경우 key에 빈 value 추가
        else:
            tmp[var] = [] # parents가 있는 경우 key에 빈 value 추가
            for elem in new_structure[var]:
                tmp[var]+=tmp[elem]
                if elem in tmp:
                    tmp[var].append(elem)
                new_value = []
                for v in tmp[var]:
                    if v not in new_value:
                        new_value.append(v)
                tmp[var] = new_value
    print("result tmp: ", tmp)

    # var_order = ['A', 'S', 'E', 'O', 'R', 'T']
    # str1 = {'A':[],'S':[],'E':['A','S'],'O':['E'],'R':['E'],'T':['O','R']}

    # tmp = {'A': [], 'S': [], 'E': ['A', 'S'], 'O': ['A', 'S', 'E'], 
    # 'R': ['A', 'S', 'E'], 'T': ['A', 'S', 'E', 'O', 'R']}
    
    for var in new_structure:
        parents = tmp[var]
        if parents == []:
            parms[var] = data.groupby(var).count()/len(data)
        else:
            unique_list = data[var].unique()
            df = data[parents]
            for unique in unique_list:
                df[unique] = (data[var] == unique)
                parms[var] = df.groupby(parents).sum()/len(data)

    return parms
                
def print_parms(var_order,parms):
    # var_order: list of learning order of variables
    # parms: dictionary of trained parameters (key=variable, value=learned parameters)
    # print the trained parameters for each variable
    for var in var_order:
        print('------------------------------------')
        print('Variable Name=%s'%(var))
        #TODO: print the trained paramters
        # for i in parms[var]:
        if parms[var].index.name:
            for i in parms[var].index:
                print('%-10s'%i, end='')
            print('\t')
            for i in list(data[var].value_counts(normalize = True)):
                print('%-10s'%i, end='')
            print('\t')
        else:
            print(parms[var])

    
data=pd.read_csv('./survey.txt', sep=' ')
str1={'A':[],'S':[],'E':['A','S'],'O':['E'],'R':['E'],'T':['O','R']} # 각 변수의 parents가 누군지 정의
order1=get_order(str1)
parms1=learn_parms(data,str1,get_order(str1))
print('----------First Structure----------')
print_parms(order1,parms1)

str2={'A':['E'],'S':['A','E'],'E':['O','R'],'O':['R','T'],'R':['T'],'T':[]}
order2=get_order(str2)
parms2=learn_parms(data,str2,get_order(str2))
print('----------Second Structure----------')
print_parms(order2,parms2)
print(parms2)
print('')