import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
import pandas as pd
import random
style.use('fivethirtyeight')

def k_nearest_neighbour(data,predict , k=3):
    if len(data)>=k:
        warnings.warn(' K is set to a value less than total voting groups')
    distances=[]
    for group in data:
        for features in data[group]:
            euclidean_dist=np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_dist,group])
    
    votes=[i[1] for i in sorted(distances)[:k]]
    vote_result= Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k
    
    # print(vote_result,confidence)
    # print(distances)
    # print(votes)
    # print(Counter(votes).most_common())
    return vote_result,confidence
accuracies=[]

for i in range(25):
    df=pd.read_csv(r"C:\Users\KRISH\Documents\breast-cancer-wisconsin.data.txt")
    df.replace('?',-99999,inplace=True)
    df.columns=['id','clump_thickness','uniform_cell_size','uniform_cell_shape','marginal_adhesion','single_epithelial_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','class']
    df.drop(columns=['id'],inplace=True)
    full_data=df.astype(float).values.tolist()  # since some values are in string , we need to convert them to float

    random.shuffle(full_data)
    
    test_size=0.4
    train_set={2:[],4:[]}
    test_set={2:[],4:[]} # 2 and 4 are the classes
    train_data=full_data[:-int(test_size*len(full_data))]
    test_data=full_data[-int(test_size*len(full_data)):] #lass 20% of the data for testing
    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    for i in test_data:
        test_set[i[-1]].append(i[:-1])
    correct=0
    total=0
    for group in  test_set:
        for data in test_set[group]:
            
            vote,confidence=k_nearest_neighbour(train_set,data,k=5)
            if group==vote:
                correct+=1
            # else:
                # print(confidence)
            total+=1
    # print('Accuracy',correct/total)
    accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies))