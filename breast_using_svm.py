import numpy as np
from sklearn import preprocessing, neighbors,svm
from sklearn.model_selection import train_test_split
import pandas as pd

df=pd.read_csv(r"C:\Users\KRISH\Documents\breast-cancer-wisconsin.data.txt")
df.replace('?',-99999,inplace=True)
df.columns=['id','clump_thickness','uniform_cell_size','uniform_cell_shape','marginal_adhesion','single_epithelial_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','class']
df.drop(columns=['id'],inplace=True)

X=np.array(df.drop(columns=['class']))
y=np.array(df['class'])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

clf=svm.SVC()
clf.fit(X_train,y_train)
# THE ACCURACY SHOULD BE MORE THAN KNEAREST NEIGHBOURS AS SVM IS MORE EFFICIENT
accuracy=clf.score(X_test,y_test)
print(accuracy)

example_measures=np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
example_measures=example_measures.reshape(len(example_measures),-1)
prediction=clf.predict(example_measures)
print(prediction)