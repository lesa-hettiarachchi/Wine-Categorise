#import libaries
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load wine dataset
from sklearn.datasets import load_wine
data = load_wine()

#conversion to pandas dataframe
df = pd.DataFrame(data.data,columns=data.feature_names)
df['target'] =data.target

#split data for x and y for test and train purposes
X = df.drop(columns=['target'])
y = df['target']

#Split data for train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=79)

#use criterion as entropy and train classifier
clf= DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train,y_train)

#Calculate Accuracy
accuracy = clf.score(X_test,y_test)
print(f"Accuracy of the Classification :{accuracy:0.4f}")

#Generate Plot tree
plt.figure(figsize=(30,20))
plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.savefig("output_graph.pdf")
plt.show()
