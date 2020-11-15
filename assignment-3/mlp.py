import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd
import matplotlib.pyplot as plt;plt.rcdefaults()

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
df = pd.read_csv(url).dropna()
df = df.reset_index(drop='True')
df_train, df_test = train_test_split(df,test_size=0.2)
df2 = df_train.reset_index(drop=True)
df3 = df_test.reset_index(drop=True)
X_train = df2.iloc[:, :-1]
y_train = df2.iloc[:, -1]
X_test = df3.iloc[:, :-1]
y_test = df3.iloc[:, -1]


model_0_vs_accuracy = []
model_vs_accuracy = []
model_1_vs_accuracy = []
model_2_vs_accuracy = []
model_3_vs_accuracy = []
lr_0_vs_accuracy = []
lr_vs_accuracy = []
lr_1_vs_accuracy = []
lr_2_vs_accuracy = []
lr_3_vs_accuracy = []


#Learning rate = 0.1
#0 hidden layers
clf0 = MLPClassifier(hidden_layer_sizes=(), solver='sgd',learning_rate_init=0.1,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 0 hidden layers and learning rate 0.1 is {} %.".format(round(clf0.score(X_test,y_test)*100,2)))
model_0_vs_accuracy.append(round(clf0.score(X_test,y_test)*100,2))
lr_0_vs_accuracy.append(round(clf0.score(X_test,y_test)*100,2))


#1 hidden layer with 2 nodes --
clf = MLPClassifier(hidden_layer_sizes=(2,), solver='sgd',learning_rate_init=0.1,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 1 hidden layer with 2 nodes and learning rate 0.1 is {} %.".format(round(clf.score(X_test,y_test)*100,2)))
model_vs_accuracy.append(round(clf.score(X_test,y_test)*100,2))
lr_0_vs_accuracy.append(round(clf.score(X_test,y_test)*100,2))


#1 hidden layer with 6 nodes --
clf1 = MLPClassifier(hidden_layer_sizes=(6,),solver='sgd',learning_rate_init=0.1,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 1 hidden layer with 6 nodes and learning rate 0.1 is {} %.".format(round(clf1.score(X_test,y_test)*100,2)))
model_1_vs_accuracy.append(round(clf1.score(X_test,y_test)*100,2))
lr_0_vs_accuracy.append(round(clf1.score(X_test,y_test)*100,2))

#2 hidden layer with 2,3 nodes --
clf2 = MLPClassifier(hidden_layer_sizes=(2,3), solver='sgd',learning_rate_init=0.1,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 2 hidden layers with 2 and 3 nodes respectively and learning rate 0.1 is {} %.".format(round(clf2.score(X_test,y_test)*100,2)))
model_2_vs_accuracy.append(round(clf2.score(X_test,y_test)*100,2))
lr_0_vs_accuracy.append(round(clf2.score(X_test,y_test)*100,2))

#2 hidden layer with 3,2 nodes --
clf3 = MLPClassifier(hidden_layer_sizes=(3,2), solver='sgd',learning_rate_init=0.1,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 2 hidden layers with 3 and 2 nodes respectively and learning rate 0.1 is {} %.".format(round(clf3.score(X_test,y_test)*100,2)))
model_3_vs_accuracy.append(round(clf3.score(X_test,y_test)*100,2))
lr_0_vs_accuracy.append(round(clf3.score(X_test,y_test)*100,2))



#Learning rate = 0.01

#0 hidden layers
clf0 = MLPClassifier(hidden_layer_sizes=(), solver='sgd',learning_rate_init=0.01,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 0 hidden layers and learning rate 0.01 is {} %.".format(round(clf0.score(X_test,y_test)*100,2)))
model_0_vs_accuracy.append(round(clf0.score(X_test,y_test)*100,2))
lr_vs_accuracy.append(round(clf0.score(X_test,y_test)*100,2))

#1 hidden layer with 2 nodes --
clf = MLPClassifier(hidden_layer_sizes=(2,), solver='sgd',learning_rate_init=0.01,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 1 hidden layer with 2 nodes and learning rate 0.01 is {} %.".format(round(clf.score(X_test,y_test)*100,2)))
model_vs_accuracy.append(round(clf.score(X_test,y_test)*100,2))
lr_vs_accuracy.append(round(clf.score(X_test,y_test)*100,2))

#1 hidden layer with 6 nodes --
clf1 = MLPClassifier(hidden_layer_sizes=(6,),solver='sgd',learning_rate_init=0.01,max_iter=2000).fit(X_train,y_train)
print("The accuracy of model with 1 hidden layer with 6 nodes and learning rate 0.01 is {} %.".format(round(clf1.score(X_test,y_test)*100,2)))
model_1_vs_accuracy.append(round(clf1.score(X_test,y_test)*100,2))
lr_vs_accuracy.append(round(clf1.score(X_test,y_test)*100,2))

#2 hidden layer with 2,3 nodes --
clf2 = MLPClassifier(hidden_layer_sizes=(2,3), solver='sgd',learning_rate_init=0.01,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 2 hidden layers with 2 and 3 nodes respectively and learning rate 0.01 is {} %.".format(round(clf2.score(X_test,y_test)*100,2)))
model_2_vs_accuracy.append(round(clf2.score(X_test,y_test)*100,2))
lr_vs_accuracy.append(round(clf2.score(X_test,y_test)*100,2))

#2 hidden layer with 3,2 nodes --
clf3 = MLPClassifier(hidden_layer_sizes=(3,2), solver='sgd',learning_rate_init=0.01,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 2 hidden layers with 3 and 2 nodes respectively and learning rate 0.01 is {} %.".format(round(clf3.score(X_test,y_test)*100,2)))
model_3_vs_accuracy.append(round(clf3.score(X_test,y_test)*100,2))
lr_vs_accuracy.append(round(clf3.score(X_test,y_test)*100,2))


#Learning rate = 0.001

#0 hidden layers
clf0 = MLPClassifier(hidden_layer_sizes=(), solver='sgd',learning_rate_init=0.001,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 0 hidden layers and learning rate 0.001 is {} %.".format(round(clf0.score(X_test,y_test)*100,2)))
model_0_vs_accuracy.append(round(clf0.score(X_test,y_test)*100,2))
lr_1_vs_accuracy.append(round(clf0.score(X_test,y_test)*100,2))

#1 hidden layer with 2 nodes --
clf = MLPClassifier(hidden_layer_sizes=(2,), solver='sgd',learning_rate_init=0.001,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 1 hidden layer with 2 nodes and learning rate 0.001 is {} %.".format(round(clf.score(X_test,y_test)*100,2)))
model_vs_accuracy.append(round(clf.score(X_test,y_test)*100,2))
lr_1_vs_accuracy.append(round(clf.score(X_test,y_test)*100,2))



#1 hidden layer with 6 nodes --
clf1 = MLPClassifier(hidden_layer_sizes=(6,),solver='sgd',learning_rate_init=0.001,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 1 hidden layer with 6 nodes and learning rate 0.001 is {} %.".format(round(clf1.score(X_test,y_test)*100,2)))
model_1_vs_accuracy.append(round(clf1.score(X_test,y_test)*100,2))
lr_1_vs_accuracy.append(round(clf1.score(X_test,y_test)*100,2))

#2 hidden layer with 2,3 nodes --
clf2 = MLPClassifier(hidden_layer_sizes=(2,3), solver='sgd',learning_rate_init=0.001,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 2 hidden layers with 2 and 3 nodes respectively and learning rate 0.001 is {} %.".format(round(clf2.score(X_test,y_test)*100,2)))
model_2_vs_accuracy.append(round(clf2.score(X_test,y_test)*100,2))
lr_1_vs_accuracy.append(round(clf2.score(X_test,y_test)*100,2))

#2 hidden layer with 3,2 nodes --
clf3 = MLPClassifier(hidden_layer_sizes=(3,2), solver='sgd',learning_rate_init=0.001,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 2 hidden layers with 3 and 2 nodes respectively and learning rate 0.001 is {} %.".format(round(clf3.score(X_test,y_test)*100,2)))
model_3_vs_accuracy.append(round(clf3.score(X_test,y_test)*100,2))
lr_1_vs_accuracy.append(round(clf3.score(X_test,y_test)*100,2))

#Learning rate = 0.0001
#0 hidden layers
clf0 = MLPClassifier(hidden_layer_sizes=(), solver='sgd',learning_rate_init=0.0001,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 0 hidden layers and learning rate 0.0001 is {} %.".format(round(clf0.score(X_test,y_test)*100,2)))
model_0_vs_accuracy.append(round(clf0.score(X_test,y_test)*100,2))
lr_2_vs_accuracy.append(round(clf0.score(X_test,y_test)*100,2))

#1 hidden layer with 2 nodes --
clf = MLPClassifier(hidden_layer_sizes=(2,), solver='sgd',learning_rate_init=0.0001,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 1 hidden layer with 2 nodes and learning rate 0.0001 is {} %.".format(round(clf.score(X_test,y_test)*100,2)))
model_vs_accuracy.append(round(clf.score(X_test,y_test)*100,2))
lr_2_vs_accuracy.append(round(clf.score(X_test,y_test)*100,2))

#1 hidden layer with 6 nodes --
clf1 = MLPClassifier(hidden_layer_sizes=(6,),solver='sgd',learning_rate_init=0.0001,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 1 hidden layer with 6 nodes and learning rate 0.0001 is {} %.".format(round(clf1.score(X_test,y_test)*100,2)))
model_1_vs_accuracy.append(round(clf1.score(X_test,y_test)*100,2))
lr_2_vs_accuracy.append(round(clf1.score(X_test,y_test)*100,2))

#2 hidden layer with 2,3 nodes --
clf2 = MLPClassifier(hidden_layer_sizes=(2,3), solver='sgd',learning_rate_init=0.0001,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 2 hidden layers with 2 and 3 nodes respectively and learning rate 0.0001 is {} %.".format(round(clf2.score(X_test,y_test)*100,2)))
model_2_vs_accuracy.append(round(clf2.score(X_test,y_test)*100,2))
lr_2_vs_accuracy.append(round(clf2.score(X_test,y_test)*100,2))
#2 hidden layer with 3,2 nodes --
clf3 = MLPClassifier(hidden_layer_sizes=(3,2), solver='sgd',learning_rate_init=0.0001,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 2 hidden layers with 3 and 2 nodes respectively and learning rate 0.0001 is {} %.".format(round(clf3.score(X_test,y_test)*100,2)))
model_3_vs_accuracy.append(round(clf3.score(X_test,y_test)*100,2))
lr_2_vs_accuracy.append(round(clf3.score(X_test,y_test)*100,2))

#Learning Rate 0.00001

#0 hidden layers
clf0 = MLPClassifier(hidden_layer_sizes=(), solver='sgd',learning_rate_init=0.00001,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 0 hidden layers and learning rate 0.00001 is {} %.".format(round(clf0.score(X_test,y_test)*100,2)))
model_0_vs_accuracy.append(round(clf0.score(X_test,y_test)*100,2))
lr_3_vs_accuracy.append(round(clf0.score(X_test,y_test)*100,2))

#1 hidden layer with 2 nodes --
clf = MLPClassifier(hidden_layer_sizes=(2,), solver='sgd',learning_rate_init=0.00001,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 1 hidden layer with 2 nodes and learning rate 0.00001 is {} %.".format(round(clf.score(X_test,y_test)*100,2)))
model_vs_accuracy.append(round(clf.score(X_test,y_test)*100,2))
lr_3_vs_accuracy.append(round(clf.score(X_test,y_test)*100,2))

#1 hidden layer with 6 nodes --
clf1 = MLPClassifier(hidden_layer_sizes=(6,),solver='sgd',learning_rate_init=0.00001,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 1 hidden layer with 6 nodes and learning rate 0.00001 is {} %.".format(round(clf1.score(X_test,y_test)*100,2)))
model_1_vs_accuracy.append(round(clf1.score(X_test,y_test)*100,2))
lr_3_vs_accuracy.append(round(clf1.score(X_test,y_test)*100,2))

#2 hidden layer with 2,3 nodes --
clf2 = MLPClassifier(hidden_layer_sizes=(2,3), solver='sgd',learning_rate_init=0.00001,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 2 hidden layers with 2 and 3 nodes respectively and learning rate 0.00001 is {} %.".format(round(clf2.score(X_test,y_test)*100,2)))
model_2_vs_accuracy.append(round(clf2.score(X_test,y_test)*100,2))
lr_3_vs_accuracy.append(round(clf2.score(X_test,y_test)*100,2))

#2 hidden layer with 3,2 nodes --
clf3 = MLPClassifier(hidden_layer_sizes=(3,2), solver='sgd',learning_rate_init=0.00001,max_iter=2000,random_state=2).fit(X_train,y_train)
print("The accuracy of model with 2 hidden layers with 3 and 2 nodes respectively and learning rate 0.00001 is {} %.".format(round(clf3.score(X_test,y_test)*100,2)))
model_3_vs_accuracy.append(round(clf3.score(X_test,y_test)*100,2))
lr_3_vs_accuracy.append(round(clf3.score(X_test,y_test)*100,2))

algo_names = ["()","(2,)","(6,)","(2,3)","(3,2)"]
lr_names = ["0.1","0.01","0.001","0.0001","0.00001"]

y_pos = np.arange(len(algo_names))
plt.barh(y_pos, lr_0_vs_accuracy, align='center', alpha=0.5)
plt.yticks(y_pos, algo_names)
plt.xlabel('Accuracy')
plt.ylabel("Hidden Layer size")
plt.title('Model vs accuracy for Learning rate 0.1')
plt.savefig("lr_0.1.png")
plt.close()

y_pos = np.arange(len(algo_names))
plt.barh(y_pos, lr_vs_accuracy, align='center', alpha=0.5)
plt.yticks(y_pos, algo_names)
plt.xlabel('Accuracy')
plt.ylabel("Hidden Layer size")
plt.title('Model vs accuracy for Learning rate 0.01')
plt.savefig("lr_0.01.png")
plt.close()


y_pos = np.arange(len(algo_names))
plt.barh(y_pos, lr_1_vs_accuracy, align='center', alpha=0.5)
plt.yticks(y_pos, algo_names)
plt.xlabel('Accuracy')
plt.ylabel("Hidden Layer size")
plt.title('Model vs accuracy for Learning rate 0.001')
plt.savefig("lr_0.001.png")
plt.close()


y_pos = np.arange(len(algo_names))
plt.barh(y_pos, lr_2_vs_accuracy, align='center', alpha=0.5)
plt.yticks(y_pos, algo_names)
plt.xlabel('Accuracy')
plt.ylabel("Hidden Layer size")
plt.title('Model vs accuracy for Learning rate 0.0001')
plt.savefig("lr_0.0001.png")
plt.close()


y_pos = np.arange(len(algo_names))
plt.barh(y_pos, lr_3_vs_accuracy, align='center', alpha=0.5)
plt.yticks(y_pos, algo_names)
plt.xlabel('Accuracy')
plt.ylabel("Hidden Layer size")
plt.title('Model vs accuracy for Learning rate 0.00001')
plt.savefig("lr_0.00001.png")
plt.close()



y_pos = np.arange(len(lr_names))
plt.barh(y_pos, model_0_vs_accuracy, align='center', alpha=0.5)
plt.yticks(y_pos, lr_names)
plt.xlabel('Accuracy')
plt.ylabel("Learning Rate")
plt.title('Learning rate vs accuracy for model with 0 hidden layers')
plt.tight_layout()
plt.savefig("0hidden.png")
plt.close()



y_pos = np.arange(len(lr_names))
plt.barh(y_pos, model_vs_accuracy, align='center', alpha=0.5)
plt.yticks(y_pos, lr_names)
plt.xlabel('Accuracy')
plt.ylabel("Learning Rate")
plt.title('Learning rate vs accuracy for model with 1 hidden layer with 2 nodes')
plt.tight_layout()
plt.savefig("1hidden2.png")
plt.close()


y_pos = np.arange(len(lr_names))
plt.barh(y_pos, model_1_vs_accuracy, align='center', alpha=0.5)
plt.yticks(y_pos, lr_names)
plt.xlabel('Accuracy')
plt.ylabel("Learning Rate")
plt.title('Learning rate vs accuracy for model with 1 hidden layer with 6 nodes')
plt.tight_layout()
plt.savefig("1hidden6.png")
plt.close()

y_pos = np.arange(len(lr_names))
plt.barh(y_pos, model_2_vs_accuracy, align='center', alpha=0.5)
plt.yticks(y_pos, lr_names)
plt.xlabel('Accuracy')
plt.ylabel("Learning Rate")
plt.title('Learning rate vs accuracy for model with 2 hidden layers with 2 and 3 nodes')
plt.tight_layout()
plt.savefig("2hidden23.png")
plt.close()

y_pos = np.arange(len(lr_names))
plt.barh(y_pos, model_3_vs_accuracy, align='center',alpha= 0.5)
plt.yticks(y_pos, lr_names)
plt.xlabel('Accuracy')
plt.ylabel("Learning Rate")
plt.title('Learning rate vs accuracy for model with 2 hidden layers with 3 and 2 nodes')
plt.savefig("2hidden32.png")
plt.close()