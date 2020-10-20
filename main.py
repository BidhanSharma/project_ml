import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

class Cleaning:
    
    def __init__(self):
        self.df = pd.read_csv('data.csv')

        # print(self.df.head(5))
        # print(self.df.shape)
        # print(self.df.isna().sum())

        '''dropping col w/ empty value'''
        self.df = self.df.dropna(axis=1)
        # print(self.df.shape)

        '''converts values of diagnosis col : B,M to 1,0'''
        # before converting
        # print(self.df.iloc[:,1].values)

        # after converting
        self.lblencoder_Y = LabelEncoder()
        # print(lblencoder_Y.fit_transform(self.df.iloc[:,1].values))

        # showing index and its value
        self.df.iloc[:,1] = self.lblencoder_Y.fit_transform(self.df.iloc[:,1].values)
        # print(self.df.iloc[:,1])

    def visualize_count(self):
        print(self.df['diagnosis'].value_counts())
        sns.countplot(x='diagnosis',data=self.df)
        plt.show()

    def pair_plot(self):
        sns.pairplot(self.df.iloc[:,1:6], hue='diagnosis')
        # sns.countplot(x='diagnosis', data=self.df)
        plt.show()
        
    def correlation(self):
        plt.figure(figsize=(10,10))
        sns.heatmap(self.df.iloc[:, 1:12].corr(),annot=True,fmt='.0%')
        plt.show()

class Model:

    def __init__(self):
        self.df = Cleaning().df
        # print(self.df)
        
        # splitting dataset into X and Y
        self.X = self.df.iloc[:,2:31].values #attributes col
        self.Y = self.df.iloc[:,1].values #diagnosis col

        # dataset : 75% training and 25% testing
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X,self.Y,test_size=0.25,random_state=0)

        # scaling data : X
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(self.X_train)
        self.X_test = self.sc.fit_transform(self.X_test)

        # print(self.X_train)

    def models(self,X_train,Y_train):
    
        # logistic Regression
        self.log = LogisticRegression(random_state=0)
        self.log.fit(self.X_train,self.Y_train)

        # KNeighbors Classifier
        self.knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
        self.knn.fit(self.X_train,self.Y_train)

        #Using SVC linear
        self.svc_lin = SVC(kernel = 'linear', random_state = 0)
        self.svc_lin.fit(self.X_train, self.Y_train)

        #Using SVC rbf
        self.svc_rbf = SVC(kernel = 'rbf', random_state = 0)
        self.svc_rbf.fit(self.X_train, self.Y_train)

        #Using GaussianNB 
        self.gauss = GaussianNB()
        self.gauss.fit(self.X_train, self.Y_train)

        #Using DecisionTreeClassifier 
        self.tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        self.tree.fit(self.X_train, self.Y_train)

        #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
        self.forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        self.forest.fit(self.X_train, self.Y_train)

        #print model accuracy on the training data.
        # print(f"Logistic Regression Training Accuracy - {self.log.score(self.X_train,self.Y_train)}")
        # print(f"K Nearest Neighbor Training Accuracy - {self.knn.score(self.X_train,self. Y_train)}")
        # print(f"Support Vector Machine (Linear Classifier) Training Accuracy - {self.svc_lin.score(self.X_train,self. Y_train)}")
        # print(f"Support Vector Machine (RBF Classifier) Training Accuracy - {self.svc_rbf.score(self.X_train,self. Y_train)}")
        # print(f"Gaussian Naive Bayes Training Accuracy - {self.gauss.score(self.X_train,self. Y_train)}")
        # print(f"Decision Tree Classifier Training Accuracy - {self.tree.score(self.X_train,self. Y_train)}")
        # print(f"Random Forest Classifier Training Accuracy - {self.forest.score(self.X_train,self. Y_train)}")

        return self.log, self.knn, self.svc_lin, self.svc_rbf, self.gauss, self.tree, self.forest

class Testing:

    def __init__(self):
        m = Model()
        self.xt = m.X_test # attributes, 
        self.yt = m.Y_test # result : (cancer / !cancer)
        self.model = m.models(m.X_train ,m.Y_train) # executes method models
        
        # checking model[0](logistic regression)
        # cm = confusion_matrix(self.yt, self.model[0].predict(self.xt))
        # print(cm)

        # checking all models
        # for i in range(len(self.model)):
        #     cm = confusion_matrix(self.yt, self.model[i].predict(self.xt))

        #     TP = cm[0][0]
        #     FP = cm[0][1]
        #     FN = cm[1][0]
        #     TN = cm[1][1]

        #     print(f"Model : {i}")
            
        #     # confusion matrix
        #     print(cm)

        #     # accuracy score
        #     print(f"Testing Accuracy : {(TP+TN)/(TP+TN+FN+FP)}");print()

    # testing classification report & accuracy score
    def testing(self):
        for i in range(len(self.model)):
            print(f"Model : {i}")

            # check precision , recall, f1-score
            print(classification_report(self.yt, self.model[i].predict(self.xt)))

            # accuracy score
            print(accuracy_score(self.yt, self.model[i].predict(self.xt)));print()
            
    # comparing model with result in dataset
    def comparing(self):
        pred = self.model[2].predict(self.xt)
        print(f"Prediction : \n{pred}");print()
        print(f"Dataset : \n{self.yt}")       

if __name__ == '__main__':

    '''cleaning data'''
    # c = Cleaning()
    # c.visualize_count()
    # c.pair_plot()
    # c.correlation()

    '''Create Model'''
    # m = Model()
    # model = m.models(m.X_train,m.Y_train)

    '''Testing Model'''
    t = Testing()
    # t.testing()
    t.comparing()

