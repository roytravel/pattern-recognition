# -*- coding:utf-8 -*-
"""
Requirement
- 와인 데이터셋의 feature를 보고서에 정리
- pandas의 scatter_matrix나 seaborn의 pairplot 함수를 이용해서 피처를 2개씩 짝지어 산포도를 출력
- pairplot을 바탕으로 육안상 분리도가 높은 특징을 2개 선택해서 제시.
- 학습70%, 검증30%로 사용.
- Logistic Regression, KNN, SVM 모델을 학습시키고 분류 정확도를 출력.
- 분류 모델별 가장 높은 분류 정확도를 보인 feature 쌍을 제시.
"""

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import pandas_profiling
from xgboost import XGBRegressor
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import metrics
from sklearn.datasets import load_wine
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from operator import itemgetter
from mlxtend.plotting import plot_decision_regions
import mglearn
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from mglearn.datasets import make_forge
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


class Preprocessing(object):
    def __init__(self):
        pass

    def profile(self):
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        sy = pd.Series(wine.target, dtype="category")
        sy = sy.cat.rename_categories(wine.target_names)
        df['class'] = sy

        pr = df.profile_report()
        pr.to_file(output_file = './pr_report.html')


class Model(object):

    def __init__(self):
        self.wine = load_wine()
        
        # print (wine['feature_names'])
        # print ("[+] class name : {}".format(self.wine['target_names']))

        self.df = pd.DataFrame(self.wine.data, columns=self.wine.feature_names)
        sy = pd.Series(self.wine.target, dtype="category")
        sy = sy.cat.rename_categories(self.wine.target_names)
        self.df['class'] = sy
        
        self.x_data = self.df.iloc[:,:-1]
        self.y_data = self.df.iloc[:,-1]

        self.x_train, self.x_test, self.y_train, self.y_test = model_selection.train_test_split(self.x_data, self.y_data, test_size=0.3)


    def logistic_regression(self):
        model = LogisticRegression(penalty='l2', C=0.1, max_iter=1000000, tol=1e-4)
        model.fit(self.x_train, self.y_train)
        y_predict = model.predict(self.x_train) 
        train_score = metrics.accuracy_score(self.y_train, y_predict)
        print ("[*] Logistic Regression")
        print("  [+] Train Accuracy : {}".format(train_score))

        y_predict = model.predict(self.x_test) 
        test_score = metrics.accuracy_score(self.y_test, y_predict)
        print("  [+] Test Accuracy : {}\n".format(test_score))

        imp_variables = {}
        importance = model.coef_[0]
        for i,v in enumerate(importance):
            imp_variables[self.wine.feature_names[i]] = v
        

        imp_variables = dict(sorted(imp_variables.items(), key=lambda x: x[1], reverse=True))
        print (imp_variables)
        top1, top2 = list(imp_variables)[0], list(imp_variables)[1]
        print (top1, top2)

        return top1, top2
        

        # plt.figure(figsize=[10,8])
        # mglearn.plots.plot_2d_classification(model, self.x_train, fill=True, eps=0.5, alpha=0.4)
        # mglearn.discrete_scatter(self.x_train, self.x_train, self.y_train)
        # imp_variables = sorted(imp_variables, key=itemgetter(''))
        # [print (key, ":", value) for (key, value) in sorted(imp_variables.items())]
        

        # plot feature importance
        # plt.scatter([x for x in range(len(importance))], importance)
        # plt.show()

        
    def liner_svc(self):
        scaler = MinMaxScaler()
        scaler.fit(self.x_data)
        x_data = scaler.transform(self.x_data)
        model = LinearSVC(C=0.1, random_state=1, max_iter=5000000) 
        model.fit(self.x_train, self.y_train)

        y_predict = model.predict(self.x_train) 
        score = metrics.accuracy_score(self.y_train, y_predict)
        print ("[*] Linear SVC")
        print("  [+] Train Accuracy : {}".format(score))

        y_predict = model.predict(self.x_test) 
        score = metrics.accuracy_score(self.y_test, y_predict)
        print("  [+] Test Accuracy : {}\n".format(score))
        


    def kneighbor_classifier(self):
        scaler = MinMaxScaler()
        scaler.fit(self.x_data)
        x_data = scaler.transform(self.x_data)

        model = KNeighborsClassifier(n_neighbors=3, metric='minkowski', weights='distance', n_jobs=-1) # -1 :모든 코어 사용
        model.fit(self.x_train, self.y_train)

        y_predict = model.predict(self.x_train) 
        score = metrics.accuracy_score(self.y_train, y_predict)
        print ("[*] KNeighbor Classifier")
        print("  [+] Train Accuracy : {}".format(score))

        y_predict = model.predict(self.x_test) 
        score = metrics.accuracy_score(self.y_test, y_predict)
        print("  [+] Test Accuracy : {}\n".format(score))

        plot_decision_regions(self.x_train, self.y_train, clf=model, legend=2)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        


    def deicisiontree_classifier(self):
        print ("[*] DecisionTree Classifier")
        model = DecisionTreeClassifier()
        model.fit(self.x_train, self.y_train)
        print ("  [+] Train Accuracy : {}".format(model.score(self.x_train, self.y_train)))
        print ("  [+] Test Accuracy: {}".format(model.score(self.x_test, self.y_test)))
        print ("  [+] Feature Importance")
        for idx in range(len(model.feature_importances_)):
            print ("    [-] {} : {}".format(self.wine.feature_names[idx], model.feature_importances_[idx]))

        # get importance
        importance = model.feature_importances_
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        plt.bar([x for x in range(len(importance))], importance)
        plt.show()



    def randomforest_classifier(self):
        print ("[*] RandomForest Classifier")
        model = RandomForestClassifier(n_estimators=3)
        model.fit(self.x_train, self.y_train)
        print ("  [+] Train Accuracy : {}".format(model.score(self.x_train, self.y_train)))
        print ("  [+] Test Accuracy: {}\n".format(model.score(self.x_test, self.y_test)))

        importance = model.feature_importances_
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        plt.bar([x for x in range(len(importance))], importance)
        plt.show()

    
    def xgboost_regressor(self):
        

        model = XGBRegressor()
        model.fit(self.x_train, self.y_train)

        y_predict = model.predict(self.x_train) 
        score = metrics.accuracy_score(self.y_train, y_predict)
        print ("[*] XGBoost regressor")
        print("  [+] Train Accuracy : {}".format(score))

        y_predict = model.predict(self.x_test) 
        score = metrics.accuracy_score(self.y_test, y_predict)
        print("  [+] Test Accuracy : {}\n".format(score))


        # get importance
        importance = model.feature_importances_
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        plt.bar([x for x in range(len(importance))], importance)
        plt.show()


    def ensemble(self):
        model = VotingClassifier(estimators=[('lr', LogisticRegression(penalty='l2', C=0.1, max_iter=1000000, tol=0.0001)),
                                             ('kn', KNeighborsClassifier(n_neighbors=3, metric='minkowski', weights='distance')),
                                             ('lsvc', LinearSVC(C=0.1, random_state=1, max_iter=5000000))], voting='hard', weights=None)
        model.fit(self.x_train, self.y_train)

        y_predict = model.predict(self.x_train)
        score = metrics.accuracy_score(self.y_train, y_predict)
        print ("[*] Ensemble")
        print("  [+] Train Accuracy : {}".format(score))

        y_predict = model.predict(self.x_test) 
        score = metrics.accuracy_score(self.y_test, y_predict)
        print("  [+] Test Accuracy : {}\n".format(score))



class Visualizer(object):

    def __init__(self, df):
        self.df = df
        self.target = self.df['class']


    def get_colors(self):
        color_wheel = {"class_0": "#0392cf", "class_1": "#7bc043", "class_2": "#ee4035"}
        colors = []

        for i in range(len(self.target)):
            colors.append(color_wheel[self.target[i]])
        
        return np.array(colors)

    def heatmap(self):

        mask = np.zeros_like(self.df.corr(), dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        plt.figure(figsize=(10, 10))
        sns.heatmap(self.df.corr(), color="k", annot=True, cbar_kws = {"shrink": .5})
        plt.legend()
        plt.show()


    def pd_scatter(self):
        pd.plotting.scatter_matrix(self.df, color=self.get_colors(), diagonal='kde', alpha=0.85, grid=True)

        # plt.draw()
        plt.savefig('./wine.png', dpi=200)
        plt.legend()
        plt.show()
        


    def sns_scatter(self):
        
        sns.pairplot(self.df, diag_kind='kde', hue="class", palette='magma_r')
        plt.legend()
        plt.show()

    def bset_scatter(self):
        plt.legend()
        plt.show()
    
    

def result(top1, top2):

    wine = load_wine()

    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    sy = pd.Series(wine.target, dtype="category")
    sy = sy.cat.rename_categories(wine.target_names)
    df['class'] = sy
    
    new_df = pd.DataFrame([df[top1], df[top2]])
    new_df = new_df.transpose()
    
    x_train, x_test, y_train, y_test = model_selection.train_test_split(new_df, wine.target, test_size=0.3)
    x_train = x_train.values
    x_test = x_test.values

    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ['darkorange', 'c', 'darkblue']

    h = .02  # step size in the mesh
    n_neighbors = 15
    clf = KNeighborsClassifier(n_neighbors, weights='distance')
    clf.fit(x_train, y_train)

    
       
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    
    Z = Z.reshape(xx.shape)
    # plt.figure(figsize=(8, 6))
    plt.subplot(121)
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    # Plot also the training points
    sns.scatterplot(x=x_train[:, 0], y=x_train[:, 1], hue=wine.target_names[y_train], palette=cmap_bold, alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    # plt.title("3-Class classification (k = %i, weights = '%s')"% (n_neighbors, 'distance'))
    plt.title("Training")
    plt.xlabel(top1)
    plt.ylabel(top2)
    
    # plt.show()


    x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
    y_min, y_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    # plt.figure(figsize=(8, 6))
    plt.subplot(122)
    plt.contourf(xx, yy, Z, cmap=cmap_light)


    
    # Plot also the test points
    sns.scatterplot(x=x_test[:, 0], y=x_test[:, 1], hue=wine.target_names[y_test], palette=cmap_bold, alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Test")
    plt.xlabel(top1)
    plt.ylabel(top2)
    plt.tight_layout()
    plt.show()    


def main():

    # 1. 모델 생성
    M = Model()
    top1, top2 = M.logistic_regression()
    # M.kneighbor_classifier()
    # M.liner_svc()
    # M.ensemble()
    # M.randomforest_classifier()
    # M.deicisiontree_classifier()
    


    # 2. 플롯 생성
    # V = Visualizer(M.df)
    # V.heatmap()
    # V.pd_scatter()
    # V.sns_scatter()
    
    # colors = df['class'].replace({'class_0':'red', 'class_1': 'blue', 'class_2':'green'})
    
    result(top1, top2)
    
    

if __name__ == "__main__":
    sys.exit(main())