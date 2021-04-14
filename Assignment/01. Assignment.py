# -*- coding:utf-8 -*-
"""
- 보고서 정리: 와인 데이터셋 피처
- 산포도 출력: 피처 쌍 별
- 피처쌍 제시: 육안상 분리도가 높은 피처
- 비율: 학습 70%, 테스트 30%
- 시각화: 모델 별 가장 높은 분류 정확도를 낸 피처 제시 및 시각화
"""

import sys
import mglearn
import numpy as np
import pandas as pd
import seaborn as sns
import pandas_profiling
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn import model_selection
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



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
        # print (len(self.wine.feature_names))

        self.df = pd.DataFrame(self.wine.data, columns=self.wine.feature_names)
        sy = pd.Series(self.wine.target, dtype="category")
        sy = sy.cat.rename_categories(self.wine.target_names)
        self.df['class'] = sy
        self.x_data = self.df.iloc[:,:-1]
        self.y_data = self.df.iloc[:,-1]
        self.x_train, self.x_test, self.y_train, self.y_test = model_selection.train_test_split(self.x_data, self.y_data, test_size=0.3)


    def logistic_regression(self):
        y = self.wine.target
        max_score = [0, 0]
        for i in range(len(self.wine.feature_names)):
            for j in range(len(self.wine.feature_names)):
                if i!=j:
                    X = self.wine.data[:, [i, j]]
                    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=2021)

                    model = LogisticRegression(penalty='l2', C=0.1, max_iter=1000000, tol=1e-4)
                    model.fit(X_train, y_train)
                    y_predict = model.predict(X_train)
                    train_score = metrics.accuracy_score(y_train, y_predict)
                    # print ("[*] Logistic Regression")
                    # print ("  [+] Used Feature: {}/{}".format(self.wine.feature_names[i], self.wine.feature_names[j]))
                    # print ("  [+] Train Accuracy : {}".format(train_score))

                    y_predict = model.predict(X_test) 
                    test_score = metrics.accuracy_score(y_test, y_predict)
                    # print("  [+] Test Accuracy : {}".format(test_score))


                    imp_variables = {}
                    importance = model.coef_[0]
                    for i,v in enumerate(importance):
                        imp_variables[self.wine.feature_names[i]] = v

                    imp_variables = dict(sorted(imp_variables.items(), key=lambda x: x[1], reverse=True))
                    # print (imp_variables)
                    top1, top2 = list(imp_variables)[0], list(imp_variables)[1]
                    # print ("  [+] Best Two Featues : {}/{}\n".format(top1, top2))

                    if test_score > max_score[1]:
                        max_score = [train_score, test_score]
                        used_feature = [self.wine.feature_names[i], self.wine.feature_names[j]]
                        best_feature = [top1, top2]

        print ("[*] Logistic Regression")
        print ("[+] Used Feature : {}/{}".format(used_feature[0], used_feature[1]))
        print ("[+] Maximum Score : {}/{}".format(max_score[0],max_score[1]))
        print ("[+] Best Feature : {}/{}".format(best_feature[0], best_feature[1]))

        return model, used_feature, max_score, best_feature, X_train, X_test, y_train, y_test



    def kneighbor_classifier(self):
        y = self.wine.target
        max_score = [0, 0]
        for i in range(len(self.wine.feature_names)):
            for j in range(len(self.wine.feature_names)):
                if i!=j:
                    X = self.wine.data[:, [i, j]]
                    scaler = MinMaxScaler()
                    scaler.fit(X)
                    X = scaler.transform(X)
                    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=2021)

                    model = KNeighborsClassifier(n_neighbors=3, metric='minkowski', weights='distance', n_jobs=-1) # -1 :모든 코어 사용
                    model.fit(X_train, y_train)

                    y_predict = model.predict(X_train)
                    train_score = metrics.accuracy_score(y_train, y_predict)
                    # print ("[*] KNeighbor Classifier")
                    # print ("  [+] Used Feature: {}/{}".format(self.wine.feature_names[i], self.wine.feature_names[j]))
                    # print ("  [+] Train Accuracy : {}".format(train_score))

                    y_predict = model.predict(X_test) 
                    test_score = metrics.accuracy_score(y_test, y_predict)
                    # print("  [+] Test Accuracy : {}\n".format(test_score))

                    if test_score > max_score[1]:
                        max_score = [train_score, test_score]
                        used_feature = [self.wine.feature_names[i], self.wine.feature_names[j]]

        print ("[*] KNeighbor Classifier")
        print ("[+] Used Feature : {}/{}".format(used_feature[0], used_feature[1]))
        print ("[+] Maximum Score : {}/{}".format(max_score[0],max_score[1]))
        

        return model, used_feature, max_score,  X_train, X_test, y_train, y_test



    def linear_svc(self):
        y = self.wine.target
        max_score = [0, 0]
        for i in range(len(self.wine.feature_names)):
            for j in range(len(self.wine.feature_names)):
                if i!=j:
                    X = self.wine.data[:, [i, j]]
                    scaler = MinMaxScaler()
                    scaler.fit(X)
                    X = scaler.transform(X)
                    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=2021)

                    model = LinearSVC(C=0.1, random_state=1, max_iter=5000000) 
                    model.fit(X_train, y_train)

                    y_predict = model.predict(X_train)
                    train_score = metrics.accuracy_score(y_train, y_predict)
                    # print ("[*] Linear SVC")
                    # print ("  [+] Used Feature: {}/{}".format(self.wine.feature_names[i], self.wine.feature_names[j]))
                    # print ("  [+] Train Accuracy : {}".format(train_score))

                    y_predict = model.predict(X_test) 
                    test_score = metrics.accuracy_score(y_test, y_predict)
                    # print("  [+] Test Accuracy : {}".format(test_score))


                    imp_variables = {}
                    importance = model.coef_[0]
                    for i,v in enumerate(importance):
                        imp_variables[self.wine.feature_names[i]] = v

                    imp_variables = dict(sorted(imp_variables.items(), key=lambda x: x[1], reverse=True))
                    # print (imp_variables)
                    top1, top2 = list(imp_variables)[0], list(imp_variables)[1]
                    # print ("  [+] Best Two Featues : {}/{}\n".format(top1, top2))

                    if test_score > max_score[1]:
                        max_score = [train_score, test_score]
                        used_feature = [self.wine.feature_names[i], self.wine.feature_names[j]]
                        best_feature = [top1, top2]

        print ("[*] Linear SVC")
        print ("[+] Used Feature : {}/{}".format(used_feature[0], used_feature[1]))
        print ("[+] Maximum Score : {}/{}".format(max_score[0],max_score[1]))
        print ("[+] Best Feature : {}/{}".format(best_feature[0], best_feature[1]))

        return model, used_feature, max_score, best_feature, X_train, X_test, y_train, y_test


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
        model = VotingClassifier(
            estimators=[('lr', LogisticRegression(penalty='l2', C=0.1, max_iter=1000000, tol=0.0001)),
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

    

    def visualize_all_models(self, model, used_feature, max_score, best_feature, x_train, x_test, y_train, y_test):
        wine = load_wine()
        
        # cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
        cmap_light = 'spring'
        cmap_bold = ['darkorange', 'c', 'darkblue']

        h = .02  # step size in the mesh
        n_neighbors = 10
        
        x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
        y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        # plt.figure(figsize=(8, 6))
        plt.subplot(131)
        plt.contourf(xx, yy, Z, cmap=cmap_light)
        # Plot also the training points
        sns.scatterplot(x=x_train[:, 0], y=x_train[:, 1], hue=wine.target_names[y_train], palette=cmap_bold, alpha=1.0, edgecolor="black")
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        # plt.title("3-Class classification (k = %i, weights = '%s')"% (n_neighbors, 'distance'))
        plt.title("training data")
        plt.xlabel(used_feature[0])
        plt.ylabel(used_feature[1])
        # plt.show()


        x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
        y_min, y_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) # Z = y_predict
        # Z = xx * yy

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        # plt.figure(figsize=(8, 6))
        plt.subplot(132)
        plt.contourf(xx, yy, Z, cmap=cmap_light)

        
        # Plot also the test points
        sns.scatterplot(x=x_test[:, 0], y=x_test[:, 1], hue=wine.target_names[y_test], palette=cmap_bold, alpha=1.0, edgecolor="black")
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("ground truth(test data)")
        plt.xlabel(used_feature[0])
        plt.ylabel(used_feature[1])



        x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
        y_min, y_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) # Z = y_predict

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        # plt.figure(figsize=(8, 6))
        plt.subplot(133)
        plt.contourf(xx, yy, Z, cmap=cmap_light)


        # Plot also the test points
        sns.scatterplot(x=x_test[:, 0], y=x_test[:, 1], hue=wine.target_names[y_test], palette=cmap_bold, alpha=1.0, edgecolor="black")
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("prediction(test data)")
        plt.xlabel(used_feature[0])
        plt.ylabel(used_feature[1])


        plt.tight_layout()  
        plt.show()
    

    def visualize_linear_svm(self, model, used_feature, max_score, best_feature, X_train, X_test, y_train, y_test):
        
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        X = pd.DataFrame([df[used_feature[0]], df[used_feature[1]]])
        X = X.transpose()
        X = X.values

        plt.figure(figsize=[10,8])
        plt.scatter(X[:,0], X[:,1], c=wine.target, cmap='spring')
        plt.xlabel(used_feature[0])
        plt.ylabel(used_feature[1])
        plt.colorbar()
        plt.show()

        # training 분류 결과
        plt.figure(figsize=[10, 8])
        mglearn.plots.plot_2d_classification(model, X_train, eps=0.5, cm = 'spring', fill=True)
        mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train)
        plt.xlabel(used_feature[0])
        plt.ylabel(used_feature[1])
        plt.title('training')
        plt.show()

        scale = 300
        xmax = X_train[:,0].max() + 1
        xmin = X_train[:,0].min() - 1
        ymax = X_train[:,1].max() + 1
        ymin = X_train[:,1].min() - 1

        xx = np.linspace(xmin, xmax, scale)
        yy = np.linspace(ymin, ymax, scale)

        data1, data2 = np.meshgrid(xx, yy)

        X_grid = np.c_[data1.ravel(), data2.ravel()]
        pred_y = model.predict(X_grid)

        plt.figure(figsize=[10, 8])
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=60)
        plt.xlabel(used_feature[0])
        plt.ylabel(used_feature[1])
        plt.title('Linear SVC - Wine', fontsize=20)
        plt.imshow(pred_y.reshape(scale, scale), interpolation=None, origin='lower', extent=[xmin, xmax, ymin, ymax], alpha=0.3, cmap='gray_r')
        plt.show()

        W = model.coef_
        b = model.intercept_
        # print (W, b)

        # test 분류 결과
        mglearn.plots.plot_2d_classification(model, X_train, cm = 'spring', eps=0.5, fill=True)
        mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test)
        plt.xlabel(used_feature[0])
        plt.ylabel(used_feature[1])
        plt.title('test')
        plt.show()

        # print (model.classes_)
        # print (model.decision_function(X_test))
        model.predict(X_test)


def main():

    # 1. 모델 & 플롯 객체 생성
    M = Model()
    V = Visualizer(M.df)


    # 2. 모델 학습 + 결과 반환 + 시각화(Logistic Regression)
    model, used_feature, max_score, best_feature, X_train, X_test, y_train, y_test = M.logistic_regression()
    V.visualize_all_models(model, used_feature, max_score, best_feature, X_train, X_test, y_train, y_test)


    # 3. 모델 학습 + 결과 반환 + 시각화(KNeighborsClassifier)
    model, used_feature, max_score, X_train, X_test, y_train, y_test = M.kneighbor_classifier()
    V.visualize_all_models(model, used_feature, max_score, best_feature, X_train, X_test, y_train, y_test)


    # 4. 모델 학습 + 결과 반환 + 시각화(LinearSVC)
    model, used_feature, max_score, best_feature, X_train, X_test, y_train, y_test = M.linear_svc()
    V.visualize_linear_svm(model, used_feature, max_score, best_feature, X_train, X_test, y_train, y_test)


    
    

if __name__ == "__main__":
    sys.exit(main())