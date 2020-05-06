
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import collections
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

us_kol = False
over_sampling = False
red_TSNE = False
random_forest = False
ADA = False
Gradient = False
Naive_bayes = False
SVM = False
Scale = False
XGB_model = False
DBSCAN = True
#-----------------------------przygotowanie danych------------------------------------------


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_sub = pd.read_csv("sampleSubmission.csv")
train = train.sample(frac = 1)
train=train[:1000]
#print(train.shape)

X=train.drop(['target','id'], axis = 1)
X_test = test.drop(['id'], axis = 1)

y=train['target']





#-------------------------------usunięcie kolumn skorelowanych---------------------------
if us_kol:

    X=X.drop(['feat_3','feat_9','feat_15','feat_30','feat_39','feat_46',
              'feat_64','feat_72','feat_84','feat_45'], axis=1)

#---------------------------------TSNE--------------------------------------------------
if red_TSNE:
    X=TSNE(n_components=3).fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#----------------------------------Scale-----------------------------------------------
if Scale:


    X = X.drop(['feat_3', 'feat_9', 'feat_15', 'feat_30', 'feat_39', 'feat_46',
                'feat_64', 'feat_72', 'feat_84', 'feat_45'], axis=1)

    from sklearn.preprocessing import scale
    X_df=scale(X)
    X_df=pd.DataFrame(X_df,columns=X.columns)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from sklearn.svm import LinearSVC
    model = LinearSVC(random_state=0, tol=1e-2, multi_class='ovr')
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    print(metrics.accuracy_score(y_test, np.round(pred)))





#--------------------------------Over_Sampling-------------------------------------------
if over_sampling:
    print("liczność klas przed modyfikacja :")
    print(sorted(collections.Counter(y).items()))



    X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    X_test, y_test = SMOTE().fit_resample(X_test, y_test)
    print("liczność klas po modyfikacji SMOTE")
    print(sorted(collections.Counter(y_train).items()))


#-----------------------------------Random_forest-------------------------
if random_forest:
    model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=3, min_samples_leaf=5)

#----------------------------------ADA_Boost_Classifier---------------------------------
if ADA:
    model = AdaBoostClassifier(n_estimators=100,learning_rate=1.0,algorithm='SAMME.R',random_state=None)



#---------------------------------XGB----------------------------------------------------
if XGB_model:
   # train['target'][train['target']=='Class_1']=1
    #train['target'][train['target']=='Class_2'] = 2
    #train['target'][train['target']=='Class_3'] = 3
    #train['target'][train['target']=='Class_4'] = 4
    #train['target'][train['target']=='Class_5'] = 5
    #train['target'][train['target']=='Class_6'] = 6
    #train['target'][train['target']=='Class_7']= 7
    #train['target'][train['target']=='Class_8'] = 8
    #train['target'][train['target']=='Class_9']= 9
    #y = train['target']

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #D_train = xgb.DMatrix(X_train, label=y_train)
    #D_test = xgb.DMatrix(X_test, label=y_test)

    #param = {'eta': 0.3,'max_depth': 9,'objective': 'multi:softprob','num_class': 9}
    #steps = 20

    import xgboost as xgb
    model = xgb.XGBClassifier(max_depth=9, n_estimators=1000, learning_rate=0.05,
                              objective="multi:softprop",num_class=9)


    #preds = model.predict(D_test)
    #best_preds = np.asarray([np.argmax(line) for line in preds])

    #print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
    #print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
    #print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))


#--------------------------------Gradient_Boost_Classifier--------------------------------
if Gradient:
    model = GradientBoostingRegressor(n_estimators=1000,max_depth=2,min_samples_split=3,learning_rate=0.01,loss='lad')

#----------------------------------NaiveBayes----------------------------------------------
if Naive_bayes:
    model = GaussianNB()

#---------------------------------SVM------------------------------------------------------
if SVM:
   #model = svm.SVC(kernel='poly',degree=3, gamma=2)

    from sklearn.svm import LinearSVC
    model=LinearSVC(random_state=0, tol=1e-2,multi_class='ovr')





model.fit(X_train, y_train)

#silhouette_score(X_test, y_test).round(4)
#model.score(X_test, y_test)
y_pred=model.labels_


print(silhouette_score(y_test, y_pred).round(4))

#final_pred = model.predict(X_test)
#my_submission = pd.DataFrame({'id':  test_index, 'type': final_pred})
#my_submission.to_csv('submission4.csv', index=False)



y=pd.DataFrame(y)
#sns.stripplot(x='feat_1', y='feat_2', hue='target',data=train, jitter=True)
#sns.countplot(x='target', data=y)











#clf = GaussianNB()
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)
from sklearn.metrics import classification_report
#print("Wynik dla niezmienionych danych  NaiveBayes ")
#print(clf.score(X_test, y_test))

#clf.fit(SMOTE_X_train, SMOTE_y_train)
#y_pred = clf.predict(X_test)
from sklearn.metrics import classification_report
#print("Wynik dla zmienionych danych  NaiveBayes ")
#print(clf.score(SMOTE_X_test, SMOTE_y_test))

#svclassifier = SVC(kernel='linear')
#svclassifier.fit(X_train, y_train)
#y_pred = svclassifier.predict(X_test)
#print("Wynik dla niezmienionych danych SVM ")
#print(svclassifier.score(X_test, y_test))
#from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))





