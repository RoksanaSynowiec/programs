# =============================================================================
# ---------------------------------PCA-----------------------------------------
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

# data -----------------------------------------------------------------------------------
train = pd.read_csv("train.csv")
train.columns = ['Id', 'Open Date', 'City', 'City Group', 'Type', 'P1', 'P2', 'P3', 'P4',
                 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15',
                 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25',
                 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35',
                 'P36', 'P37', 'revenue']
test = pd.read_csv("test.csv")
test.columns = ['Id', 'Open Date', 'City', 'City Group', 'Type', 'P1', 'P2', 'P3', 'P4',
                'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15',
                'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25',
                'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35',
                'P36', 'P37']
sample_sub = pd.read_csv("sampleSubmission.csv")
sample_sub.columns


# ------------------------------------------------------------------------------------------

# Function extracting year form date instead of loop----------------------------------------
def get_year(x):
    x = datetime.strptime(x, '%m/%d/%Y')
    x = datetime.date(x)
    return (x.year)


train['Open Date'] = train['Open Date'].apply(get_year)

test['Open Date'] = test['Open Date'].apply(get_year)
# ------------------------------------------------------------------------------------------


# preparation to do correlation matrix-----------------------------------------------------
train = train.drop(['P12', 'P13', 'P18', 'P15', 'P26', 'P24', 'P32','P34', 'P35', 'P36', 'P9'], axis=1)
X = train.drop(['revenue', 'Id', 'City', 'City Group', 'Type'], axis=1)
test = test.drop(['Id', 'City', 'City Group', 'Type'], axis=1)
y = train['revenue']


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# encoding year data from int to categorical variable. in fact it's not string. it's still int but smaller
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
years = [i for i in range(1990, 2019)]
years = labelencoder.fit_transform(years)
X.loc[:, "Open Date"] = labelencoder.transform(X.loc[:, "Open Date"])
# X.loc[:,"Open Date"] = labelencoder.transform(X.loc[:,"Open Date"])
# ------------------------------------------------------------------------------------------

# PCA---------------------------------------------------------------------------------------
minmax = MinMaxScaler().fit_transform(X)
pca = PCA().fit(minmax)
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('number of components')
#plt.ylabel('cumulative explained variance');





def PCA_model(m):
    pca = PCA(n_components=m)
    principalComponents = pca.fit_transform(minmax)

    pca.explained_variance_ratio_
    pca_df = pd.DataFrame(data=principalComponents, columns=[(i + 1) for i in range(0, m)])


   # pca_train, pca_test, y_train, y_test = train_test_split(pca_df, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(pca_df, y)
    R2 = model.score(pca_df, y)
    p = len(pca_df.columns)
    n = len(pca_df[pca_df.columns[1]])
    ADJR2 = 1 + (R2 - 1) * (n - 1) / (n - p - 1)

    print('Model dla PCA-transofrmacji z', m, 'zmiennymi:\n')
    print('Liczba wierszy = ' + str(n))
    print('Liczba zmiennych = ' + str(p))
    print('R^2 = ' + str(round(R2, 4)))
    print('Skorygowane R^2 = ' + str(round(ADJR2, 4)))
    print('Intercept = ' + str(round(model.intercept_, 4)))
    print(pca_df)
    return (round(ADJR2, 4))


##### SVR przepuszczony po danych z PCA dla mm=6 #####
######################################################

m=6
pca = PCA(n_components=m)
principalComponents = pca.fit_transform(minmax)
pca.explained_variance_ratio_
pca_df = pd.DataFrame(data=principalComponents, columns=[(i + 1) for i in range(0, m)])

from sklearn.svm import SVR
regressor=SVR(kernel='linear',C=10e4, gamma='scale', epsilon=30*10e2)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
pca_train, pca_test, y_train, y_test = train_test_split(pca_df, y, test_size=0.25, random_state=0)
regressor.fit(pca_train,y_train)
pred=regressor.predict(pca_test)
#print(regressor.score(pca_df,y))
R2=regressor.score(pca_test,y_test)
p = len(pca_test.columns)
n = len(pca_test[pca_test.columns[1]])
ADJR2 = 1 + (R2 - 1)*(n - 1)/(n-p-1)
print(" ADJR2: ")
print(ADJR2)



print(regressor.score(pca_test,y_test))
from sklearn.metrics import r2_score
#print(r2_score(y,pred))

import math
#wektor = np.array(pca_df[pca_df.columns[5]])
#plt.plot(wektor, y, '.')
#plt.show()


#PCA_model(4)  # 0.059
#PCA_model(5)  # 0.105
#PCA_model(6)  # 0.115 !!!
#PCA_model(7)  # 0.107
#PCA_model(8)  # 0.101
#PCA_model(9)  # 0.103
#PCA_model(10)  # 0.094

