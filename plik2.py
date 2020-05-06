import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime as dt
from datetime import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import normalize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
pd.options.mode.chained_assignment = None

# USTAWIENIA 
one_hot = True  # czy zastąpić zmienne kategoryczne binarnymi?
imputacja = True  # czy podmienić 0 na średnią w kolumnie?
outliery_out = True  # czy wyrzucić wartości odstające revenue?
skorelowane_out = False  # czy wyrzucić skorelowane kolumny?
redukcja_pca = False  # czy zastosować redukcję do 6 wymiarów pca?
normalizacja = True  # czy znormalizować dane przed treninowaniem modelu?
model = "RF"  # "linear" "GBM" "RF" "lasso" "SVR"

# ------------------------------------ przygotowanie danych ----------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# sample_sub = pd.read_csv("sampleSubmission.csv")


def get_year(x):
    x = datetime.strptime(x, '%m/%d/%Y')
    x = datetime.date(x)
    return (x.year)


train['Open Date'] = 2014 - train['Open Date'].apply(get_year)
test['Open Date'] = 2014 - test['Open Date'].apply(get_year)

# ----------------------------------- wykorzystanie algorytmu Przemka--------------------------
if one_hot:
    ##### Zamiana MB (którego nie ma w train) na DT (w pliku test)
    test['Type'][test['Type'] == 'MB'] = 'DT'

    ##### Rozbicie kolumny Type na 3 (one hot?) + rozbicie city group na 1/-1
    ##### Można chyba też na 2? Bo nie da się jednocześnie być w kilku typach,
    ##### więc wiadomo, że jeśli w dwóch typach jest -1 to znaczy, że jest w tym trzecim typie
    ### plik test
    test['DT'] = test['Type']
    test['IL'] = test['Type']
    test['FC'] = test['Type']
    test['City Group Bin'] = test['City Group']

    test['City Group Bin'][test['City Group Bin'] != 'Other'] = 1
    test['City Group Bin'][test['City Group Bin'] == 'Other'] = -1
    test['DT'][test['DT'] != 'DT'] = -1
    test['DT'][test['DT'] == 'DT'] = 1
    test['IL'][test['IL'] != 'IL'] = -1
    test['IL'][test['IL'] == 'IL'] = 1
    test['FC'][test['FC'] != 'FC'] = -1
    test['FC'][test['FC'] == 'FC'] = 1

    ### plik train

    train['DT'] = train['Type']
    train['IL'] = train['Type']
    train['FC'] = train['Type']
    train['City Group Bin'] = train['City Group']

    train['City Group Bin'][train['City Group Bin'] != 'Other'] = 1
    train['City Group Bin'][train['City Group Bin'] == 'Other'] = -1
    train['DT'][train['DT'] != 'DT'] = -1
    train['DT'][train['DT'] == 'DT'] = 1
    train['IL'][train['IL'] != 'IL'] = -1
    train['IL'][train['IL'] == 'IL'] = 1
    train['FC'][train['FC'] != 'FC'] = -1
    train['FC'][train['FC'] == 'FC'] = 1

##### Uzupełnienie brakujących danych
if imputacja:
    for i in train.columns[5:]:
        ### uzupełnienie średnią z pozostałych wartości
        train[i][train[i] == 0] = train[i][train[i] != 0].mean()

        ### uzupełnienie najczęśniej występującą wartością (poza zerem)
        # train[i][train[i]==0] = train[i][train[i]!=0].value_counts().idxmax()
# ----------------------------------------------------------------------------------

if outliery_out:
    train = train[train['revenue'] < 16000000]

X = train.drop(['revenue', 'Id', 'City', 'City Group', 'Type'], axis=1)

if skorelowane_out:
    X = X.drop(['P12', 'P13', 'P18', 'P15', 'P26', 'P24', 'P32', 'P34', 'P35', 'P36', 'P9'], axis=1)
columns = X.columns
y = np.array(train['revenue'])
X = np.array(X)

# --------------------------------------------------------------------------------------
# ------------------------------------------ PCA -----------------------------------------
if redukcja_pca:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler

    minmax = MinMaxScaler().fit_transform(X)
    pca = PCA().fit(minmax)

    m = 6
    pca = PCA(n_components=m)
    principalComponents = pca.fit_transform(minmax)
    pca.explained_variance_ratio_
    pca_df = pd.DataFrame(data=principalComponents, columns=[(i + 1) for i in range(0, m)])
    X = pca_df

if normalizacja:

    X = normalize(X)


r2 = 0
for i in range(30):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    if model == "RF":

        from sklearn.ensemble import RandomForestRegressor
        M = RandomForestRegressor(n_estimators=1000)
        M.fit(X_train, y_train)


    if model == "GBM":
        params = {'n_estimators': 1000, 'max_depth': 2, 'min_samples_split': 3,
                  'learning_rate': 0.01, 'loss': 'lad'}
        M = GradientBoostingRegressor(**params)

    if model == "lasso":
        M = Lasso(alpha=10000, max_iter=10e5)

    if model == "linear":
        M = LinearRegression()
    if model == "SVR":
        M = SVR(kernel='poly', C=10e4, gamma='scale', epsilon=30 * 10e2)


    M.fit(X_train, y_train)
    # print(M.score(X_test, y_test))
    r2 += M.score(X_test, y_test)
    #feature_importances = pd.DataFrame(M.feature_importances_, index=pd.DataFrame(X).columns, columns=['importance']).sort_values('importance', ascending=False)
importances = M.feature_importances_
plt.figure()
plt.title("Feature importances")
plt.barh(columns, importances,align='center')
plt.show()
print("Model", model, "\nŚrednie R2 w cross walidacji:", r2 / 30)
