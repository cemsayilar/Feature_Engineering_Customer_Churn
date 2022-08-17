import pandas as pd
import numpy as np
import math
import scipy.stats as st
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.stats._stats_py import ttest_ind
import matplotlib as mt
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from datetime import date
# Measurement
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
# Recomendation Systems
from mlxtend.frequent_patterns import apriori, association_rules
# Measurement
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
# Feature Engineering
import missingno as msno
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
pd.set_option('display.max_columns',None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

### Import Data
df_ = pd.read_csv('/Users/buraksayilar/Desktop/feature_engineering/datasets/Telco-Customer-Churn.csv')
df = df_.copy()
df.columns = [col.upper() for col in df.columns]
### First Look
def check_df(dataframe, head=5):
    print(f'{" Shape ":-^100}')
    print(dataframe.shape)
    print(f'{" Info":-^100}')
    print(dataframe.info(head))
    print(f'{" Head ":-^100}')
    print(dataframe.head(head))
    print(f'{" Tail ":-^100}')
    print(dataframe.tail(head))
    print(f'{" NA ":-^100}')
    print(dataframe.isnull().sum())
    print(f'{" Quantiles ":-^100}')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)
### Exploretery Data Analysis
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)
df['TOTALCHARGES'] = pd.to_numeric(df['TOTALCHARGES'], errors='coerce')
## Checking Numerical Values

## Checking Cat. Values

### Outlier Analysis
def outlier_thresholds(dataframe, col_name, q1 = 0.25, q3 = 0.90):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    iqr = quantile3 - quantile1
    up_limit = quantile3 + iqr * 1.5
    low_limit = quantile1 - iqr * 1.5
    return low_limit, up_limit
def check_outlier(dataframe,col_name):
    low_limit, up_limit = outlier_thresholds(dataframe,col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
       return True
    else:
        return False
## Check
for col in num_cols:
    check_outlier(df, col)
## Threshold Analysis
### Missing Value Analysis
## Check
for col in num_cols:
    if df[col].nunique() > 10:
        df[col].replace(0,np.nan,inplace=True)
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns
missing_values_table(df)
## Solving
df.dropna(inplace=True)

### Feature Engineering
## Feature Extraction
df.head()

df['TENURE_YEAR'] = df['TENURE'] / 12
df['LOYAL_USER'] = np.where(df['TENURE_YEAR'] > 4, 1, 0)

# Binary Features (Flag, Bool)
# Text Features
# Feature Extraction with Regex (Regular Expressions)

## Feature Interaction
df['ALONE_USER'] = np.where(df['PARTNER'] == 'No', 1, 0)
df['ALONE_and_SENIOR_USER'] = np.where((df['SENIORCITIZEN'] == 1) & (df['PARTNER'] == 'No'), 1, 0)
df['ADVANCE_USER'] = np.where((df['INTERNETSERVICE'] == 'Fiber optic') & (df['ONLINESECURITY'] == 'Yes') &
                              (df['ONLINEBACKUP'] == 'Yes') & (df['DEVICEPROTECTION'] == 'Yes'), 1, 0)
# .groupby relevant features
### Encoding (For Categoric Features)

## Label Encoding (Binary Encoding)
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
binary_col = [col for col in df.columns if (df[col].dtype in [object]) and (df[col].nunique() < 5)]
## One-Hot-Encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
# cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Now, maybe I want to ignore my target value, or I already
# label encoded my 'Sex' feature. So I can select categorical variables
# for One Hot Encoding.
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
dff = one_hot_encoder(df, binary_col)
dff.head()
check_df(dff)
### Feature Scaling (For Numeric Features)

## Standart Scaling
scaler = StandardScaler()
dff[num_cols] = scaler.fit_transform(df[num_cols])
dff[num_cols].head()
dff.head()
## MinMax Scaling
## Robust Scaler (no NaN effect)



### Model Building
y = dff['CHURN_Yes']
X = dff.drop(['CUSTOMERID','CHURN_Yes'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.ensemble import RandomForestClassifier
X.shape
y.shape
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

accuracy_score(y_pred, y_test)


#############################################
# Without any processing?
#############################################

df0 = pd.read_csv('/Users/buraksayilar/Desktop/feature_engineering/Telco-Customer-Churn.csv')
df0.columns = [col.upper() for col in df0.columns]
df0['TOTALCHARGES'] = pd.to_numeric(df['TOTALCHARGES'], errors='coerce')


df0.dropna(inplace=True)
# Encoding
binary_col = [col.upper() for col in binary_col]
df0 = pd.get_dummies(df0, columns=binary_col, drop_first=True)
df0['CHURN'] = np.where(df['CHURN'] == 'Yes',1,0)
# Building Model
df0.drop('CHURN_Yes',inplace=True, axis=1)
y = df0["CHURN"]
X = df0.drop(["CUSTOMERID",'CHURN'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=46)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)