from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import kpss
import pickle
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_python_lib().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('AQI_prediction_dataset.csv')

df

df.shape

df.isnull().sum()

df.info()

df.describe()

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

df.drop(["max_temp", "min_temp"], axis=1, inplace=True)

sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')

sns.set_theme(style="whitegrid")

for i in df.columns[1:15]:
    sns.boxplot(x=df[i])
    plt.title('Boxplot of the sensors data')
    plt.show()

Q1 = df.quantile(0.25)  # first 25% of the data
Q3 = df.quantile(0.75)  # first 75% of the data
IQR = Q3 - Q1  # IQR = InterQuartile Range

scale = 2  # For Normal Distributions, scale = 1.5
lower_lim = Q1 - scale*IQR
upper_lim = Q3 + scale*IQR

lower_outliers = (df[df.columns[1:15]] < lower_lim)
upper_outliers = (df[df.columns[1:15]] > upper_lim)

df[df.columns[1:15]][(lower_outliers | upper_outliers)].info()

num_cols = list(df.columns[1:15])
df_out_IQR = df[~((df[num_cols] < (Q1 - 2 * IQR)) |
                  (df[num_cols] > (Q3 + 2 * IQR))).any(axis=1)]
df_out_IQR.info()

df_filt = df_out_IQR.dropna(how='any', axis=0)
df_filt.reset_index(drop=True, inplace=True)
df_filt.info()


def kpss_test(timeseries):
    print('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=[
                            'Test Statistic', 'p-value', '#Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output)


df.head()

kpss_test(df['PM2.5'])

"""PM2.5 is Non-Stationary"""

kpss_test(df['PM10'])

"""PM10 is Non-Stationary"""

kpss_test(df['NO2'])

"""NO2 is Non-Stationary"""

kpss_test(df['NOx'])

"""NOx is Non-Stationary"""

kpss_test(df['CO'])

"""CO is Stationary"""

kpss_test(df['SO2'])

"""SO2 is Non-Stationary"""

kpss_test(df['O3'])

"""O3 is Stationary"""

kpss_test(df['temp'])

"""temp is Stationary"""

kpss_test(df['humid'])

"""humid is Stationary"""

kpss_test(df['visible'])

"""visible is Stationary"""

kpss_test(df['wind'])

"""wind is Stationary"""


"""After Converting to Stationary"""

df_log1 = np.sqrt(df['PM2.5'])
df_diff1 = df_log1.diff().dropna()
kpss_test(df_diff1)

df_log2 = np.sqrt(df['PM10'])
df_diff2 = df_log2.diff().dropna()
kpss_test(df_diff2)

df_log3 = np.sqrt(df['NO2'])
df_diff3 = df_log3.diff().dropna()
kpss_test(df_diff3)

df_log4 = np.sqrt(df['NOx'])
df_diff4 = df_log4.diff().dropna()
kpss_test(df_diff4)

df_log5 = np.sqrt(df['SO2'])
df_diff5 = df_log5.diff().dropna()
kpss_test(df_diff5)


df_filt['Week Day'] = df_filt['Date'].dt.day_name()

cols = df_filt.columns.tolist()
cols = cols[:1] + cols[-1:] + cols[1:11]
ar_filt = df_filt[cols]
ar_filt.head(10)

sns.barplot(x='Week Day', y='PM2.5', data=df_filt)
plt.title('Mean Values of PM2.5 on Week Days')
plt.xticks(rotation=90)
plt.show()

sns.barplot(x='Week Day', y='PM10', data=df_filt)
plt.title('Mean Values of PM10 on Week Days')
plt.xticks(rotation=90)
plt.show()

sns.barplot(x='Week Day', y='NO2', data=df_filt)
plt.title('Mean Values of NO2 on Week Days')
plt.xticks(rotation=90)
plt.show()

sns.barplot(x='Week Day', y='NOx', data=df_filt)
plt.title('Mean Values of NOx on Week Days')
plt.xticks(rotation=90)
plt.show()

sns.barplot(x='Week Day', y='CO', data=df_filt)
plt.title('Mean Values of CO on Week Days')
plt.xticks(rotation=90)
plt.show()

sns.barplot(x='Week Day', y='SO2', data=df_filt)
plt.title('Mean Values of SO2 on Week Days')
plt.xticks(rotation=90)
plt.show()

sns.barplot(x='Week Day', y='O3', data=df_filt)
plt.title('Mean Values of O3 on Week Days')
plt.xticks(rotation=90)
plt.show()


df_filt


y = df_filt['AQI']
x = df_filt.drop(['Date', 'Week Day', 'AQI'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
y_train = sc.fit_transform(np.array(y_train).reshape(-1, 1))
y_test = sc.transform(np.array(y_test).reshape(-1, 1))

y_test[:5]

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_prediction = regressor.predict(x_test)
y_prediction

score = r2_score(y_test, y_prediction)
mean_error = mean_squared_error(y_test, y_prediction)
mae = mean_absolute_error(y_test, y_prediction)
print("R2 SCORE is", score)
print("mean_sqrd_error is ", mean_error)
print("Root mean squared error of is", np.sqrt(mean_error))
print("Mean Absolute error is", mae)

df1 = pd.DataFrame({'Actual': y_test[100:120].flatten(
), 'Predicted': y_prediction[100:120].flatten()})
df1.plot(kind='line')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

result = pd.DataFrame()
result["ACTUAL AQI"] = y_test.flatten()
result["PREDICTED AQI"] = y_prediction
result["DIFFERENCE"] = result["ACTUAL AQI"] - result["PREDICTED AQI"]
result["PERCENTAGE ERROR"] = (
    abs(result["ACTUAL AQI"] - result["PREDICTED AQI"]) / result["ACTUAL AQI"]) * 100
result.head(15)


X, y = make_regression(n_samples=1000, n_features=11,
                       n_informative=10, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

base_estimator = LinearRegression()

bagging = BaggingRegressor(base_estimator=base_estimator, n_estimators=10,
                           max_samples=0.8, oob_score=True, max_features=0.5, random_state=1)

bagging.fit(X_train, y_train)

y_pred = bagging.predict(X_test)

bagging.oob_score_

accuracy1 = r2_score(y_test, y_pred)
print(f"Accuracy score: {accuracy1}")


bagging = BaggingRegressor(base_estimator=base_estimator, n_estimators=100,
                           max_samples=0.8, oob_score=True, max_features=0.5, random_state=1)
bagging.fit(X_train, y_train)

y_pred = bagging.predict(X_test)

bagging.oob_score_

accuracy2 = r2_score(y_test, y_pred)
print(f"Accuracy score: {accuracy2}")


bagging = BaggingRegressor(base_estimator=base_estimator, n_estimators=50,
                           max_samples=0.8, oob_score=True, max_features=0.5, random_state=1)
bagging.fit(X_train, y_train)

y_pred = bagging.predict(X_test)

bagging.oob_score_

accuracy3 = r2_score(y_test, y_pred)
print(f"Accuracy score: {accuracy3}")


bagging = BaggingRegressor(base_estimator=base_estimator, n_estimators=200,
                           max_samples=0.8, oob_score=True, max_features=0.5, random_state=1)
bagging.fit(X_train, y_train)

y_pred = bagging.predict(X_test)

bagging.oob_score_

accuracy4 = r2_score(y_test, y_pred)
print(f"Accuracy score: {accuracy4}")


bagging = BaggingRegressor(base_estimator=base_estimator, n_estimators=1000,
                           max_samples=0.8, oob_score=True, max_features=0.5, random_state=1)
bagging.fit(X_train, y_train)

y_pred = bagging.predict(X_test)

bagging.oob_score_

accuracy5 = r2_score(y_test, y_pred)
print(f"Accuracy score: {accuracy5}")

final_accuracy = (accuracy1+accuracy2+accuracy3+accuracy4+accuracy5)/5
print(f"final Accuracy score: {final_accuracy}")

pickle.dump(regressor, open('model_mlr.pkl', 'wb'))
model = pickle.load(open('model_mlr.pkl', 'rb'))
