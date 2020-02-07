import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.ensemble import IsolationForest

import pyod

df = pd.read_csv('I:/Data Requests/Ross_F/SP__Py/Test_Data.csv')

df['locate_ref'] = df['Start_Lat'] + df['Start_Lng']

#print(df)

####################################################################

plt.scatter(range(df.shape[0]), np.sort(df['locate_ref'].values))
plt.xlabel('index')
plt.ylabel('Location')
plt.title("Location distribution")

sns.despine()

df['locate_ref'].describe()

print("Skewness: %f" % df['locate_ref'].skew())
print("Kurtosis: %f" % df['locate_ref'].kurt())

plt.show()

####################################################################

plt.scatter(range(df.shape[0]), np.sort(df['Severity'].values))
plt.xlabel('index')
plt.ylabel('Severity')
plt.title("Severity distribution")

sns.despine()

df['Severity'].describe()


print("Skewness: %f" % df['Severity'].skew())
print("Kurtosis: %f" % df['Severity'].kurt())

plt.show()

####################################################################
#Univariate Anomaly Detection on Location

isolation_forest = IsolationForest(n_estimators=100)
isolation_forest.fit(df['locate_ref'].values.reshape(-1, 1))
xx = np.linspace(df['locate_ref'].min(), df['locate_ref'].max(), len(df)).reshape(-1,1)
anomaly_score = isolation_forest.decision_function(xx)
outlier = isolation_forest.predict(xx)
plt.figure(figsize=(10,4))
plt.plot(xx, anomaly_score, label='anomaly score')
plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 
                 where=outlier==-1, color='r', 
                 alpha=.4, label='outlier region')
plt.legend()
plt.ylabel('anomaly score')
plt.xlabel('locate_ref')
plt.show()

####################################################################
#Investigate

print(df.iloc[])
print(df.iloc[])

####################################################################
#Multivariate Anomaly Detection

sns.regplot(x="locate_ref", y="Severity", data=df)
sns.despine()
plt.show()

####################################################################

#print(df.iloc[:,3])
#print(df.iloc[:,18])

plt.xlabel('Severity')
plt.ylabel('Location')
plt.plot(df.iloc[:,3],df.iloc[:,18],'bx')
plt.show()


##https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1
