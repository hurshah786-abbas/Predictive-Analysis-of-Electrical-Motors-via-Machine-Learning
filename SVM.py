import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

motordata = pd.read_csv("pdm.csv")

motordata = motordata.reset_index()
motordata.replace([np.inf, -np.inf], np.nan, inplace=True)

pd.DataFrame.isin
pd.DataFrame.any
#motordata[~motordata.isin([np.nan, np.inf, -np.inf]).any(1)]
motordata = motordata[np.isfinite(motordata).all(1)]

motordata.head()

motordata.shape


X = motordata.drop('class', axis=1)
y = motordata['class']



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

"""
np.isnan(X)
np.where(np.isnan(X))
np.nan_to_num(X)
pd.DataFrame(X).fillna(X_test.mean())"""


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)


y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))