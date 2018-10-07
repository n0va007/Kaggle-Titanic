import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




dataset =pd.read_csv('train.csv')

test =pd.read_csv('test.csv')

x_train=dataset.iloc[:,4:8].values

x_test=test.iloc[:,3:7].values

y_train=dataset.iloc[:,1].values


from sklearn.preprocessing import Imputer



imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)



imputer=imputer.fit(x_train[:,1:2])



x_train[:,1:2]=imputer.transform(x_train[:,1:2])

imputer=imputer.fit(x_test[:,1:2])



x_test[:,1:2]=imputer.transform(x_test[:,1:2])

from sklearn.preprocessing import LabelEncoder,OneHotEncoder



le=LabelEncoder()


x_train[:,0]=le.fit_transform(x_train[:,0])

x_test[:,0]=le.fit_transform(x_test[:,0])

oe=OneHotEncoder(categorical_features=[0])



x_train=oe.fit_transform(x_train).toarray()

x_test=oe.fit_transform(x_test).toarray()


from sklearn.preprocessing import StandardScaler


sc=StandardScaler()


x_train=sc.fit_transform(x_train)


x_test=sc.transform(x_test)


from sklearn.decomposition import PCA

pca =PCA(n_components=3)

x_train=pca.fit_transform(x_train)

x_test=pca.transform(x_test)



exp_var=pca.explained_variance_ratio_



from sklearn.svm import SVC

classifier= SVC()

classifier.fit(x_train,y_train)


y_pred=classifier.predict(x_test)


submission =pd.DataFrame({
        "PassengerId":test["PassengerId"],
        "Survived":y_pred
        })

submission.to_csv('titanic.csv',index=False)

submission.head()

#accuracy = 77.99% after submission










