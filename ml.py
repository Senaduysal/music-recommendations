import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib    #tanımlamalarımı sürekli yapmamak için yazıyorum
 
music = pd.read_csv('music.csv')
X = music.drop(columns = ['genre']) #INPUT DATA SET
Y = music ['genre'] #OUTPUT DATA SET BİZE SADECE SONUCLARI DÖNDÜRÜR
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8)

##Decision learn sklearn ile cagirilir

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

joblib.dump(model, 'music-recommender.joblib')

#predictions = model.predict([ [21, 1], [22,0] ]) #21 yasındaki erkekleri ve 22 yasındakı kadınların predicitionunu yapıyor.

predictions = model.predict(X_test)


#calculating model accuracy

score= accuracy_score(Y_test, predictions) 
score
