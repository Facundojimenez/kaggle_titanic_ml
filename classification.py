from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

# from data_old import load_data, preprocess
from data2 import load_data, preprocess
from sklearn import svm

data = load_data()
X, y = preprocess(data)



norm = MinMaxScaler()
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.25)

norm.fit(X_train)
X_train = norm.transform(X_train)
X_test = norm.transform(X_test)


#USING K NEAREST NEIGHBORS
print("-----KNN MODEL-----")
for k in [ 1, 3, 5, 9, 10, 20, 30, 50, 100 ]:
    model = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    model.fit(X_train, y_train)
    y_hat = model.predict(X_train)
    trerr = sum([ 1 if yp != yhp else 0 for yp, yhp in zip(y_train, y_hat) ]) / len(y_train)
    y_hat = model.predict(X_test)
    teerr = sum([ 1 if yp != yhp else 0 for yp, yhp in zip(y_test, y_hat) ]) / len(y_test)
    print(k, 'train:', trerr, 'test:', teerr)

#Using Decission trees
print("-----DECISSION TREES MODEL-----")
model = DecisionTreeClassifier(random_state=0, max_depth=2)
model.fit(X_train, y_train)
y_hat = model.predict(X_train)
trerr = sum([ 1 if yp != yhp else 0 for yp, yhp in zip(y_train, y_hat) ]) / len(y_train)
y_hat = model.predict(X_test)
teerr = sum([ 1 if yp != yhp else 0 for yp, yhp in zip(y_test, y_hat) ]) / len(y_test)
print(k, 'train:', trerr, 'test:', teerr)

#Using SVM
print("-----SVM MODEL-----")
model = svm.SVC()
model.fit(X_train, y_train)
y_hat = model.predict(X_train)
trerr = sum([ 1 if yp != yhp else 0 for yp, yhp in zip(y_train, y_hat) ]) / len(y_train)
y_hat = model.predict(X_test)
teerr = sum([ 1 if yp != yhp else 0 for yp, yhp in zip(y_test, y_hat) ]) / len(y_test)
print(k, 'train:', trerr, 'test:', teerr)

#Using SVM
print("-----Random Forest MODEL-----")
model = RandomForestClassifier(max_depth=7, random_state=0)
model.fit(X_train, y_train)
y_hat = model.predict(X_train)
trerr = sum([ 1 if yp != yhp else 0 for yp, yhp in zip(y_train, y_hat) ]) / len(y_train)
y_hat = model.predict(X_test)
teerr = sum([ 1 if yp != yhp else 0 for yp, yhp in zip(y_test, y_hat) ]) / len(y_test)
print(k, 'train:', trerr, 'test:', teerr)
