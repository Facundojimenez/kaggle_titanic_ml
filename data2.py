import pandas as pd

def load_data():
	data = pd.read_csv('./train.csv')
	return data

def preprocess(data):
	y = data["Survived"]

	data = pd.get_dummies(data, columns=["Embarked"], prefix=["Embarked"])
	sex_encoding_dict = {"Sex": {"male": 0, "female": 1}}
	data = data.replace(sex_encoding_dict)
	data.fillna(data.mean(numeric_only=True).round(0), inplace=True)
	X = data.drop(columns=["PassengerId", "Survived", "Name", "Ticket", "Cabin", "Parch"])
	print(X.head()) #Prints the first top rows of the X dataframe

	return X.to_numpy(), y.to_numpy() #Converts from Pandas DataFrame type to Numpy Array for easier processing later

X, y = preprocess(load_data())
