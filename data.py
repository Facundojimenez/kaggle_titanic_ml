import pandas as pd

def load_data(path):
	data = pd.read_csv(path)
	return data

def preprocess(data):
	X = pd.DataFrame()
	y = pd.DataFrame()

	#As SibSp and Parch are very related to family, I merged them into only one feature
	data["FamilyMembers"] = data['SibSp'] + data['Parch']

	#One-Hot encoding for the "Embarked" feature
	data = pd.get_dummies(data, columns=["Embarked"], prefix=["Embarked"])

	#Binary encoding for the "Sex" feature
	sex_encoding_dict = {"Sex": {"male": 0, "female": 1}}
	data = data.replace(sex_encoding_dict)

	#Filling missing ages with the median value
	data.fillna(data.median(numeric_only=True).round(0), inplace=True)

	if "Survived" in data.columns:
		#Asigning survived values to the "y" array for comparing results
		y = data["Survived"]
	#Deleting irrelevant features
	X = data[["Pclass", "Sex", "Age", "Fare", "FamilyMembers", "Embarked_C", "Embarked_Q", "Embarked_S"]]

	# print(X.head()) #Prints the first top rows of the X dataframe

	return X.to_numpy(), y.to_numpy() #Converts from Pandas DataFrame type to Numpy Array for easier processing later

X, y = preprocess(load_data('input/train.csv'))
# X, y = preprocess(load_data('input/test.csv'))
