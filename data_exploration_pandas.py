import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
This script is used for check if a feature is relevant 
to determinate if a passenger will survive or not
"""

train = pd.read_csv('./train.csv')

# Shape and preview
print('Train set shape:', train.shape)
train.head()

print('TRAIN SET MISSING VALUES:')
print(train.isna().sum())

print("TRAIN SET DATA TYPES")
print(train.dtypes)


# Figure size
# plt.figure(figsize=(6,6))

# Pie plot
# train['Survived'].value_counts().plot.pie(explode=[0.1,0.1], autopct='%1.1f%%', shadow=True, textprops={'fontsize':16}).set_title("Target distribution")



#--------------PClass Histogram--------------
#People in 1st class tend to survive while people in 2nd class is even and people in 3rd class is likely not to survive

sns.histplot(data=train, x='Pclass', hue='Survived',multiple="stack")
plt.title('Pclass distribution')
plt.xlabel('Pclass (1-3)')

plt.show()

#------------ Sex Histogram-------------------
#Women are a bit more likely to survive than men
sns.histplot(data=train, x='Sex', hue='Survived', binwidth=1, multiple="stack")
plt.title('Sex distribution')
plt.xlabel('Sex')

plt.show()

#------------ Age Histogram----------------
#Small children have more chances of surviving
sns.histplot(data=train, x='Age', hue='Survived', binwidth=1, kde=True, multiple="stack")
plt.title('Age distribution')
plt.xlabel('Age (years)')

plt.show()

#--------------Embarked Histogram (Port of departure)--------------
#**Not sure** --> People from Southhampton is less likely to survive than the rest
sns.displot(data=train, x='Embarked', hue='Survived', multiple="stack")
plt.title('Embarked distribution')
plt.xlabel('Embarked')

plt.show()

#--------------Parch Histogram (Number of parents/children on the ship)--------------
#**Not sure**
sns.displot(data=train, x='Parch', hue='Survived', multiple="stack")
plt.title('Parch distribution')
plt.xlabel('Parch')

plt.show()

#--------------Fare Histogram--------------
#People with higher fare (>100) are more likely to survive
sns.histplot(data=train, x='Fare', hue='Survived', multiple="stack")
plt.title('Fare distribution')
plt.xlabel('Fare')

plt.show()

#--------------SibSp Histogram--------------
#People with SibSp 1 and 2 are more likely to survive than the rest
sns.histplot(data=train, x='SibSp', hue='Survived',multiple="stack")
plt.title('SibSp distribution')
plt.xlabel('SibSp (0-8)')

plt.show()
