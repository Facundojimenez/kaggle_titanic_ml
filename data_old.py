import csv
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    fd = open('./train.csv')
    data = []
    for line in csv.reader(fd):
        data.append(line)
    return data[1:]


def preprocess(data):
    X, y, z = [], [], []

    data_without_missings = []
    for reg in data:
        if '' not in reg:
            reg[0] = int(reg[0])
            reg[1] = int(reg[1])
            reg[2] = True if int(reg[2]) == 1 else False
            reg[5] = float(reg[5])
            data_without_missings.append(reg)
    data = data_without_missings


    for PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked in data:

        x = [ PassengerId, Pclass , Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked]

        X.append(x)
        y.append(Survived)

        # z is both X and Y merged
        z.append(x + [Survived])


    return X, y,z


X, y, data = preprocess(load_data())
#
#
Pclass = [ ]
Sex = []
Age = []

for reg in data:
    Pclass.append(reg[1])
    Sex.append(reg[2])
    Age.append(reg[3])
print(Age)

fig, ax = plt.subplots()
#--------------- PClass Histogram
#
# ax.hist(Pclass, bins=10, linewidth=0.5, edgecolor="white")
# ax.set(xlim=(1, 4), xticks=range(1, 4))
# ax.set_title("PClass - Survivors")
# ax.set_xlabel("PClass")
# ax.set_ylabel("Amount of survived")
#
# plt.show()

# #--------------- Sex Histogram
# ax.hist(Sex, linewidth=0.5, edgecolor="white")
# # ax.set(xlim=(0, 4), xticks=range(0, 4))
# ax.set_title("Sex - Survivors")
# ax.set_xlabel("Sex")
# ax.set_ylabel("Amount of survived")
#
# plt.show()

#--------------- Age Histogram
# plt.figure(figsize=(10,4))
sns.histplot(data=Age,  binwidth=1, kde=True)
sns.displot(data=Age,  binwidth=1, kde=True, col=Age)
# ax.hist(Age, bins=80, linewidth=0.5, edgecolor="white")
# ax.set(xlim=(0, 80), xticks=range(0, 80))
ax.set_title("Age - Survivors")
ax.set_xlabel("Age")
ax.set_ylabel("Amount of survived")

#
plt.show()


# print(Pclass)
# print("------------------")
# print(y)