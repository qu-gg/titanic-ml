import pandas as pd

df = pd.read_csv("train.csv")


# True labels
labels = df["Survived"]
print(labels)

labels.to_csv("labels.csv", index=False)

# Train vectors
train = df[["Pclass", "Sex", "Age", "Fare"]]

train['Sex'] = pd.factorize(train['Sex'])[0]

isnull = train["Age"].isnull()
sample = train["Age"].dropna().sample(isnull.sum(), replace=True).values
train.loc[isnull, 'Age'] = sample

train.to_csv("dataset.csv", index=False)

# Train vectors
df = pd.read_csv("test.csv")
test = df[["Pclass", "Sex", "Age", "Fare"]]

test['Sex'] = pd.factorize(test['Sex'])[0]

isnull = test["Age"].isnull()
sample = test["Age"].dropna().sample(isnull.sum(), replace=True).values
test.loc[isnull, 'Age'] = sample

test.to_csv("test_set.csv", index=False)

test_id = df["PassengerId"]
test_id.to_csv("test_ids.csv", index=False)