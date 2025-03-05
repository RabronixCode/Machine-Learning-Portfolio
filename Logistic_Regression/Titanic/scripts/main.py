from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

pd.set_option("display.max_rows", None)

df = pd.read_csv(r"D:\Python_Projects\Machine_Learning_Portfolio\Logistic_Regression\Titanic\data\train.csv")
df_test = pd.read_csv(r"D:\Python_Projects\Machine_Learning_Portfolio\Logistic_Regression\Titanic\data\test.csv")


print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.dtypes)
print(df.describe())
print(df.duplicated().sum())
print(df.nunique())

print(df_test.isnull().sum())

# We can see that we don't need some columns (PassengerId, Name, Ticket, Embarked, Cabin) since these values do not decide whether someone survived or not
# Numerical features - ints (Survived (target), Pclass, SibSp, Parch), floats (Age, Fare)
# Categorical features - object (Sex)
# Also we see that there are a lot of missing values for Cabin (687) which can maybe lead to dropping that column and also in Age (177), Embarked (2)
# There are NO DUPLICATES

# TRAIN DATASET
# Dropped unnecessary columns
df_cleaned = df.drop(['Name', 'Ticket', 'Embarked', 'Cabin'], axis=1)
df_cleaned['Age'] = df_cleaned.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median())) # CONSIDER REMOVING AGE

# TEST DATASET
df_test = df_test.drop(['Name', 'Ticket', 'Embarked', 'Cabin'], axis=1)
df_test['Age'] = df_test.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median())) # CONSIDER REMOVING AGE

numerical_features = ['Survived', 'Pclass', 'SibSp', 'Parch', 'Age', 'Fare']
categorical_features = ['Sex']
"""
# UNIVARIATE
df_cleaned.hist(grid=False, figsize=(15,15), edgecolor='black', alpha=0.7, bins=15)
plt.show()

fig = plt.figure(figsize=(10,10))
df[numerical_features].boxplot(patch_artist=True, notch=True, grid=False)
plt.show()

sns.barplot(x='Survived', y='Fare', data=df_cleaned)
plt.show()



# BIVARIATE
sns.pairplot(df_cleaned, hue='Sex', palette='coolwarm')
plt.show()


fig = plt.figure(figsize=(15,15))
ax1 = fig.add_subplot(1,3,1)
ax1 = sns.violinplot(x='Survived', y='Fare', data=df_cleaned)
ax2 = fig.add_subplot(1,3,2)
ax2 = sns.violinplot(x='Survived', y='Age', data=df_cleaned)
ax3 = fig.add_subplot(1,3,3)
ax3 = sns.violinplot(x='Survived', y='Pclass', data=df_cleaned)

plt.show()

#sns.stripplot(data=df_cleaned, x='Survived', y='Fare')
#sns.stripplot(data=df_cleaned, x='Survived', y='Age')
sns.stripplot(data=df_cleaned, x='Survived', y='SibSp')
plt.show()

#sns.kdeplot(data=df_cleaned, x='Survived', y='Age')
sns.kdeplot(data=df_cleaned, x='Survived', y='Fare')
plt.show()

# CORRELATION HEATMAP
corr_matrix = df_cleaned[numerical_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=1, linecolor="black")
plt.show()

"""


# TRAIN DATASET ______________________
#df_cleaned = df_cleaned[(df_cleaned['Fare'] < (df_cleaned.Fare.quantile(0.75) + 2.5*(df_cleaned.Fare.quantile(0.75) - df_cleaned.Fare.quantile(0.25)))) & (df_cleaned['Fare'] != 0)]

# BINNING Age
bin_edges = [0, 16, 35, 50, 100]
bin_labels = ['Kid', 'Adult', 'Middle-aged', 'Senior']

df_cleaned['Age'] = pd.cut(df_cleaned['Age'], bins=bin_edges, labels=bin_labels, include_lowest=False)

l_encoder = LabelEncoder()
df_cleaned['Age']= l_encoder.fit_transform(df_cleaned['Age']) # 1, 0, 2, 3 ORDER (MAYBE CUSTOM MAP IT)

df_cleaned['Sex'] = l_encoder.fit_transform(df_cleaned['Sex'])

# LOG TRANSFORMATION on Fare
df_cleaned['Fare'] = np.log1p(df_cleaned['Fare'])


# TEST DATASET __________________
df_test['Age'] = df_test.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))
df_test['Fare'].fillna(df_test['Fare'].median(), inplace=True)

# BINNING Age
bin_edges = [0, 16, 35, 50, 100]
bin_labels = ['Kid', 'Adult', 'Middle-aged', 'Senior']

df_test['Age'] = pd.cut(df_test['Age'], bins=bin_edges, labels=bin_labels, include_lowest=False)

l_encoder = LabelEncoder()
df_test['Age']= l_encoder.fit_transform(df_test['Age']) # 1, 0, 2, 3 ORDER (MAYBE CUSTOM MAP IT)

df_test['Sex'] = l_encoder.fit_transform(df_test['Sex'])

# LOG TRANSFORMATION on Fare
df_test['Fare'] = np.log1p(df_test['Fare'])



X_train = df_cleaned.drop('Survived', axis=1)
y_train = df_cleaned['Survived']

X_test = df_test




from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2', 'elasticnet'],  # L1 = Lasso, L2 = Ridge
    'solver': ['liblinear', 'saga', 'sag', 'newton-cholesky', 'newton-cg', 'lbfgs']
}

# Perform Grid Search
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Train with best parameters
best_model = grid_search.best_estimator_

model = LogisticRegression(C=1, penalty='l2', solver='newton-cholesky')
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)

# Accuracy & Classification Report
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

print("Classification Report:\n", classification_report(y_train, y_train_pred))

# HIGHER THRESHOLD
# Get probability predictions
y_train_probs = model.predict_proba(X_train)[:, 1]  # Probabilities for "Survived" class
# Change the threshold from 0.5 to 0.4
y_train_pred_new = (y_train_probs > 0.59).astype(int)
train_accuracy = accuracy_score(y_train, y_train_pred_new)
print(f"Training Accuracy: {train_accuracy:.4f}")
# Evaluate new model
print("New Classification Report:\n", classification_report(y_train, y_train_pred_new))



# Predict survival on test.csv
y_pred_test = model.predict(X_test)

# Create a submission DataFrame
submission = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    'Survived': y_pred_test  # Predicted survival (0 or 1)
})

# Save to CSV
submission.to_csv(r"D:\Python_Projects\Machine_Learning_Portfolio\Logistic_Regression\Titanic\data\submission.csv", index=False)

print("Submission file saved as submission.csv")