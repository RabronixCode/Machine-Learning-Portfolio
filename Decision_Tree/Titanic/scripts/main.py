import time
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import skew, kurtosis
import scipy.stats as stats
from sklearn import tree

df = pd.read_csv(r"D:\Python_Projects\Machine_Learning_Portfolio\Decision_Tree\Titanic\data\train.csv")
df_test = pd.read_csv(r"D:\Python_Projects\Machine_Learning_Portfolio\Decision_Tree\Titanic\data\test.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.dtypes)
print(df.describe())
print(df.duplicated().sum())
print(df.nunique())

df_cleaned = df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df_test = df_test.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

"""
df_cleaned.hist(grid=False)
plt.show()

plt.boxplot(df_cleaned['Fare'])
plt.show()
# MAYBE CAP FARE

sns.pairplot(df_cleaned)
plt.show()
# Most people are between 20 and 50 - Fare is mostly low - Survived is half as not survived


corr = df_cleaned.drop(['Sex', 'Embarked'], axis=1).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=1, linecolor="black")
plt.show()
# No Correlations

fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(2,3,1)
ax1 = sns.violinplot(data=df_cleaned, x='Survived', y='Fare')

ax2 = fig.add_subplot(2,3,2)
ax2 = sns.violinplot(data=df_cleaned, x='Survived', y='Age')

ax3 = fig.add_subplot(2,3,3)
ax3 = sns.violinplot(data=df_cleaned, x='Survived', y='Embarked')

ax4 = fig.add_subplot(2,3,4)
ax4 = sns.violinplot(data=df_cleaned, x='Survived', y='Sex')

ax4 = fig.add_subplot(2,3,5)
ax4 = sns.violinplot(data=df_cleaned, x='Survived', y='SibSp')

ax5 = fig.add_subplot(2,3,6)
ax5 = sns.violinplot(data=df_cleaned, x='Survived', y='Parch')
plt.show()
# More females, higher fare, less siblings = better survival rate


sns.kdeplot(data=df_cleaned, x='Survived', y='Pclass')
plt.show()
# More people in lower class died


sns.countplot(data=df_cleaned, x='Pclass')
plt.show()
# Most people in 3rd class


sns.scatterplot(data=df_cleaned, x='Age', y='Fare', hue='Survived')
plt.show()
sns.scatterplot(data=df_cleaned, x='Pclass', y='Fare', hue='Survived')
plt.show()
sns.scatterplot(data=df_cleaned, x='Parch', y='Fare', hue='Survived')
plt.show()
sns.stripplot(data=df_cleaned, x='Parch', y='Fare', hue='Survived')
plt.show()
sns.stripplot(data=df_cleaned, x='Pclass', y='Fare', hue='Survived')
plt.show()
sns.stripplot(data=df_cleaned, x='Age', y='Fare', hue='Survived')
plt.show()
# Higher class, higher fare = survival rate higher
"""


#age_skew = skew(df_cleaned['Age'].dropna())
#age_kurt = kurtosis(df_cleaned['Age'].dropna())
#print(age_skew, age_kurt)

#stats.probplot(df_cleaned['Age'].dropna(), dist="norm", plot=plt)
#plt.show()
# Approximately normally distributed

#df_cleaned = df_cleaned.dropna(subset=['Embarked'])
df_cleaned = df_cleaned.fillna(df_cleaned['Age'].mean())


l_encoder = LabelEncoder()

# BINNING Age
bin_edges = [0, 16, 35, 50, 100]
bin_labels = ['Kid', 'Adult', 'Middle-aged', 'Senior']
df_cleaned['Age'] = pd.cut(df_cleaned['Age'], bins=bin_edges, labels=bin_labels, include_lowest=False)
df_cleaned['Age']= l_encoder.fit_transform(df_cleaned['Age']) # 1, 0, 2, 3 ORDER (MAYBE CUSTOM MAP IT)

df_cleaned['Sex'] = l_encoder.fit_transform(df_cleaned['Sex'])

# LOG TRANSFORMATION on Fare
df_cleaned['Fare'] = np.log1p(df_cleaned['Fare'])



# TEST DATASET ______
df_test = df_test.fillna(df_test['Age'].mean())

# BINNING Age
bin_edges = [0, 16, 35, 50, 100]
bin_labels = ['Kid', 'Adult', 'Middle-aged', 'Senior']
df_test['Age'] = pd.cut(df_test['Age'], bins=bin_edges, labels=bin_labels, include_lowest=False)
df_test['Age']= l_encoder.fit_transform(df_test['Age']) # 1, 0, 2, 3 ORDER (MAYBE CUSTOM MAP IT)

df_test['Sex'] = l_encoder.fit_transform(df_test['Sex'])

# LOG TRANSFORMATION on Fare
df_test['Fare'] = np.log1p(df_test['Fare'])


X_train = df_cleaned.drop('Survived', axis=1)
y_train = df_cleaned['Survived']

X_test = df_test


"""
param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [1, 3, 5, 9, 12, 15],
    'max_features': ['sqrt', 'log2'],
    'splitter': ['best', 'random']
}

# Perform Grid Search
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy', verbose=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)


param_grid = {
    'min_samples_split': [1, 5, 10, 15, 25, 40, 50, 100],
    'min_samples_leaf': [1, 5, 10, 15, 20, 50]
}

# Perform Grid Search
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy', verbose=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)


param_grid = {
    'min_weight_fraction_leaf': [0, 0.15, 0.25, 0.5],
    'ccp_alpha': [0, 0.1, 0.25, 0.5, 1],
    'max_leaf_nodes': [1, 5, 10, 25, 50, 100, 200]
}

# Perform Grid Search
grid_search = GridSearchCV(DecisionTreeClassifier(criterion='entropy'), param_grid, cv=5, scoring='accuracy', verbose=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

"""
dt = DecisionTreeClassifier(criterion='entropy', max_depth=9, max_features='log2', splitter='best', min_samples_leaf=5, min_samples_split=25, ccp_alpha=0, max_leaf_nodes=5, min_weight_fraction_leaf=0, random_state=41)

dt.fit(X_train, y_train)
"""
# Get feature importance 
# HERE THERE CAN BE SO MUCH OPTIMIZATION>>
importances = dt.feature_importances_
feature_names = X_train.columns

# Sort & plot
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 5))
plt.title("Feature Importance")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.show()

low_importance_features = ['Fare', 'Parch', 'Age', 'PassengerId']  # Replace with actual features
X_train = X_train.drop(columns=low_importance_features)
X_test = X_test.drop(columns=low_importance_features)

dt.fit(X_train, y_train)
"""
y_train_pred = dt.predict(X_train)

# Accuracy & Classification Report
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

print("Classification Report:\n", classification_report(y_train, y_train_pred))

y_pred_test = dt.predict(X_test)

# Create a submission DataFrame
submission = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    'Survived': y_pred_test  # Predicted survival (0 or 1)
})

# Save to CSV
submission.to_csv(r"D:\Python_Projects\Machine_Learning_Portfolio\Decision_Tree\Titanic\data\submission.csv", index=False)

print("Submission file saved as submission.csv")

tree.plot_tree(dt)
plt.show()