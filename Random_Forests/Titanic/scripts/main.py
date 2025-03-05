import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
import plotting as plot
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv(r"D:\Python_Projects\Machine_Learning_Portfolio\Random_Forests\Titanic\data\train.csv")
df_test = pd.read_csv(r"D:\Python_Projects\Machine_Learning_Portfolio\Random_Forests\Titanic\data\test.csv")

print(df_train.describe())
print(df_train.isnull().sum())
print(df_train['Ticket'].duplicated().sum())

df_train = df_train.drop('Cabin', axis=1) # Too many missing values
df_train = df_train.dropna(subset='Embarked') # Dropped 2 rows because embarked is missing

df_test = df_test.drop('Cabin', axis=1) # Too many missing values
df_test = df_test.dropna(subset='Embarked') # Dropped 2 rows because embarked is missing

# BINNING Age and then IMPUTING IT WITH MODE (Most common category!)
bin_edges = [0, 16, 35, 50, 100]
bin_labels = ['Kid', 'Adult', 'Middle-aged', 'Senior']

df_train['Age'] = pd.cut(df_train['Age'], bins=bin_edges, labels=bin_labels, include_lowest=False)

df_train["Age"].fillna(df_train['Age'].mode()[0], inplace=True)

numerical_features = df_train.select_dtypes(include='number').columns
categorical_features = df_train.select_dtypes(include='object').columns

#plot.heatmap_only_features(df_train, numerical_features)
# NO CORRELATIONS

custom_qual = {'Kid': 0, 'Adult': 1, 'Middle-ages': 2, 'Senior': 3}
df_train['Age'] = df_train['Age'].map(custom_qual)

df_train['Title'] = df_train['Name'].str.extract(r'([A-Za-z]+)\.')  # Extracts title before "."
df_train.drop('Name', axis=1, inplace=True)
df_train['Title_Frequency'] = df_train['Title'].map(df_train['Title'].value_counts(normalize=True)) # Frequency Encoding for TITLES
df_train.drop('Title', axis=1, inplace=True)

# ONE HOT ENCODING SEX
df_train = pd.get_dummies(df_train, columns=['Sex'], drop_first=True)
df_train['Sex_male'] = df_train['Sex_male'].astype(int)

# ONE HOT ENCODING EMBARKED
df_train = pd.get_dummies(df_train, columns=['Embarked'])
df_train['Embarked_C'] = df_train['Embarked_C'].astype(int)
df_train['Embarked_Q'] = df_train['Embarked_Q'].astype(int)
df_train['Embarked_S'] = df_train['Embarked_S'].astype(int)


df_train['Ticket'] = df_train['Ticket'].map(df_train['Ticket'].value_counts(normalize=True)) # There are duplicates so we are using FREQUENCY ENCODING



# BINNING Age and then IMPUTING IT WITH MODE (Most common category!)
bin_edges = [0, 16, 35, 50, 100]
bin_labels = ['Kid', 'Adult', 'Middle-aged', 'Senior']

df_test['Age'] = pd.cut(df_test['Age'], bins=bin_edges, labels=bin_labels, include_lowest=False)

df_test["Age"].fillna(df_test['Age'].mode()[0], inplace=True)

numerical_features = df_test.select_dtypes(include='number').columns
categorical_features = df_test.select_dtypes(include='object').columns

#plot.heatmap_only_features(df_train, numerical_features)
# NO CORRELATIONS

custom_qual = {'Kid': 0, 'Adult': 1, 'Middle-ages': 2, 'Senior': 3}
df_test['Age'] = df_test['Age'].map(custom_qual)

df_test['Title'] = df_test['Name'].str.extract(r'([A-Za-z]+)\.')  # Extracts title before "."
df_test.drop('Name', axis=1, inplace=True)
df_test['Title_Frequency'] = df_test['Title'].map(df_test['Title'].value_counts(normalize=True)) # Frequency Encoding for TITLES
df_test.drop('Title', axis=1, inplace=True)

# ONE HOT ENCODING SEX
df_test = pd.get_dummies(df_test, columns=['Sex'], drop_first=True)
df_test['Sex_male'] = df_test['Sex_male'].astype(int)

# ONE HOT ENCODING EMBARKED
df_test = pd.get_dummies(df_test, columns=['Embarked'])
df_test['Embarked_C'] = df_test['Embarked_C'].astype(int)
df_test['Embarked_Q'] = df_test['Embarked_Q'].astype(int)
df_test['Embarked_S'] = df_test['Embarked_S'].astype(int)


df_test['Ticket'] = df_test['Ticket'].map(df_test['Ticket'].value_counts(normalize=True)) # There are duplicates so we are using FREQUENCY ENCODING


X_train = df_train.drop('Survived', axis=1)
y_train = df_train['Survived']

X_test = df_test

feature_names = [i for i in X_train]

model = RandomForestClassifier()
model.fit(X_train, y_train)

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()



# KEEP ONLY IMPORTANT FEATURES
importance_threshold = 0.05
important_features = [feature for feature, importance in zip(feature_names, importances) if importance > importance_threshold]

X_train_selected = X_train[important_features]
X_test_selected = X_test[important_features]
"""
param_grid = {
    'n_estimators': [100, 200, 300],
    'criterion': ['gini', 'log_loss', 'entropy'],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2'],
    'max_leaf_nodes': [None, 25, 50, 100]
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=3, n_jobs=-1)
grid_search.fit(X_train_selected, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)
time.sleep(199)
"""
rf = RandomForestClassifier(criterion='gini', max_depth=10, max_features='log2', max_leaf_nodes=None, min_samples_split=5, n_estimators=200)
rf.fit(X_train_selected, y_train)

y_train_pred = rf.predict(X_train_selected)

accuracy = accuracy_score(y_train, y_train_pred)
precision = precision_score(y_train, y_train_pred, zero_division=1)
recall = recall_score(y_train, y_train_pred)
f1 = f1_score(y_train, y_train_pred)

# Confusion Matrix
conf_matrix = confusion_matrix(y_train, y_train_pred)

# Print Results
print("Confusion Matrix:\n", conf_matrix)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_train, y_train_pred, zero_division=1))


y_pred_test = rf.predict(X_test_selected)
print(y_pred_test)
# Create a submission DataFrame
submission = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    'Survived': y_pred_test
})

# Save to CSV
submission.to_csv(r"D:\Python_Projects\Machine_Learning_Portfolio\Random_Forests\Titanic\data\submission.csv", index=False)
