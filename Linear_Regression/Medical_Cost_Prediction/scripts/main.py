import time
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

# We need to predict charges based on some information because we want to know an estimate/prediction
# In the dataset we have columns: feature_columns = [age, sex, bmi, children, smoker, region], target_column = [charges]

# Now we are going to gather & understand the data
# Inspect the structure
df = pd.read_csv(r"D:\Python_Projects\Machine_Learning_Portfolio\Linear_Regression\Medical_Cost_Prediction\data\insurance.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.dtypes)
print(df.describe())
print(df.duplicated().sum())
print(df.nunique())

# We Have 1338 rows, no null cells, (age, children) = int64, (bmi, charges) = float64, (sex, smoker, region) = object
# Now we will visualize the data

numerical_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

"""
# Histogram
df[numerical_features].hist(figsize=(10,10), grid=False, edgecolor='black', alpha=0.7, bins=15)
plt.xlabel('Value')
plt.ylabel('Frequency')
#plt.show()


fig = plt.figure(figsize=(10,10))
df[numerical_features].boxplot(patch_artist=True, notch=True, grid=False)
plt.show()

# Scatter plot to check the outliers - seems like the charges price varies and is not specifically high or low so yes indeed we can delete these rows and consider these bmi values as outliers
plt.scatter(df[df['bmi'] > df.bmi.quantile(0.75) + 1.5*(df.bmi.quantile(0.75) - df.bmi.quantile(0.25))]['bmi'], df[df['bmi'] > df.bmi.quantile(0.75) + 1.5*(df.bmi.quantile(0.75) - df.bmi.quantile(0.25))]['charges'], alpha=0.7)
#plt.show()


# Can see that there are almost twice as much paitents who smoke then who don't smoke
fig = plt.figure(figsize=(10,10))
plt.bar(df['sex'], df['charges'])
plt.bar(df['smoker'], df['charges'])
plt.bar(df['region'], df['charges'])
plt.show()
"""

# We can see here that only people who smoke have some important vaiability in charges..rest of categorical have no significant impact
# Create a single figure with 3 subplots
fig = plt.figure(figsize=(12, 5))

# First scatter plot
ax1 = fig.add_subplot(1, 3, 1)  # 1 row, 3 columns, first plot
ax1 = sns.scatterplot(data=df, x='charges', y='bmi', hue='smoker', palette="Set1")

# Second scatter plot
ax3 = fig.add_subplot(1, 3, 2)  # Third plot
ax4 = sns.scatterplot(data=df, x='charges', y='bmi', hue='region', palette="Set1")

# Third scatter plot
ax3 = fig.add_subplot(1, 3, 3)  # Third plot
ax1 = sns.scatterplot(data=df, x='charges', y='bmi', hue='sex', palette="Set1")

# Adjust layout and show all plots in one window
plt.tight_layout()
plt.show()

# Heatmap
dataplot = sns.heatmap(df.corr(numeric_only=True), cmap="YlGnBu", annot=True)
plt.show()

