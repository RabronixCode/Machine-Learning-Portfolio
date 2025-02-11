from matplotlib import pyplot as plt
import pandas as pd

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

numerical_features = ['age', 'bmi', 'children', 'charges']
categorical_features = ['sex', 'smoker', 'region']

# Histogram
df[numerical_features].hist(figsize=(10,10), grid=False, edgecolor='black', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()


fig = plt.figure(figsize=(10,10))
df[numerical_features].boxplot(patch_artist=True, notch=True, vert=0, grid=False)
plt.show()

fig = plt.figure(figsize=(10,10))
plt.bar(df['sex'], df['charges'])
plt.bar(df['smoker'], df['charges'])
plt.bar(df['region'], df['charges'])
plt.show()


plt.scatter(df['age'], df['charges'])
plt.show()