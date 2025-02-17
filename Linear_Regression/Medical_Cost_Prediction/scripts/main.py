import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures, StandardScaler


pd.set_option('display.max_rows', None)
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
df[numerical_features].hist(figsize=(10,10), grid=False, edgecolor='black', alpha=0.7, bins=15)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Checking whether relationship between age and charges is linear
sns.scatterplot(data=df, x='age', y='charges', hue='smoker')
plt.show()


fig = plt.figure(figsize=(10,10))
df[numerical_features].boxplot(patch_artist=True, notch=True, grid=False)
plt.show()

# Scatter plot to check the outliers - seems like the charges price varies and is not specifically high or low so yes indeed we can delete these rows and consider these bmi values as outliers
plt.scatter(df[df['bmi'] > df.bmi.quantile(0.75) + 1.5*(df.bmi.quantile(0.75) - df.bmi.quantile(0.25))]['bmi'], df[df['bmi'] > df.bmi.quantile(0.75) + 1.5*(df.bmi.quantile(0.75) - df.bmi.quantile(0.25))]['charges'], alpha=0.7)
plt.show()


# Can see that there are almost twice as much paitents who smoke then who don't smoke
fig = plt.figure(figsize=(10,10))
plt.bar(df['sex'], df['charges'])
plt.bar(df['smoker'], df['charges'])
plt.bar(df['region'], df['charges'])
plt.show()


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


# CATEGORICAL BOX PLOTS
fig = plt.figure(figsize=(15,10))

ax1 = fig.add_subplot(1,3,1)
ax1 = sns.boxplot(data=df, x='bmi', y='region', hue='smoker')

ax2 = fig.add_subplot(1,3,2)
ax2 = sns.boxplot(data=df, x='bmi', y='sex')

ax3 = fig.add_subplot(1,3,3)
ax3 = sns.boxplot(data=df, x='bmi', y='smoker')

plt.tight_layout()
plt.show()


# BOXPLOT WITH SWARM PLOT
ax3 = sns.boxplot(data=df, x='smoker', y='bmi')
sns.swarmplot(x='smoker', y='bmi', data=df, color='black', alpha=0.6)
plt.show()

# VIOLIN PLOT
sns.violinplot(x='smoker', y='bmi', data=df)
plt.show()

# Heatmap
#dataplot = sns.heatmap(df.corr(numeric_only=True), cmap="YlGnBu", annot=True)
#plt.show()

# We can see that there is no correlation between values and also that smokers have signifanctly higher charges
corr_matrix = df[numerical_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=1, linecolor="black")
plt.show()

pivot_table = df.pivot_table(index='sex', columns='smoker', values='charges', aggfunc='mean')

plt.figure(figsize=(6, 5))
sns.heatmap(pivot_table, annot=True, cmap="Blues", linewidths=0.5)
plt.show()

pivot_table = df.pivot_table(index='children', columns='sex', values='bmi', aggfunc='mean')

plt.figure(figsize=(6, 5))
sns.heatmap(pivot_table, annot=True, cmap="Blues", linewidths=0.5)
plt.show()


# We can see smokers are the worst :)
sns.pairplot(df, hue='smoker', palette='coolwarm')
plt.show()

# Just in case (pairplot with regression line) because we will be predicting values
sns.pairplot(df, kind='reg')
plt.show()

df_wo_outliers = df[df['bmi'] < (df.bmi.quantile(0.75) + 1.5*(df.bmi.quantile(0.75) - df.bmi.quantile(0.25)))]

#time.sleep(19)
# Scaling AGE with MinMaxScaler
min_max = MinMaxScaler()
df_wo_outliers['age'] = min_max.fit_transform(df_wo_outliers[['age']])

# Scaling AGE with Bins
#df_wo_outliers['age'] = pd.cut(df_wo_outliers['age'], bins=3, labels=['Young', 'Middle Age', 'Old'])

# Scaling BMI with Standard Scaler
z_score = StandardScaler()
df_wo_outliers['bmi'] = z_score.fit_transform(df_wo_outliers[['bmi']])

# Log Transforming CHARGES
df_wo_outliers['charges'] = np.log1p(df_wo_outliers['charges'])
#sns.scatterplot(data=df_wo_outliers, x='bmi', y='charges', hue='age')
#plt.show()

encoder = OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False)
encoded_array = encoder.fit_transform(df_wo_outliers[['smoker', 'sex', 'region']])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['smoker', 'sex', 'region']))
# Had to reset index since the dataframes seemed to have filled some rows with NaNs
df_wo_outliers.reset_index(drop=True, inplace=True)
encoded_df.reset_index(drop=True, inplace=True)
df_encoded = pd.concat([df_wo_outliers.drop(columns=['smoker', 'sex', 'region']), encoded_df], axis=1)

df_encoded = df_encoded.drop(columns=['region_northeast'])

df_encoded['bmi_smoker'] = df_encoded['bmi'] * df_encoded['smoker_yes']
df_encoded['age_smoker'] = df_encoded['age'] * df_encoded['smoker_yes']
df_encoded['age_bmi'] = df_encoded['age'] * df_encoded['bmi']
#df_encoded['children_smoker'] = df_encoded['children'] * df_encoded['smoker_yes']

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = df_encoded.columns
vif_data["VIF"] = [variance_inflation_factor(df_encoded.values, i) for i in range(df_encoded.shape[1])]

print(vif_data)


X = df_encoded.drop(columns='charges')
y = df_encoded['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ridge = Ridge()
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

mse = mean_squared_error(y_test, ridge_pred)
rmse = root_mean_squared_error(y_test, ridge_pred)
mae = mean_absolute_error(y_test, ridge_pred)
r2 = r2_score(y_test, ridge_pred)
print("MEAN SQUARE ERROR RIDGE",mse)
print("ROOT MEAN SQUARE ERROR RIDGE",rmse)
print("MEAN ABSOLUTE ERROR RIDGE",mae)
print("R2 SCORE RIDGE",r2)

sns.residplot(x=ridge_pred, y=y_test-ridge_pred, lowess=True, line_kws={"color": "red"})
plt.show()

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MEAN SQUARE ERROR ",mse)
print("ROOT MEAN SQUARE ERROR ",rmse)
print("MEAN ABSOLUTE ERROR ",mae)
print("R2 SCORE ",r2)


sns.residplot(x=y_pred, y=y_test-y_pred, lowess=True, line_kws={"color": "red"})
plt.show()