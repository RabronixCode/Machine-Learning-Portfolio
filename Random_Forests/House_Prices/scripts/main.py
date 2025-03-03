from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler

import plotting as plot

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df_train = pd.read_csv(r"D:\Python_Projects\Machine_Learning_Portfolio\Random_Forests\House_Prices\data\train.csv")
df_test = pd.read_csv(r"D:\Python_Projects\Machine_Learning_Portfolio\Random_Forests\House_Prices\data\test.csv")

print(df_train.describe())
print(df_train.isnull().sum())
print(df_train['SalePrice'].isnull().sum())

df_train['LotFrontage'] = df_train['LotFrontage'].replace("NA", 0) # Filling NA with 0 since it probably means there is no connection to the street
df_train = df_train.drop('MasVnrType', axis=1) # Dropped because too many NaNs
df_train['GarageYrBlt'] = df_train['GarageYrBlt'].replace("NA", 0)
df_train = df_train.drop(df_train[(df_train.Electrical == 'NA')].index)
df_train = df_train.drop('Neighborhood', axis=1) # Too many unique values
df_train = df_train.drop('Exterior1st', axis=1) # Useless info because we have ExterQual
df_train = df_train.drop('Exterior2nd', axis=1) # Useless info because we have ExterQual

# _-----------------------------------
df_test['LotFrontage'] = df_test['LotFrontage'].replace("NA", 0) # Filling NA with 0 since it probably means there is no connection to the street
df_test = df_test.drop('MasVnrType', axis=1) # Dropped because too many NaNs
df_test['GarageYrBlt'] = df_test['GarageYrBlt'].replace("NA", 0)
df_test = df_test.drop(df_test[(df_test.Electrical == 'NA')].index)
df_test = df_test.drop('Neighborhood', axis=1) # Too many unique values
df_test = df_test.drop('Exterior1st', axis=1) # Useless info because we have ExterQual
df_test = df_test.drop('Exterior2nd', axis=1) # Useless info because we have ExterQual


numerical_features = df_train.select_dtypes(include='number').columns
categorical_features = df_train.select_dtypes(include='object').columns

#plot.heatmap_only_target(df_train, numerical_features, 'SalePrice')
# IMPORTANT COLUMNS (IN CORRELATION WITH THE TARGET COLUMN)
# OverallQual - 0.79

ordinal_features = ['LandSlope', 'BsmtExposure',
                    'PavedDrive', 'Utilities']

nominal_features = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'LotConfig', 'Condition1', 'Condition2',
                    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Foundation', 'Heating', 'CentralAir', 'Electrical',
                    'GarageType', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder()

for c in nominal_features:
    df_train[c] = label_encoder.fit_transform(df_train[c])
    df_test[c] = label_encoder.fit_transform(df_test[c])
for c in ordinal_features:
    df_train[c] = label_encoder.fit_transform(df_train[c])
    df_test[c] = label_encoder.fit_transform(df_test[c])


# ExterQual, ExterCond, BsmtQual, BsmtCond,   HeatingQC, KitchenQual, FIreplaceQual, GarageQual, GarageCond, PoolQC
custom_qual = {'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4}
df_train['ExterQual'] = df_train['ExterQual'].map(custom_qual)
df_train['ExterCond'] = df_train['ExterCond'].map(custom_qual)
df_train['BsmtQual'] = df_train['BsmtQual'].map(custom_qual)
df_train['BsmtCond'] = df_train['BsmtCond'].map(custom_qual)
df_train['HeatingQC'] = df_train['HeatingQC'].map(custom_qual)
df_train['KitchenQual'] = df_train['KitchenQual'].map(custom_qual)
df_train['FireplaceQu'] = df_train['FireplaceQu'].map(custom_qual)
df_train['GarageQual'] = df_train['GarageQual'].map(custom_qual)
df_train['GarageCond'] = df_train['GarageCond'].map(custom_qual)
df_train['PoolQC'] = df_train['PoolQC'].map(custom_qual)
# BsmtExposure
custom_qual = {'Gd': 0, 'Av': 1, 'Mn': 2, 'No': 3, 'nan': 4}
df_train['BsmtExposure'] = df_train['BsmtExposure'].map(custom_qual)
# BsmtFinType1, BsmtFinType2
custom_qual = {'GLQ': 0, 'ALQ': 1, 'BLQ': 2, 'Rec': 3, 'LwQ': 4, 'Unf': 5, 'nan': 6}
df_train['BsmtFinType1'] = df_train['BsmtFinType1'].map(custom_qual)
df_train['BsmtFinType2'] = df_train['BsmtFinType2'].map(custom_qual)
# GarageFinish
custom_qual = {'Fin': 0, 'RFn': 1, 'Unf': 2, 'nan': 3}
df_train['GarageFinish'] = df_train['GarageFinish'].map(custom_qual)
# Functional
custom_qual = {'Typ': 0, 'Min1': 1, 'Min2': 2, 'Mod': 3, 'Maj1': 4, 'Maj2': 5, 'Sev': 6, 'Sal': 7}
df_train['Functional'] = df_train['Functional'].map(custom_qual)



# _----------------------------------------
# ExterQual, ExterCond, BsmtQual, BsmtCond,   HeatingQC, KitchenQual, FIreplaceQual, GarageQual, GarageCond, PoolQC
custom_qual = {'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4}
df_test['ExterQual'] = df_test['ExterQual'].map(custom_qual)
df_test['ExterCond'] = df_test['ExterCond'].map(custom_qual)
df_test['BsmtQual'] = df_test['BsmtQual'].map(custom_qual)
df_test['BsmtCond'] = df_test['BsmtCond'].map(custom_qual)
df_test['HeatingQC'] = df_test['HeatingQC'].map(custom_qual)
df_test['KitchenQual'] = df_test['KitchenQual'].map(custom_qual)
df_test['FireplaceQu'] = df_test['FireplaceQu'].map(custom_qual)
df_test['GarageQual'] = df_test['GarageQual'].map(custom_qual)
df_test['GarageCond'] = df_test['GarageCond'].map(custom_qual)
df_test['PoolQC'] = df_test['PoolQC'].map(custom_qual)
# BsmtExposure
custom_qual = {'Gd': 0, 'Av': 1, 'Mn': 2, 'No': 3, 'nan': 4}
df_test['BsmtExposure'] = df_test['BsmtExposure'].map(custom_qual)
# BsmtFinType1, BsmtFinType2
custom_qual = {'GLQ': 0, 'ALQ': 1, 'BLQ': 2, 'Rec': 3, 'LwQ': 4, 'Unf': 5, 'nan': 6}
df_test['BsmtFinType1'] = df_test['BsmtFinType1'].map(custom_qual)
df_test['BsmtFinType2'] = df_test['BsmtFinType2'].map(custom_qual)
# GarageFinish
custom_qual = {'Fin': 0, 'RFn': 1, 'Unf': 2, 'nan': 3}
df_test['GarageFinish'] = df_test['GarageFinish'].map(custom_qual)
# Functional
custom_qual = {'Typ': 0, 'Min1': 1, 'Min2': 2, 'Mod': 3, 'Maj1': 4, 'Maj2': 5, 'Sev': 6, 'Sal': 7}
df_test['Functional'] = df_test['Functional'].map(custom_qual)


X_train = df_train.drop('SalePrice', axis=1)
y_train = df_train['SalePrice']

X_test = df_test

feature_names = [i for i in X_train]
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

importances = rf_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
#plt.show()


# KEEP ONLY IMPORTANT FEATURES
importance_threshold = 0.005
important_features = [feature for feature, importance in zip(feature_names, importances) if importance > importance_threshold]

X_train_selected = X_train[important_features]
X_test_selected = X_test[important_features]
#plot.heatmap_only_features(df_train, important_features)
#plot.feature_hist(X_train_selected, important_features)
plot.feature_box_plot(X_train_selected, important_features)

X_train_selected = X_train_selected.drop('GarageYrBlt', axis=1) # Heatmap 
X_train_selected = X_train_selected.drop('GarageCars', axis=1) # Heatmap


X_test_selected = X_test_selected.drop('GarageYrBlt', axis=1) # Heatmap 
X_test_selected = X_test_selected.drop('GarageCars', axis=1) # Heatmap

# LotArea - a lot of outliers - check how many above Q3
# OverallQual - under Q1 outlier drop

# BsmtFinSF1 - above Q3 outliers
# BsmtUnfSF - above Q3
# TotalBsmtSF - above Q3 - 1 below Q1
# 1stFlrSF - Above Q3
# GrLivArea - above Q3
# GarageArea - above Q3
"""
param_grid = {
    'n_estimators': [100, 200, 300],
    'criterion': ['squared_error', 'absolute_error'],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2'],
    'max_leaf_nodes': [None, 25, 50, 100]
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=3, n_jobs=-1)
grid_search.fit(X_train_selected, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)
"""

rf = RandomForestRegressor(criterion='squared_error', max_depth=20, max_features='sqrt', max_leaf_nodes=None, min_samples_split=2, n_estimators=100)
rf.fit(X_train_selected, y_train)

y_train_pred = rf.predict(X_train_selected)

mse = mean_squared_error(y_train, y_train_pred)
print(mse)
rmse = root_mean_squared_error(y_train, y_train_pred)
print(rmse)
mae = mean_absolute_error(y_train, y_train_pred)
print(mae)
r2 = r2_score(y_train, y_train_pred)
print(r2)


y_pred_test = rf.predict(X_test_selected)
print(y_pred_test)
# Create a submission DataFrame
submission = pd.DataFrame({
    'Id': df_test['Id'],
    'SalePrice': y_pred_test
})

# Save to CSV
submission.to_csv(r"D:\Python_Projects\Machine_Learning_Portfolio\Random_Forests\House_Prices\data\submission.csv", index=False)
