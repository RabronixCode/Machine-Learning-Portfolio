# PROJECT OVERVIEW

We used a Linear Regression model to predict yearly medical charges for a person based on age, sex, BMI, region, number of children, and smoking status. The goal is to identify key factors affecting medical costs, enabling predictions based on a regular check-up. This analysis not only helps in predicting medical charges but also provides insights into how lifestyle changes (e.g., lowering BMI) can reduce costs.

To get the best predictions, we tested a couple of different approaches:
1. **Ridge Regression** – Uses L2 regularization to reduce overfitting by penalizing large coefficients.
2. **Regular Linear Regression** – Predicts charges based on input features without regularization.


## Installation

To run this project, follow these steps:

1. **Clone the repository** (if using Git):
git clone https://github.com/RabronixCode/Machine-Learning-Portfolio.git
cd Machine-Learning-Portfolio/Linear_Regression/Medical_Cost_Prediction

2. **Install dependencies** from the `requirements.txt` file:
pip install -r requirements.txt

**Python Version:** Ensure you have Python **3.8+** installed.

Now you are ready to run the project!

## Data Description

The dataset used in this project comes from **medical insurance records** and is sourced from **Kaggle**. It consists of **1,338 entries and 7 features**, where **6 features serve as predictors**, and the **target variable** is `charges`, representing yearly medical costs.

### Dataset Features

| Feature   | Type         | Description |
|-----------|-------------|-------------|
| `age`     | Integer     | Age of the individual |
| `sex`     | Categorical | Gender (`male` or `female`) |
| `bmi`     | Float       | Body Mass Index (BMI), a measure of body fat |
| `children`| Integer     | Number of children/dependents covered by insurance |
| `smoker`  | Categorical | Whether the person is a smoker (`yes` or `no`) |
| `region`  | Categorical | Residential region (`northeast`, `northwest`, `southeast`, `southwest`) |
| `charges` | Float       | Medical insurance cost (target variable) |


## Exploratory Data Analysis (EDA)

To better understand the data, we performed **detailed visualizations**. These plots helped identify feature distributions, relationships between variables, and potential outliers that could impact model performance. The following analysis includes **univariate analysis, bivariate analysis, and correlation heatmaps**.


### Univariate Analysis (Single Feature Distribution)
Univariate analysis helps us understand how each individual feature is distributed.

#### **Histograms**  
We plotted histograms for **`age`**, **`bmi`**, **`children`**, and **`charges`** to observe their distributions:  
- **`age`** – Almost uniformly distributed, with a noticeable spike around **20 years old**, suggesting a concentration of younger individuals in the dataset.  
- **`bmi`** – Follows a **Gaussian (normal) distribution**, making it a good candidate for standardization.  
- **`children`** – Right-skewed, but since it contains only **six distinct values** (0 to 5 children), it is categorical in nature.  
- **`charges`** – Also right-skewed, indicating the presence of high medical costs for some individuals.

#### **Box Plots**  
Box plots were used to **detect outliers** and compare feature distributions:  
- **`age`** and **`children`** had **no outliers**, confirming stable distributions.  
- **`bmi`** and **`charges`** contained outliers, but we **only removed extreme BMI values** since they were **rare, extreme cases** that wouldn't generalize well.  
- We also plotted **box plots for `region`, `sex`, and `smoker` against `bmi`** to check for differences across groups. For example, smokers tend to have a higher BMI range than non-smokers.

#### **Bar Plots**  
Bar plots helped us analyze categorical feature distributions:  
- The dataset is fairly balanced in terms of **sex** and **region** distributions.  
- However, there is a noticeable imbalance in the **smoker vs. non-smoker** ratio, with **significantly more non-smokers** in the dataset.

#### **Box Plot with Swarm Plot**  
A **box plot with a swarm plot overlay** was used to visualize **how BMI varies across smokers and non-smokers**. This visualization confirmed that **BMI is widely spread among smokers**, indicating a higher variance in weight-related health factors.



### Bivariate Analysis (Feature Relationships)
Bivariate analysis examines how different features interact with each other and their impact on `charges`.

#### **Scatter Plots**
- **`charges` vs. `age`** → We introduced a **hue for `smoker` status** and observed a **clear linear trend**, where older individuals (especially smokers) tend to have higher charges.  
- **BMI outliers vs. `charges`** → We plotted BMI outliers against charges to check for any patterns. Since **no meaningful pattern emerged**, we justified dropping those extreme BMI values.  
- **`bmi` vs. `charges` (with different hues)** →  
  - **Smoker as hue** → Smokers consistently have higher medical charges, regardless of BMI.  
  - **Region as hue** → No significant impact on charges.  
  - **Sex as hue** → Minimal difference between males and females.

#### **Pair Plots**
- **Pair plots were used to examine linear relationships** between numerical variables.  
- We plotted pair plots **with and without regression lines** to visually inspect **linear correlations** and detect possible **multicollinearity**.



### Correlation Heatmaps & Pivot Tables
#### **Correlation Heatmap**
- A **heatmap** of numerical features revealed **no strong correlations** between independent variables.  
- This suggests that **each feature contributes independently to the prediction of `charges`**.

#### **Pivot Tables**
We used **pivot tables to analyze category-based trends**:
- **`sex` vs. `smoker` with `charges` (Mean Value)** → Smokers have **significantly higher charges** than non-smokers.  
- **`children` vs. `sex` with `bmi` (Mean Value)** → The average **BMI remains consistent across sexes and number of children**, indicating that these features may have minimal influence on BMI.



## Data Preprocessing

Before training the model, **data preprocessing** was performed to improve model performance and ensure proper feature scaling. The following steps were taken:

1. **Outlier Removal:**  
   - After analyzing the **BMI distribution using visualizations**, we identified extreme outliers that could negatively impact the model.
   - **Removed high BMI values** that fell beyond the interquartile range (IQR) threshold.

2. **Feature Scaling:**  
   - **`age`** → Scaled using **MinMaxScaler** since it has clear lower (`0`) and upper (`100`) boundaries.  
   - **`bmi`** → Standardized using **StandardScaler** because it follows a **Gaussian distribution**.  
   - **`charges`** → Applied **log(1+x) transformation** to normalize its **right-skewed distribution** and then standardized it using **StandardScaler** because the distribution was normalized.

3. **Encoding Categorical Variables:**  
   - Converted **`smoker`**, **`sex`**, and **`region`** into numerical values using **OneHotEncoding**.  
   - Dropped one category for **binary variables** to prevent multicollinearity.  
   - Dropped **one region column** based on **Variance Inflation Factor (VIF) analysis**, as it was redundant.

4. **Feature Engineering:**  
   - Created new **interaction features** to improve model performance:  
     - **`bmi_smoker`** → Interaction between **BMI and smoker status**.  
     - **`age_smoker`** → Interaction between **age and smoker status**.  
     - **`age_bmi`** → Interaction between **age and BMI**.


## Model Training

After preprocessing the data, we trained two regression models to predict medical charges:  
1. **Linear Regression** – A baseline model without regularization.  
2. **Ridge Regression** – A regularized model using **L2 penalty** to prevent overfitting.

---

### Train-Test Split

The dataset was split into **training (80%) and testing (20%) subsets** using `train_test_split()`:
```python
X = df_encoded.drop(columns='charges')
y = df_encoded['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

X_train, X_test -> Contain independent features
y_train, y_test -> Contain the target variable

### Ridge Regression (L2 Regularization)

Ridge Regression was chosen to reduce overfitting by penalizing large coefficients.

```python
ridge = Ridge()
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
```

Reduces variance in the model while maintaining predictive accuracy.

#### Ridge Regression Metrics

```python
mse = mean_squared_error(y_test, ridge_pred)
rmse = root_mean_squared_error(y_test, ridge_pred)
mae = mean_absolute_error(y_test, ridge_pred)
r2 = r2_score(y_test, ridge_pred)
```

MSE (Mean Squared Error) : Measures average squared errors
RMSE (Root Mean Squared Error) : Provides error magnitude in the original scale
MAE (Mean Absolute Error) : Measures average absolute errors
R^2 Score : Explains how much variance in `charges` is captured by the model

#### Residual Plot for Ridge Regression

```python
sns.residplot(x=ridge_pred, y=y_test - ridge_pred, lowess=True, line_kws={"color": "red"})
plt.show()
```

Help detect patterns in prediction errors.
A good model should show residuals randomly scattered around zero, indicating minimal bias.


### Linear Regression (Baseline Model)

To compare performance we also trained a regular Linear Regression model

```python
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
```

No regularization applied, meaning it could overfit if features are highly correlated.

#### Linear Regression Metrics

Same metrics used to compare to Ridge Regression.

#### Residual Plot for Linear Regression

If residuals form a pattern, it suggests the model is missing key relationships.
A random spread indicates a well-fitted model.

### Model Comparison & Insights

| Model               | MSE (Lower is better) | RMSE | MAE  | R² Score (Higher is better) |
|---------------------|----------------------|------|------|----------------------------|
| **Linear Regression** | 0.190 | 0.436 | 0.2194 | 0.7975 |
| **Ridge Regression**  | 0.188 | 0.434 | 0.2199 | 0.7992 |


**Key Insights from Model Performance:**
- Ridge Regression **did not significantly improve performance** over standard Linear Regression.
- **Smoking had the strongest correlation** with higher charges, reinforcing its impact in insurance cost prediction.
- **Residual plots confirmed** that our model captures key trends, but potential **missing variables** (e.g., diet, genetics) could improve performance.


## Conclusion & Future Work

### Summary

This project aimed to predict **yearly medical insurance charges** using **Linear Regression** and **Ridge Regression**. By analyzing the dataset, we identified key factors influencing medical costs, such as **smoking status, BMI, and age**. Our **data preprocessing pipeline** included **outlier removal, feature scaling, encoding categorical variables, and feature engineering**, ensuring the dataset was optimized for model training.

Through **Exploratory Data Analysis (EDA)**, we discovered that **smoking is the most significant factor in determining medical charges**, followed by **BMI and age**. Additionally, we applied **log transformation** to normalize the highly skewed charges distribution, improving model performance.

### Limitations of the Current Approach

While the model performed well, a few limitations should be noted:
- **Feature Engineering Scope** – We only created **three interaction features** (`bmi_smoker`, `age_smoker`, `age_bmi`). More **feature interactions** could be explored.
- **Dataset Size & Diversity** – The dataset contains **1,338 rows**, which may not be large enough for highly generalized predictions.
- **Exclusion of External Factors** – Variables like **exercise habits, diet, genetic predispositions, and socioeconomic status** could significantly impact insurance charges but are not included in this dataset.
- **Imbalanced Data** – The dataset has **more non-smokers than smokers**, which could impact the model’s ability to generalize to smoker-related predictions.

### Future Work & Improvements

To further improve model accuracy and generalizability, the following enhancements could be made:
- **Test More Regression Models** – Explore **Lasso Regression, ElasticNet, or Decision Trees** to compare performance.
- **Hyperparameter Tuning** – Optimize Ridge Regression by fine-tuning the **alpha parameter** for better regularization.
- **Incorporate More Features** – Include external datasets or synthetic features (e.g., lifestyle habits) to improve predictions.
- **Outlier Handling** – Use **advanced anomaly detection techniques** rather than simple IQR-based removal.
- **Cross-Validation** – Implement **K-Fold Cross-Validation** to ensure model performance is robust across different data splits.