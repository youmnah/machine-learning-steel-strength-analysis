import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import roc_curve
from plotly.subplots import make_subplots
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

## Data Loading: Read CSV file into a DataFrame
df = pd.read_csv('https://storage.googleapis.com/kagglesdsdata/datasets/2681508/4604033/steel_strength.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240610%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240610T084620Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=604a2bcfbae02b017ae01b6c495fb7ee786b7ebc3500e37c1bf2128bcb75d1e788819f2c53ea72f6e1e56c63d9a15e9057714a5d5fac0aaef21bb48aaa9a88b9329f83ff3dc0bede7e31817f8b206aa414bff0d276157970d9808c6b2d80954d18ac1221f4b441d8ec4ca70cda03bc1ffc26f11fe3e158d2474ae2bc4ba831ea5f6e736fbe8bcffb7e314da912e62ef413f75ad4bc1f40508d6e6cc26928802e7bf4eb7d040073e6d418e29dc78a8a7ab47573f138781d44eba4d2d8e6c971f2e765b488063e48a8652b3cf25246bd52b12e666627547d96110f5580dcb34bb0221d826c8c6b28af1434d09a3d2eadcdeec9da29f15ca5d29d2fe1d881f9c883')
"""
## Data Understanding
print("Rows Count:" + str(df.shape[0]))
print("Columns Count:" + str(df.shape[1]))
print("First 5 rows:")
print(df.head())
print("Last 5 rows:")
print(df.tail())
print("Random 5 rows:")
print(df.sample(5))

# Data types and null values
print("Data types and null values:")
print(df.info())
# Check the column with missing values
print("Check the column with missing values:")
print(df.isna().sum())
# Check the number of 0 values in each column
zero_counts = df.eq(0).sum()
print("Check the number of 0 values in each column:")
print(zero_counts)
# Matrix Visualization to check the missing values in the dataset
msno.matrix(df)
# Summary Statistics
print("Summary Statistics:")
print(df.describe(include='all'))
"""

## Data Manipulation
# Drop formula column as it is not significant for the dataset analysis
df = df.drop(['formula'], axis = 1)
# Remove rows where carbon or elongation is 0 or missing
df = df.dropna(subset=['c','elongation']).loc[(df[['c','elongation']] != 0).all(axis=1)]
"""
print("After Data Manipulation:")
print("Rows Count:" + str(df.shape[0]))
print("Columns Count:" + str(df.shape[1]))
print("Random 5 rows:")
print(df.sample(5))

# Outliers
def detect_outliers_iqr_columns(df, columns):
    outliers = pd.DataFrame()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR        
        outliers_in_col = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers = pd.concat([outliers, outliers_in_col])
    return outliers.drop_duplicates()

num_cols = ['c','mn','si','cr','ni','mo','v','n','nb','co','w','al','ti']

outliers_iqr = detect_outliers_iqr_columns(df, num_cols)
print("Outliers detected using IQR method:")
print(outliers_iqr)

colors = sns.color_palette("husl", len(num_cols))
plt.figure(figsize=(10, len(num_cols)))
for i, col in enumerate(num_cols, 1):
  plt.subplot(len(num_cols), 1, i)
  sns.boxplot(x=df[col], orient='h', color=colors[i-1])
  plt.xlabel(col)
  plt.tight_layout()

## Data Visualization
# Correlation Heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')

features =  ['c','mn','si','cr','ni','mo','v','n','nb','co','w','al','ti']
# Yield Strength Correlation
label = "yield strength"
correlation_matrix = df.drop(['tensile strength'], axis = 1).drop(['elongation'], axis = 1).corrwith(df[label]).sort_values(ascending=False)
#print("Yield Strength Correlation:")
#print(correlation_matrix)
# Plot the correlation using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix.to_frame(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Yield Strength Correlation Heatmap')

# Yield Strength Scatter Plot
sns.pairplot(df, x_vars=features, y_vars=label, kind='scatter', diag_kind='kde')
plt.title('Yield Strength Scatter Plot Matrix')

# Tensile Strength Correlation
label = "tensile strength"
correlation_matrix = df.drop(['yield strength'], axis = 1).drop(['elongation'], axis = 1).corrwith(df[label]).sort_values(ascending=False)
#print("Tensile Strength Correlation:")
#print(correlation_matrix)
# Plot the correlation using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix.to_frame(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Tensile Strength Correlation Heatmap')

# Tensile Strength Scatter Plot
sns.pairplot(df, x_vars=features, y_vars=label, kind='scatter', diag_kind='kde')
plt.title('Tensile Strength Scatter Plot Matrix')
"""

## Data Splitting
# Split the data into training and testing where yield strength is dependent variable and the independant variables are without tensile strength and elongation
YX = df.drop(['yield strength','tensile strength','elongation'], axis=1)
Yy = df['yield strength']
YX_train, YX_test, Yy_train, Yy_test = train_test_split(YX, Yy, test_size=0.2, random_state=42)
print('Yield Strength Training data shape:', YX_train.shape)
print('Yield Strength Testing data shape:', YX_test.shape)

# Split the data into training and testing where tensile strength is dependent variable and the independant variables are without yield strength and elongation
TX = df.drop(['yield strength','tensile strength','elongation'], axis=1)
Ty = df['tensile strength']
TX_train, TX_test, Ty_train, Ty_test = train_test_split(TX, Ty, test_size=0.2, random_state=42)
print('Tensile Strength Training data shape:', TX_train.shape)
print('Tensile Strength Testing data shape:', TX_test.shape)

num_cols = ['c','mn','si','cr','ni','mo','v','n','nb','co','w','al','ti']
# Scale Training Data
scaler = StandardScaler()
YX_train[num_cols] = scaler.fit_transform(YX_train[num_cols])
# Apply the same transformation learned from the training to test data
YX_test[num_cols] = scaler.transform(YX_test[num_cols])

## Model Training and Evaluation
"""
# KNN
knn_model = KNeighborsRegressor(n_neighbors=3).fit(YX_train, Yy_train)
y_pred_knn = knn_model.predict(YX_test)
mae = metrics.mean_absolute_error(Yy_test, y_pred_knn)
mse = metrics.mean_squared_error(Yy_test, y_pred_knn)
rmse = np.sqrt(metrics.mean_squared_error(Yy_test, y_pred_knn))
r2score = metrics.r2_score(Yy_test, y_pred_knn)
print("Evaluation of KNN model:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2score}")

plt.figure(figsize=(8, 6))
plt.scatter(Yy_test, y_pred_knn, color='blue', label='Actual vs Predicted')
plt.plot([min(Yy_test), max(Yy_test)], [min(Yy_test), max(Yy_test)], linestyle='--', color='red', label='Perfect Prediction')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# SVR
svr_model = SVR(kernel='linear').fit(YX_train, Yy_train)
y_pred_svr = svr_model.predict(YX_test)
svrmae = metrics.mean_absolute_error(Yy_test, y_pred_svr)
svrmse = metrics.mean_squared_error(Yy_test, y_pred_svr)
svrrmse = np.sqrt(metrics.mean_squared_error(Yy_test, y_pred_svr))
svrr2score = metrics.r2_score(Yy_test, y_pred_svr)
print("Evaluation of Support Vector Regressor:")
print(f"MAE: {svrmae}")
print(f"MSE: {svrmse}")
print(f"RMSE: {svrrmse}")
print(f"R2 Score: {svrr2score}")

plt.figure(figsize=(8, 6))
plt.scatter(Yy_test, y_pred_svr, color='blue', label='Actual vs Predicted')
plt.plot([min(Yy_test), max(Yy_test)], [min(Yy_test), max(Yy_test)], linestyle='--', color='red', label='Perfect Prediction')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# Random Forest
rf_model = RandomForestRegressor().fit(YX_train, Yy_train)
y_pred_rf = rf_model.predict(YX_test)
rfmae = metrics.mean_absolute_error(Yy_test, y_pred_rf)
rfmse = metrics.mean_squared_error(Yy_test, y_pred_rf)
rfrmse = np.sqrt(metrics.mean_squared_error(Yy_test, y_pred_rf))
rfr2score = metrics.r2_score(Yy_test, y_pred_rf)
print("Evaluation of Random Forest Regressor:")
print(f"MAE: {rfmae}")
print(f"MSE: {rfmse}")
print(f"RMSE: {rfrmse}")
print(f"R2 Score: {rfr2score}")

plt.figure(figsize=(8, 6))
plt.scatter(Yy_test, y_pred_rf, color='blue', label='Actual vs Predicted')
plt.plot([min(Yy_test), max(Yy_test)], [min(Yy_test), max(Yy_test)], linestyle='--', color='red', label='Perfect Prediction')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# Linear Regression
linear_regression_model = LinearRegression().fit(YX_train, Yy_train)
y_pred_linear = linear_regression_model.predict(YX_test)
linear_mae = metrics.mean_absolute_error(Yy_test, y_pred_linear)
linear_mse = metrics.mean_squared_error(Yy_test, y_pred_linear)
linear_rmse = np.sqrt(metrics.mean_squared_error(Yy_test, y_pred_linear))
linear_r2score = metrics.r2_score(Yy_test, y_pred_linear)
print("Evaluation of Linear Regression:")
print(f"MAE: {linear_mae}")
print(f"MSE: {linear_mse}")
print(f"RMSE: {linear_rmse}")
print(f"R2 Score: {linear_r2score}")

plt.figure(figsize=(8, 6))
plt.scatter(Yy_test, y_pred_linear, color='blue', label='Actual vs Predicted')
plt.plot([min(Yy_test), max(Yy_test)], [min(Yy_test), max(Yy_test)], linestyle='--', color='red', label='Perfect Prediction')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# Decision Tree
decision_tree_model = DecisionTreeRegressor().fit(YX_train, Yy_train)
y_pred_dt = decision_tree_model.predict(YX_test)
dt_mae = metrics.mean_absolute_error(Yy_test, y_pred_dt)
dt_mse = metrics.mean_squared_error(Yy_test, y_pred_dt)
dt_rmse = np.sqrt(metrics.mean_squared_error(Yy_test, y_pred_dt))
dt_r2score = metrics.r2_score(Yy_test, y_pred_dt)
print("Evaluation of Decision Tree Regressor:")
print(f"MAE: {dt_mae}")
print(f"MSE: {dt_mse}")
print(f"RMSE: {dt_rmse}")
print(f"R2 Score: {dt_r2score}")

plt.figure(figsize=(8, 6))
plt.scatter(Yy_test, y_pred_dt, color='blue', label='Actual vs Predicted')
plt.plot([min(Yy_test), max(Yy_test)], [min(Yy_test), max(Yy_test)], linestyle='--', color='red', label='Perfect Prediction')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# Adaboost
adaboost_model = AdaBoostRegressor().fit(YX_train, Yy_train)
y_pred_adaboost = adaboost_model.predict(YX_test)
adaboost_mae = metrics.mean_absolute_error(Yy_test, y_pred_adaboost)
adaboost_mse = metrics.mean_squared_error(Yy_test, y_pred_adaboost)
adaboost_rmse = np.sqrt(metrics.mean_squared_error(Yy_test, y_pred_adaboost))
adaboost_r2score = metrics.r2_score(Yy_test, y_pred_adaboost)
print("Evaluation of Adaboost Regressor:")
print(f"MAE: {adaboost_mae}")
print(f"MSE: {adaboost_mse}")
print(f"RMSE: {adaboost_rmse}")
print(f"R2 Score: {adaboost_r2score}")

plt.figure(figsize=(8, 6))
plt.scatter(Yy_test, y_pred_adaboost, color='blue', label='Actual vs Predicted')
plt.plot([min(Yy_test), max(Yy_test)], [min(Yy_test), max(Yy_test)], linestyle='--', color='red', label='Perfect Prediction')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# Gradient Boosting
gb_model = GradientBoostingRegressor().fit(YX_train, Yy_train)
y_pred_gb = gb_model.predict(YX_test)
gb_mae = metrics.mean_absolute_error(Yy_test, y_pred_gb)
gb_mse = metrics.mean_squared_error(Yy_test, y_pred_gb)
gb_rmse = np.sqrt(metrics.mean_squared_error(Yy_test, y_pred_gb))
gb_r2score = metrics.r2_score(Yy_test, y_pred_gb)
print("Evaluation of Gradient Boosting Regressor:")
print(f"MAE: {gb_mae}")
print(f"MSE: {gb_mse}")
print(f"RMSE: {gb_rmse}")
print(f"R2 Score: {gb_r2score}")

plt.figure(figsize=(8, 6))
plt.scatter(Yy_test, y_pred_gb, color='blue', label='Actual vs Predicted')
plt.plot([min(Yy_test), max(Yy_test)], [min(Yy_test), max(Yy_test)], linestyle='--', color='red', label='Perfect Prediction')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# Voting
# Define the individual regressors
estimators = list()
estimators.append(('knn', knn_model))
estimators.append(('svr', svr_model))
estimators.append(('rf', rf_model))
estimators.append(('linear', linear_regression_model))
estimators.append(('dt', decision_tree_model))
estimators.append(('adaboost', adaboost_model))
estimators.append(('gb', gb_model))

voting_model = VotingRegressor(estimators=estimators)
voting_model.fit(YX_train, Yy_train)
y_pred_voting = voting_model.predict(YX_test)
voting_mae = metrics.mean_absolute_error(Yy_test, y_pred_voting)
voting_mse = metrics.mean_squared_error(Yy_test, y_pred_voting)
voting_rmse = np.sqrt(metrics.mean_squared_error(Yy_test, y_pred_voting))
voting_r2score = metrics.r2_score(Yy_test, y_pred_voting)
print("Evaluation of Voting Regressor:")
print(f"MAE: {voting_mae}")
print(f"MSE: {voting_mse}")
print(f"RMSE: {voting_rmse}")
print(f"R2 Score: {voting_r2score}")

plt.figure(figsize=(8, 6))
plt.scatter(Yy_test, y_pred_voting, color='blue', label='Actual vs Predicted')
plt.plot([min(Yy_test), max(Yy_test)], [min(Yy_test), max(Yy_test)], linestyle='--', color='red', label='Perfect Prediction')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

results = {}
results['Algorithm'] =  ['KNN', 'SVR', 'Random Forest', 'Linear Regression', 'Decision Tree', 'AdaBoost', 'Gradient Boosting', 'Voting']
results['MAE'] = [mae,svrmae,rfmae,linear_mae,dt_mae,adaboost_mae,gb_mae,voting_mae]
results['MSE'] = [mse,svrmse,rfmse,linear_mse,dt_mse,adaboost_mse,gb_mse,voting_mse]
results['RMSE'] = [rmse,svrrmse,rfrmse,linear_rmse,dt_rmse,adaboost_rmse,gb_rmse,voting_rmse]
results['R2 Score'] = [r2score,svrr2score,rfr2score,linear_r2score,dt_r2score,adaboost_r2score,gb_r2score,voting_r2score]
results_df = pd.DataFrame(results)
# Comparison table
print(results_df)

## Visualization
# Bar Chart
algorithms = results_df['Algorithm']
MAE = results_df['MAE']
MSE = results_df['MSE']
RMSE = results_df['RMSE']
R2Score = results_df['R2 Score']

plt.figure(figsize=(12, 8))
plt.bar(algorithms, MAE, color='skyblue', label='MAE')
plt.xlabel('MAE')
plt.title('Metrix Comparison')
plt.legend()

plt.figure(figsize=(12, 8))
plt.bar(algorithms, MSE, color='grey', label='MSE')
plt.xlabel('MSE')
plt.title('Metrix Comparison')
plt.legend()

plt.figure(figsize=(12, 8))
plt.bar(algorithms, RMSE, color='green', label='RMSE')
plt.xlabel('RMSE')
plt.title('Metrix Comparison')
plt.legend()

plt.figure(figsize=(12, 8))
plt.bar(algorithms, R2Score, color='yellow', label='R2 Score')
plt.xlabel('R2 Score')
plt.title('Metrix Comparison')
plt.legend()

# Plotting RMSE on the primary y-axis
fig, ax1 = plt.subplots(figsize=(15, 10))
ax1.set_ylabel('RMSE')
ax1.bar(algorithms, RMSE, color='r', alpha=0.5, label='RMSE')
ax1.tick_params(axis='y')
# Creating a secondary y-axis for R-squared
ax2 = ax1.twinx()
ax2.set_ylabel('R² Score')
ax2.plot(algorithms, R2Score, color='purple', linestyle='-', marker='o', label='R² Score')
ax2.tick_params(axis='y')
# Show legend for both axes
fig.legend(loc='upper right')
# Adding x-axis labels
plt.xticks(algorithms)
# Show the plot
plt.tight_layout()
plt.title('Comparison of RMSE and R2 Score')
"""

# KNN
knn_model = KNeighborsRegressor(n_neighbors=3).fit(TX_train, Ty_train)
y_pred_knn = knn_model.predict(TX_test)
mae = metrics.mean_absolute_error(Ty_test, y_pred_knn)
mse = metrics.mean_squared_error(Ty_test, y_pred_knn)
rmse = np.sqrt(metrics.mean_squared_error(Ty_test, y_pred_knn))
r2score = metrics.r2_score(Ty_test, y_pred_knn)

# SVR
svr_model = SVR(kernel='linear').fit(TX_train, Ty_train)
y_pred_svr = svr_model.predict(TX_test)
svrmae = metrics.mean_absolute_error(Ty_test, y_pred_svr)
svrmse = metrics.mean_squared_error(Ty_test, y_pred_svr)
svrrmse = np.sqrt(metrics.mean_squared_error(Ty_test, y_pred_svr))
svrr2score = metrics.r2_score(Ty_test, y_pred_svr)

# Random Forest
rf_model = RandomForestRegressor().fit(TX_train, Ty_train)
y_pred_rf = rf_model.predict(TX_test)
rfmae = metrics.mean_absolute_error(Ty_test, y_pred_rf)
rfmse = metrics.mean_squared_error(Ty_test, y_pred_rf)
rfrmse = np.sqrt(metrics.mean_squared_error(Ty_test, y_pred_rf))
rfr2score = metrics.r2_score(Ty_test, y_pred_rf)

# Linear Regression
linear_regression_model = LinearRegression().fit(TX_train, Ty_train)
y_pred_linear = linear_regression_model.predict(TX_test)
linear_mae = metrics.mean_absolute_error(Ty_test, y_pred_linear)
linear_mse = metrics.mean_squared_error(Ty_test, y_pred_linear)
linear_rmse = np.sqrt(metrics.mean_squared_error(Ty_test, y_pred_linear))
linear_r2score = metrics.r2_score(Ty_test, y_pred_linear)

# Decision Tree
decision_tree_model = DecisionTreeRegressor().fit(TX_train, Ty_train)
y_pred_dt = decision_tree_model.predict(TX_test)
dt_mae = metrics.mean_absolute_error(Ty_test, y_pred_dt)
dt_mse = metrics.mean_squared_error(Ty_test, y_pred_dt)
dt_rmse = np.sqrt(metrics.mean_squared_error(Ty_test, y_pred_dt))
dt_r2score = metrics.r2_score(Ty_test, y_pred_dt)

# Adaboost
adaboost_model = AdaBoostRegressor().fit(TX_train, Ty_train)
y_pred_adaboost = adaboost_model.predict(TX_test)
adaboost_mae = metrics.mean_absolute_error(Ty_test, y_pred_adaboost)
adaboost_mse = metrics.mean_squared_error(Ty_test, y_pred_adaboost)
adaboost_rmse = np.sqrt(metrics.mean_squared_error(Ty_test, y_pred_adaboost))
adaboost_r2score = metrics.r2_score(Ty_test, y_pred_adaboost)

# Gradient Boosting
gb_model = GradientBoostingRegressor().fit(TX_train, Ty_train)
y_pred_gb = gb_model.predict(TX_test)
gb_mae = metrics.mean_absolute_error(Ty_test, y_pred_gb)
gb_mse = metrics.mean_squared_error(Ty_test, y_pred_gb)
gb_rmse = np.sqrt(metrics.mean_squared_error(Ty_test, y_pred_gb))
gb_r2score = metrics.r2_score(Ty_test, y_pred_gb)

# Voting
# Define the individual regressors
estimators = list()
estimators.append(('knn', knn_model))
estimators.append(('svr', svr_model))
estimators.append(('rf', rf_model))
estimators.append(('linear', linear_regression_model))
estimators.append(('dt', decision_tree_model))
estimators.append(('adaboost', adaboost_model))
estimators.append(('gb', gb_model))

voting_model = VotingRegressor(estimators=estimators)
voting_model.fit(TX_train, Ty_train)
y_pred_voting = voting_model.predict(TX_test)
voting_mae = metrics.mean_absolute_error(Ty_test, y_pred_voting)
voting_mse = metrics.mean_squared_error(Ty_test, y_pred_voting)
voting_rmse = np.sqrt(metrics.mean_squared_error(Ty_test, y_pred_voting))
voting_r2score = metrics.r2_score(Ty_test, y_pred_voting)

results = {}
results['Algorithm'] =  ['KNN', 'SVR', 'Random Forest', 'Linear Regression', 'Decision Tree', 'AdaBoost', 'Gradient Boosting', 'Voting']
results['MAE'] = [mae,svrmae,rfmae,linear_mae,dt_mae,adaboost_mae,gb_mae,voting_mae]
results['MSE'] = [mse,svrmse,rfmse,linear_mse,dt_mse,adaboost_mse,gb_mse,voting_mse]
results['RMSE'] = [rmse,svrrmse,rfrmse,linear_rmse,dt_rmse,adaboost_rmse,gb_rmse,voting_rmse]
results['R2 Score'] = [r2score,svrr2score,rfr2score,linear_r2score,dt_r2score,adaboost_r2score,gb_r2score,voting_r2score]
results_df = pd.DataFrame(results)
# Comparison table
print(results_df)


## Visualization
# Bar Chart
algorithms = results_df['Algorithm']
MAE = results_df['MAE']
MSE = results_df['MSE']
RMSE = results_df['RMSE']
R2Score = results_df['R2 Score']

plt.figure(figsize=(12, 8))
plt.bar(algorithms, MAE, color='skyblue', label='MAE')
plt.xlabel('MAE')
plt.title('Metrix Comparison')
plt.legend()

plt.figure(figsize=(12, 8))
plt.bar(algorithms, MSE, color='grey', label='MSE')
plt.xlabel('MSE')
plt.title('Metrix Comparison')
plt.legend()

plt.figure(figsize=(12, 8))
plt.bar(algorithms, RMSE, color='green', label='RMSE')
plt.xlabel('RMSE')
plt.title('Metrix Comparison')
plt.legend()

plt.figure(figsize=(12, 8))
plt.bar(algorithms, R2Score, color='yellow', label='R2 Score')
plt.xlabel('R2 Score')
plt.title('Metrix Comparison')
plt.legend()

# Plotting RMSE on the primary y-axis
fig, ax1 = plt.subplots(figsize=(15, 10))
ax1.set_ylabel('RMSE')
ax1.bar(algorithms, RMSE, color='r', alpha=0.5, label='RMSE')
ax1.tick_params(axis='y')
# Creating a secondary y-axis for R-squared
ax2 = ax1.twinx()
ax2.set_ylabel('R² Score')
ax2.plot(algorithms, R2Score, color='purple', linestyle='-', marker='o', label='R² Score')
ax2.tick_params(axis='y')
# Show legend for both axes
fig.legend(loc='upper right')
# Adding x-axis labels
plt.xticks(algorithms)
# Show the plot
plt.tight_layout()
plt.title('Comparison of RMSE and R2 Score')

# To display the plot and show the matrix use this 
plt.show()

