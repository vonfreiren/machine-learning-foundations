import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

# Load the dataset
ccpp_data = pd.read_csv('/Users/javier/PycharmProjects/machine-learning-foundations/folder/CCPP_data.csv')

# Define the target and features
target = ['PE']
features = ['AT', 'V', 'AP', 'RH']

# Split the data into input features (X) and target variable (y)
X = ccpp_data[features]
y = ccpp_data[target]

# Create a Linear Regression model
model = LinearRegression()

# Perform cross-validation (e.g., 5-fold)
num_folds = 5
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
r2_scorer = make_scorer(r2_score)
mse_scores = -cross_val_score(model, X, y, cv=num_folds, scoring=mse_scorer)
r2_scores = cross_val_score(model, X, y, cv=num_folds, scoring=r2_scorer)

# Calculate the mean and standard deviation of MSE and R-squared scores
mean_mse = mse_scores.mean()
std_mse = mse_scores.std()
mean_r2 = r2_scores.mean()
std_r2 = r2_scores.std()

print(f"Mean Squared Error (cross-validated): {mean_mse} +/- {std_mse}")
print(f"R-squared (cross-validated): {mean_r2} +/- {std_r2}")