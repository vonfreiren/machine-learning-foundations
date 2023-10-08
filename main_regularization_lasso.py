import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
ccpp_data = pd.read_csv('/Users/javier/PycharmProjects/machine-learning-foundations/folder/CCPP_data.csv')

# Define the target and features
target = ['PE']
features = ['AT', 'V', 'AP', 'RH']

# Split the data into training and testing sets
X = ccpp_data[features]
y = ccpp_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Lasso Regression model with regularization strength (alpha)
alpha = 1.0  # You can adjust alpha as needed
lasso_model = Lasso(alpha=alpha)

# Train the Lasso model
lasso_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = lasso_model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
