import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



# - Temperature (T) in the range 1.81°C to 37.11°C,
# - Ambient Pressure (AP) in the range 992.89-1033.30 milibar,
# - Relative Humidity (RH) in the range 25.56% to 100.16%
# - Exhaust Vacuum (V) in the range 25.36-81.56 cm Hg
# - Net hourly electrical energy output (PE) 420.26-495.76 MW (Target we are trying to predict)


ccpp_data = pd.read_csv('/Users/javier/PycharmProjects/machine-learning-foundations/folder/CCPP_data.csv')

target = ['PE']
features = ['AT', 'V', 'AP', 'RH']



# Split the data into training and testing sets
X = ccpp_data[features]
y = ccpp_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

print(y_pred)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mean Squared Error (MSE): The MSE measures the average squared difference between the actual and predicted values.
# In your case, an MSE of 20.27 means that, on average, the model's predictions are off by approximately 20.27 units squared.
# Lower MSE values are generally better, but the "goodness" of MSE depends on the specific context of your problem and the scale of your target variable.
print(f"Mean Squared Error: {mse}")



# 1.	R-squared (R2): The R2 score measures the proportion of the variance in the target variable that is predictable from the independent variables (features).
# An R2 value of 0.93 indicates that approximately 93% of the variance in the target variable is explained by the features used in the model.
# Higher R2 values are better, and a value close to 1 indicates a good fit between the model and the data.


print(f"R-squared: {r2}")