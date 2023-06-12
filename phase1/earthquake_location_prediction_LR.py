import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt


# Read the dataset
data = pd.read_pickle('catalog.pickle')

# Convert data types of various columns in the DataFrame to float
data.Lat_ref = data.Lat_ref.astype(float)
data.Long_ref = data.Long_ref.astype(float)
data.Lat_bias = data.Lat_bias.astype(float)
data.Long_bias = data.Long_bias.astype(float)
data.Depth_ref = data.Depth_ref.astype(float)
data.Depth_bias = data.Depth_bias.astype(float)

# Define the feature columns (bias lat, long, depth)
features = ['Lat_bias', 'Long_bias', 'Depth_bias']

# Define the target columns (ref lat, long, depth)
targets = ['Lat_ref', 'Long_ref', 'Depth_ref']

# Split the data into training and testing sets
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Create the linear regression model
# model = LinearRegression()
# Fit the model on the training data
# model.fit(train_data[features], train_data[targets])
model = GradientBoostingRegressor(random_state=42)
# Create a dictionary to store the models for each target
models = {}

for target in targets:
    # Create the Gradient Boosting Regressor model
    model = GradientBoostingRegressor(random_state=42)
    # Fit the model to the data
    model.fit(train_data[features], train_data[target])
    # Store the model in the dictionary
    models[target] = model
    # Evaluate the model on the testing data
    score = model.score(test_data[features], test_data[targets])
    print(f'R^2 score: {score:.2f}')



# Create a dictionary to store the predictions for each target
_predictions = {}

for target in targets:
    # Get the model for this target
    model = models[target]
    # Use the model to predict the target value for the test data
    _predictions[target] = model.predict(test_data[features])

# Convert the dictionary of predictions to a DataFrame
predictions = pd.DataFrame(_predictions)

# The DataFrame predictions_df now contains the predicted values for each target.

# Create a dictionary to store the predictions for each target
_training = {}

for target in targets:
    # Get the model for this target
    model = models[target]
    # Use the model to predict the target value for the test data
    _training[target] = model.predict(test_data[features])

# Convert the dictionary of predictions to a DataFrame
training = pd.DataFrame(_training)

# The DataFrame predictions_df now contains the predicted values for each target.


# Make predictions using the model
# predictions = model.predict(test_data[features])
# training = model.predict(train_data[features])

# Plot the predictions against the actual values
plt.figure(figsize=(10, 3))
# depth_mean = data.Depth_ref.mean()
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.scatter(test_data[features[i]], test_data[targets[i]], s=5, alpha=0.5, color='blue', label='Actual')
    plt.scatter(predictions[:, i], test_data[targets[i]],  s=5, alpha=0.5, color='red', label='Predicted')
    # plt.scatter(training[:, i], train_data[targets[i]], s=5, alpha=0.5, color='green', label='Trained')
    feature_mean = test_data[features[i]].mean()
    plt.axline((feature_mean, feature_mean), slope=1)
    plt.xlabel(f'{features[i]}')
    plt.ylabel(f'{targets[i]}')
    xmin = min(data[targets[i]].min(), predictions[:, i].min())
    xmax = max(data[targets[i]].max(), predictions[:, i].max())
    plt.xlim(left=xmin, right=xmax)
    plt.ylim(bottom=xmin, top=xmax)
    plt.gca().set_aspect("equal")
    plt.legend()

plt.tight_layout()
plt.show()
