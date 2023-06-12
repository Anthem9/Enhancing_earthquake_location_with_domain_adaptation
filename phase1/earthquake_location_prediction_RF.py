import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

# Create the random forest regression model
model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=1)

# Fit the model on the training data
model.fit(train_data[features], train_data[targets])

# Evaluate the model on the testing data
score = model.score(test_data[features], test_data[targets])
print(f'R^2 score: {score:.2f}')

# Make predictions using the model
predictions = model.predict(test_data[features])
training = model.predict(train_data[features])

# Plot the predictions against the actual values
plt.figure(figsize=(10, 8))
for i in range(3):
    plt.subplot(1, 3, i+1)
    # plt.scatter(test_data[features[i]], test_data[targets[i]], s=5, alpha=0.5, color='blue', label='Actual')
    plt.scatter(training[:, i], train_data[targets[i]], s=5, alpha=0.5, color='green', label='Test')
    plt.scatter(predictions[:, i], test_data[targets[i]], s=5, alpha=0.5, color='blue', label='Train')
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
