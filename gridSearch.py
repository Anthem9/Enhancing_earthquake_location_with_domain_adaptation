from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


# Read the dataset
data = pd.read_pickle('phase1/catalog.pickle')

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
model = RandomForestRegressor(n_estimators=400, random_state=42, min_samples_leaf=10)

# Set the parameters by cross-validation
param_grid = {
    'n_estimators': [300, 400, 500],     # number of trees in the forest
    'max_depth': [10, 12, 14, 16, 18,
                  20, 22, 24, 26, 28],   # maximum number of levels in tree
    'min_samples_leaf': [10, 20],       # minimum number of samples required to be at a leaf node
}

# Create the grid search object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

# Fit the grid search object to the data
grid_search.fit(train_data[features], train_data[targets])

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters: ", best_params)

# Create a new random forest model with the best parameters
best_model = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                   # max_features=best_params['max_features'],
                                   max_depth=best_params['max_depth'],
                                   min_samples_leaf=best_params['min_samples_leaf'],
                                   random_state=42)

# Fit the new model to the data
best_model.fit(train_data[features], train_data[targets])

# Evaluate the new model on the testing data
score = best_model.score(test_data[features], test_data[targets])
print(f'Improved R^2 score: {score:.2f}')