# Orient Express
A library to accelerate model deployments to Vertex AI directly from colab notebooks

![train-resized](https://github.com/user-attachments/assets/f1ed32ec-07d9-4d48-8b96-3323db6b5091)

## Installation

```
pip install orient_express
```

## Example

### Train Model

Train a regular model. In the example below, it's xgboost model, trained on the Titanic dataset.

```python

# Import necessary libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the Titanic dataset
data = sns.load_dataset('titanic').dropna(subset=['survived'])  # Dropping rows with missing target labels

# Select features and target
X = data[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
y = data['survived']

# Define preprocessing for numeric columns (impute missing values and scale features)
numeric_features = ['age', 'fare', 'sibsp', 'parch']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical columns (impute missing values and one-hot encode)
categorical_features = ['pclass', 'sex', 'embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline that first transforms the data, then trains an XGBoost model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)
```

## Upload Model To Model Registry

```python

model_wrapper  = ModelExpress(model=model,
                             project_name='my-project-name',
                             region='us-central1',
                             bucket_name='my-artifacts-bucket',
                             model_name='titanic')
model_wrapper.upload()
```

## Local Inference (Without Online Prediction Endpoint)

The following code will download the last model from the model registry and run the inference locally.

```python

# create input dataframe
titanic_data = {
    "pclass": [1],          # Passenger class (1st, 2nd, 3rd)
    "sex": ["female"],      # Gender
    "age": [29],            # Age
    "sibsp": [0],           # Number of siblings/spouses aboard
    "parch": [0],           # Number of parents/children aboard
    "fare": [100.0],        # Ticket fare
    "embarked": ["S"]       # Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
}
input_df = pd.DataFrame(titanic_data)

# init the model wrapper
model_wrapper  = ModelExpress(project_name='my-project-name',
                             region='us-central1',
                             model_name='titanic')

# Run inference locally
# It will download the most recent version from the model registry automatically
model_wrapper.local_predict(input_df)
```

## Pin Model Version

In many cases, the pipeline should be pinned to a specific model version so the model can only
be updated explicitly. Just pass a `model_version` parameter when instantiating the ModelExpress wrapper.

```python

# init the model wrapper
model_wrapper  = ModelExpress(project_name='my-project-name',
                             region='us-central1',
                             model_name='titanic',
                             model_version=11)
```

## Remote Inference (With Online Prediction Endpoint)

Make sure the model is deployed:
```python

model_wrapper  = ModelExpress(model=model,
                             project_name='my-project-name',
                             region='us-central1',
                             bucket_name='my-artifacts-bucket',
                             model_name='titanic')

# upload the version to the registry and deploy it to the endpoint
model_wrapper.deploy()
```

Run inference with `remote_predict` method. It will make a remote call to the endpoint without fetching the model locally.

```python

titanic_data = {
    "pclass": [1],             # Passenger class (1st, 2nd, 3rd)
    "sex": ["female"],         # Gender
    "age": [29],               # Age
    "sibsp": [0],              # Number of siblings/spouses aboard
    "parch": [0],              # Number of parents/children aboard
    "fare": [100.0],           # Ticket fare
    "embarked": ["S"]          # Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
}
df = pd.DataFrame(titanic_data)

model_wrapper.remote_predict(df)
```
