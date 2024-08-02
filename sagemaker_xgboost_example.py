import boto3
import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import joblib  # For saving the scaler

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
# Initialize Sagemaker Session
role = 'sagremaker-execution-role'
session = sagemaker.Session()
bucket = 'sagemaker-examples-ip-exhaustion'
prefix = 'subnet-prediction'

# Load the synthetic data
data = pd.read_csv("synthetic_data.csv")

# Feature Engineering
data['ips_allocated_last_month'] = data.groupby('subnet_id')['ips_allocated'].shift(1)
data['ips_allocated_last_month'] = data['ips_allocated_last_month'].fillna(0)
data['ips_allocated_last_2months'] = data.groupby('subnet_id')['ips_allocated'].shift(2)
data['ips_allocated_last_2months'] = data['ips_allocated_last_2months'].fillna(0)
data['avg_ips_allocated_last_3months'] = data.groupby('subnet_id')['ips_allocated'].rolling(window=3).mean().reset_index(level=0, drop=True)
data['avg_ips_allocated_last_3months'] = data['avg_ips_allocated_last_3months'].fillna(0)

# Define features and target
features = ['subnet_id', 'subnet_size', 'month', 'year', 'ips_allocated_last_month', 'ips_allocated_last_2months', 'avg_ips_allocated_last_3months']
target = 'ips_allocated'

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')

# Save training and validation data to CSV
train_data = pd.concat([pd.DataFrame(X_train_scaled, columns=features), y_train.reset_index(drop=True)], axis=1)
val_data = pd.concat([pd.DataFrame(X_val_scaled, columns=features), y_val.reset_index(drop=True)], axis=1)

train_data.to_csv('synthetic_train_data.csv', index=False)
val_data.to_csv('synthetic_val_data.csv', index=False)

# Upload the training and validation data to S3
train_s3_path = sagemaker_session.upload_data('synthetic_train_data.csv', bucket=bucket, key_prefix=prefix)
val_s3_path = sagemaker_session.upload_data('synthetic_val_data.csv', bucket=bucket, key_prefix=prefix)

# Define the XGBoost container image
container = get_image_uri(sagemaker_session.boto_region_name, 'xgboost', '1.2-2')

# Hyperparameter tuning job configuration
xgb = sagemaker.estimator.Estimator(container,
                                    role,
                                    instance_count=1,
                                    instance_type='ml.m5.xlarge',
                                    output_path=f's3://{bucket}/xgboost/output',
                                    sagemaker_session=sagemaker_session)

xgb.set_hyperparameters(objective='reg:squarederror',
                        eval_metric='rmse',
                        num_round=300)

# Define the hyperparameter ranges
hyperparameter_ranges = {
    'eta': sagemaker.parameter.ContinuousParameter(0.01, 0.3),
    'max_depth': sagemaker.parameter.IntegerParameter(3, 10),
    'subsample': sagemaker.parameter.ContinuousParameter(0.5, 1.0),
    'colsample_bytree': sagemaker.parameter.ContinuousParameter(0.5, 1.0)
}

# Set up the tuner
tuner = sagemaker.tuner.HyperparameterTuner(xgb,
                                            objective_metric_name='validation:rmse',
                                            hyperparameter_ranges=hyperparameter_ranges,
                                            max_jobs=20,
                                            max_parallel_jobs=3,
                                            objective_type='Minimize')

# Define the inputs for the training job
train_input = TrainingInput(train_s3_path, content_type='csv')
val_input = TrainingInput(val_s3_path, content_type='csv')

# Start the hyperparameter tuning job
tuner.fit({'train': train_input, 'validation': val_input})

# Get the best training job
best_training_job_name = tuner.best_training_job()
print(f"Best training job name: {best_training_job_name}")

# Get the model data path for the best training job
client = boto3.client('sagemaker')
response = client.describe_training_job(TrainingJobName=best_training_job_name)
model_data_path = response['ModelArtifacts']['S3ModelArtifacts']
print(f"Model data path: {model_data_path}")

# Save the model data path to a file for later use
with open('model_data_path.json', 'w') as f:
    json.dump({'model_data_path': model_data_path}, f)
