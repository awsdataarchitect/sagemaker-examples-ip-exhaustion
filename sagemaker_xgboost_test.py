import boto3
import sagemaker
from sagemaker.model import Model
import pandas as pd
import numpy as np
import json
import joblib  

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
# Initialize Sagemaker Session
role = 'sagemaker-execution-role'
session = sagemaker.Session()
bucket = 'sagemaker-examples-ip-exhaustion'
prefix = 'subnet-prediction'

# Define features and target
features = ['subnet_id', 'subnet_size', 'month', 'year', 'ips_allocated_last_month', 'ips_allocated_last_2months', 'avg_ips_allocated_last_3months']
target = 'ips_allocated'

# Load the model data path
with open('model_data_path.json', 'r') as f:
    model_data_info = json.load(f)
model_data_path = model_data_info['model_data_path']
print(f"Loaded model data path: {model_data_path}")

# Create and deploy the model
model = Model(
    model_data=model_data_path,
    image_uri=sagemaker.image_uris.retrieve('xgboost', boto3.Session().region_name, '1.3-1'),
    role=role,
)

# Preprocess the test data before batch transform
test_data = pd.read_csv("synthetic_val_data.csv")

X_test = test_data[features]

# Load the scaler and standardize the test data
scaler = joblib.load('scaler.joblib')
X_test_scaled = scaler.transform(X_test)
test_data_scaled = pd.DataFrame(X_test_scaled, columns=features)
test_data_scaled.to_csv('synthetic_test_data_scaled.csv', index=False, header=False)  # Save without headers

# Upload the preprocessed test data to S3
test_s3_path = sagemaker_session.upload_data('synthetic_test_data_scaled.csv', bucket=bucket, key_prefix=prefix)

# Batch Transform
transformer = model.transformer(
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path=f's3://{bucket}/{prefix}/transform-output'
)

transformer.transform(
    data=test_s3_path,
    content_type='text/csv',
    split_type='Line',
    input_filter='$'  # Include all rows
)

transformer.wait()
transform_output_path = transformer.output_path

output_key = f"{prefix}/transform-output/synthetic_test_data_scaled.csv.out"
s3_client = boto3.client('s3')
response = s3_client.get_object(Bucket=bucket, Key=output_key)
transformed_output = response['Body'].read().decode('utf-8')

# Split the transformed output into lines and then parse each line to extract values
transformed_output_lines = transformed_output.strip().split('\n')
transformed_output_array = []

for line in transformed_output_lines:
    values = line.split(',')  # Assuming the values are comma-separated
    transformed_output_array.append([float(value) for value in values])

# Convert the list of lists to a numpy array
transformed_output_array = np.array(transformed_output_array)

# Reshape the transformed output array to be a 2D array with a single column
transformed_output_array = transformed_output_array.reshape(-1, 1)

# Create a temporary array with the same shape as the original scaler input
temp_array = np.zeros((transformed_output_array.shape[0], len(features)))
# Replace the last column (assuming the prediction column) with the transformed output
temp_array[:, -1] = transformed_output_array[:, 0]

# Inverse transform the temporary array
inverse_transformed_array = scaler.inverse_transform(temp_array)

# Extract the original predictions from the last column
original_output = inverse_transformed_array[:, -1]

# Convert the array to a DataFrame
original_output_df = pd.DataFrame(original_output, columns=[target])

# Display the DataFrame with the original scale

# Display the DataFrame with the original scale
print(original_output_df)
