import boto3
import sagemaker
from sagemaker.transformer import Transformer
import pandas as pd
import numpy as np
import json
import joblib
from time import gmtime, strftime

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = 'sagemaker-execution-role'  # Replace with your SageMaker execution role
bucket = 'sagemaker-examples-ip-exhaustion'
prefix = 'subnet-prediction'

# Define features and target
features = [
    'subnet_id', 'subnet_size', 'month', 'year',
    'ips_allocated_last_month', 'ips_allocated_last_2months', 'avg_ips_allocated_last_3months'
]
target = 'ips_allocated'

# Load the model data path
with open('model_data_path.json', 'r') as f:
    model_data_info = json.load(f)
model_data_path = model_data_info['model_data_path']
print(f"Loaded model data path: {model_data_path}")

# Initialize SageMaker client
sagemaker_client = boto3.client('sagemaker')

# Specify your model package group name
model_package_group_name = 'my-xgboost-packages'  # Replace with your model package group name

# Step 1: Check if the model package group already exists
try:
    response = sagemaker_client.list_model_package_groups()
    existing_groups = [group['ModelPackageGroupName'] for group in response['ModelPackageGroupSummaryList']]
    if model_package_group_name in existing_groups:
        print(f"Model package group '{model_package_group_name}' already exists.")
    else:
        # Create a new model package group if it doesn't exist
        response = sagemaker_client.create_model_package_group(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageGroupDescription='Model package group for XGBoost models'
        )
        print(f"Model package group created with ARN: {response['ModelPackageGroupArn']}")
except boto3.exceptions.Boto3Error as e:
    print(f"An error occurred while checking or creating the model package group: {e}")

# Step 2: Create a new model package
try:
    response = sagemaker_client.create_model_package(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageDescription='Demo model package for XGBoost',
        InferenceSpecification={
            'Containers': [
                {
                    'Image': sagemaker.image_uris.retrieve('xgboost', boto3.Session().region_name, '1.3-1'),
                    'ModelDataUrl': model_data_path
                }
            ],
            'SupportedContentTypes': ['text/csv'],
            'SupportedResponseMIMETypes': ['text/csv']
        },
        ModelApprovalStatus='Approved'
    )
    model_package_arn = response['ModelPackageArn']
    print(f"Model registered with ARN: {model_package_arn}")
except boto3.exceptions.Boto3Error as e:
    print(f"An error occurred while creating the model package: {e}")

# Step 3: Create a SageMaker Model from the model package ARN
model_name = f"xgboost-ip-prediction-model-{strftime('%Y-%m-%d-%H-%M-%S', gmtime())}"
print(f"Model name: {model_name}")
container_list = [{"ModelPackageName": model_package_arn}]

# Get the role ARN dynamically
try:
    iam_client = boto3.client('iam')
    role_response = iam_client.get_role(RoleName=role)
    role_arn = role_response['Role']['Arn']
    print(f"Role ARN obtained: {role_arn}")
except boto3.exceptions.Boto3Error as e:
    print(f"An error occurred while fetching the role ARN: {e}")
    role_arn = None

try:
    create_model_response = sagemaker_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        Containers=container_list
    )
    print(f"Model ARN: {create_model_response['ModelArn']}")
except boto3.exceptions.Boto3Error as e:
    print(f"An error occurred while creating the SageMaker model: {e}")

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

# Step 4: Create Batch Transform job using the SageMaker model
try:
    transformer = Transformer(
        model_name=model_name,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        output_path=f's3://{bucket}/{prefix}/transform-output',
        sagemaker_session=sagemaker_session
    )

    transformer.transform(
        data=test_s3_path,
        content_type='text/csv',
        split_type='Line',
        input_filter='$'  # Include all rows
    )

    transformer.wait()
    transform_output_path = transformer.output_path

    # Retrieve and process the transformed output
    s3_client = boto3.client('s3')
    output_key = f"{prefix}/transform-output/synthetic_test_data_scaled.csv.out"
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
    print(original_output_df)
except boto3.exceptions.Boto3Error as e:
    print(f"An error occurred during the batch transform job: {e}")
