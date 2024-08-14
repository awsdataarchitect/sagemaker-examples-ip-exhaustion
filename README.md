# Executing SageMaker Jobs from AWS CloudShell

Step-by-step guide to running SageMaker jobs using AWS CloudShell. 
Follow these instructions to ensure that your setup works correctly and that SageMaker jobs run as expected.

## AWS Environment Setup

Before running the tests, ensure that you have completed the following:

1. **AWS CloudShell:** Open AWS CloudShell from the AWS Management Console.
2. **SageMaker Execution Role:** Create a SageMaker execution role using the `sagemaker-execution-role-template.yaml` template.
3. **Repository Cloned:** Clone this GitHub repository containing the SageMaker scripts.
4. **Check Installed Packages:** Confirm that the required Python packages are installed:
    `pip show sagemaker scikit-learn matplotlib`
    
* [Refer to blog post for step by step guide](https://vivek-aws.medium.com/4-ways-to-get-hands-on-with-sagemaker-for-free-41ff9bee0d54).

* [SageMaker Example using SageMaker Model Registry for model deployment and batch transform](https://vivek-aws.medium.com/using-aws-cloudshell-for-automating-xgboost-model-deployment-and-batch-transform-with-aws-sagemaker-2adedc4d2b02).

# Using SageMaker Studio to manage the Model Registry and Training Jobs

Creating Studio User and Domain using AWS-CDK and Leveraging the Model Registry

* [Maximizing ML Efficiency with SageMaker Studio](https://medium.com/@vivek-aws/maximizing-ml-efficiency-with-sagemaker-studio-a55030da2a45).
