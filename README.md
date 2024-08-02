# Testing SageMaker Jobs in AWS CloudShell

This document provides a step-by-step guide to testing SageMaker jobs using AWS CloudShell. 
Follow these instructions to ensure that your setup works correctly and that SageMaker jobs run as expected.

## Test Setup

Before running the tests, ensure that you have completed the following:

1. **AWS CloudShell:** Open AWS CloudShell from the AWS Management Console.
2. **SageMaker Execution Role:** Create a SageMaker execution role using the `sagemaker-execution-role-template.yaml` template.
3. **Repository Cloned:** Clone this GitHub repository containing the SageMaker scripts.

## Test Steps

### 1. Verify Environment Setup

**Check Installed Packages:**

   Confirm that the required Python packages are installed:

   `pip show sagemaker scikit-learn matplotlib`

   * [Refer to blog post for step by step guide](https://vivek-aws.medium.com/4-ways-to-get-hands-on-with-sagemaker-for-free-41ff9bee0d54).

