import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as sagemaker from 'aws-cdk-lib/aws-sagemaker';
import * as iam from 'aws-cdk-lib/aws-iam';

export class SageMakerStudioStack extends cdk.Stack {
    constructor(scope: Construct, id: string, props?: cdk.StackProps) {
        super(scope, id, props);

        // Create a VPC with public subnets only and 2 max availability zones
        const vpc = new ec2.Vpc(this, 'MyVpc', {
            maxAzs: 2,
            subnetConfiguration: [
                {
                    cidrMask: 24,
                    name: 'public-subnet',
                    subnetType: ec2.SubnetType.PUBLIC,
                },
            ],
        });


        // Get the existing IAM role by its name
        const sageMakerRole = iam.Role.fromRoleName(this, 'SageMakerRole', 'sagemaker-execution-role');

        // Attach additional managed policies to the existing role
        sageMakerRole.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess'));
        sageMakerRole.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonS3FullAccess'));

        const smDomain = new sagemaker.CfnDomain(this, 'SageMakerStudioDomain', {
            appNetworkAccessType: 'PublicInternetOnly',
            authMode: 'IAM',
            defaultUserSettings: {
                executionRole: sageMakerRole.roleArn,
            },
            domainName: 'lab-sagemaker-domain',
            subnetIds: vpc.publicSubnets.map(subnet => subnet.subnetId), // Use public subnets
            vpcId: vpc.vpcId,

        });

        // Create a UserProfile resource
        const userProfile = new sagemaker.CfnUserProfile(this, 'UserProfile', {
            domainId: smDomain.attrDomainId,
            userProfileName: 'lab-user', // Replace with your desired user profile name
        });

        // Make the UserProfile resource depend on the Domain resource
        userProfile.node.addDependency(smDomain);

    }
}

const app = new cdk.App();
new SageMakerStudioStack(app, 'SageMakerStudioStack');
