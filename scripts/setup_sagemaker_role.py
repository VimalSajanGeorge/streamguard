
"""Create IAM role for SageMaker with enhanced permissions."""

import boto3
import json

def create_sagemaker_role():
    iam = boto3.client('iam')

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }

    try:
        response = iam.create_role(
            RoleName='StreamGuardSageMakerRoleV3',
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Enhanced role for StreamGuard v3.0 ML training'
        )

        role_arn = response['Role']['Arn']
        print(f"✅ Created role: {role_arn}")

        # Attach policies
        policies = [
            'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
            'arn:aws:iam::aws:policy/AmazonS3FullAccess',
            'arn:aws:iam::aws:policy/CloudWatchLogsFullAccess'
        ]

        for policy in policies:
            iam.attach_role_policy(
                RoleName='StreamGuardSageMakerRoleV3',
                PolicyArn=policy
            )
            print(f"✅ Attached policy: {policy}")

        return role_arn

    except iam.exceptions.EntityAlreadyExistsException:
        print("Role already exists")
        response = iam.get_role(RoleName='StreamGuardSageMakerRoleV3')
        return response['Role']['Arn']

if __name__ == "__main__":
    role_arn = create_sagemaker_role()
    print(f"\n📋 Save this ARN: {role_arn}")

    # Save to .env file
    with open('.env', 'a') as f:
        f.write(f"\nSAGEMAKER_ROLE_ARN={role_arn}\n")
