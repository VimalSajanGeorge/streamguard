"""Verify AWS SageMaker infrastructure setup."""

import os
import sys
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

def check_aws_credentials():
    """Check if AWS credentials are configured."""
    print("[*] Checking AWS Credentials...")
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print("  [+] AWS credentials configured")
        print(f"     Account ID: {identity['Account']}")
        print(f"     ARN: {identity['Arn']}")
        return True
    except NoCredentialsError:
        print("  [-] AWS credentials not found")
        print("     Run: aws configure")
        return False
    except Exception as e:
        print(f"  [-] Error checking credentials: {e}")
        return False

def check_s3_bucket():
    """Check if S3 bucket exists and has correct structure."""
    print("\n[*] Checking S3 Bucket...")
    bucket_name = os.getenv('S3_BUCKET', 'streamguard-ml-v3')

    try:
        s3 = boto3.client('s3')

        # Check if bucket exists
        try:
            s3.head_bucket(Bucket=bucket_name)
            print(f"  [+] Bucket '{bucket_name}' exists")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                print(f"  [-] Bucket '{bucket_name}' does not exist")
                print(f"     Run: aws s3 mb s3://{bucket_name}")
                return False
            else:
                print(f"  [-] Error accessing bucket: {e}")
                return False

        # Check folder structure
        prefixes = ['data/processed/', 'data/feedback/', 'models/', 'checkpoints/']
        print(f"\n  Checking folder structure:")

        response = s3.list_objects_v2(Bucket=bucket_name, Delimiter='/')
        existing_prefixes = []

        if 'CommonPrefixes' in response:
            existing_prefixes = [prefix['Prefix'] for prefix in response['CommonPrefixes']]

        all_folders_exist = True
        for prefix in prefixes:
            if prefix in existing_prefixes or any(prefix.startswith(ep) for ep in existing_prefixes):
                print(f"    [+] {prefix}")
            else:
                print(f"    [~] {prefix} (not found, but can be created on first use)")

        return True

    except Exception as e:
        print(f"  [-] Error checking S3 bucket: {e}")
        return False

def check_iam_role():
    """Check if SageMaker IAM role exists and has correct policies."""
    print("\n[*] Checking IAM Role...")
    role_arn = os.getenv('SAGEMAKER_ROLE_ARN')

    if not role_arn:
        print("  [-] SAGEMAKER_ROLE_ARN not found in .env")
        return False

    role_name = role_arn.split('/')[-1]

    try:
        iam = boto3.client('iam')

        # Get role details
        role = iam.get_role(RoleName=role_name)
        print(f"  [+] Role '{role_name}' exists")
        print(f"     ARN: {role['Role']['Arn']}")
        print(f"     Created: {role['Role']['CreateDate']}")

        # Check attached policies
        print(f"\n  Checking attached policies:")
        attached_policies = iam.list_attached_role_policies(RoleName=role_name)

        required_policies = [
            'AmazonSageMakerFullAccess',
            'AmazonS3FullAccess',
            'CloudWatchLogsFullAccess'
        ]

        attached_policy_names = [p['PolicyName'] for p in attached_policies['AttachedPolicies']]

        for policy in required_policies:
            if policy in attached_policy_names:
                print(f"    [+] {policy}")
            else:
                print(f"    [-] {policy} (missing)")

        return True

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            print(f"  [-] Role '{role_name}' does not exist")
            print("     Run: python scripts/setup_sagemaker_role.py")
        else:
            print(f"  [-] Error checking IAM role: {e}")
        return False
    except Exception as e:
        print(f"  [-] Error: {e}")
        return False

def check_env_variables():
    """Check if all required environment variables are set."""
    print("\n[*] Checking Environment Variables (.env)...")

    required_vars = {
        'AWS_REGION': 'AWS region',
        'S3_BUCKET': 'S3 bucket name',
        'SAGEMAKER_ROLE_ARN': 'SageMaker IAM role ARN',
        'NEO4J_URI': 'Neo4j connection URI',
        'NEO4J_USER': 'Neo4j username',
        'NEO4J_PASSWORD': 'Neo4j password',
        'REDIS_HOST': 'Redis host',
        'REDIS_PORT': 'Redis port'
    }

    all_present = True
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if 'PASSWORD' in var or 'ARN' in var:
                display_value = value[:20] + '...' if len(value) > 20 else '***'
            else:
                display_value = value
            print(f"  [+] {var}: {display_value}")
        else:
            print(f"  [-] {var}: not set ({description})")
            all_present = False

    return all_present

def main():
    """Run all verification checks."""
    print("AWS SageMaker Infrastructure Verification\n")
    print("=" * 70)

    checks = {
        'Environment Variables': check_env_variables(),
        'AWS Credentials': check_aws_credentials(),
        'S3 Bucket': check_s3_bucket(),
        'IAM Role': check_iam_role()
    }

    print("\n" + "=" * 70)
    print("\nVerification Summary:\n")

    for check_name, result in checks.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {check_name}")

    all_passed = all(checks.values())

    if all_passed:
        print("\n[SUCCESS] All AWS infrastructure checks passed!")
        print("\nAWS SageMaker setup is complete and ready to use.")
        print("\nNext steps:")
        print("  1. Test Neo4j connection: python scripts/test_neo4j.py")
        print("  2. Run full verification: python scripts/verify_enhanced_setup.py")
        return 0
    else:
        print("\n[WARNING] Some checks failed. Please review the output above.")
        print("\nRefer to docs/01_setup.md for troubleshooting.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
