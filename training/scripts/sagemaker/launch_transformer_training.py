"""
SageMaker Launcher for Enhanced SQL Intent Transformer

Features:
- Spot instance configuration with checkpointing
- Custom Docker container with dependencies
- S3 data integration
- CloudWatch metrics
- Hyperparameter passing
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

try:
    import boto3
    import sagemaker
    from sagemaker.pytorch import PyTorch
    SAGEMAKER_AVAILABLE = True
except ImportError:
    SAGEMAKER_AVAILABLE = False
    print("[!] SageMaker SDK not available. Install: pip install sagemaker boto3")


def get_execution_role():
    """Get SageMaker execution role ARN."""
    # Try environment variable first
    role = os.environ.get('SAGEMAKER_EXECUTION_ROLE')

    if role:
        return role

    # Try getting from SageMaker session
    try:
        role = sagemaker.get_execution_role()
        return role
    except:
        pass

    # Prompt user
    print("[!] SageMaker execution role not found.")
    print("    Set SAGEMAKER_EXECUTION_ROLE environment variable or provide via --role")
    return None


def launch_training(
    train_data_s3: str,
    val_data_s3: str,
    test_data_s3: str = None,
    output_s3: str = None,
    role_arn: str = None,
    instance_type: str = 'ml.g4dn.xlarge',
    use_spot: bool = True,
    max_run_hours: int = 4,
    custom_image_uri: str = None,
    quick_test: bool = False,
    # Hyperparameters
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_seq_len: int = 512,
    dropout: float = 0.1,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    early_stopping_patience: int = 2,
    use_weights: bool = False,
    mixed_precision: bool = True
):
    """
    Launch transformer training on SageMaker.

    Args:
        train_data_s3: S3 path to training data
        val_data_s3: S3 path to validation data
        test_data_s3: S3 path to test data (optional)
        output_s3: S3 output path
        role_arn: SageMaker execution role ARN
        instance_type: Instance type
        use_spot: Use Spot instances
        max_run_hours: Maximum runtime hours
        custom_image_uri: Custom Docker image URI
        quick_test: Quick test mode
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_seq_len: Max sequence length
        dropout: Dropout rate
        weight_decay: Weight decay
        warmup_ratio: Warmup ratio
        early_stopping_patience: Early stopping patience
        use_weights: Use sample weights
        mixed_precision: Use mixed precision
    """

    if not SAGEMAKER_AVAILABLE:
        raise ImportError("SageMaker SDK required. Install: pip install sagemaker boto3")

    # Get role
    if not role_arn:
        role_arn = get_execution_role()

    if not role_arn:
        raise ValueError("SageMaker execution role required")

    print(f"[*] Using execution role: {role_arn[:50]}...")

    # SageMaker session
    session = sagemaker.Session()
    region = session.boto_region_name

    # Default output path
    if not output_s3:
        bucket = session.default_bucket()
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        output_s3 = f's3://{bucket}/streamguard/training/transformer/{timestamp}'

    print(f"[*] Output S3 path: {output_s3}")

    # Checkpoint S3 path
    checkpoint_s3 = f"{output_s3}/checkpoints"

    # Hyperparameters
    hyperparameters = {
        'epochs': epochs,
        'batch-size': batch_size,
        'lr': learning_rate,
        'max-seq-len': max_seq_len,
        'dropout': dropout,
        'weight-decay': weight_decay,
        'warmup-ratio': warmup_ratio,
        'early-stopping-patience': early_stopping_patience,
        's3-bucket': output_s3.split('/')[2],  # Extract bucket name
        's3-prefix': '/'.join(output_s3.split('/')[3:]) + '/checkpoints'
    }

    if use_weights:
        hyperparameters['use-weights'] = ''

    if mixed_precision:
        hyperparameters['mixed-precision'] = ''

    if quick_test:
        hyperparameters['quick-test'] = ''

    print(f"\n[*] Hyperparameters:")
    for k, v in hyperparameters.items():
        print(f"    {k}: {v}")

    # Estimator configuration
    estimator_kwargs = {
        'entry_point': 'train_transformer.py',
        'source_dir': str(Path(__file__).parent.parent.parent),  # training/ directory
        'role': role_arn,
        'instance_count': 1,
        'instance_type': instance_type,
        'hyperparameters': hyperparameters,
        'output_path': output_s3,
        'base_job_name': 'streamguard-transformer',
        'max_run': max_run_hours * 3600,  # Convert to seconds
        'checkpoint_s3_uri': checkpoint_s3,
        'checkpoint_local_path': '/opt/ml/checkpoints',
        'metric_definitions': [
            {'Name': 'train:loss', 'Regex': 'Train Loss: ([0-9\\.]+)'},
            {'Name': 'val:loss', 'Regex': 'Val Loss: ([0-9\\.]+)'},
            {'Name': 'val:accuracy', 'Regex': 'Val Accuracy: ([0-9\\.]+)'},
            {'Name': 'val:f1_vulnerable', 'Regex': 'Val F1 \\(vulnerable\\): ([0-9\\.]+)'},
        ]
    }

    # Spot instances
    if use_spot:
        max_wait_hours = max_run_hours + 1  # Allow 1 hour buffer
        estimator_kwargs.update({
            'use_spot_instances': True,
            'max_wait': max_wait_hours * 3600
        })
        print(f"[+] Spot instances enabled (max wait: {max_wait_hours}h)")

    # Custom Docker image
    if custom_image_uri:
        estimator_kwargs['image_uri'] = custom_image_uri
        estimator_kwargs.pop('entry_point')
        estimator_kwargs.pop('source_dir')
        print(f"[+] Using custom image: {custom_image_uri}")
    else:
        # Use default PyTorch container
        estimator_kwargs['framework_version'] = '2.1.0'
        estimator_kwargs['py_version'] = 'py310'
        print(f"[+] Using default PyTorch 2.1.0 container")

    # Create estimator
    estimator = PyTorch(**estimator_kwargs)

    # Input data channels
    inputs = {
        'train': train_data_s3,
        'val': val_data_s3
    }

    if test_data_s3:
        inputs['test'] = test_data_s3

    print(f"\n[*] Input data channels:")
    for channel, path in inputs.items():
        print(f"    {channel}: {path}")

    # Launch training
    print(f"\n{'='*70}")
    print("LAUNCHING SAGEMAKER TRAINING JOB")
    print(f"{'='*70}\n")

    estimator.fit(
        inputs=inputs,
        wait=True,
        logs='All'
    )

    print(f"\n{'='*70}")
    print("TRAINING JOB COMPLETE")
    print(f"{'='*70}")
    print(f"Model artifacts: {estimator.model_data}")
    print(f"Training job name: {estimator.latest_training_job.name}")

    return estimator


def main():
    parser = argparse.ArgumentParser(description="Launch Transformer training on SageMaker")

    # Data
    parser.add_argument('--train-data-s3', type=str, required=True, help='S3 path to training data')
    parser.add_argument('--val-data-s3', type=str, required=True, help='S3 path to validation data')
    parser.add_argument('--test-data-s3', type=str, default=None, help='S3 path to test data')
    parser.add_argument('--output-s3', type=str, default=None, help='S3 output path')

    # Infrastructure
    parser.add_argument('--role', type=str, default=None, help='SageMaker execution role ARN')
    parser.add_argument('--instance-type', type=str, default='ml.g4dn.xlarge', help='Instance type')
    parser.add_argument('--use-spot', action='store_true', default=True, help='Use Spot instances')
    parser.add_argument('--no-spot', dest='use_spot', action='store_false', help='Disable Spot instances')
    parser.add_argument('--max-run-hours', type=int, default=4, help='Maximum runtime hours')
    parser.add_argument('--custom-image', type=str, default=None, help='Custom Docker image URI')

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max-seq-len', type=int, default=512, help='Max sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup-ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--early-stopping-patience', type=int, default=2, help='Early stopping patience')
    parser.add_argument('--use-weights', action='store_true', help='Use sample weights')
    parser.add_argument('--mixed-precision', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode')

    args = parser.parse_args()

    # Launch training
    estimator = launch_training(
        train_data_s3=args.train_data_s3,
        val_data_s3=args.val_data_s3,
        test_data_s3=args.test_data_s3,
        output_s3=args.output_s3,
        role_arn=args.role,
        instance_type=args.instance_type,
        use_spot=args.use_spot,
        max_run_hours=args.max_run_hours,
        custom_image_uri=args.custom_image,
        quick_test=args.quick_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        early_stopping_patience=args.early_stopping_patience,
        use_weights=args.use_weights,
        mixed_precision=args.mixed_precision
    )

    print(f"\n[+] Training complete!")
    print(f"    Job name: {estimator.latest_training_job.name}")
    print(f"    Model artifacts: {estimator.model_data}")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
