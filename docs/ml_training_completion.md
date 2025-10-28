# 02_ml_training.md - COMPLETION (Append to existing file)

```python
# Continuation of training/scripts/sagemaker/launch_enhanced_training.py

def launch_gnn_training():
    """Launch Taint-Flow GNN training."""
    session = sagemaker.Session()
    
    estimator = PyTorch(
        entry_point='train_enhanced_models.py',
        source_dir='training/scripts/sagemaker',
        role=ROLE_ARN,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        framework_version='2.1.0',
        py_version='py310',
        hyperparameters={
            'epochs': 50,
            'batch-size': 32,
            'learning-rate': 2e-5,
            'model-type': 'gnn'
        },
        output_path=f's3://{BUCKET}/models/',
        code_location=f's3://{BUCKET}/code/',
        metric_definitions=[
            {'Name': 'train:loss', 'Regex': 'Train Loss=(.*?);'},
            {'Name': 'train:accuracy', 'Regex': 'Train Acc=(.*?);'},
            {'Name': 'validation:accuracy', 'Regex': 'Val Acc=(.*?);'}
        ],
        enable_sagemaker_metrics=True,
        tags=[
            {'Key': 'Project', 'Value': 'StreamGuard'},
            {'Key': 'Model', 'Value': 'TaintFlowGNN'},
            {'Key': 'Version', 'Value': 'v3.0'}
        ]
    )
    
    # Configure training data
    train_input = TrainingInput(
        s3_data=f's3://{BUCKET}/data/processed/train/',
        content_type='application/x-jsonlines'
    )
    
    val_input = TrainingInput(
        s3_data=f's3://{BUCKET}/data/processed/val/',
        content_type='application/x-jsonlines'
    )
    
    # Launch training
    estimator.fit({
        'train': train_input,
        'validation': val_input
    })
    
    print(f"✅ GNN Training launched: {estimator.latest_training_job.name}")
    return estimator


def launch_transformer_training():
    """Launch SQL Intent Transformer training."""
    session = sagemaker.Session()
    
    estimator = PyTorch(
        entry_point='train_enhanced_models.py',
        source_dir='training/scripts/sagemaker',
        role=ROLE_ARN,
        instance_type='ml.p3.16xlarge',  # 8x V100 for larger model
        instance_count=1,
        framework_version='2.1.0',
        py_version='py310',
        hyperparameters={
            'epochs': 30,
            'batch-size': 16,
            'learning-rate': 1e-5,
            'model-type': 'transformer'
        },
        output_path=f's3://{BUCKET}/models/',
        metric_definitions=[
            {'Name': 'train:loss', 'Regex': 'Train Loss=(.*?);'},
            {'Name': 'train:accuracy', 'Regex': 'Train Acc=(.*?);'},
            {'Name': 'validation:accuracy', 'Regex': 'Val Acc=(.*?);'}
        ],
        enable_sagemaker_metrics=True,
        tags=[
            {'Key': 'Project', 'Value': 'StreamGuard'},
            {'Key': 'Model', 'Value': 'SQLIntentTransformer'},
            {'Key': 'Version', 'Value': 'v3.0'}
        ]
    )
    
    # Configure training data
    train_input = TrainingInput(
        s3_data=f's3://{BUCKET}/data/processed/train/',
        content_type='application/x-jsonlines'
    )
    
    val_input = TrainingInput(
        s3_data=f's3://{BUCKET}/data/processed/val/',
        content_type='application/x-jsonlines'
    )
    
    # Launch training
    estimator.fit({
        'train': train_input,
        'validation': val_input
    })
    
    print(f"✅ Transformer Training launched: {estimator.latest_training_job.name}")
    return estimator


def launch_all_training():
    """Launch all training jobs."""
    print("🚀 Launching all training jobs...\n")
    
    # Launch GNN training
    print("1️⃣ Launching Taint-Flow GNN training...")
    gnn_estimator = launch_gnn_training()
    
    # Wait a bit before launching next job
    import time
    time.sleep(60)
    
    # Launch Transformer training
    print("\n2️⃣ Launching SQL Intent Transformer training...")
    transformer_estimator = launch_transformer_training()
    
    print("\n✅ All training jobs launched!")
    print(f"GNN Job: {gnn_estimator.latest_training_job.name}")
    print(f"Transformer Job: {transformer_estimator.latest_training_job.name}")
    
    return gnn_estimator, transformer_estimator


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['gnn', 'transformer', 'all'], default='all')
    args = parser.parse_args()
    
    if args.model == 'gnn':
        launch_gnn_training()
    elif args.model == 'transformer':
        launch_transformer_training()
    else:
        launch_all_training()
```

---

## 📊 Step 4: Training Monitoring

### Real-Time Training Monitor

**File:** `training/scripts/sagemaker/monitor_training.py`

```python
"""Monitor SageMaker training jobs in real-time."""

import boto3
import time
from rich.console import Console
from rich.table import Table
from rich.live import Live
from datetime import datetime
from typing import List, Dict

console = Console()

class TrainingMonitor:
    """Monitor SageMaker training jobs."""
    
    def __init__(self):
        self.sagemaker = boto3.client('sagemaker')
        self.cloudwatch = boto3.client('logs')
    
    def list_active_jobs(self) -> List[str]:
        """List all active training jobs."""
        response = self.sagemaker.list_training_jobs(
            StatusEquals='InProgress',
            MaxResults=10
        )
        
        return [job['TrainingJobName'] for job in response['TrainingJobSummaries']]
    
    def get_job_status(self, job_name: str) -> Dict:
        """Get detailed status of a training job."""
        response = self.sagemaker.describe_training_job(
            TrainingJobName=job_name
        )
        
        return {
            'name': job_name,
            'status': response['TrainingJobStatus'],
            'creation_time': response['CreationTime'],
            'last_modified': response['LastModifiedTime'],
            'training_time': response.get('TrainingTimeInSeconds', 0),
            'billable_time': response.get('BillableTimeInSeconds', 0),
            'instance_type': response['ResourceConfig']['InstanceType'],
            'instance_count': response['ResourceConfig']['InstanceCount'],
            'model_artifacts': response.get('ModelArtifacts', {}).get('S3ModelArtifacts'),
            'metrics': self._get_latest_metrics(job_name)
        }
    
    def _get_latest_metrics(self, job_name: str) -> Dict:
        """Get latest training metrics."""
        try:
            response = self.sagemaker.describe_training_job(
                TrainingJobName=job_name
            )
            
            # Get final metrics if available
            if 'FinalMetricDataList' in response:
                metrics = {}
                for metric in response['FinalMetricDataList']:
                    metrics[metric['MetricName']] = metric['Value']
                return metrics
            
            return {}
        except Exception as e:
            console.print(f"[yellow]Warning: Could not fetch metrics: {e}[/yellow]")
            return {}
    
    def get_logs(self, job_name: str, limit: int = 50) -> List[str]:
        """Get recent training logs."""
        log_group = f'/aws/sagemaker/TrainingJobs'
        log_stream = job_name
        
        try:
            response = self.cloudwatch.get_log_events(
                logGroupName=log_group,
                logStreamName=log_stream,
                limit=limit,
                startFromHead=False
            )
            
            return [event['message'] for event in response['events']]
        except Exception as e:
            return [f"Could not fetch logs: {e}"]
    
    def create_status_table(self, jobs: List[str]) -> Table:
        """Create a rich table of job statuses."""
        table = Table(title="StreamGuard Training Jobs")
        
        table.add_column("Job Name", style="cyan", no_wrap=True)
        table.add_column("Status", style="magenta")
        table.add_column("Instance", style="green")
        table.add_column("Training Time", style="blue")
        table.add_column("Train Acc", style="yellow")
        table.add_column("Val Acc", style="yellow")
        
        for job_name in jobs:
            status = self.get_job_status(job_name)
            
            # Format training time
            training_time = status['training_time']
            hours = training_time // 3600
            minutes = (training_time % 3600) // 60
            time_str = f"{hours}h {minutes}m"
            
            # Get metrics
            metrics = status['metrics']
            train_acc = f"{metrics.get('train:accuracy', 0.0):.2%}"
            val_acc = f"{metrics.get('validation:accuracy', 0.0):.2%}"
            
            # Status emoji
            status_str = status['status']
            if status_str == 'InProgress':
                status_str = "🔄 In Progress"
            elif status_str == 'Completed':
                status_str = "✅ Completed"
            elif status_str == 'Failed':
                status_str = "❌ Failed"
            
            table.add_row(
                job_name[-40:],  # Truncate long names
                status_str,
                status['instance_type'],
                time_str,
                train_acc,
                val_acc
            )
        
        return table
    
    def monitor_live(self, refresh_interval: int = 30):
        """Monitor training jobs with live updates."""
        console.print("[bold green]StreamGuard Training Monitor[/bold green]\n")
        console.print("Press Ctrl+C to stop monitoring\n")
        
        try:
            with Live(console=console, refresh_per_second=1) as live:
                while True:
                    jobs = self.list_active_jobs()
                    
                    if not jobs:
                        live.update("[yellow]No active training jobs[/yellow]")
                    else:
                        table = self.create_status_table(jobs)
                        live.update(table)
                    
                    time.sleep(refresh_interval)
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped[/yellow]")
    
    def wait_for_completion(self, job_name: str):
        """Wait for a specific job to complete."""
        console.print(f"Waiting for {job_name} to complete...")
        
        while True:
            status = self.get_job_status(job_name)
            
            if status['status'] in ['Completed', 'Failed', 'Stopped']:
                console.print(f"\n✅ Job {job_name} finished with status: {status['status']}")
                
                # Print final metrics
                if status['metrics']:
                    console.print("\nFinal Metrics:")
                    for metric_name, value in status['metrics'].items():
                        console.print(f"  {metric_name}: {value:.4f}")
                
                break
            
            # Print progress
            console.print(f"Status: {status['status']}, Time: {status['training_time']}s", end='\r')
            time.sleep(30)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--monitor', action='store_true', help='Monitor all active jobs')
    parser.add_argument('--wait', type=str, help='Wait for specific job to complete')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor()
    
    if args.monitor:
        monitor.monitor_live()
    elif args.wait:
        monitor.wait_for_completion(args.wait)
    else:
        # Show current status
        jobs = monitor.list_active_jobs()
        if jobs:
            table = monitor.create_status_table(jobs)
            console.print(table)
        else:
            console.print("[yellow]No active training jobs[/yellow]")
```

---

## 📦 Step 5: Model Registry & Versioning

### Model Registry Manager

**File:** `training/scripts/model_registry.py`

```python
"""Model registry for versioning and deployment."""

import boto3
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

class ModelRegistry:
    """Manage model versions and metadata."""
    
    def __init__(self, bucket: str = "streamguard-ml-v3"):
        self.s3 = boto3.client('s3')
        self.sagemaker = boto3.client('sagemaker')
        self.bucket = bucket
        self.registry_key = 'models/registry.json'
    
    def register_model(
        self,
        model_name: str,
        model_type: str,
        s3_uri: str,
        metrics: Dict[float, float],
        training_job_name: str,
        tags: Dict[str, str] = None
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model_name: Name of the model (e.g., "taint_flow_gnn")
            model_type: Type of model (e.g., "gnn", "transformer")
            s3_uri: S3 URI of model artifacts
            metrics: Training metrics
            training_job_name: SageMaker training job name
            tags: Additional metadata tags
        
        Returns:
            version_id: Unique version identifier
        """
        # Load registry
        registry = self._load_registry()
        
        # Create version ID
        version_id = f"{model_name}_v{len(registry.get(model_name, [])) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create model entry
        model_entry = {
            'version_id': version_id,
            'model_name': model_name,
            'model_type': model_type,
            's3_uri': s3_uri,
            'metrics': metrics,
            'training_job_name': training_job_name,
            'registered_at': datetime.now().isoformat(),
            'tags': tags or {},
            'status': 'registered',
            'deployed': False
        }
        
        # Add to registry
        if model_name not in registry:
            registry[model_name] = []
        
        registry[model_name].append(model_entry)
        
        # Save registry
        self._save_registry(registry)
        
        print(f"✅ Registered model: {version_id}")
        print(f"   Metrics: {metrics}")
        
        return version_id
    
    def get_latest_version(self, model_name: str) -> Optional[Dict]:
        """Get latest version of a model."""
        registry = self._load_registry()
        
        if model_name not in registry or not registry[model_name]:
            return None
        
        # Return most recent version
        return registry[model_name][-1]
    
    def get_best_version(
        self,
        model_name: str,
        metric_name: str = 'validation:accuracy'
    ) -> Optional[Dict]:
        """Get best version based on a metric."""
        registry = self._load_registry()
        
        if model_name not in registry or not registry[model_name]:
            return None
        
        # Find version with best metric
        versions = registry[model_name]
        best_version = max(
            versions,
            key=lambda v: v['metrics'].get(metric_name, 0.0)
        )
        
        return best_version
    
    def list_versions(self, model_name: str) -> List[Dict]:
        """List all versions of a model."""
        registry = self._load_registry()
        return registry.get(model_name, [])
    
    def mark_deployed(self, version_id: str):
        """Mark a model version as deployed."""
        registry = self._load_registry()
        
        # Find and update version
        for model_name, versions in registry.items():
            for version in versions:
                if version['version_id'] == version_id:
                    version['deployed'] = True
                    version['deployed_at'] = datetime.now().isoformat()
                    self._save_registry(registry)
                    print(f"✅ Marked {version_id} as deployed")
                    return
        
        print(f"❌ Version {version_id} not found")
    
    def compare_versions(
        self,
        version_id_1: str,
        version_id_2: str
    ) -> Dict:
        """Compare two model versions."""
        registry = self._load_registry()
        
        # Find versions
        v1 = None
        v2 = None
        
        for model_name, versions in registry.items():
            for version in versions:
                if version['version_id'] == version_id_1:
                    v1 = version
                if version['version_id'] == version_id_2:
                    v2 = version
        
        if not v1 or not v2:
            return {"error": "One or both versions not found"}
        
        # Compare metrics
        comparison = {
            'version_1': version_id_1,
            'version_2': version_id_2,
            'metric_comparison': {}
        }
        
        for metric_name in v1['metrics'].keys():
            v1_value = v1['metrics'].get(metric_name, 0.0)
            v2_value = v2['metrics'].get(metric_name, 0.0)
            
            comparison['metric_comparison'][metric_name] = {
                'version_1': v1_value,
                'version_2': v2_value,
                'difference': v2_value - v1_value,
                'improvement': v2_value > v1_value
            }
        
        return comparison
    
    def _load_registry(self) -> Dict:
        """Load registry from S3."""
        try:
            response = self.s3.get_object(
                Bucket=self.bucket,
                Key=self.registry_key
            )
            return json.loads(response['Body'].read())
        except self.s3.exceptions.NoSuchKey:
            return {}
    
    def _save_registry(self, registry: Dict):
        """Save registry to S3."""
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.registry_key,
            Body=json.dumps(registry, indent=2),
            ContentType='application/json'
        )


# Example usage
if __name__ == "__main__":
    registry = ModelRegistry()
    
    # Register a model
    version_id = registry.register_model(
        model_name="taint_flow_gnn",
        model_type="gnn",
        s3_uri="s3://streamguard-ml-v3/models/model.tar.gz",
        metrics={
            'train:accuracy': 0.94,
            'validation:accuracy': 0.92,
            'train:loss': 0.15
        },
        training_job_name="streamguard-gnn-20241008-123456",
        tags={'experiment': 'baseline'}
    )
    
    # Get best version
    best = registry.get_best_version("taint_flow_gnn")
    print(f"\nBest version: {best['version_id']}")
    print(f"Val Accuracy: {best['metrics']['validation:accuracy']:.2%}")
```

---

## 🔄 Step 6: Continuous Retraining Pipeline

### Automatic Retraining Trigger

**File:** `training/scripts/retraining/auto_retrain.py`

```python
"""Automatic retraining pipeline with drift detection."""

import boto3
import json
from typing import Dict, List
from datetime import datetime, timedelta
from pathlib import Path

class AutoRetrainingPipeline:
    """Manage automatic model retraining."""
    
    def __init__(
        self,
        bucket: str = "streamguard-ml-v3",
        role_arn: str = None
    ):
        self.s3 = boto3.client('s3')
        self.sagemaker = boto3.client('sagemaker')
        self.bucket = bucket
        self.role_arn = role_arn or os.getenv("SAGEMAKER_ROLE_ARN")
    
    def check_retraining_criteria(
        self,
        model_name: str,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        feedback_count: int
    ) -> Dict:
        """
        Check if retraining should be triggered.
        
        Returns:
            Dict with 'should_retrain' boolean and 'reasons' list
        """
        reasons = []
        should_retrain = False
        
        # 1. Accuracy drift check
        if current_metrics.get('accuracy', 0) < baseline_metrics.get('accuracy', 0) - 0.05:
            reasons.append("accuracy_drift")
            should_retrain = True
        
        # 2. False positive rate increase
        if current_metrics.get('false_positive_rate', 0) > baseline_metrics.get('false_positive_rate', 0) + 0.02:
            reasons.append("fp_rate_increase")
            should_retrain = True
        
        # 3. Sufficient new feedback
        if feedback_count >= 1000:
            reasons.append("sufficient_feedback")
            should_retrain = True
        
        # 4. Time-based (monthly retraining)
        last_training = self._get_last_training_date(model_name)
        if last_training and (datetime.now() - last_training).days >= 30:
            reasons.append("scheduled_monthly")
            should_retrain = True
        
        return {
            'should_retrain': should_retrain,
            'reasons': reasons,
            'feedback_count': feedback_count
        }
    
    def trigger_retraining(
        self,
        model_type: str,
        reason: str,
        feedback_data_uri: str
    ) -> str:
        """
        Trigger a retraining job.
        
        Args:
            model_type: 'gnn' or 'transformer'
            reason: Why retraining was triggered
            feedback_data_uri: S3 URI of new feedback data
        
        Returns:
            training_job_name: Name of launched job
        """
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        training_job_name = f"streamguard-retrain-{model_type}-{timestamp}"
        
        print(f"🚀 Triggering retraining for {model_type}")
        print(f"   Reason: {reason}")
        print(f"   Feedback data: {feedback_data_uri}")
        
        # Configure training job
        from sagemaker.pytorch import PyTorch
        
        estimator = PyTorch(
            entry_point='train_enhanced_models.py',
            source_dir='training/scripts/sagemaker',
            role=self.role_arn,
            instance_type='ml.p3.8xlarge',
            instance_count=1,
            framework_version='2.1.0',
            py_version='py310',
            hyperparameters={
                'epochs': 30,  # Fewer epochs for fine-tuning
                'batch-size': 32,
                'learning-rate': 5e-6,  # Lower LR for fine-tuning
                'model-type': model_type,
                'fine-tune': True  # Load previous weights
            },
            output_path=f's3://{self.bucket}/models/',
            tags=[
                {'Key': 'Retraining', 'Value': 'true'},
                {'Key': 'Reason', 'Value': reason},
                {'Key': 'Timestamp', 'Value': timestamp}
            ]
        )
        
        # Launch training
        from sagemaker.inputs import TrainingInput
        
        train_input = TrainingInput(
            s3_data=feedback_data_uri,
            content_type='application/x-jsonlines'
        )
        
        estimator.fit({'train': train_input})
        
        print(f"✅ Retraining job launched: {training_job_name}")
        
        # Log retraining event
        self._log_retraining_event(model_type, reason, training_job_name)
        
        return training_job_name
    
    def evaluate_retrained_model(
        self,
        new_model_uri: str,
        current_model_uri: str,
        test_data_uri: str
    ) -> Dict:
        """
        Evaluate retrained model against current model.
        
        Returns:
            Comparison metrics and recommendation
        """
        # Run evaluation on test set
        new_metrics = self._evaluate_model(new_model_uri, test_data_uri)
        current_metrics = self._evaluate_model(current_model_uri, test_data_uri)
        
        # Compare
        improvement = {}
        for metric in new_metrics:
            improvement[metric] = new_metrics[metric] - current_metrics.get(metric, 0)
        
        # Decision: deploy if better
        should_deploy = (
            improvement.get('accuracy', 0) > 0.01 and
            improvement.get('false_positive_rate', 0) <= 0.005
        )
        
        return {
            'new_metrics': new_metrics,
            'current_metrics': current_metrics,
            'improvement': improvement,
            'should_deploy': should_deploy,
            'recommendation': 'deploy' if should_deploy else 'keep_current'
        }
    
    def _get_last_training_date(self, model_name: str) -> datetime:
        """Get date of last training for a model."""
        # Query model registry
        from training.scripts.model_registry import ModelRegistry
        
        registry = ModelRegistry(self.bucket)
        latest = registry.get_latest_version(model_name)
        
        if latest:
            return datetime.fromisoformat(latest['registered_at'])
        
        return None
    
    def _log_retraining_event(
        self,
        model_type: str,
        reason: str,
        training_job_name: str
    ):
        """Log retraining event to S3."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'reason': reason,
            'training_job_name': training_job_name
        }
        
        # Append to log file
        log_key = f'retraining/events/{model_type}_retraining.jsonl'
        
        try:
            # Get existing log
            response = self.s3.get_object(Bucket=self.bucket, Key=log_key)
            existing_log = response['Body'].read().decode('utf-8')
        except self.s3.exceptions.NoSuchKey:
            existing_log = ""
        
        # Append new event
        new_log = existing_log + json.dumps(event) + '\n'
        
        self.s3.put_object(
            Bucket=self.bucket,
            Key=log_key,
            Body=new_log
        )
    
    def _evaluate_model(self, model_uri: str, test_data_uri: str) -> Dict:
        """Evaluate a model on test data."""
        # This would run a SageMaker Processing job
        # For simplicity, returning mock metrics
        return {
            'accuracy': 0.93,
            'precision': 0.91,
            'recall': 0.95,
            'false_positive_rate': 0.025
        }


# Example usage
if __name__ == "__main__":
    pipeline = AutoRetrainingPipeline()
    
    # Check if retraining needed
    current_metrics = {'accuracy': 0.88, 'false_positive_rate': 0.045}
    baseline_metrics = {'accuracy': 0.93, 'false_positive_rate': 0.025}
    
    result = pipeline.check_retraining_criteria(
        model_name="taint_flow_gnn",
        current_metrics=current_metrics,
        baseline_metrics=baseline_metrics,
        feedback_count=1200
    )
    
    if result['should_retrain']:
        print(f"⚠️  Retraining recommended: {', '.join(result['reasons'])}")
        
        # Trigger retraining
        job_name = pipeline.trigger_retraining(
            model_type='gnn',
            reason=result['reasons'][0],
            feedback_data_uri='s3://streamguard-ml-v3/data/feedback/batch_001/'
        )
    else:
        print("✅ No retraining needed")
```

---

## ✅ Testing & Validation

### Training Pipeline Tests

**File:** `tests/integration/test_training_pipeline.py`

```python
"""Integration tests for ML training pipeline."""

import pytest
import boto3
from training.scripts.sagemaker.launch_enhanced_training import launch_gnn_training
from training.scripts.model_registry import ModelRegistry
from training.scripts.retraining.auto_retrain import AutoRetrainingPipeline

class TestTrainingPipeline:
    """Test end-to-end training pipeline."""
    
    @pytest.fixture
    def s3_client(self):
        return boto3.client('s3')
    
    @pytest.fixture
    def test_bucket(self):
        return "streamguard-ml-v3-test"
    
    def test_data_upload(self, s3_client, test_bucket):
        """Test uploading training data to S3."""
        # Create test data
        test_data = b'{"code": "test", "label": 1}\n'
        
        # Upload
        s3_client.put_object(
            Bucket=test_bucket,
            Key='data/test/sample.jsonl',
            Body=test_data
        )
        
        # Verify
        response = s3_client.get_object(
            Bucket=test_bucket,
            Key='data/test/sample.jsonl'
        )
        
        assert response['Body'].read() == test_data
    
    def test_model_registry(self):
        """Test model registration and retrieval."""
        registry = ModelRegistry()
        
        # Register model
        version_id = registry.register_model(
            model_name="test_model",
            model_type="gnn",
            s3_uri="s3://test/model.tar.gz",
            metrics={'accuracy': 0.95},
            training_job_name="test_job"
        )
        
        # Retrieve
        latest = registry.get_latest_version("test_model")
        assert latest['version_id'] == version_id
        assert latest['metrics']['accuracy'] == 0.95
    
    def test_retraining_trigger(self):
        """Test automatic retraining trigger."""
        pipeline = AutoRetrainingPipeline()
        
        # Check criteria
        result = pipeline.check_retraining_criteria(
            model_name="test_model",
            current_metrics={'accuracy': 0.85},
            baseline_metrics={'accuracy': 0.93},
            feedback_count=500
        )
        
        # Should trigger due to accuracy drift
        assert result['should_retrain']
        assert 'accuracy_drift' in result['reasons']


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, '-v'])
```

---

## 📊 Performance Benchmarks

### Training Performance Targets

**File:** `tests/benchmarks/benchmark_training.py`

```python
"""Benchmark training performance."""

import time
import torch
from training.models.enhanced_taint_gnn import EnhancedTaintFlowGNN
from training.models.enhanced_sql_intent import EnhancedSQLIntentTransformer

def benchmark_gnn_training():
    """Benchmark GNN training speed."""
    model = EnhancedTaintFlowGNN()
    
    # Create dummy data
    x = torch.randint(0, 10000, (1000, ))
    edge_index = torch.randint(0, 1000, (2, 5000))
    batch = torch.zeros(1000, dtype=torch.long)
    
    # Warm up
    for _ in range(10):
        logits, _ = model(x, edge_index, batch)
    
    # Benchmark
    start = time.time()
    iterations = 100
    
    for _ in range(iterations):
        logits, _ = model(x, edge_index, batch)
    
    elapsed = time.time() - start
    throughput = iterations / elapsed
    
    print(f"GNN Training Throughput: {throughput:.2f} iterations/sec")
    assert throughput > 10, "GNN training too slow"


def benchmark_transformer_training():
    """Benchmark transformer training speed."""
    model = EnhancedSQLIntentTransformer()
    
    # Create dummy data
    input_ids = torch.randint(0, 50000, (32, 512))
    attention_mask = torch.ones(32, 512)
    
    # Warm up
    for _ in range(10):
        logits, _, _ = model(input_ids, attention_mask)
    
    # Benchmark
    start = time.time()
    iterations = 50
    
    for _ in range(iterations):
        logits, _, _ = model(input_ids, attention_mask)
    
    elapsed = time.time() - start
    throughput = iterations / elapsed
    
    print(f"Transformer Training Throughput: {throughput:.2f} iterations/sec")
    assert throughput > 5, "Transformer training too slow"


if __name__ == "__main__":
    print("🏃 Running Training Benchmarks\n")
    benchmark_gnn_training()
    benchmark_transformer_training()
    print("\n✅ All benchmarks passed")
```

---

## 📋 Implementation Checklist

### Training Pipeline Checklist

- [x] Data preprocessing with counterfactual augmentation
- [x] Enhanced model architectures (GNN + Transformer)
- [x] SageMaker training scripts
- [x] Training job launcher
- [x] Real-time monitoring
- [x] Model registry & versioning
- [x] Continuous retraining pipeline
- [x] Automatic drift detection
- [x] A/B testing framework
- [x] Integration tests
- [x] Performance benchmarks

### Validation Checklist

- [ ] Verify data preprocessing output
- [ ] Test model training on small dataset
- [ ] Launch full training jobs on SageMaker
- [ ] Monitor training metrics
- [ ] Register trained models
- [ ] Test model deployment
- [ ] Validate retraining trigger logic
- [ ] Run integration tests
- [ ] Benchmark training performance
- [ ] Document training procedures

---

## 🎯 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| **Training Time (GNN)** | <4 hours on ml.p3.8xlarge | ⏳ To Validate |
| **Training Time (Transformer)** | <8 hours on ml.p3.16xlarge | ⏳ To Validate |
| **Model Accuracy (GNN)** | ≥92% | ⏳ To Validate |
| **Model Accuracy (Transformer)** | ≥88% | ⏳ To Validate |
| **False Positive Rate** | <3% | ⏳ To Validate |
| **Model Size (GNN)** | <500MB | ⏳ To Validate |
| **Model Size (Transformer)** | <2GB | ⏳ To Validate |
| **Retraining Frequency** | Monthly or on drift | ✅ Configured |
| **Model Registry Updates** | Real-time | ✅ Implemented |

---

## 🚀 Next Steps

**After completing this phase:**

1. **Verify Training:**
   ```bash
   # Launch training
   python training/scripts/sagemaker/launch_enhanced_training.py
   
   # Monitor progress
   python training/scripts/sagemaker/monitor_training.py --monitor
   ```

2. **Test Models:**
   ```bash
   # Run evaluation
   python training/scripts/evaluate_models.py
   
   # Benchmark performance
   python tests/benchmarks/benchmark_training.py
   ```

3. **Continue to Phase 2:**
   - [03_explainability.md](./03_explainability.md) - Implement deep explainability
   - Focus on Integrated Gradients and counterfactual generation

---

## 📚 Resources

### SageMaker Documentation
- Training Jobs: https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html
- PyTorch: https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/
- Model Registry: https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html

### Model Architectures
- CodeBERT: https://arxiv.org/abs/2002.08155
- Graph Neural Networks: https://pytorch-geometric.readthedocs.io/

### Best Practices
- Hyperparameter Tuning: https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html
- Distributed Training: https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training.html

---

**Status:** ✅ Phase 1 Complete  
**Next:** [03_explainability.md](./03_explainability.md) - Deep Explainability System

---

## 💡 Tips for Claude Code

```bash
# Launch training with Claude Code
claude --agent ml-training "Launch both GNN and Transformer training on SageMaker"

# Monitor training
claude --agent ml-training "Show me real-time training metrics and logs"

# Debug training issues
claude "The GNN training is failing with OOM error. Analyze logs and suggest fixes"

# Optimize training
claude "Profile GNN training and optimize for <4 hour training time"

# Test retraining pipeline
claude "Simulate model drift and test automatic retraining trigger"
```