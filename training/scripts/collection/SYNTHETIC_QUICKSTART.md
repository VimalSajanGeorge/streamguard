# Synthetic Data Generator - Quick Start Guide

## 1. Generate 500 Samples (Default)

```bash
cd training/scripts/collection
python synthetic_generator.py --validate
```

**Output:** `data/raw/synthetic/synthetic_data.jsonl` (500 samples)

## 2. View Statistics

```bash
python synthetic_generator.py --validate
```

**Expected Output:**
- Total samples: 500
- Vulnerable: 250 (50%)
- Safe: 250 (50%)
- 5 vulnerability types (100 samples each)
- 4 programming languages (Python, JavaScript, PHP, Java)

## 3. Programmatic Usage

```python
from synthetic_generator import SyntheticGenerator

# Initialize generator
generator = SyntheticGenerator(output_dir="data/raw/synthetic", seed=42)

# Generate samples
samples = generator.generate_all(total_samples=500)

# Save to file
output_path = generator.save_samples(samples, "synthetic_data.jsonl")

# Validate
stats = generator.validate_dataset(samples)
print(f"Generated {stats['total_samples']} samples")
```

## 4. Generate Specific Vulnerability Type

```python
# Generate 20 SQL injection samples (10 pairs)
samples = generator.generate_for_type("sql_injection_concat", num_pairs=10)

# Available types:
# - sql_injection_concat
# - xss_output
# - command_injection
# - path_traversal
# - ssrf
```

## 5. Run Examples

```bash
python example_synthetic_usage.py
```

This runs 6 comprehensive examples demonstrating all features.

## 6. Sample Output Format

```json
{
  "code": "query = \"SELECT * FROM users WHERE id=\" + user_id",
  "vulnerable": true,
  "vulnerability_type": "sql_injection_concat",
  "counterfactual": "cursor.execute(\"SELECT * FROM users WHERE id=?\", (user_id,))",
  "source": "synthetic",
  "language": "python",
  "generated_at": "2025-10-14T10:30:00.000000"
}
```

## 7. Integration with Training Pipeline

```python
# Load synthetic data
import json

samples = []
with open("data/raw/synthetic/synthetic_data.jsonl", "r") as f:
    for line in f:
        samples.append(json.loads(line))

# Filter by type
sql_samples = [s for s in samples if s["vulnerability_type"] == "sql_injection_concat"]

# Filter by language
python_samples = [s for s in samples if s["language"] == "python"]

# Get vulnerable/safe pairs
vulnerable = [s for s in samples if s["vulnerable"]]
safe = [s for s in samples if not s["vulnerable"]]
```

## 8. Command Line Options

```bash
# Custom output location
python synthetic_generator.py --output-dir my/custom/path

# Different sample count (must be even)
python synthetic_generator.py --num-samples 1000

# Different random seed
python synthetic_generator.py --seed 123

# Custom output filename
python synthetic_generator.py --output-file custom_data.jsonl

# All options combined
python synthetic_generator.py \
    --output-dir data/raw/synthetic \
    --num-samples 500 \
    --seed 42 \
    --output-file synthetic_data.jsonl \
    --validate
```

## 9. Expected Results

After running the default command:

- **File**: `training/scripts/collection/data/raw/synthetic/synthetic_data.jsonl`
- **Size**: ~100KB
- **Samples**: 500 total
  - 250 vulnerable
  - 250 safe
  - 100 samples per vulnerability type
  - Distributed across Python, JavaScript, PHP, Java

## 10. Troubleshooting

### "total_samples must be even"
Provide an even number for `--num-samples` (e.g., 500, 1000, 2000)

### Import error
Make sure you're in the correct directory:
```bash
cd training/scripts/collection
```

### File not found
The generator automatically creates directories. Check the output path in the console.

## 11. Next Steps

1. **Generate data**: Run the generator
2. **Inspect samples**: Open the JSONL file and review
3. **Integrate**: Use with preprocessing pipeline
4. **Train model**: Include synthetic data in training dataset
5. **Evaluate**: Test model performance on synthetic vs. real data

## Documentation

For full documentation, see `SYNTHETIC_GENERATOR_README.md`
