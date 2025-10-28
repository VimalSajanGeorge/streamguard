# Synthetic Data Generator for StreamGuard

## Overview

The Synthetic Data Generator creates artificial code samples with vulnerability patterns and their safe counterparts. It generates **500 samples** (250 vulnerable + 250 safe) across 5 vulnerability types.

## Features

- **Template-based generation**: Uses predefined templates for realistic code patterns
- **Counterfactual pairs**: Each vulnerable sample has a corresponding safe version
- **Multi-language support**: Python, JavaScript, PHP, Java
- **5 vulnerability types**: SQL Injection, XSS, Command Injection, Path Traversal, SSRF
- **Reproducible**: Seeded random generation for consistent results

## Installation

No additional dependencies required. Uses the base collector framework.

```bash
cd training/scripts/collection
```

## Quick Start

### Command Line Usage

```bash
# Generate default 500 samples
python synthetic_generator.py

# Generate with validation
python synthetic_generator.py --validate

# Custom number of samples (must be even)
python synthetic_generator.py --num-samples 1000

# Specify output directory and file
python synthetic_generator.py --output-dir data/raw/synthetic --output-file my_data.jsonl

# Set random seed for reproducibility
python synthetic_generator.py --seed 123
```

### Programmatic Usage

```python
from synthetic_generator import SyntheticGenerator

# Create generator
generator = SyntheticGenerator(
    output_dir="data/raw/synthetic",
    seed=42
)

# Generate all samples
samples = generator.generate_all(total_samples=500)

# Save to file
output_path = generator.save_samples(samples, "synthetic_data.jsonl")
print(f"Saved to: {output_path}")
```

## Vulnerability Types

### 1. SQL Injection (Concatenation)
**Vulnerable:**
```python
query = "SELECT * FROM users WHERE id=" + user_id
```

**Safe:**
```python
cursor.execute("SELECT * FROM users WHERE id=?", (user_id,))
```

### 2. Cross-Site Scripting (XSS)
**Vulnerable:**
```javascript
return "<div>" + user_input + "</div>"
```

**Safe:**
```javascript
return "<div>" + escape(user_input) + "</div>"
```

### 3. Command Injection
**Vulnerable:**
```python
os.system("ping " + user_input)
```

**Safe:**
```python
subprocess.run(["ping", user_input], shell=False)
```

### 4. Path Traversal
**Vulnerable:**
```python
with open("/var/www/files/" + user_input, "r") as f:
```

**Safe:**
```python
safe_path = os.path.basename(user_input)
with open("/var/www/files/" + safe_path, "r") as f:
```

### 5. Server-Side Request Forgery (SSRF)
**Vulnerable:**
```python
response = requests.get(user_input)
```

**Safe:**
```python
if user_input.startswith("https://"):
    response = requests.get(user_input)
```

## Output Format

Each sample is a JSON object with the following fields:

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

## Advanced Usage

### Generate Specific Vulnerability Type

```python
from synthetic_generator import SyntheticGenerator

generator = SyntheticGenerator()

# Generate 10 pairs (20 samples) of SQL injection
samples = generator.generate_for_type("sql_injection_concat", num_pairs=10)
```

### Create Custom Counterfactual Pairs

```python
# Create a specific vulnerable/safe pair
vuln_sample, safe_sample = generator.create_vulnerable_safe_pair(
    vuln_type="xss_output",
    vuln_idx=0,  # First vulnerable template
    safe_idx=0   # First safe template
)

print("Vulnerable:", vuln_sample['code'])
print("Safe:", safe_sample['code'])
```

### Validate Generated Dataset

```python
# Generate samples
samples = generator.generate_all(total_samples=500)

# Run validation
stats = generator.validate_dataset(samples)

print(f"Total: {stats['total_samples']}")
print(f"Vulnerable: {stats['vulnerable_count']}")
print(f"Safe: {stats['safe_count']}")
print(f"By type: {stats['by_type']}")
print(f"By language: {stats['by_language']}")
```

## Command Line Options

```
usage: synthetic_generator.py [-h] [--output-dir OUTPUT_DIR]
                               [--num-samples NUM_SAMPLES]
                               [--seed SEED] [--output-file OUTPUT_FILE]
                               [--validate]

Generate synthetic vulnerability samples for StreamGuard

optional arguments:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
                        Output directory for synthetic data (default: data/raw/synthetic)
  --num-samples NUM_SAMPLES
                        Total number of samples to generate, must be even (default: 500)
  --seed SEED           Random seed for reproducibility (default: 42)
  --output-file OUTPUT_FILE
                        Output filename (default: synthetic_data.jsonl)
  --validate            Run validation after generation
```

## Examples

See `example_synthetic_usage.py` for comprehensive examples:

```bash
python example_synthetic_usage.py
```

Examples include:
1. Basic generation (500 samples)
2. Generate specific vulnerability type
3. Generate and validate
4. Create custom counterfactual pairs
5. Show all vulnerability types
6. Load and analyze generated data

## Template Structure

Templates are defined in `SyntheticGenerator.vulnerability_templates`:

```python
vulnerability_templates = {
    "vulnerability_type_name": {
        "vulnerable": [
            "template1 with {placeholders}",
            "template2 with {placeholders}",
        ],
        "safe": [
            "safe_template1 with {placeholders}",
            "safe_template2 with {placeholders}",
        ]
    }
}
```

Placeholders are expanded using the vocabulary:
- `{table}`: Database table names
- `{column}`: Database column names
- `{user_input}`: User input variables
- `{value}`: Generic values
- `{element}`: DOM element names

## Extending the Generator

### Add New Vulnerability Type

```python
# In SyntheticGenerator.__init__
self.vulnerability_templates["new_vuln_type"] = {
    "vulnerable": [
        "vulnerable code template with {placeholders}",
    ],
    "safe": [
        "safe code template with {placeholders}",
    ]
}
```

### Add New Vocabulary

```python
# In SyntheticGenerator.__init__
self.vocabulary["new_placeholder"] = [
    "option1",
    "option2",
    "option3",
]
```

### Add Language Hints

```python
# In SyntheticGenerator.__init__
self.language_hints["pattern_to_match"] = "language_name"
```

## Output Location

By default, synthetic data is saved to:
```
data/raw/synthetic/synthetic_data.jsonl
```

## Statistics

For a 500-sample dataset:
- **Total samples**: 500
- **Vulnerable**: 250 (50%)
- **Safe**: 250 (50%)
- **Pairs per type**: 50 (100 samples per vulnerability type)
- **Languages**: Python, JavaScript, PHP, Java
- **Format**: JSONL (one JSON object per line)

## Validation Checks

The validator checks:
- Total sample count
- Vulnerable/safe balance
- Distribution by vulnerability type
- Distribution by language
- Counterfactual presence
- Average code length

## Integration with StreamGuard

The synthetic data generator integrates with the StreamGuard training pipeline:

1. **Data Collection**: Generates synthetic samples
2. **Data Storage**: Saves to `data/raw/synthetic/`
3. **Data Processing**: Processed by preprocessing pipeline
4. **Model Training**: Used for training the vulnerability detection model

## Best Practices

1. **Use consistent seed**: Set `--seed` for reproducible datasets
2. **Validate output**: Always run with `--validate` flag
3. **Even numbers**: Ensure `--num-samples` is even (pairs)
4. **Review samples**: Check a few samples manually for quality
5. **Mix with real data**: Combine with CVE and repository data

## Troubleshooting

### Issue: "total_samples must be even"
**Solution**: Provide an even number for `--num-samples` (e.g., 500, 1000)

### Issue: Output directory doesn't exist
**Solution**: The generator automatically creates the directory

### Issue: Import error for BaseCollector
**Solution**: Ensure you're in the correct directory and base_collector.py exists

## Performance

- Generation speed: ~1000 samples/second
- Memory usage: Low (< 50MB for 500 samples)
- Disk usage: ~100KB per 500 samples (JSONL format)

## Quality Assurance

Each generated sample:
- Has a valid counterfactual pair
- Contains realistic code patterns
- Is properly labeled with vulnerability type
- Includes language detection
- Has timestamp metadata

## Future Enhancements

Potential improvements:
- More vulnerability types (XXE, CSRF, Deserialization)
- More language support (Ruby, Go, C#)
- Context-aware code generation
- Variable naming conventions
- More complex code patterns
- Multi-line code blocks

## License

Part of the StreamGuard project.

## Support

For issues or questions, refer to the main StreamGuard documentation.
