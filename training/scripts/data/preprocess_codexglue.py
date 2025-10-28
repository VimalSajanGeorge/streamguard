"""
CodeXGLUE Data Preprocessing with Production Safety Checks

Converts raw CodeXGLUE JSONL to standardized format with:
- Token offsets for Integrated Gradients visualization
- AST nodes with fallback strategies
- Vulnerable code-aware trimming
- Memory profiling for GNN batch sizing

Safety features:
- Fast tokenizer validation (Rust-backed required)
- AST parsing with multiple fallbacks
- Graph statistics analysis
- Dataset checksum generation
"""

import json
import hashlib
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import re
import sys

try:
    from transformers import AutoTokenizer
    import numpy as np
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("WARNING: transformers not installed. Install with: pip install transformers")

try:
    import tree_sitter_c
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    print("WARNING: tree-sitter or tree-sitter-c not installed. Install with: pip install tree-sitter tree-sitter-c")


class SafeTokenizer:
    """Tokenizer wrapper with safety checks for token offsets."""

    def __init__(self, model_name: str = "microsoft/codebert-base"):
        """
        Initialize tokenizer with validation.

        Args:
            model_name: HuggingFace model name

        Raises:
            RuntimeError: If tokenizer is not fast or doesn't support offsets
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required. Install: pip install transformers")

        print(f"[*] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # SAFETY CHECK 1: Must be fast (Rust-backed)
        if not self.tokenizer.is_fast:
            raise RuntimeError(
                f"Tokenizer {model_name} is not fast (Rust-backed). "
                "Token offsets require fast tokenizer. "
                "Use models like: microsoft/codebert-base, microsoft/graphcodebert-base"
            )

        # SAFETY CHECK 2: Must support offset_mapping
        test_output = self.tokenizer(
            "test code",
            return_offsets_mapping=True,
            truncation=False,
            padding=False
        )
        if 'offset_mapping' not in test_output:
            raise RuntimeError(
                f"Tokenizer does not support offset_mapping. "
                "This is required for Integrated Gradients visualization."
            )

        print(f"[+] Tokenizer validated: fast={self.tokenizer.is_fast}, "
              f"supports_offsets=True")

    def encode_with_offsets(
        self,
        text: str,
        max_length: int = 512
    ) -> Dict[str, Any]:
        """
        Encode text with token offsets.

        Args:
            text: Input code text
            max_length: Maximum sequence length

        Returns:
            Dictionary with tokens, offsets, attention_mask
        """
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True
        )

        return {
            'tokens': encoding['input_ids'],
            'token_offsets': encoding['offset_mapping'],
            'attention_mask': encoding['attention_mask']
        }


class ASTParser:
    """AST parser with fallback strategies."""

    def __init__(self, language: str = "c"):
        """
        Initialize tree-sitter parser.

        Args:
            language: Programming language (c, cpp, python, etc.)
        """
        if not TREE_SITTER_AVAILABLE:
            raise ImportError("tree-sitter required. Install: pip install tree-sitter")

        self.language = language
        self.parser = None
        self.ts_language = None

        # Try to load pre-built library
        self._load_or_build_language()

    def _load_or_build_language(self):
        """Load tree-sitter language using new API (v0.22+)."""
        try:
            # New API: tree-sitter >= 0.22 uses language-specific packages
            print(f"[*] Loading tree-sitter-{self.language} language")

            if self.language == 'c':
                import tree_sitter_c
                self.ts_language = Language(tree_sitter_c.language())
            elif self.language == 'python':
                try:
                    import tree_sitter_python
                    self.ts_language = Language(tree_sitter_python.language())
                except ImportError:
                    print(f"[!] tree-sitter-python not installed")
                    return
            elif self.language == 'javascript':
                try:
                    import tree_sitter_javascript
                    self.ts_language = Language(tree_sitter_javascript.language())
                except ImportError:
                    print(f"[!] tree-sitter-javascript not installed")
                    return
            else:
                print(f"[!] Unsupported language: {self.language}")
                return

            # New API: pass language to Parser constructor or use .language property
            self.parser = Parser(self.ts_language)
            print(f"[+] Tree-sitter parser ready for {self.language}")

        except Exception as e:
            print(f"[!] Failed to load language: {e}")
            print("    AST parsing will use fallback mode")

    def preprocess_code_for_parsing(self, code: str) -> str:
        """
        Preprocess code to improve parsing success.

        Args:
            code: Raw code string

        Returns:
            Preprocessed code
        """
        # Remove common preprocessor directives that confuse parser
        code = re.sub(r'^\s*#\s*include\s+[<"].*?[>"]\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^\s*#\s*define\s+.*?$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^\s*#\s*ifdef\s+.*?$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^\s*#\s*ifndef\s+.*?$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^\s*#\s*endif\s*$', '', code, flags=re.MULTILINE)

        # Remove excessive whitespace
        code = re.sub(r'\n\n+', '\n\n', code)

        return code.strip()

    def extract_ast_nodes(self, node, node_list: List[int], depth: int = 0) -> None:
        """
        Recursively extract AST node types.

        Args:
            node: Tree-sitter node
            node_list: List to append node type IDs
            depth: Current depth (limit to prevent infinite recursion)
        """
        if depth > 50:  # Safety limit
            return

        # Map node type to integer (simple hash for now)
        node_type_id = hash(node.type) % 1000
        node_list.append(node_type_id)

        for child in node.children:
            self.extract_ast_nodes(child, node_list, depth + 1)

    def extract_edges(self, node, edges: List[List[int]], node_id: int = 0, depth: int = 0) -> int:
        """
        Extract AST edges (parent-child relationships).

        Args:
            node: Tree-sitter node
            edges: List to append edges [parent, child]
            node_id: Current node ID
            depth: Current depth

        Returns:
            Next available node ID
        """
        if depth > 50:
            return node_id

        current_id = node_id

        for child in node.children:
            node_id += 1
            edges.append([current_id, node_id])  # Parent to child
            edges.append([node_id, current_id])  # Child to parent (bidirectional)
            node_id = self.extract_edges(child, edges, node_id, depth + 1)

        return node_id

    def parse_with_fallback(self, code: str) -> Tuple[List[int], List[List[int]], bool]:
        """
        Parse code to AST with fallback strategies.

        Strategy:
        1. Try full parse
        2. If errors, try partial parse (extract what works)
        3. If fails, fallback to token sequence graph

        Args:
            code: Source code string

        Returns:
            Tuple of (ast_nodes, edge_index, success_flag)
        """
        # FALLBACK LEVEL 0: No parser available
        if self.parser is None:
            return self._create_token_sequence_graph(code)

        # Preprocess code
        code_clean = self.preprocess_code_for_parsing(code)

        try:
            # STRATEGY 1: Full parse
            tree = self.parser.parse(bytes(code_clean, 'utf8'))
            root = tree.root_node

            # Check for parse errors
            if not root.has_error:
                # Success! Extract nodes and edges
                ast_nodes = []
                edges = []
                self.extract_ast_nodes(root, ast_nodes)
                self.extract_edges(root, edges)

                return ast_nodes, edges, True

            # STRATEGY 2: Partial parse (has errors but some structure)
            ast_nodes = []
            edges = []
            self.extract_ast_nodes(root, ast_nodes)  # Extract what we can
            self.extract_edges(root, edges)

            if len(ast_nodes) > 0:
                return ast_nodes, edges, False  # Partial success

            # STRATEGY 3: Complete failure, use fallback
            return self._create_token_sequence_graph(code)

        except Exception as e:
            print(f"[!] AST parsing failed: {e}")
            # FALLBACK: Token sequence graph
            return self._create_token_sequence_graph(code)

    def _create_token_sequence_graph(self, code: str) -> Tuple[List[int], List[List[int]], bool]:
        """
        Create simple token sequence graph as fallback.

        Args:
            code: Source code

        Returns:
            Tuple of (node_types, edges, False)
        """
        # Split into tokens (simple whitespace split)
        tokens = code.split()

        # Create node IDs (hash of token)
        node_types = [hash(token) % 1000 for token in tokens[:512]]  # Limit to 512

        # Create sequential edges
        edges = []
        for i in range(len(node_types) - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])  # Bidirectional

        return node_types, edges, False  # False = fallback mode


class VulnerableCodeTrimmer:
    """Trim code while preserving vulnerable sections."""

    # Common vulnerable API patterns
    VULN_PATTERNS = [
        # C/C++ unsafe functions
        r'\bstrcpy\b', r'\bstrcat\b', r'\bsprintf\b', r'\bgets\b',
        r'\bmemcpy\b', r'\bmemmove\b', r'\bscanf\b',
        # Command injection
        r'\bsystem\b', r'\bexec\b', r'\bpopen\b', r'\bshell_exec\b',
        # Code injection
        r'\beval\b', r'\bunserialize\b',
        # SQL patterns
        r'\bexecute\b.*\+', r'\bquery\b.*\+',
        # String concatenation (potential injection)
        r'"\s*\+\s*', r"'\s*\+\s*",
        # Pointer arithmetic
        r'\[\s*\w+\s*\+\s*\w+\s*\]',
    ]

    def __init__(self):
        """Initialize trimmer with compiled patterns."""
        self.patterns = [re.compile(p) for p in self.VULN_PATTERNS]

    def find_vulnerable_spans(self, code: str) -> List[Tuple[int, int]]:
        """
        Find character spans of potential vulnerabilities.

        Args:
            code: Source code

        Returns:
            List of (start, end) character positions
        """
        spans = []

        for pattern in self.patterns:
            for match in pattern.finditer(code):
                spans.append((match.start(), match.end()))

        return spans

    def trim_with_context(
        self,
        code: str,
        token_ids: List[int],
        token_offsets: List[Tuple[int, int]],
        max_length: int = 512
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        Trim to max_length while preserving vulnerable code.

        Args:
            code: Source code
            token_ids: Token IDs from tokenizer
            token_offsets: Token character offsets
            max_length: Maximum tokens

        Returns:
            Tuple of (trimmed_token_ids, trimmed_offsets)
        """
        if len(token_ids) <= max_length:
            return token_ids, token_offsets

        # Find vulnerable spans
        vuln_spans = self.find_vulnerable_spans(code)

        if not vuln_spans:
            # No vulnerabilities detected, use simple truncation
            return token_ids[:max_length], token_offsets[:max_length]

        # Find tokens that overlap with vulnerable code
        vuln_token_indices = set()
        for token_idx, (start, end) in enumerate(token_offsets):
            for vuln_start, vuln_end in vuln_spans:
                if not (end <= vuln_start or start >= vuln_end):
                    # Token overlaps with vulnerable span
                    vuln_token_indices.add(token_idx)

        if not vuln_token_indices:
            # No overlap found, use simple truncation
            return token_ids[:max_length], token_offsets[:max_length]

        # Find middle of vulnerable region
        middle_idx = sorted(vuln_token_indices)[len(vuln_token_indices) // 2]

        # Take window around middle
        start_idx = max(0, middle_idx - max_length // 2)
        end_idx = min(len(token_ids), start_idx + max_length)

        # Adjust start if end hit boundary
        if end_idx - start_idx < max_length:
            start_idx = max(0, end_idx - max_length)

        return token_ids[start_idx:end_idx], token_offsets[start_idx:end_idx]


class GraphStatistics:
    """Collect graph statistics for GNN batch sizing."""

    def __init__(self):
        """Initialize statistics collector."""
        self.node_counts = []
        self.edge_counts = []

    def add_sample(self, ast_nodes: List[int], edges: List[List[int]]):
        """Add sample statistics."""
        self.node_counts.append(len(ast_nodes))
        self.edge_counts.append(len(edges))

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dictionary with statistics and batch size recommendation
        """
        if not self.node_counts:
            return {}

        node_counts = np.array(self.node_counts)
        edge_counts = np.array(self.edge_counts)

        stats = {
            'total_samples': len(self.node_counts),
            'avg_nodes': float(np.mean(node_counts)),
            'median_nodes': float(np.median(node_counts)),
            'p95_nodes': float(np.percentile(node_counts, 95)),
            'max_nodes': int(np.max(node_counts)),
            'avg_edges': float(np.mean(edge_counts)),
            'p95_edges': float(np.percentile(edge_counts, 95)),
        }

        # Recommend batch size based on memory
        # Assumptions:
        # - GPU: 16GB (T4)
        # - Hidden dim: 256
        # - 4 bytes per float32
        # - Safety margin: 0.5
        gpu_mem_bytes = 16 * 1e9
        hidden_dim = 256
        bytes_per_param = 4
        safety_margin = 0.5

        mem_per_node = hidden_dim * bytes_per_param
        recommended_batch = int(
            (gpu_mem_bytes * safety_margin) / (stats['p95_nodes'] * mem_per_node)
        )
        recommended_batch = max(1, min(recommended_batch, 64))  # Clamp to [1, 64]

        stats['recommended_batch_size'] = recommended_batch

        return stats


def compute_checksum(file_path: Path) -> str:
    """
    Compute SHA256 checksum of file.

    Args:
        file_path: Path to file

    Returns:
        Hex digest of SHA256
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def preprocess_dataset(
    input_path: Path,
    output_path: Path,
    split_name: str,
    tokenizer: SafeTokenizer,
    ast_parser: ASTParser,
    trimmer: VulnerableCodeTrimmer,
    graph_stats: GraphStatistics,
    max_seq_len: int = 512,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Preprocess dataset with all safety checks.

    Args:
        input_path: Input JSONL file
        output_path: Output JSONL file
        split_name: Split name (train/valid/test)
        tokenizer: Safe tokenizer instance
        ast_parser: AST parser instance
        trimmer: Code trimmer instance
        graph_stats: Statistics collector
        max_seq_len: Maximum sequence length
        max_samples: Maximum samples to process (for testing)

    Returns:
        Statistics dictionary
    """
    print(f"\n[*] Processing {split_name} split: {input_path}")

    samples_processed = 0
    samples_with_ast = 0
    samples_with_fallback = 0
    samples_trimmed = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            if max_samples and samples_processed >= max_samples:
                break

            try:
                sample = json.loads(line.strip())

                # Extract fields
                code = sample.get('func', '')
                label = sample.get('target', 0)
                orig_id = sample.get('idx', f'{split_name}_{line_num}')

                if not code:
                    continue

                # Tokenize with offsets
                encoding = tokenizer.encode_with_offsets(code, max_length=max_seq_len)
                tokens = encoding['tokens']
                token_offsets = encoding['token_offsets']

                # Trim if needed
                if len(tokens) > max_seq_len:
                    tokens, token_offsets = trimmer.trim_with_context(
                        code, tokens, token_offsets, max_seq_len
                    )
                    samples_trimmed += 1

                # Parse AST with fallback
                ast_nodes, edge_index, ast_success = ast_parser.parse_with_fallback(code)

                if ast_success:
                    samples_with_ast += 1
                else:
                    samples_with_fallback += 1

                # Collect graph statistics
                graph_stats.add_sample(ast_nodes, edge_index)

                # Create standardized sample
                output_sample = {
                    'id': f'CODEXGLUE-{split_name.upper()}-{orig_id}',
                    'code': code,
                    'language': 'c',  # CodeXGLUE is C/C++
                    'tokens': tokens,
                    'token_offsets': token_offsets,
                    'ast_nodes': ast_nodes,
                    'edge_index': edge_index,
                    'label': label,
                    'metadata': {
                        'source': 'codexglue',
                        'split': split_name,
                        'orig_id': str(orig_id),
                        'ast_success': ast_success,
                        'num_tokens': len(tokens),
                        'num_ast_nodes': len(ast_nodes)
                    }
                }

                # Write to output
                f_out.write(json.dumps(output_sample) + '\n')
                samples_processed += 1

                # Progress
                if samples_processed % 1000 == 0:
                    print(f"    Processed {samples_processed} samples "
                          f"(AST: {samples_with_ast}, Fallback: {samples_with_fallback})")

            except Exception as e:
                print(f"[!] Error processing line {line_num}: {e}")
                continue

    # Statistics
    stats = {
        'split': split_name,
        'total_processed': samples_processed,
        'ast_success': samples_with_ast,
        'ast_fallback': samples_with_fallback,
        'ast_success_rate': samples_with_ast / max(samples_processed, 1),
        'samples_trimmed': samples_trimmed,
        'output_path': str(output_path),
        'output_checksum': compute_checksum(output_path)
    }

    print(f"\n[+] {split_name.upper()} complete:")
    print(f"    Total: {samples_processed}")
    print(f"    AST Success: {samples_with_ast} ({stats['ast_success_rate']:.1%})")
    print(f"    AST Fallback: {samples_with_fallback}")
    print(f"    Trimmed: {samples_trimmed}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CodeXGLUE dataset with safety checks"
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/raw/codexglue'),
        help='Input directory with CodeXGLUE JSONL files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/processed/codexglue'),
        help='Output directory for processed files'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='microsoft/codebert-base',
        help='Tokenizer model name'
    )
    parser.add_argument(
        '--max-seq-len',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Process only 100 samples per split for testing'
    )

    args = parser.parse_args()

    print("="*70)
    print("CodeXGLUE Preprocessing with Safety Checks")
    print("="*70)

    # Initialize components
    print("\n[*] Initializing components...")

    try:
        tokenizer = SafeTokenizer(args.tokenizer)
    except Exception as e:
        print(f"[!] Failed to initialize tokenizer: {e}")
        return 1

    ast_parser = ASTParser(language='c')
    trimmer = VulnerableCodeTrimmer()
    graph_stats = GraphStatistics()

    # Process splits
    splits = ['train', 'valid', 'test']
    all_stats = {}

    max_samples = 100 if args.quick_test else None

    for split in splits:
        input_file = args.input_dir / f"{split}.jsonl"
        output_file = args.output_dir / f"{split}.jsonl"

        if not input_file.exists():
            print(f"[!] Input file not found: {input_file}")
            continue

        stats = preprocess_dataset(
            input_file,
            output_file,
            split,
            tokenizer,
            ast_parser,
            trimmer,
            graph_stats,
            max_seq_len=args.max_seq_len,
            max_samples=max_samples
        )
        all_stats[split] = stats

    # Graph statistics
    print("\n" + "="*70)
    print("Graph Statistics & GNN Batch Size Recommendation")
    print("="*70)

    graph_summary = graph_stats.get_summary()
    for key, value in graph_summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'tokenizer': args.tokenizer,
        'max_seq_len': args.max_seq_len,
        'quick_test': args.quick_test,
        'splits': all_stats,
        'graph_statistics': graph_summary
    }

    metadata_path = args.output_dir / 'preprocessing_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[+] Metadata saved to: {metadata_path}")

    # Summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)

    total_samples = sum(s['total_processed'] for s in all_stats.values())
    total_ast_success = sum(s['ast_success'] for s in all_stats.values())

    print(f"\nTotal samples processed: {total_samples}")
    print(f"Overall AST success rate: {total_ast_success/max(total_samples, 1):.1%}")
    print(f"Recommended GNN batch size: {graph_summary.get('recommended_batch_size', 'N/A')}")

    print("\nOutput files:")
    for split, stats in all_stats.items():
        print(f"  {split}: {stats['output_path']}")

    print("\n[+] Ready for training!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
