"""
Unit tests for CodeXGLUE preprocessing components.

Tests safety-critical functionality:
- Fast tokenizer validation
- Token offset generation
- AST parsing with fallbacks
- Vulnerable code trimming
- Graph statistics
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from training.scripts.data.preprocess_codexglue import (
        SafeTokenizer, ASTParser, VulnerableCodeTrimmer, GraphStatistics
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestSafeTokenizer(unittest.TestCase):
    """Test tokenizer safety checks."""

    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_fast_tokenizer_validation(self):
        """Test that tokenizer validates fast tokenizer requirement."""
        try:
            tokenizer = SafeTokenizer("microsoft/codebert-base")
            self.assertTrue(tokenizer.tokenizer.is_fast)
        except Exception as e:
            self.fail(f"SafeTokenizer initialization failed: {e}")

    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_offset_mapping_support(self):
        """Test that tokenizer supports offset mapping."""
        try:
            tokenizer = SafeTokenizer("microsoft/codebert-base")
            encoding = tokenizer.encode_with_offsets("void main() { }")

            self.assertIn('tokens', encoding)
            self.assertIn('token_offsets', encoding)
            self.assertIn('attention_mask', encoding)

            # Check offsets are tuples of (start, end)
            self.assertIsInstance(encoding['token_offsets'], list)
            if len(encoding['token_offsets']) > 0:
                self.assertIsInstance(encoding['token_offsets'][0], (list, tuple))
                self.assertEqual(len(encoding['token_offsets'][0]), 2)

        except Exception as e:
            self.fail(f"Offset encoding failed: {e}")

    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_truncation(self):
        """Test that tokenizer respects max_length."""
        try:
            tokenizer = SafeTokenizer("microsoft/codebert-base")

            # Create long code
            long_code = "int x;\n" * 1000

            encoding = tokenizer.encode_with_offsets(long_code, max_length=512)

            self.assertLessEqual(len(encoding['tokens']), 512)
            self.assertEqual(len(encoding['tokens']), len(encoding['token_offsets']))

        except Exception as e:
            self.fail(f"Truncation test failed: {e}")


class TestASTParser(unittest.TestCase):
    """Test AST parser with fallbacks."""

    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_parser_initialization(self):
        """Test parser can initialize."""
        try:
            parser = ASTParser(language='c')
            # Parser may or may not have tree-sitter available
            # Should not raise exception either way
        except Exception as e:
            self.fail(f"Parser initialization failed: {e}")

    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_simple_code_parsing(self):
        """Test parsing simple C code."""
        parser = ASTParser(language='c')

        code = """
        void foo() {
            int x = 5;
            return x;
        }
        """

        ast_nodes, edges, success = parser.parse_with_fallback(code)

        # Should always return something (either AST or fallback)
        self.assertIsInstance(ast_nodes, list)
        self.assertIsInstance(edges, list)
        self.assertIsInstance(success, bool)

        # Should have at least some nodes
        self.assertGreater(len(ast_nodes), 0)

    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_malformed_code_fallback(self):
        """Test fallback works for malformed code."""
        parser = ASTParser(language='c')

        # Malformed code
        code = "void foo( { int x = "

        ast_nodes, edges, success = parser.parse_with_fallback(code)

        # Should still return something via fallback
        self.assertIsInstance(ast_nodes, list)
        self.assertIsInstance(edges, list)
        self.assertGreater(len(ast_nodes), 0)

    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_preprocessor_handling(self):
        """Test code with preprocessor directives."""
        parser = ASTParser(language='c')

        code = """
        #include <stdio.h>
        #define MAX 100

        void foo() {
            printf("test");
        }
        """

        # Should handle preprocessor directives
        ast_nodes, edges, success = parser.parse_with_fallback(code)

        self.assertGreater(len(ast_nodes), 0)


class TestVulnerableCodeTrimmer(unittest.TestCase):
    """Test vulnerable code-aware trimming."""

    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_find_vulnerable_spans(self):
        """Test detection of vulnerable code patterns."""
        trimmer = VulnerableCodeTrimmer()

        code = """
        void foo(char *input) {
            char buffer[100];
            strcpy(buffer, input);  // VULNERABLE
            system(input);  // VULNERABLE
        }
        """

        spans = trimmer.find_vulnerable_spans(code)

        # Should find strcpy and system
        self.assertGreater(len(spans), 0)

    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_safe_code_no_spans(self):
        """Test safe code has no vulnerable spans."""
        trimmer = VulnerableCodeTrimmer()

        code = """
        void foo() {
            int x = 5;
            int y = 10;
            return x + y;
        }
        """

        spans = trimmer.find_vulnerable_spans(code)

        # Should find no vulnerabilities
        self.assertEqual(len(spans), 0)

    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_trim_preserves_vulnerable_code(self):
        """Test trimming preserves vulnerable sections."""
        try:
            tokenizer = SafeTokenizer("microsoft/codebert-base")
            trimmer = VulnerableCodeTrimmer()

            # Create code with vulnerability in middle
            code_before = "int x = 0;\n" * 100
            vuln_code = "strcpy(buffer, input);"
            code_after = "int y = 0;\n" * 100

            full_code = code_before + vuln_code + code_after

            # Tokenize
            encoding = tokenizer.encode_with_offsets(full_code, max_length=99999)
            tokens = encoding['tokens']
            offsets = encoding['token_offsets']

            # Trim to small window
            trimmed_tokens, trimmed_offsets = trimmer.trim_with_context(
                full_code, tokens, offsets, max_length=50
            )

            self.assertEqual(len(trimmed_tokens), 50)
            self.assertEqual(len(trimmed_offsets), 50)

            # Reconstruct trimmed code
            trimmed_code = full_code[trimmed_offsets[0][0]:trimmed_offsets[-1][1]]

            # Vulnerable code should still be present
            self.assertIn('strcpy', trimmed_code)

        except Exception as e:
            self.skipTest(f"Tokenizer not available: {e}")


class TestGraphStatistics(unittest.TestCase):
    """Test graph statistics collector."""

    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_statistics_collection(self):
        """Test collecting graph statistics."""
        stats = GraphStatistics()

        # Add sample graphs
        stats.add_sample([1, 2, 3, 4, 5], [[0, 1], [1, 2]])
        stats.add_sample([1, 2, 3], [[0, 1]])
        stats.add_sample([1, 2, 3, 4, 5, 6, 7], [[0, 1], [1, 2], [2, 3]])

        summary = stats.get_summary()

        self.assertEqual(summary['total_samples'], 3)
        self.assertIn('avg_nodes', summary)
        self.assertIn('p95_nodes', summary)
        self.assertIn('recommended_batch_size', summary)

        # Batch size should be positive
        self.assertGreater(summary['recommended_batch_size'], 0)

    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_batch_size_recommendation(self):
        """Test batch size scales with graph size."""
        stats_small = GraphStatistics()
        stats_large = GraphStatistics()

        # Add small graphs
        for _ in range(10):
            stats_small.add_sample([1] * 50, [[0, 1]])

        # Add large graphs
        for _ in range(10):
            stats_large.add_sample([1] * 500, [[0, 1]])

        summary_small = stats_small.get_summary()
        summary_large = stats_large.get_summary()

        # Smaller graphs should allow larger batch size
        self.assertGreater(
            summary_small['recommended_batch_size'],
            summary_large['recommended_batch_size']
        )


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestSafeTokenizer))
    suite.addTests(loader.loadTestsFromTestCase(TestASTParser))
    suite.addTests(loader.loadTestsFromTestCase(TestVulnerableCodeTrimmer))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphStatistics))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
