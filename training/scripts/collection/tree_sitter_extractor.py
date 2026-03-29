"""
Tree-Sitter Function Extractor with Smart Slicer

Extracts complete function bodies from source code with:
- Full function body extraction
- Smart slicing for oversized functions (token limit handling)
- Context window fallback
- Multi-language support (Python, JavaScript/TypeScript)
"""

import re
from typing import List, Optional, Tuple, Dict
from pathlib import Path

try:
    from tree_sitter import Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    Parser = None

# Try to import tree-sitter-languages (pre-built binaries for all languages)
try:
    from tree_sitter_languages import get_language, get_parser
    TREE_SITTER_LANGUAGES_AVAILABLE = True
except ImportError:
    TREE_SITTER_LANGUAGES_AVAILABLE = False
    get_language = None
    get_parser = None


class TreeSitterFunctionExtractor:
    """Extract function bodies using tree-sitter with smart slicing."""
    
    MAX_TOKENS = 1000  # Transformer-friendly limit
    CONTEXT_LINES = 50  # Lines of context for smart slicing
    
    def __init__(self):
        """Initialize function extractor."""
        self.parsers = {}
        self.stats = {
            'full_function': 0,
            'smart_sliced': 0,
            'context_fallback': 0,
            'failed': 0
        }
        
        self._init_parsers()
    
    def _init_parsers(self):
        """Initialize tree-sitter parsers for supported languages."""
        # Method 1: Try tree-sitter-languages (pre-built, recommended)
        if TREE_SITTER_LANGUAGES_AVAILABLE:
            try:
                self.parsers['python'] = get_parser('python')
                self.parsers['javascript'] = get_parser('javascript')
                self.parsers['typescript'] = get_parser('typescript')
                print("Using tree-sitter-languages (pre-built binaries)")
                return
            except Exception as e:
                print(f"tree-sitter-languages failed: {e}")
        
        # Method 2: Try individual language packages (requires Rust)
        if TREE_SITTER_AVAILABLE:
            try:
                import tree_sitter_python
                import tree_sitter_javascript
                from tree_sitter import Language
                
                # Python parser
                PY_LANGUAGE = Language(tree_sitter_python.language())
                py_parser = Parser(PY_LANGUAGE)
                self.parsers['python'] = py_parser
                
                # JavaScript/TypeScript parser
                JS_LANGUAGE = Language(tree_sitter_javascript.language())
                js_parser = Parser(JS_LANGUAGE)
                self.parsers['javascript'] = js_parser
                self.parsers['typescript'] = js_parser
                print("Using tree-sitter with individual language packages")
                return
            except Exception as e:
                print(f"Individual language packages not available: {e}")
        
        # If we get here, no parser method worked
        if not self.parsers:
            print("Warning: No tree-sitter parsers available. Using context window extraction only.")
    
    def extract_function_body(
        self,
        file_content: str,
        changed_lines: List[int],
        language: str
    ) -> Optional[str]:
        """
        Extract function containing changed lines.
        
        Args:
            file_content: Full file content
            changed_lines: Line numbers that were changed
            language: Programming language
            
        Returns:
            Extracted function code or None
        """
        if not changed_lines:
            return None
        
        # Normalize language
        language = language.lower()
        if language in ['js', 'jsx', 'tsx', 'ts']:
            language = 'javascript'
        
        # Try tree-sitter extraction
        if language in self.parsers:
            try:
                result = self._extract_with_treesitter(
                    file_content, changed_lines, language
                )
                if result:
                    return result
            except Exception as e:
                print(f"Tree-sitter extraction failed: {e}")
        
        # Fallback to context window
        return self._fallback_context_window(file_content, changed_lines)
    
    def _extract_with_treesitter(
        self,
        file_content: str,
        changed_lines: List[int],
        language: str
    ) -> Optional[str]:
        """
        Extract function using tree-sitter parsing.
        
        Args:
            file_content: Full file content
            changed_lines: Line numbers that were changed
            language: Programming language
            
        Returns:
            Extracted function code or None
        """
        parser = self.parsers.get(language)
        if not parser:
            return None
        
        # Parse the file
        tree = parser.parse(bytes(file_content, 'utf-8'))
        root_node = tree.root_node
        
        # Find the function node containing the changed lines
        func_node = self._find_enclosing_function(root_node, changed_lines, language)
        
        if func_node is None:
            self.stats['failed'] += 1
            return None
        
        # Extract function text
        func_code = self._extract_node_text(file_content, func_node)
        
        # Check if function is too large (smart slicer)
        if self._estimate_tokens(func_code) > self.MAX_TOKENS:
            self.stats['smart_sliced'] += 1
            return self._smart_slice(file_content, func_node, changed_lines)
        
        self.stats['full_function'] += 1
        return func_code
    
    def _find_enclosing_function(
        self,
        node: 'Node',
        changed_lines: List[int],
        language: str
    ) -> Optional['Node']:
        """
        Find the function node that encloses the changed lines.
        
        Args:
            node: Tree-sitter node to search
            changed_lines: Line numbers that were changed
            language: Programming language
            
        Returns:
            Function node or None
        """
        # Function node types by language
        function_types = {
            'python': ['function_definition', 'async_function_definition'],
            'javascript': [
                'function_declaration',
                'function_expression',
                'arrow_function',
                'method_definition'
            ]
        }
        
        target_types = function_types.get(language, [])
        
        def search_node(n: 'Node') -> Optional['Node']:
            """Recursively search for function containing changed lines."""
            # Check if this node is a function
            if n.type in target_types:
                start_line = n.start_point[0] + 1  # Convert to 1-indexed
                end_line = n.end_point[0] + 1
                
                # Check if any changed line is within this function
                for line in changed_lines:
                    if start_line <= line <= end_line:
                        return n
            
            # Search children
            for child in n.children:
                result = search_node(child)
                if result:
                    return result
            
            return None
        
        return search_node(node)
    
    def _extract_node_text(self, file_content: str, node: 'Node') -> str:
        """
        Extract text for a tree-sitter node.
        
        Args:
            file_content: Full file content
            node: Tree-sitter node
            
        Returns:
            Node text
        """
        start_byte = node.start_byte
        end_byte = node.end_byte
        return file_content[start_byte:end_byte]
    
    def _estimate_tokens(self, code: str) -> int:
        """
        Estimate token count for code.
        
        Uses rough heuristic: ~4 characters per token.
        
        Args:
            code: Code string
            
        Returns:
            Estimated token count
        """
        return len(code) // 4
    
    def _smart_slice(
        self,
        file_content: str,
        func_node: 'Node',
        changed_lines: List[int]
    ) -> str:
        """
        Smart slice a large function to keep it under token limit.
        
        Strategy:
        1. Keep function signature (first 2-3 lines)
        2. Keep context around changed lines (50 lines before/after)
        3. Add truncation markers
        
        Args:
            file_content: Full file content
            func_node: Function node
            changed_lines: Line numbers that were changed
            
        Returns:
            Sliced function code
        """
        lines = file_content.split('\n')
        func_start = func_node.start_point[0]
        func_end = func_node.end_point[0]
        
        sliced_parts = []
        
        # 1. Extract function signature
        signature_end = self._find_signature_end(func_node, lines)
        signature_lines = lines[func_start:signature_end]
        sliced_parts.append('\n'.join(signature_lines))
        sliced_parts.append('    # ... (function body truncated for length) ...')
        
        # 2. Extract context around each changed line
        included_ranges = []
        for change_line in sorted(changed_lines):
            # Convert to 0-indexed
            change_idx = change_line - 1
            
            # Calculate context window
            start = max(signature_end, change_idx - self.CONTEXT_LINES)
            end = min(func_end, change_idx + self.CONTEXT_LINES + 1)
            
            # Merge overlapping ranges
            merged = False
            for i, (existing_start, existing_end) in enumerate(included_ranges):
                if start <= existing_end and end >= existing_start:
                    # Merge ranges
                    included_ranges[i] = (
                        min(start, existing_start),
                        max(end, existing_end)
                    )
                    merged = True
                    break
            
            if not merged:
                included_ranges.append((start, end))
        
        # 3. Add context sections
        for start, end in sorted(included_ranges):
            if start > signature_end:
                sliced_parts.append('    # ...')
            sliced_parts.append('\n'.join(lines[start:end]))
        
        return '\n'.join(sliced_parts)
    
    def _find_signature_end(
        self,
        func_node: 'Node',
        lines: List[str]
    ) -> int:
        """
        Find the end of function signature.
        
        Args:
            func_node: Function node
            lines: File lines
            
        Returns:
            Line index where signature ends
        """
        func_start = func_node.start_point[0]
        
        # Look for the first line with a colon (Python) or opening brace (JS)
        for i in range(func_start, min(func_start + 10, len(lines))):
            line = lines[i].rstrip()
            if line.endswith(':') or line.endswith('{'):
                return i + 1
        
        # Default: first 3 lines
        return min(func_start + 3, len(lines))
    
    def _fallback_context_window(
        self,
        file_content: str,
        changed_lines: List[int]
    ) -> str:
        """
        Fallback extraction using context window.
        
        Used when tree-sitter parsing fails.
        
        Args:
            file_content: Full file content
            changed_lines: Line numbers that were changed
            
        Returns:
            Code with context
        """
        self.stats['context_fallback'] += 1
        
        lines = file_content.split('\n')
        
        if not changed_lines:
            return ""
        
        # Get min/max changed lines
        min_line = min(changed_lines) - 1  # Convert to 0-indexed
        max_line = max(changed_lines) - 1
        
        # Expand context
        start = max(0, min_line - self.CONTEXT_LINES)
        end = min(len(lines), max_line + self.CONTEXT_LINES + 1)
        
        # Try to find function start by looking backwards for def/function
        for i in range(min_line, max(0, min_line - 100), -1):
            line = lines[i].strip()
            if (line.startswith('def ') or 
                line.startswith('async def ') or
                'function ' in line or
                '=>' in line):
                start = i
                break
        
        return '\n'.join(lines[start:end])
    
    def parse_diff_line_numbers(self, diff_text: str) -> Dict[str, List[int]]:
        """
        Parse diff to extract changed line numbers.
        
        Args:
            diff_text: Git diff text
            
        Returns:
            Dict with 'removed' and 'added' line numbers
        """
        removed_lines = []
        added_lines = []
        
        current_old_line = 0
        current_new_line = 0
        
        for line in diff_text.split('\n'):
            # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
            if line.startswith('@@'):
                match = re.search(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
                if match:
                    current_old_line = int(match.group(1))
                    current_new_line = int(match.group(2))
                continue
            
            if line.startswith('-') and not line.startswith('---'):
                # Removed line
                removed_lines.append(current_old_line)
                current_old_line += 1
            elif line.startswith('+') and not line.startswith('+++'):
                # Added line
                added_lines.append(current_new_line)
                current_new_line += 1
            elif line.startswith(' '):
                # Context line
                current_old_line += 1
                current_new_line += 1
        
        return {
            'removed': removed_lines,
            'added': added_lines
        }
    
    def get_stats(self) -> Dict:
        """Get extraction statistics."""
        total = sum(self.stats.values())
        return {
            **self.stats,
            'total': total,
            'full_function_pct': f"{(self.stats['full_function']/total*100):.1f}%" if total > 0 else "0%",
            'smart_sliced_pct': f"{(self.stats['smart_sliced']/total*100):.1f}%" if total > 0 else "0%",
            'fallback_pct': f"{(self.stats['context_fallback']/total*100):.1f}%" if total > 0 else "0%"
        }
    
    def print_stats(self):
        """Print extraction statistics."""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("Function Extraction Statistics")
        print("="*60)
        print(f"Total extractions:   {stats['total']}")
        print(f"Full function:       {stats['full_function']} ({stats['full_function_pct']})")
        print(f"Smart sliced:        {stats['smart_sliced']} ({stats['smart_sliced_pct']})")
        print(f"Context fallback:    {stats['context_fallback']} ({stats['fallback_pct']})")
        print(f"Failed:              {stats['failed']}")
        print("="*60)


def test_extractor():
    """Test the function extractor."""
    # Sample Python code
    python_code = '''
def calculate_total(items):
    total = 0
    for item in items:
        # Vulnerable: SQL injection
        query = "SELECT * FROM products WHERE id=" + item.id
        result = db.execute(query)
        total += result.price
    return total

def safe_calculate_total(items):
    total = 0
    for item in items:
        # Fixed: Parameterized query
        query = "SELECT * FROM products WHERE id=?"
        result = db.execute(query, (item.id,))
        total += result.price
    return total
'''
    
    extractor = TreeSitterFunctionExtractor()
    
    # Test extraction
    changed_lines = [6]  # Line with SQL injection
    result = extractor.extract_function_body(python_code, changed_lines, 'python')
    
    print("Extracted function:")
    print(result)
    print("\nStats:")
    extractor.print_stats()


if __name__ == '__main__':
    test_extractor()
