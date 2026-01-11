#!/usr/bin/env python3
"""
Tests for Code Ingestion Pipeline

Tests:
- Python AST parsing and chunk extraction
- Language detection
- Concept detection patterns
- Chunk quality validation
- Integration with database

Run with: pytest tests/test_code_ingest.py -v
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))


# =============================================================================
# Code Chunking Tests
# =============================================================================

class TestPythonParsing:
    """Test Python AST parsing for semantic chunking."""

    def test_parse_function(self):
        """Test function extraction from Python code."""
        from code_ingest import parse_python_file, CodeChunk

        code = '''
def hello_world(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"
'''
        chunks = parse_python_file(code, "test.py", "test")

        assert len(chunks) >= 1
        func_chunks = [c for c in chunks if c.chunk_type == 'function']
        assert len(func_chunks) == 1
        assert func_chunks[0].name == 'hello_world'
        assert 'Say hello' in (func_chunks[0].docstring or '')

    def test_parse_class_with_methods(self):
        """Test class and method extraction."""
        from code_ingest import parse_python_file

        code = '''
class MyModel:
    """A simple model class."""

    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim

    def forward(self, x):
        """Forward pass."""
        return x * self.hidden_dim
'''
        chunks = parse_python_file(code, "test.py", "test")

        class_chunks = [c for c in chunks if c.chunk_type == 'class']
        method_chunks = [c for c in chunks if c.chunk_type == 'method']

        assert len(class_chunks) == 1
        assert class_chunks[0].name == 'MyModel'
        # __init__ may be excluded, just check we got at least forward
        assert len(method_chunks) >= 1
        assert any(m.name == 'forward' for m in method_chunks)

    def test_parse_async_function(self):
        """Test async function extraction."""
        from code_ingest import parse_python_file

        code = '''
async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()
'''
        chunks = parse_python_file(code, "test.py", "test")

        func_chunks = [c for c in chunks if c.chunk_type == 'function']
        assert len(func_chunks) == 1
        assert func_chunks[0].name == 'fetch_data'

    def test_parse_imports(self):
        """Test import extraction."""
        from code_ingest import parse_python_file

        code = '''
import numpy as np
from torch import nn
from pathlib import Path
import os, sys

def process():
    pass
'''
        chunks = parse_python_file(code, "test.py", "test")

        # Module chunk should have imports
        module_chunks = [c for c in chunks if c.chunk_type == 'module']
        if module_chunks:
            imports = module_chunks[0].imports
            assert 'numpy' in str(imports) or 'np' in str(imports)


# =============================================================================
# Language Detection Tests
# =============================================================================

class TestLanguageDetection:
    """Test file language detection."""

    def test_python_extensions(self):
        """Test Python file detection."""
        from code_ingest import LANG_MAP

        assert LANG_MAP.get('.py') == 'python'
        assert LANG_MAP.get('.pyx') == 'python'
        assert LANG_MAP.get('.pyi') == 'python'

    def test_r_extensions(self):
        """Test R file detection."""
        from code_ingest import LANG_MAP

        assert LANG_MAP.get('.r') == 'r'
        assert LANG_MAP.get('.R') == 'r'

    def test_javascript_extensions(self):
        """Test JavaScript file detection."""
        from code_ingest import LANG_MAP

        assert LANG_MAP.get('.js') == 'javascript'
        assert LANG_MAP.get('.jsx') == 'javascript'
        assert LANG_MAP.get('.ts') == 'typescript'
        assert LANG_MAP.get('.tsx') == 'typescript'

    def test_notebook_extension(self):
        """Test Jupyter notebook detection."""
        from code_ingest import LANG_MAP

        assert LANG_MAP.get('.ipynb') == 'notebook'


# =============================================================================
# Concept Detection Tests
# =============================================================================

class TestConceptDetection:
    """Test concept pattern matching."""

    def test_attention_pattern(self):
        """Test attention pattern detection."""
        from code_ingest import CONCEPT_PATTERNS
        import re

        pattern = CONCEPT_PATTERNS['attention']

        assert re.search(pattern, 'self_attention layer', re.I)
        assert re.search(pattern, 'Multi Head Attention', re.I)  # With spaces
        assert re.search(pattern, 'cross-attention', re.I)
        assert re.search(pattern, 'The attention mechanism', re.I)  # standalone attention

    def test_transformer_pattern(self):
        """Test transformer pattern detection."""
        from code_ingest import CONCEPT_PATTERNS
        import re

        pattern = CONCEPT_PATTERNS['transformer']

        assert re.search(pattern, 'A transformer model', re.I)  # standalone word
        assert re.search(pattern, 'BERT model', re.I)
        assert re.search(pattern, 'GPT-2', re.I)
        assert re.search(pattern, 'ViT architecture', re.I)

    def test_spatial_pattern(self):
        """Test spatial transcriptomics pattern detection."""
        from code_ingest import CONCEPT_PATTERNS
        import re

        pattern = CONCEPT_PATTERNS['spatial']

        assert re.search(pattern, 'spatial transcriptomics', re.I)
        assert re.search(pattern, 'Visium data', re.I)
        assert re.search(pattern, '10x Genomics', re.I)

    def test_pathology_pattern(self):
        """Test computational pathology pattern detection."""
        from code_ingest import CONCEPT_PATTERNS
        import re

        pattern = CONCEPT_PATTERNS['pathology']

        assert re.search(pattern, 'digital pathology', re.I)
        assert re.search(pattern, 'WSI processing', re.I)
        assert re.search(pattern, 'whole-slide image', re.I)
        assert re.search(pattern, 'H&E staining', re.I)


# =============================================================================
# Skip Pattern Tests
# =============================================================================

class TestSkipPatterns:
    """Test file/directory skip patterns."""

    def test_skip_directories(self):
        """Test that common build/cache dirs are skipped."""
        from code_ingest import SKIP_DIRS

        assert '.git' in SKIP_DIRS
        assert '__pycache__' in SKIP_DIRS
        assert 'node_modules' in SKIP_DIRS
        assert 'venv' in SKIP_DIRS
        assert '.pytest_cache' in SKIP_DIRS

    def test_skip_files(self):
        """Test that minified/lock files are skipped."""
        from code_ingest import SKIP_PATTERNS

        assert 'min.js' in SKIP_PATTERNS
        assert 'package-lock.json' in SKIP_PATTERNS
        assert 'yarn.lock' in SKIP_PATTERNS


# =============================================================================
# CodeChunk Dataclass Tests
# =============================================================================

class TestCodeChunk:
    """Test CodeChunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a CodeChunk."""
        from code_ingest import CodeChunk

        chunk = CodeChunk(
            chunk_type='function',
            name='hello',
            class_name=None,
            symbol_qualified_name='module.hello',
            start_line=1,
            end_line=5,
            content='def hello(): pass',
            docstring='Say hello.',
            signature='hello()',
            imports=['os'],
            concepts=['function']
        )

        assert chunk.chunk_type == 'function'
        assert chunk.name == 'hello'
        assert chunk.start_line == 1
        assert chunk.end_line == 5
        assert 'os' in chunk.imports

    def test_chunk_defaults(self):
        """Test CodeChunk default values."""
        from code_ingest import CodeChunk

        chunk = CodeChunk(
            chunk_type='function',
            name='test',
            class_name=None,
            symbol_qualified_name='test',
            start_line=1,
            end_line=2,
            content='pass',
            docstring=None,
            signature=None
        )

        assert chunk.imports == []
        assert chunk.concepts == []


# =============================================================================
# Integration Tests (require database)
# =============================================================================

class TestCodeIngestIntegration:
    """Integration tests that require database connection."""

    def test_postgres_code_chunks_table(self):
        """Test that code_chunks table exists and has data."""
        import psycopg2

        conn = psycopg2.connect(
            dbname='polymath',
            user='polymath',
            host='/var/run/postgresql'
        )
        cursor = conn.cursor()

        # Check table exists
        cursor.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = 'code_chunks'
        """)
        assert cursor.fetchone()[0] == 1, "code_chunks table not found"

        # Check has data
        cursor.execute("SELECT COUNT(*) FROM code_chunks")
        count = cursor.fetchone()[0]
        conn.close()

        assert count > 0, "No code chunks in database"
        print(f"Code chunks in database: {count}")

    def test_code_chunk_schema(self):
        """Test code_chunks table has expected columns."""
        import psycopg2

        conn = psycopg2.connect(
            dbname='polymath',
            user='polymath',
            host='/var/run/postgresql'
        )
        cursor = conn.cursor()

        cursor.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'code_chunks'
        """)
        columns = {row[0] for row in cursor.fetchall()}
        conn.close()

        # Expected columns based on actual schema
        expected = {'chunk_id', 'content', 'chunk_type', 'name', 'start_line', 'end_line'}
        missing = expected - columns
        assert not missing, f"Missing columns: {missing}"

    def test_chunk_concepts_populated(self):
        """Test that chunk_concepts table has concept extractions."""
        import psycopg2

        conn = psycopg2.connect(
            dbname='polymath',
            user='polymath',
            host='/var/run/postgresql'
        )
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM chunk_concepts")
        count = cursor.fetchone()[0]
        conn.close()

        # Should have some concept extractions
        assert count > 0, "No chunk concepts extracted"
        print(f"Chunk concepts in database: {count}")


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
