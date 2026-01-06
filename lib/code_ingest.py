#!/usr/bin/env python3
"""
Code Repository Ingestion Pipeline

Ingests code repos into Postgres with:
- Citation-grade provenance (commit SHA, file path, line ranges)
- Semantic chunking (functions, methods, classes)
- Full-text search (tsvector)
- Optional embeddings (ChromaDB)

Usage:
    python3 -m lib.code_ingest mahmoodlab/UNI
    python3 -m lib.code_ingest --tier 1
    python3 -m lib.code_ingest --all --workers 4
"""
import ast
import hashlib
import logging
import os
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Config
REPOS_DIR = Path("/home/user/work/polymax/data/github_repos")
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB
SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'env',
             'build', 'dist', '.eggs', '*.egg-info', '.tox', '.pytest_cache',
             'vendor', 'third_party', '.next', 'coverage'}
SKIP_PATTERNS = {'min.js', 'bundle.js', '.min.css', 'package-lock.json',
                 'yarn.lock', 'poetry.lock', 'Pipfile.lock'}

# Language detection
LANG_MAP = {
    '.py': 'python',
    '.pyx': 'python',
    '.pyi': 'python',
    '.r': 'r',
    '.R': 'r',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.java': 'java',
    '.go': 'go',
    '.rs': 'rust',
    '.cpp': 'cpp',
    '.c': 'c',
    '.h': 'c',
    '.hpp': 'cpp',
    '.md': 'markdown',
    '.ipynb': 'notebook',
}

# Concept patterns for auto-tagging
CONCEPT_PATTERNS = {
    'attention': r'\b(attention|self[-_]?attention|cross[-_]?attention|mha|multihead)\b',
    'transformer': r'\b(transformer|bert|gpt|vit|encoder[-_]?decoder)\b',
    'contrastive': r'\b(contrastive|simclr|moco|clip|infonce|ntxent)\b',
    'gnn': r'\b(gnn|gcn|gat|message[-_]?passing|graph[-_]?conv)\b',
    'vae': r'\b(vae|variational|elbo|kl[-_]?divergence|reparameterization)\b',
    'spatial': r'\b(spatial|visium|10x|xenium|spot|tile|patch)\b',
    'pathology': r'\b(pathology|histology|wsi|whole[-_]?slide|h&?e)\b',
    'single_cell': r'\b(single[-_]?cell|scrna|scatac|anndata|scanpy)\b',
    'mil': r'\b(mil|multiple[-_]?instance|bag[-_]?of|aggregat)\b',
    'unet': r'\b(unet|u[-_]?net|encoder[-_]?decoder|skip[-_]?connection)\b',
    'resnet': r'\b(resnet|residual|skip[-_]?connection|bottleneck)\b',
    'loss_function': r'\b(loss|criterion|objective|bce|mse|cross[-_]?entropy)\b',
    'optimizer': r'\b(optim|adam|sgd|learning[-_]?rate|lr[-_]?scheduler)\b',
    'data_loader': r'\b(dataloader|dataset|batch|collate|sampler)\b',
    'augmentation': r'\b(augment|transform|flip|rotate|crop|normalize)\b',
}


@dataclass
class CodeChunk:
    """A semantic code unit."""
    chunk_type: str  # 'function', 'method', 'class', 'module'
    name: str
    class_name: Optional[str]
    symbol_qualified_name: str
    start_line: int
    end_line: int
    content: str
    docstring: Optional[str]
    signature: Optional[str]
    imports: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)


@dataclass
class RepoInfo:
    """Repository metadata."""
    name: str
    path: Path
    url: Optional[str]
    branch: str
    commit_sha: str


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        dbname='polymath',
        user='polymath',
        host='/var/run/postgresql'
    )


def get_repo_info(repo_path: Path) -> RepoInfo:
    """Extract git metadata from repo."""
    name = repo_path.name
    url = None
    branch = 'main'
    commit_sha = 'unknown'

    git_dir = repo_path / '.git'
    if git_dir.exists():
        try:
            # Get remote URL
            result = subprocess.run(
                ['git', '-C', str(repo_path), 'remote', 'get-url', 'origin'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                url = result.stdout.strip()
                # Extract org/repo from URL
                match = re.search(r'github\.com[:/](.+?)(?:\.git)?$', url)
                if match:
                    name = match.group(1)

            # Get current branch
            result = subprocess.run(
                ['git', '-C', str(repo_path), 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                branch = result.stdout.strip()

            # Get HEAD commit SHA
            result = subprocess.run(
                ['git', '-C', str(repo_path), 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                commit_sha = result.stdout.strip()

        except Exception as e:
            logger.warning(f"Git metadata extraction failed for {repo_path}: {e}")

    return RepoInfo(name=name, path=repo_path, url=url, branch=branch, commit_sha=commit_sha)


def should_skip_file(file_path: Path, repo_root: Path) -> bool:
    """Check if file should be skipped."""
    rel_path = str(file_path.relative_to(repo_root))

    # Skip directories
    for part in file_path.parts:
        if part in SKIP_DIRS or part.endswith('.egg-info'):
            return True

    # Skip patterns
    for pattern in SKIP_PATTERNS:
        if pattern in file_path.name:
            return True

    # Skip large files
    try:
        if file_path.stat().st_size > MAX_FILE_SIZE:
            return True
    except:
        return True

    # Skip generated files
    if 'generated' in rel_path.lower() or 'auto_' in file_path.name.lower():
        return True

    return False


def detect_language(file_path: Path) -> Optional[str]:
    """Detect programming language from file extension."""
    return LANG_MAP.get(file_path.suffix.lower())


def compute_file_hash(content: str) -> str:
    """Compute SHA256 hash of file content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def compute_chunk_id(repo_name: str, file_path: str, start_line: int, end_line: int, chunk_hash: str) -> str:
    """Compute deterministic chunk ID."""
    key = f"{repo_name}:{file_path}:{start_line}:{end_line}:{chunk_hash}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def extract_concepts(content: str) -> List[str]:
    """Auto-tag concepts from code content."""
    concepts = []
    content_lower = content.lower()
    for concept, pattern in CONCEPT_PATTERNS.items():
        if re.search(pattern, content_lower, re.IGNORECASE):
            concepts.append(concept)
    return concepts


def parse_python_file(content: str, file_path: str, module_name: str) -> List[CodeChunk]:
    """Parse Python file into semantic chunks using AST."""
    chunks = []
    lines = content.split('\n')

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        logger.debug(f"Syntax error in {file_path}: {e}")
        # Fallback: treat whole file as one chunk
        return [CodeChunk(
            chunk_type='module',
            name=module_name,
            class_name=None,
            symbol_qualified_name=module_name,
            start_line=1,
            end_line=len(lines),
            content=content[:5000],  # Limit size
            docstring=ast.get_docstring(ast.parse("")) if content.startswith('"""') else None,
            signature=None,
            imports=[],
            concepts=extract_concepts(content)
        )]

    # Extract imports
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    # Module docstring
    module_doc = ast.get_docstring(tree)
    if module_doc:
        chunks.append(CodeChunk(
            chunk_type='module',
            name=module_name,
            class_name=None,
            symbol_qualified_name=module_name,
            start_line=1,
            end_line=min(20, len(lines)),  # Just the header
            content=module_doc[:2000],
            docstring=module_doc[:1000],
            signature=None,
            imports=imports[:20],
            concepts=extract_concepts(module_doc)
        ))

    # Process top-level definitions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Standalone function
            chunk = _extract_function_chunk(node, lines, module_name, None, imports)
            if chunk:
                chunks.append(chunk)

        elif isinstance(node, ast.ClassDef):
            # Class definition
            class_name = node.name
            class_start = node.lineno
            class_end = node.end_lineno or class_start

            # Class summary chunk
            class_doc = ast.get_docstring(node)
            class_content = '\n'.join(lines[class_start-1:min(class_start+10, class_end)])

            chunks.append(CodeChunk(
                chunk_type='class',
                name=class_name,
                class_name=None,
                symbol_qualified_name=f"{module_name}.{class_name}",
                start_line=class_start,
                end_line=min(class_start + 20, class_end),  # Just class header
                content=class_content,
                docstring=class_doc[:1000] if class_doc else None,
                signature=f"class {class_name}",
                imports=imports[:10],
                concepts=extract_concepts(class_content + (class_doc or ''))
            ))

            # Each method as separate chunk
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunk = _extract_function_chunk(item, lines, module_name, class_name, [])
                    if chunk:
                        chunks.append(chunk)

    return chunks


def _extract_function_chunk(node, lines: List[str], module_name: str,
                           class_name: Optional[str], imports: List[str]) -> Optional[CodeChunk]:
    """Extract a function/method as a chunk."""
    try:
        start = node.lineno
        end = node.end_lineno or start

        # Skip tiny functions
        if end - start < 2:
            return None

        # Build signature
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                try:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                except:
                    pass
            args.append(arg_str)

        returns = ""
        if node.returns:
            try:
                returns = f" -> {ast.unparse(node.returns)}"
            except:
                pass

        is_async = isinstance(node, ast.AsyncFunctionDef)
        prefix = "async def" if is_async else "def"
        signature = f"{prefix} {node.name}({', '.join(args)}){returns}"

        # Get content
        content = '\n'.join(lines[start-1:end])
        docstring = ast.get_docstring(node)

        # Build qualified name
        if class_name:
            qualified = f"{module_name}.{class_name}.{node.name}"
            chunk_type = 'method'
        else:
            qualified = f"{module_name}.{node.name}"
            chunk_type = 'function'

        return CodeChunk(
            chunk_type=chunk_type,
            name=node.name,
            class_name=class_name,
            symbol_qualified_name=qualified,
            start_line=start,
            end_line=end,
            content=content[:5000],  # Limit
            docstring=docstring[:1000] if docstring else None,
            signature=signature,
            imports=imports[:10],
            concepts=extract_concepts(content + (docstring or ''))
        )
    except Exception as e:
        logger.debug(f"Error extracting function {node.name}: {e}")
        return None


def scan_repo(repo_path: Path) -> Tuple[RepoInfo, List[Tuple[Path, str, str]]]:
    """Scan repository for code files.

    Returns:
        Tuple of (repo_info, list of (file_path, language, content))
    """
    repo_info = get_repo_info(repo_path)
    files = []

    for file_path in repo_path.rglob('*'):
        if not file_path.is_file():
            continue

        if should_skip_file(file_path, repo_path):
            continue

        lang = detect_language(file_path)
        if not lang:
            continue

        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            if len(content.strip()) < 50:  # Skip near-empty files
                continue
            files.append((file_path, lang, content))
        except Exception as e:
            logger.debug(f"Error reading {file_path}: {e}")
            continue

    return repo_info, files


def ingest_repo(repo_path: Path, conn=None) -> Dict:
    """Ingest a single repository.

    Returns:
        Stats dict with files_processed, chunks_created, etc.
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    stats = {'files': 0, 'chunks': 0, 'errors': 0}

    try:
        repo_info, files = scan_repo(repo_path)
        logger.info(f"Scanning {repo_info.name}: {len(files)} files found")

        cur = conn.cursor()

        for file_path, lang, content in files:
            try:
                rel_path = str(file_path.relative_to(repo_path))
                file_hash = compute_file_hash(content)

                # Check if already ingested (same commit + file)
                cur.execute("""
                    SELECT file_id FROM code_files
                    WHERE repo_name = %s AND file_path = %s AND head_commit_sha = %s
                """, (repo_info.name, rel_path, repo_info.commit_sha))

                if cur.fetchone():
                    continue  # Already ingested

                # Insert file
                file_id = str(uuid.uuid4())
                cur.execute("""
                    INSERT INTO code_files
                    (file_id, repo_name, repo_url, repo_root, default_branch, head_commit_sha,
                     file_path, language, file_hash, file_size_bytes, loc, is_generated)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (repo_name, file_path, head_commit_sha) DO NOTHING
                    RETURNING file_id
                """, (
                    file_id, repo_info.name, repo_info.url, str(repo_path),
                    repo_info.branch, repo_info.commit_sha, rel_path, lang,
                    file_hash, len(content), content.count('\n'), False
                ))

                result = cur.fetchone()
                if not result:
                    continue
                file_id = result[0]
                stats['files'] += 1

                # Parse and chunk based on language
                chunks = []
                if lang == 'python':
                    module_name = rel_path.replace('/', '.').replace('.py', '')
                    chunks = parse_python_file(content, rel_path, module_name)
                elif lang == 'markdown':
                    # Simple markdown chunking
                    chunks = [CodeChunk(
                        chunk_type='module',
                        name=file_path.stem,
                        class_name=None,
                        symbol_qualified_name=rel_path,
                        start_line=1,
                        end_line=content.count('\n'),
                        content=content[:5000],
                        docstring=content[:1000],
                        signature=None,
                        imports=[],
                        concepts=extract_concepts(content)
                    )]
                else:
                    # Generic chunking for other languages
                    chunks = [CodeChunk(
                        chunk_type='module',
                        name=file_path.stem,
                        class_name=None,
                        symbol_qualified_name=rel_path,
                        start_line=1,
                        end_line=content.count('\n'),
                        content=content[:5000],
                        docstring=None,
                        signature=None,
                        imports=[],
                        concepts=extract_concepts(content)
                    )]

                # Insert chunks
                for chunk in chunks:
                    chunk_hash = compute_file_hash(chunk.content)
                    embedding_id = compute_chunk_id(
                        repo_info.name, rel_path,
                        chunk.start_line, chunk.end_line, chunk_hash
                    )

                    cur.execute("""
                        INSERT INTO code_chunks
                        (chunk_id, file_id, chunk_type, name, class_name, symbol_qualified_name,
                         start_line, end_line, content, chunk_hash, docstring, signature,
                         imports, concepts, embedding_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (
                        str(uuid.uuid4()), file_id, chunk.chunk_type, chunk.name,
                        chunk.class_name, chunk.symbol_qualified_name,
                        chunk.start_line, chunk.end_line, chunk.content,
                        chunk_hash, chunk.docstring, chunk.signature,
                        chunk.imports, chunk.concepts, embedding_id
                    ))
                    stats['chunks'] += 1

            except Exception as e:
                logger.debug(f"Error processing {file_path}: {e}")
                stats['errors'] += 1

        conn.commit()
        logger.info(f"✓ {repo_info.name}: {stats['files']} files, {stats['chunks']} chunks")

    except Exception as e:
        logger.error(f"Error ingesting {repo_path}: {e}")
        conn.rollback()
        stats['errors'] += 1

    finally:
        if close_conn:
            conn.close()

    return stats


def ingest_all_repos(workers: int = 4, tier: Optional[int] = None):
    """Ingest all repositories.

    Args:
        workers: Number of parallel workers
        tier: Optional tier filter (1-4)
    """
    # Define tiers
    tier1 = {'UNI', 'CONCH', 'CLAM', 'HIPT', 'squidpy', 'cell2location', 'scvi-tools',
             'spatial_CRC_atlas', 'dropkick', 'pCreode', 'PathomicFusion', 'MCAT'}
    tier2 = {'dspy', 'pytorch_geometric', 'transformers', 'scikit-learn', 'deepchem'}
    tier3 = {'giotto-tda', 'pymdp', 'papers-we-love'}

    repos = list(REPOS_DIR.iterdir())
    repos = [r for r in repos if r.is_dir()]

    if tier == 1:
        repos = [r for r in repos if r.name in tier1]
    elif tier == 2:
        repos = [r for r in repos if r.name in tier2]
    elif tier == 3:
        repos = [r for r in repos if r.name in tier3]
    elif tier == 4:
        all_tiers = tier1 | tier2 | tier3
        repos = [r for r in repos if r.name not in all_tiers]

    # Sort by priority (tier1 first)
    def priority(r):
        if r.name in tier1: return 0
        if r.name in tier2: return 1
        if r.name in tier3: return 2
        return 3
    repos.sort(key=priority)

    logger.info(f"Ingesting {len(repos)} repositories with {workers} workers")

    total_stats = {'files': 0, 'chunks': 0, 'errors': 0, 'repos': 0}

    if workers == 1:
        for repo in repos:
            stats = ingest_repo(repo)
            total_stats['files'] += stats['files']
            total_stats['chunks'] += stats['chunks']
            total_stats['errors'] += stats['errors']
            total_stats['repos'] += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(ingest_repo, r): r for r in repos}
            for future in as_completed(futures):
                repo = futures[future]
                try:
                    stats = future.result()
                    total_stats['files'] += stats['files']
                    total_stats['chunks'] += stats['chunks']
                    total_stats['errors'] += stats['errors']
                    total_stats['repos'] += 1
                except Exception as e:
                    logger.error(f"Error ingesting {repo}: {e}")
                    total_stats['errors'] += 1

    logger.info(f"\n=== COMPLETE ===")
    logger.info(f"Repos: {total_stats['repos']}")
    logger.info(f"Files: {total_stats['files']}")
    logger.info(f"Chunks: {total_stats['chunks']}")
    logger.info(f"Errors: {total_stats['errors']}")

    return total_stats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Ingest code repositories')
    parser.add_argument('repo', nargs='?', help='Specific repo to ingest')
    parser.add_argument('--all', action='store_true', help='Ingest all repos')
    parser.add_argument('--tier', type=int, choices=[1,2,3,4], help='Ingest specific tier')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    args = parser.parse_args()

    if args.repo:
        repo_path = REPOS_DIR / args.repo
        if not repo_path.exists():
            # Try as full path
            repo_path = Path(args.repo)
        if repo_path.exists():
            ingest_repo(repo_path)
        else:
            print(f"Repo not found: {args.repo}")
    elif args.all or args.tier:
        ingest_all_repos(workers=args.workers, tier=args.tier)
    else:
        parser.print_help()
