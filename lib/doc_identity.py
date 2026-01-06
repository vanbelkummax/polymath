#!/usr/bin/env python3
"""
Document Identity Management - Deterministic UUIDv5 System

CRITICAL: Identity is COMPUTED, not discovered. DB lookups are just caching for upsert.

Version: HARDENING_2026-01-05
"""
import uuid
import hashlib
import re
from typing import Optional, List
import psycopg2
import logging

logger = logging.getLogger(__name__)

# Standard DNS namespace for deterministic UUIDs
NAMESPACE_POLYMATH = uuid.NAMESPACE_DNS


def normalize_title(title: str) -> str:
    """
    Normalize title for deduplication.

    Rules:
    - Lowercase
    - Strip punctuation except spaces
    - Collapse whitespace
    """
    # Lowercase
    t = title.lower()
    # Strip punctuation except spaces
    t = re.sub(r'[^\w\s]', '', t)
    # Collapse whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def compute_doc_id(
    title: str,
    authors: List[str],
    year: int,
    doi: Optional[str] = None,
    pmid: Optional[str] = None,
    arxiv_id: Optional[str] = None
) -> uuid.UUID:
    """
    Compute deterministic doc_id using UUIDv5.

    IDENTITY RULES (deterministic):
    1. DOI (if provided) → uuid5(namespace, "doi:"+doi_lower)
    2. PMID (if provided) → uuid5(namespace, "pmid:"+pmid)
    3. arXiv (if provided) → uuid5(namespace, "arxiv:"+base_id) [v1/v2 normalized]
    4. Title hash fallback → uuid5(namespace, "th:"+sha256[:16]+"|y:"+year+"|a:"+author1)

    Same inputs → same UUID. ALWAYS.
    DB lookups happen AFTER, for upsert.

    Args:
        title: Paper title
        authors: List of author names
        year: Publication year
        doi: DOI if known
        pmid: PubMed ID if known
        arxiv_id: arXiv ID if known

    Returns:
        Deterministic UUID for this document
    """
    # Priority 1: DOI
    if doi:
        canonical = f"doi:{doi.lower().strip()}"
        return uuid.uuid5(NAMESPACE_POLYMATH, canonical)

    # Priority 2: PMID
    if pmid:
        canonical = f"pmid:{pmid.strip()}"
        return uuid.uuid5(NAMESPACE_POLYMATH, canonical)

    # Priority 3: arXiv (normalize v1/v2 → same base)
    if arxiv_id:
        arxiv_base = arxiv_id.split('v')[0]  # "2301.12345v2" → "2301.12345"
        canonical = f"arxiv:{arxiv_base}"
        return uuid.uuid5(NAMESPACE_POLYMATH, canonical)

    # Priority 4: Title hash fallback
    title_norm = normalize_title(title)
    author_norm = authors[0].lower().strip() if authors else "unknown"
    title_hash = hashlib.sha256(title_norm.encode()).hexdigest()[:16]
    canonical = f"th:{title_hash}|y:{year}|a:{author_norm}"
    return uuid.uuid5(NAMESPACE_POLYMATH, canonical)


def upsert_document(
    doc_id: uuid.UUID,
    title: str,
    authors: List[str],
    year: int,
    doi: Optional[str] = None,
    pmid: Optional[str] = None,
    arxiv_id: Optional[str] = None,
    venue: Optional[str] = None,
    publication_type: Optional[str] = None,
    parser_version: Optional[str] = None,
    db_conn=None
) -> uuid.UUID:
    """
    Upsert document using COMPUTED doc_id.

    Stores aliases for all identifiers (doi, pmid, arxiv, title_hash).

    Args:
        doc_id: Pre-computed doc_id from compute_doc_id()
        title: Paper title
        authors: List of author names
        year: Publication year
        doi: DOI if known
        pmid: PubMed ID if known
        arxiv_id: arXiv ID if known
        venue: Publication venue
        publication_type: Type of publication ('primary', 'review', etc.)
        parser_version: Version of parser used
        db_conn: Database connection (creates new if None)

    Returns:
        The doc_id (same as input, for chaining)
    """
    # Database connection
    close_conn = False
    if db_conn is None:
        db_conn = psycopg2.connect("dbname=polymath user=polymath host=/var/run/postgresql")
        close_conn = True

    cursor = db_conn.cursor()

    try:
        # Check if exists
        cursor.execute("SELECT doc_id FROM documents WHERE doc_id = %s", (str(doc_id),))
        existing = cursor.fetchone()

        # Compute title_hash for deduplication
        title_hash = hashlib.sha256(normalize_title(title).encode()).hexdigest()[:16]

        if existing:
            # Update metadata (year/venue may improve over time)
            cursor.execute("""
                UPDATE documents
                SET doi = COALESCE(%s, doi),
                    pmid = COALESCE(%s, pmid),
                    arxiv_id = COALESCE(%s, arxiv_id),
                    year = COALESCE(%s, year),
                    venue = COALESCE(%s, venue),
                    publication_type = COALESCE(%s, publication_type),
                    parser_version = COALESCE(%s, parser_version),
                    title_hash = COALESCE(%s, title_hash)
                WHERE doc_id = %s
            """, (doi, pmid, arxiv_id, year, venue, publication_type, parser_version,
                  title_hash, str(doc_id)))
            logger.info(f"Updated document {doc_id}")
        else:
            # Insert new document
            cursor.execute("""
                INSERT INTO documents (doc_id, doi, pmid, arxiv_id, title, authors, year,
                                      venue, publication_type, parser_version, title_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (str(doc_id), doi, pmid, arxiv_id, title, authors, year, venue,
                  publication_type, parser_version, title_hash))
            logger.info(f"Inserted document {doc_id}")

        # Store aliases
        if doi:
            cursor.execute("""
                INSERT INTO doc_aliases (doc_id, alias_type, alias_value)
                VALUES (%s, 'doi', %s)
                ON CONFLICT (alias_type, alias_value) DO NOTHING
            """, (str(doc_id), doi))

        if pmid:
            cursor.execute("""
                INSERT INTO doc_aliases (doc_id, alias_type, alias_value)
                VALUES (%s, 'pmid', %s)
                ON CONFLICT (alias_type, alias_value) DO NOTHING
            """, (str(doc_id), pmid))

        if arxiv_id:
            arxiv_base = arxiv_id.split('v')[0]
            cursor.execute("""
                INSERT INTO doc_aliases (doc_id, alias_type, alias_value)
                VALUES (%s, 'arxiv', %s)
                ON CONFLICT (alias_type, alias_value) DO NOTHING
            """, (str(doc_id), arxiv_base))

        db_conn.commit()
        return doc_id

    except Exception as e:
        db_conn.rollback()
        logger.error(f"Error upserting document {doc_id}: {e}")
        raise

    finally:
        cursor.close()
        if close_conn:
            db_conn.close()


def get_or_create_doc_id(
    title: str,
    authors: List[str],
    year: int,
    doi: Optional[str] = None,
    pmid: Optional[str] = None,
    arxiv_id: Optional[str] = None,
    **kwargs
) -> uuid.UUID:
    """
    Convenience function: compute doc_id and upsert in one call.

    This is the primary entry point for most ingestion code.

    Args:
        title: Paper title
        authors: List of author names
        year: Publication year
        doi: DOI if known
        pmid: PubMed ID if known
        arxiv_id: arXiv ID if known
        **kwargs: Additional fields passed to upsert_document

    Returns:
        Deterministic doc_id
    """
    doc_id = compute_doc_id(title, authors, year, doi, pmid, arxiv_id)
    upsert_document(doc_id, title, authors, year, doi, pmid, arxiv_id, **kwargs)
    return doc_id


def lookup_doc_by_identifier(
    doi: Optional[str] = None,
    pmid: Optional[str] = None,
    arxiv_id: Optional[str] = None,
    db_conn=None
) -> Optional[uuid.UUID]:
    """
    Look up doc_id by any known identifier.

    This is useful for finding existing documents without full metadata.

    Args:
        doi: DOI to search for
        pmid: PMID to search for
        arxiv_id: arXiv ID to search for
        db_conn: Database connection

    Returns:
        doc_id if found, None otherwise
    """
    close_conn = False
    if db_conn is None:
        db_conn = psycopg2.connect("dbname=polymath user=polymath host=/var/run/postgresql")
        close_conn = True

    cursor = db_conn.cursor()

    try:
        # Try DOI first
        if doi:
            cursor.execute("""
                SELECT doc_id FROM doc_aliases
                WHERE alias_type = 'doi' AND alias_value = %s
            """, (doi,))
            result = cursor.fetchone()
            if result:
                return uuid.UUID(result[0])

        # Try PMID
        if pmid:
            cursor.execute("""
                SELECT doc_id FROM doc_aliases
                WHERE alias_type = 'pmid' AND alias_value = %s
            """, (pmid,))
            result = cursor.fetchone()
            if result:
                return uuid.UUID(result[0])

        # Try arXiv (normalize first)
        if arxiv_id:
            arxiv_base = arxiv_id.split('v')[0]
            cursor.execute("""
                SELECT doc_id FROM doc_aliases
                WHERE alias_type = 'arxiv' AND alias_value = %s
            """, (arxiv_base,))
            result = cursor.fetchone()
            if result:
                return uuid.UUID(result[0])

        return None

    finally:
        cursor.close()
        if close_conn:
            db_conn.close()


if __name__ == "__main__":
    # Test determinism
    doc_id_1 = compute_doc_id(
        title="GraphST: Spatially informed graph attention network",
        authors=["Long, Yahui", "Ang, Kok Siong"],
        year=2023,
        doi="10.1038/s41467-023-36796-3"
    )

    doc_id_2 = compute_doc_id(
        title="GraphST: Spatially informed graph attention network",
        authors=["Long, Yahui", "Ang, Kok Siong"],
        year=2023,
        doi="10.1038/s41467-023-36796-3"
    )

    print(f"doc_id_1: {doc_id_1}")
    print(f"doc_id_2: {doc_id_2}")
    print(f"Deterministic: {doc_id_1 == doc_id_2}")

    # Test arXiv normalization
    arxiv_v1 = compute_doc_id(
        title="Test Paper",
        authors=["Smith, John"],
        year=2024,
        arxiv_id="2401.12345v1"
    )

    arxiv_v2 = compute_doc_id(
        title="Test Paper",
        authors=["Smith, John"],
        year=2024,
        arxiv_id="2401.12345v2"
    )

    print(f"\narXiv v1: {arxiv_v1}")
    print(f"arXiv v2: {arxiv_v2}")
    print(f"arXiv normalization works: {arxiv_v1 == arxiv_v2}")
