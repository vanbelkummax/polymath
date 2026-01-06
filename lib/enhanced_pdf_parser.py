#!/usr/bin/env python3
"""
Enhanced PDF Parser with Page Boundaries and Evidence Provenance

Features:
- Page-local character offsets (not global)
- Infinite loop guardrail in chunking
- True sliding window overlap
- Section classification
- Quality scoring
- Deterministic passage_id generation

Version: HARDENING_2026-01-05
"""
import uuid
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Namespace for deterministic UUIDs
NAMESPACE_POLYMATH = uuid.NAMESPACE_DNS


@dataclass
class Passage:
    """A passage of text with provenance metadata."""
    passage_id: uuid.UUID
    doc_id: uuid.UUID
    page_num: int
    page_char_start: int          # Page-local offset
    page_char_end: int            # Page-local offset
    section: str
    passage_text: str
    quality_score: float
    parser_version: Optional[str] = None


class EnhancedPDFParser:
    """
    Parse PDFs with page/char boundary preservation.

    CRITICAL: Coordinates are PAGE-LOCAL, not global.
    This prevents overlap bugs and enables reliable passage lookup.
    """

    def __init__(self):
        self.parser_method = 'unknown'

    def extract_with_provenance(self, pdf_path: str, doc_id: uuid.UUID) -> List[Passage]:
        """
        Extract text preserving page-local character offsets.

        CRITICAL: Coordinates are PAGE-LOCAL, not global.
        This prevents overlap bugs and enables reliable passage lookup.

        Args:
            pdf_path: Path to PDF file
            doc_id: Deterministic doc_id from doc_identity.compute_doc_id()

        Returns:
            List of Passage objects with provenance metadata
        """
        passages = []
        parser_method = 'unknown'

        try:
            # Primary: pdfplumber (best for coordinates)
            import pdfplumber
            passages = self._extract_pdfplumber(pdf_path, doc_id)
            parser_method = 'pdfplumber_v1'
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, trying PyMuPDF")
            try:
                # Fallback: PyMuPDF
                import fitz
                passages = self._extract_pymupdf(pdf_path, doc_id)
                parser_method = 'pymupdf_fallback'
            except Exception as e2:
                logger.error(f"PyMuPDF failed: {e2}, trying OCR")
                try:
                    # Last resort: OCR (pytesseract)
                    passages = self._extract_ocr(pdf_path, doc_id)
                    parser_method = 'ocr_tesseract'
                except Exception as e3:
                    logger.error(f"All parsers failed: {e3}")
                    raise RuntimeError(f"Failed to parse PDF {pdf_path}: all methods exhausted")

        # Tag parser version
        for p in passages:
            p.parser_version = parser_method

        logger.info(f"Extracted {len(passages)} passages from {Path(pdf_path).name} using {parser_method}")
        return passages

    def _extract_pdfplumber(self, pdf_path: str, doc_id: uuid.UUID) -> List[Passage]:
        """Primary extraction with pdfplumber."""
        import pdfplumber

        passages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue  # Skip empty pages

                # Classify section
                section = self._classify_section(page_text, page_num, len(pdf.pages))

                # Chunk with overlap, preserving PAGE-LOCAL coordinates
                page_passages = self._chunk_page(
                    page_text=page_text,
                    doc_id=doc_id,
                    page_num=page_num,
                    section=section,
                    chunk_size=512,
                    overlap=50
                )
                passages.extend(page_passages)

        return passages

    def _extract_pymupdf(self, pdf_path: str, doc_id: uuid.UUID) -> List[Passage]:
        """Fallback extraction with PyMuPDF (fitz)."""
        import fitz

        passages = []
        doc = fitz.open(pdf_path)

        for page_num in range(1, len(doc) + 1):
            page = doc[page_num - 1]
            page_text = page.get_text()

            if not page_text.strip():
                continue

            section = self._classify_section(page_text, page_num, len(doc))
            page_passages = self._chunk_page(
                page_text=page_text,
                doc_id=doc_id,
                page_num=page_num,
                section=section,
                chunk_size=512,
                overlap=50
            )
            passages.extend(page_passages)

        doc.close()
        return passages

    def _extract_ocr(self, pdf_path: str, doc_id: uuid.UUID) -> List[Passage]:
        """Last resort: OCR with pytesseract."""
        try:
            from pdf2image import convert_from_path
            import pytesseract
        except ImportError:
            raise RuntimeError("OCR dependencies not installed (pdf2image, pytesseract)")

        passages = []
        images = convert_from_path(pdf_path)

        for page_num, image in enumerate(images, start=1):
            page_text = pytesseract.image_to_string(image)

            if not page_text.strip():
                continue

            section = self._classify_section(page_text, page_num, len(images))
            page_passages = self._chunk_page(
                page_text=page_text,
                doc_id=doc_id,
                page_num=page_num,
                section=section,
                chunk_size=512,
                overlap=50
            )
            passages.extend(page_passages)

        return passages

    def _chunk_page(
        self,
        page_text: str,
        doc_id: uuid.UUID,
        page_num: int,
        section: str,
        chunk_size: int,
        overlap: int
    ) -> List[Passage]:
        """
        Chunk page text with PAGE-LOCAL coordinates.

        Returns passages with (page_char_start, page_char_end) that refer
        to the original page_text string.

        CRITICAL FIXES (Code Review):
        - Infinite loop guardrail (min_end)
        - True sliding window (start = end - overlap)
        - Deterministic passage_id (includes END offset)
        """
        passages = []
        page_len = len(page_text)
        start = 0

        while start < page_len:
            end = min(start + chunk_size, page_len)

            # Find sentence/word boundary (don't split mid-word)
            # GUARDRAIL: Don't shrink below 50% of chunk_size (prevents infinite loop on long strings)
            min_end = start + (chunk_size // 2)
            if end < page_len and not page_text[end].isspace():
                # Search backward for space
                found_space = False
                for i in range(end, max(min_end, start), -1):
                    if page_text[i].isspace():
                        end = i
                        found_space = True
                        break
                # If no space found, hard chop at original end (don't create empty chunks)
                if not found_space:
                    logger.warning(
                        f"Long string detected at page {page_num}, char {start}-{end}, hard chopping"
                    )

            passage_text = page_text[start:end]

            # Skip empty passages (can happen at page boundaries)
            if not passage_text.strip():
                start = end
                continue

            # Generate deterministic passage_id (includes END to handle boundary tweaks)
            passage_canonical = f"doc:{doc_id}|p:{page_num}|s:{start}|e:{end}"
            passage_id = uuid.uuid5(NAMESPACE_POLYMATH, passage_canonical)

            passages.append(Passage(
                passage_id=passage_id,
                doc_id=doc_id,
                page_num=page_num,
                page_char_start=start,  # PAGE-LOCAL (not global)
                page_char_end=end,      # PAGE-LOCAL
                section=section,
                passage_text=passage_text,
                quality_score=self._score_quality(section),
                parser_version=None  # Set by caller
            ))

            # Advance with overlap (sliding window)
            # Use actual end position, then back up by overlap
            start = end - overlap
            if start >= page_len or end >= page_len:
                break

        return passages

    def _classify_section(self, text: str, page_num: int, total_pages: int) -> str:
        """
        Heuristic section detection.

        Uses rule-based classification with page number heuristics.
        """
        text_lower = text.lower()

        # Rule-based classification
        if page_num == 1 or 'abstract' in text_lower[:200]:
            return 'abstract'
        elif any(kw in text_lower for kw in ['methods', 'materials', 'experimental']):
            return 'methods'
        elif any(kw in text_lower for kw in ['results', 'findings']):
            return 'results'
        elif any(kw in text_lower for kw in ['discussion', 'conclusion']):
            return 'discussion'
        elif 'references' in text_lower or 'bibliography' in text_lower:
            return 'references'
        elif page_num > total_pages * 0.9:  # Last 10% of paper
            return 'references'  # Likely references/appendix
        else:
            return 'body'

    def _score_quality(self, section: str) -> float:
        """Section-based quality scoring."""
        return {
            'abstract': 1.0,
            'methods': 1.0,
            'results': 1.0,
            'discussion': 0.8,
            'body': 0.7,
            'references': 0.3,
            'acknowledgments': 0.1
        }.get(section, 0.5)


if __name__ == "__main__":
    # Test on a sample PDF
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 enhanced_pdf_parser.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # Generate test doc_id
    from lib.doc_identity import compute_doc_id
    doc_id = compute_doc_id(
        title="Test Document",
        authors=["Test Author"],
        year=2024
    )

    # Parse PDF
    parser = EnhancedPDFParser()
    passages = parser.extract_with_provenance(pdf_path, doc_id)

    print(f"\n{'='*60}")
    print(f"Parsed {len(passages)} passages")
    print(f"{'='*60}\n")

    # Show first 3 passages
    for i, passage in enumerate(passages[:3], 1):
        print(f"Passage {i}:")
        print(f"  Page: {passage.page_num}")
        print(f"  Section: {passage.section}")
        print(f"  Quality: {passage.quality_score}")
        print(f"  Char range: {passage.page_char_start}-{passage.page_char_end}")
        print(f"  Text: {passage.passage_text[:100]}...")
        print()
