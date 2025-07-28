#!/usr/bin/env python3
"""
PDF Outline Extractor for Adobe India Hackathon 2025 - Challenge 1A

This script processes PDF files from an input directory, extracts their outline structure,
and saves the results as JSON files in the output directory.

Usage:
    docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output:/app/output --network none pdf-processor
"""

import os
import json
import time
import logging
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('extractor.log')
    ]
)

# Constants
MARGIN_THRESHOLD = 0.1  # 10% of page width/height for header/footer detection
LINE_SPACING_RATIO = 1.2  # Max line spacing to consider as same paragraph
BLOCK_MERGE_DISTANCE = 5.0  # Points between blocks to consider for merging
MIN_HEADING_SIZE = 10  # Minimum font size to consider as heading
VERSION = "1.0.0"

@dataclass
class TextBlock:
    """Represents a block of text with spatial and formatting information."""
    
    def __init__(self, text: str, bbox: Tuple[float, float, float, float], 
                 size: float, is_bold: bool = False, is_italic: bool = False, 
                 font: str = "", page_num: int = 0, confidence: float = 1.0):
        self.text = text.strip()
        self.bbox = bbox  # (x0, y0, x1, y1)
        self.size = size
        self.is_bold = is_bold
        self.is_italic = is_italic
        self.font = font
        self.page_num = page_num
        self.confidence = confidence
    
    @property
    def area(self) -> float:
        """Calculate the area of the bounding box."""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get the center point of the bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )
    
    def distance_to(self, other: 'TextBlock') -> float:
        """Calculate the minimum distance between two text blocks."""
        dx = max(0, max(self.bbox[0] - other.bbox[2], other.bbox[0] - self.bbox[2]))
        dy = max(0, max(self.bbox[1] - other.bbox[3], other.bbox[1] - self.bbox[3]))
        return (dx**2 + dy**2) ** 0.5
    
    def is_same_line(self, other: 'TextBlock', page_width: float) -> bool:
        """Check if two blocks are on the same line."""
        # Check vertical overlap
        vertical_overlap = (self.bbox[1] <= other.bbox[3] and 
                          self.bbox[3] >= other.bbox[1])
        
        # Check horizontal proximity (within 5% of page width)
        horizontal_proximity = abs(self.bbox[0] - other.bbox[0]) < (0.05 * page_width)
        
        return vertical_overlap and horizontal_proximity

def extract_blocks_from_page(page: fitz.Page, page_num: int) -> List[TextBlock]:
    """Extract text blocks from a PDF page with formatting information."""
    blocks = []
    page_width = page.rect.width
    
    # Get page text with detailed information
    blocks_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)
    
    for block in blocks_dict.get("blocks", []):
        if "lines" not in block:
            continue
            
        for line in block["lines"]:
            for span in line["spans"]:
                # Skip empty spans
                if not span["text"].strip():
                    continue
                
                # Skip very small text (likely page numbers, footnotes)
                if span["size"] < 6:
                    continue
                    
                # Create a text block
                text_block = TextBlock(
                    text=span["text"],
                    bbox=span["bbox"],
                    size=span["size"],
                    is_bold="bold" in span["font"].lower(),
                    is_italic="italic" in span["font"].lower(),
                    font=span["font"],
                    page_num=page_num + 1,  # Convert to 1-based
                    confidence=1.0
                )
                
                blocks.append(text_block)
    
    return blocks

def merge_blocks(blocks: List[TextBlock], page_width: float) -> List[TextBlock]:
    """Merge text blocks that are part of the same logical line or paragraph."""
    if not blocks:
        return []
    
    # Sort blocks by vertical position, then horizontal position
    blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    
    merged = [blocks[0]]
    
    for block in blocks[1:]:
        last = merged[-1]
        
        # Check if blocks should be merged
        if (block.page_num == last.page_num and 
            (block.is_same_line(last, page_width) or 
             (block.bbox[1] <= last.bbox[3] + 5 and  # Close vertically
              abs(block.bbox[0] - last.bbox[0]) < page_width * 0.1))):  # Similar x-position
            
            # Merge blocks
            new_bbox = (
                min(block.bbox[0], last.bbox[0]),
                min(block.bbox[1], last.bbox[1]),
                max(block.bbox[2], last.bbox[2]),
                max(block.bbox[3], last.bbox[3])
            )
            
            # Create new merged block
            new_text = f"{last.text} {block.text}"
            new_block = TextBlock(
                text=new_text,
                bbox=new_bbox,
                size=max(last.size, block.size),
                is_bold=last.is_bold or block.is_bold,
                is_italic=last.is_italic or block.is_italic,
                font=last.font if last.size >= block.size else block.font,
                page_num=last.page_num,
                confidence=min(last.confidence, block.confidence)
            )
            
            merged[-1] = new_block
        else:
            merged.append(block)
    
    return merged

def identify_headings(blocks: List[TextBlock]) -> List[Dict]:
    """Identify headings from text blocks."""
    if not blocks:
        return []
    
    # Filter potential heading blocks
    potential_headings = [b for b in blocks if b.size >= MIN_HEADING_SIZE]
    
    if not potential_headings:
        return []
    
    # Group by font size
    size_groups = defaultdict(list)
    for block in potential_headings:
        # Round to nearest 0.5 to group similar sizes
        size = round(block.size * 2) / 2
        size_groups[size].append(block)
    
    # Sort sizes in descending order
    sorted_sizes = sorted(size_groups.keys(), reverse=True)
    
    # Assign heading levels (H1, H2, etc.)
    headings = []
    for i, size in enumerate(sorted_sizes):
        level = min(i + 1, 6)  # Max heading level is H6
        
        for block in size_groups[size]:
            # Skip very short text (likely not a heading)
            if len(block.text) < 3:
                continue
                
            # Skip text in margins (likely page numbers or headers/footers)
            if (block.bbox[0] < 50 or block.bbox[2] > 550):
                continue
                
            # Calculate confidence based on formatting and position
            confidence = 0.7  # Base confidence
            if block.is_bold:
                confidence += 0.2
            if block.is_italic:
                confidence += 0.1
                
            # Reduce confidence for text that looks like a page number
            if block.text.strip().isdigit() and 0 < int(block.text) < 1000:
                confidence -= 0.3
            
            # Only include headings with sufficient confidence
            if confidence >= 0.6:
                headings.append({
                    'level': f"H{level}",
                    'text': block.text,
                    'page': block.page_num,
                    'confidence': min(1.0, confidence),  # Cap at 1.0
                    'is_bold': block.is_bold,
                    'is_italic': block.is_italic
                })
    
    return headings

def extract_title(blocks: List[TextBlock]) -> str:
    """Extract the document title from the first page."""
    if not blocks:
        return ""
    
    # Get blocks from the first page
    first_page_blocks = [b for b in blocks if b.page_num == 1]
    
    if not first_page_blocks:
        return ""
    
    # Find the largest text block on the first page
    # (weight size more than bold/italic)
    def score_block(block: TextBlock) -> float:
        size_score = block.size
        if block.is_bold:
            size_score *= 1.2
        if block.is_italic:
            size_score *= 1.1
        return size_score
    
    # Get the highest scoring block with reasonable length
    title_block = max(
        [b for b in first_page_blocks if 5 <= len(b.text) <= 200],
        key=score_block,
        default=None
    )
    
    return title_block.text if title_block else ""

def process_pdf(pdf_path: str) -> Dict:
    """Process a single PDF file and return the outline."""
    start_time = time.time()
    doc = fitz.open(pdf_path)
    
    all_blocks = []
    
    # Process each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_blocks = extract_blocks_from_page(page, page_num)
        page_blocks = merge_blocks(page_blocks, page.rect.width)
        all_blocks.extend(page_blocks)
    
    # Extract title and headings
    title = extract_title(all_blocks)
    headings = identify_headings(all_blocks)
    
    # Build the result
    result = {
        "title": title,
        "outline": headings,
        "metadata": {
            "processing_time": time.time() - start_time,
            "page_count": len(doc),
            "block_count": len(all_blocks),
            "heading_count": len(headings),
            "version": VERSION
        }
    }
    
    doc.close()
    return result

def process_pdfs():
    """Process all PDFs in the input directory and save results to the output directory."""
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all PDF files in the input directory
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logging.warning(f"No PDF files found in {input_dir}")
        return
    
    for pdf_file in pdf_files:
        try:
            logging.info(f"Processing {pdf_file.name}...")
            
            # Process the PDF
            result = process_pdf(str(pdf_file))
            
            # Save the result as JSON
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Saved outline to {output_file}")
            
        except Exception as e:
            logging.error(f"Error processing {pdf_file.name}: {str(e)}", exc_info=True)

if __name__ == "__main__":
    logging.info("Starting PDF outline extraction...")
    start_time = time.time()
    
    try:
        process_pdfs()
        logging.info(f"Completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        raise
