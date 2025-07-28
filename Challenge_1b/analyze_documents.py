#!/usr/bin/env python3
"""
Multi-Collection PDF Analyzer for Adobe India Hackathon 2025 - Challenge 1B

This script analyzes multiple PDF collections based on a specific persona and task,
extracting relevant sections and generating a structured analysis.
"""

import os
import json
import logging
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import fitz  # PyMuPDF
from datetime import datetime

# Set environment variables for offline mode
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install the required packages using:")
    print("pip install -r requirements.txt")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('analyzer.log')
    ]
)

# Constants
VERSION = "1.0.0"
MIN_SECTION_LENGTH = 50  # Minimum characters for a section to be considered
MAX_SECTIONS = 20  # Maximum number of sections to return
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score to include a section
MODEL_NAME = "all-MiniLM-L6-v2"  # Lightweight model that fits within size constraints

@dataclass
class DocumentSection:
    """Represents a section of text from a document."""
    document: str
    text: str
    page_number: int
    score: float = 0.0
    metadata: Dict = None

class DocumentAnalyzer:
    """Analyzes documents based on a specific persona and task."""
    
    def __init__(self, input_data: Dict):
        """Initialize the analyzer with input configuration."""
        self.challenge_info = input_data.get("challenge_info", {})
        self.documents = input_data.get("documents", [])
        self.persona = input_data.get("persona", {})
        self.task = input_data.get("job_to_be_done", {})
        self.sections: List[DocumentSection] = []
        
        # Initialize the embedding model
        try:
            print(f"Loading model: {MODEL_NAME}")
            self.model = SentenceTransformer(MODEL_NAME)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure the model is downloaded and available locally.")
            print("You can download it using:")
            print(f"python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{MODEL_NAME}')\"")
            raise
        
    def extract_text_from_pdf(self, pdf_path: Path) -> List[DocumentSection]:
        """Extract text sections from a PDF file."""
        try:
            doc = fitz.open(pdf_path)
            sections = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                
                # Split text into paragraphs
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                
                for para in paragraphs:
                    if len(para) >= MIN_SECTION_LENGTH:
                        sections.append(DocumentSection(
                            document=pdf_path.name,
                            text=para,
                            page_number=page_num + 1,
                            metadata={
                                "char_count": len(para),
                                "word_count": len(para.split())
                            }
                        ))
            
            return sections
            
        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {str(e)}")
            return []
    
    def calculate_section_scores(self, query_embedding: np.ndarray) -> None:
        """Calculate relevance scores for each section based on the query."""
        if not self.sections:
            return
            
        # Extract text from sections
        texts = [section.text for section in self.sections]
        
        # Get embeddings for all sections
        section_embeddings = self.model.encode(texts, convert_to_tensor=True)
        
        # Calculate cosine similarity with query
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            section_embeddings.cpu().numpy()
        )[0]
        
        # Update section scores
        for i, section in enumerate(self.sections):
            section.score = float(similarities[i])
    
    def rank_sections(self, top_n: int = MAX_SECTIONS) -> List[DocumentSection]:
        """Rank sections by relevance score and return top N."""
        # Filter out low-scoring sections
        filtered = [s for s in self.sections if s.score >= SIMILARITY_THRESHOLD]
        
        # Sort by score (descending)
        filtered.sort(key=lambda x: x.score, reverse=True)
        
        # Return top N sections
        return filtered[:top_n]
    
    def analyze(self, collection_dir: Path) -> Dict:
        """Analyze documents in the collection based on the persona and task."""
        start_time = time.time()
        self.sections = []
        
        # Create query from persona and task
        query = self._create_query()
        logging.info(f"Analysis query: {query}")
        
        # Get query embedding
        query_embedding = self.model.encode(query, convert_to_tensor=True).cpu().numpy()
        
        # Process each document in the collection
        pdf_dir = collection_dir / "PDFs"
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDFs directory not found: {pdf_dir}")
        
        # Get all PDFs from the PDFs directory
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logging.error(f"No PDF files found in {pdf_dir}")
            return
        
        logging.info(f"Found {len(pdf_files)} PDFs in {pdf_dir}")
        
        # Process each PDF in the directory
        for pdf_path in pdf_files:
            logging.info(f"Processing PDF: {pdf_path.name}")
            try:
                # Create a document info dictionary for this PDF
                doc_info = {
                    "filename": pdf_path.name,
                    "title": pdf_path.stem.replace('_', ' ').title()
                }
                
                # Extract text and add to sections
                extracted_sections = self.extract_text_from_pdf(pdf_path)
                for section in extracted_sections:
                    section.doc_info = doc_info
                self.sections.extend(extracted_sections)
                
                logging.info(f"Extracted {len(extracted_sections)} sections from {pdf_path.name}")
                
            except Exception as e:
                logging.error(f"Error processing {pdf_path}: {e}", exc_info=True)
                continue
        
        # Calculate relevance scores
        self.calculate_section_scores(query_embedding)
        
        # Get top sections
        top_sections = self.rank_sections()
        
        # Calculate processing time safely
        try:
            processing_time = (datetime.now() - start_time).total_seconds()
        except Exception as e:
            logging.warning(f"Error calculating processing time: {e}")
            processing_time = 0.0
        
        # Count relevant sections safely
        try:
            relevant_sections = len([s for s in self.sections if hasattr(s, 'score') and s.score >= SIMILARITY_THRESHOLD])
        except Exception as e:
            logging.warning(f"Error counting relevant sections: {e}")
            relevant_sections = 0
        
        result = {
            "metadata": {
                "challenge_info": self.challenge_info or {},
                "persona": self.persona or {},
                "job_to_be_done": self.task or {},
                "processing_time_seconds": processing_time,
                "document_count": len(pdf_files) if pdf_files else 0,
                "section_count": len(self.sections) if hasattr(self, 'sections') else 0,
                "relevant_sections": relevant_sections,
                "version": VERSION
            },
            "extracted_sections": [
                {
                    "document": section.document,
                    "section_title": section.text[:100] + ("..." if len(section.text) > 100 else ""),
                    "importance_rank": i + 1,
                    "page_number": section.page_number,
                    "relevance_score": section.score,
                    "char_count": section.metadata.get("char_count", 0),
                    "word_count": section.metadata.get("word_count", 0)
                }
                for i, section in enumerate(top_sections)
            ],
            "subsection_analysis": [
                {
                    "document": section.document,
                    "refined_text": section.text,
                    "page_number": section.page_number,
                    "relevance_score": section.score
                }
                for section in top_sections
            ]
        }
        
        return result
    
    def _create_query(self) -> str:
        """Create a search query from persona and task information."""
        query_parts = []
        
        # Add persona information
        if "role" in self.persona:
            query_parts.append(f"Role: {self.persona['role']}")
        if "interests" in self.persona and isinstance(self.persona["interests"], list):
            query_parts.append(f"Interests: {', '.join(self.persona['interests'])}")
        
        # Add task information
        if "task" in self.task:
            query_parts.append(f"Task: {self.task['task']}")
        if "objectives" in self.task and isinstance(self.task["objectives"], list):
            query_parts.append("Objectives: " + ". ".join(self.task["objectives"]))
        
        return ". ".join(query_parts)

def load_input_data(input_path: Path) -> Dict:
    """Load input data from JSON file."""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading input file {input_path}: {str(e)}")
        raise

def save_output(output_path: Path, data: Dict) -> None:
    """Save analysis results to a JSON file."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved results to {output_path}")
    except Exception as e:
        logging.error(f"Error saving output to {output_path}: {str(e)}")
        raise

def process_collection(collection_dir: Path):
    """Process a single collection directory."""
    logging.info(f"Processing collection: {collection_dir.name}")
    
    # Paths
    input_path = collection_dir / "challenge1b_input.json"
    output_path = collection_dir / "challenge1b_output.json"
    
    if not input_path.exists():
        logging.warning(f"Input file not found: {input_path}")
        return
    
    # Load input data
    input_data = load_input_data(input_path)
    
    # Initialize and run analyzer
    analyzer = DocumentAnalyzer(input_data)
    result = analyzer.analyze(collection_dir)
    
    # Save results
    save_output(output_path, result)

def main():
    """Main function to process all collections."""
    import time
    import argparse
    import os
    
    start_time = time.time()
    
    try:
        # Set up argument parsing
        parser = argparse.ArgumentParser(description='Analyze PDF documents for a specific persona and task.')
        parser.add_argument('--collection', type=str, help='Path to the collection directory')
        args = parser.parse_args()
        
        # Get collection path from environment variable or use the one from args
        collection_path = os.environ.get('COLLECTION_PATH')
        if collection_path:
            collection_path = Path(collection_path)
        elif args.collection:
            collection_path = Path(args.collection)
        else:
            # Default to looking in the current directory
            collection_path = Path.cwd()
        
        logging.info(f"Using collection path: {collection_path.absolute()}")
        
        # Check if this is a collection directory (contains challenge1b_input.json)
        if not (collection_path / 'challenge1b_input.json').exists():
            # Look for collection subdirectories
            collections = [d for d in collection_path.iterdir() 
                          if d.is_dir() and (d / 'challenge1b_input.json').exists()]
            if not collections:
                logging.error(f"No valid collection directories found in {collection_path}")
                return
        else:
            collections = [collection_path]
        
        # Process each collection
        for collection_dir in collections:
            try:
                process_collection(collection_dir)
            except Exception as e:
                logging.error(f"Error processing collection {collection_dir.name}: {str(e)}")
                continue
        
        logging.info(f"Completed processing in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import sys
    import time
    
    logging.info("Starting document analysis...")
    start_time = time.time()
    
    try:
        main()
        logging.info(f"Analysis completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)
