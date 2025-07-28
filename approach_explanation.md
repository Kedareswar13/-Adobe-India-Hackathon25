# Approach Explanation: Adobe India Hackathon 2025

## Round 1A: PDF Outline Extraction

### Overview
This solution extracts document structure (title and headings) from PDFs by analyzing font sizes and page layout. The approach is lightweight, fast, and works entirely offline.

### Technical Implementation
1. **PDF Parsing**:
   - Utilizes PyMuPDF (fitz) for efficient PDF text extraction
   - Processes each page to extract text blocks with their associated font sizes

2. **Title Detection**:
   - Identifies the largest font size in the document
   - Selects the longest text span with this size from the first page as the title

3. **Heading Extraction**:
   - Ranks remaining font sizes in descending order
   - Maps the top three sizes to heading levels H1, H2, and H3
   - Extracts all text spans matching these sizes as document headings

4. **Performance Optimization**:
   - Single-pass processing of PDF content
   - Minimal memory footprint
   - Efficient text processing with built-in string operations

### Constraints Compliance
- **Offline Operation**: No network dependencies
- **Size Limit**: Minimal dependencies (<200MB)
- **Performance**: Processes documents in <10 seconds for typical PDFs
- **Compatibility**: Works with most PDFs containing extractable text

## Round 1B: Persona-Based Document Analysis

### Overview
This solution provides intelligent document analysis by ranking document sections based on their relevance to a given persona and task, using semantic search techniques.

### Technical Implementation
1. **Document Processing**:
   - Extracts text content from PDFs while preserving page structure
   - Handles various PDF formats and layouts

2. **Semantic Search**:
   - Employs the `all-MiniLM-L6-v2` sentence transformer model
   - Generates dense vector embeddings for document sections and queries
   - Computes cosine similarity between query and document embeddings

3. **Ranking & Analysis**:
   - Ranks document sections by relevance to the query
   - Returns top-k most relevant sections with their context
   - Provides structured output with document references and analysis

### Model Selection
- **Model**: `all-MiniLM-L6-v2`
  - Size: ~80MB (well under 1GB limit)
  - Performance: Optimized for speed and accuracy
  - Capabilities: Strong semantic understanding with relatively low computational requirements

### Constraints Compliance
- **Offline Operation**: All models are pre-downloaded during build
- **Size Limit**: Total package <1GB (actual model ~80MB)
- **Performance**: Processes documents in <60 seconds for typical inputs
- **Resource Usage**: Efficient CPU-only operation

## Challenges & Solutions

### Round 1A Challenges
1. **Font Size Variability**:
   - Challenge: Different documents use varying font size hierarchies
   - Solution: Relative size ranking instead of absolute size thresholds

2. **Noise in Text Extraction**:
   - Challenge: Extraneous elements (headers, footers, page numbers)
   - Solution: Focus on dominant text patterns and size distributions

### Round 1B Challenges
1. **Efficient Text Processing**:
   - Challenge: Processing large documents within time constraints
   - Solution: Page-level chunking and parallel processing

2. **Semantic Understanding**:
   - Challenge: Capturing nuanced meaning with limited model size
   - Solution: Leveraging pre-trained sentence transformers fine-tuned for semantic similarity

## Future Improvements
1. **Enhanced Structure Detection**:
   - Implement more sophisticated layout analysis
   - Detect and handle multi-column documents

2. **Improved Semantic Analysis**:
   - Add entity recognition for better content understanding
   - Implement topic modeling for section summarization

3. **Performance Optimization**:
   - Add parallel processing for multi-document analysis
   - Implement caching for repeated queries
