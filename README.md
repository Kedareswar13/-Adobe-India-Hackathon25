# Adobe India Hackathon 2025 Submission

This repository contains solutions for both Challenge 1A and Challenge 1B of the Adobe India Hackathon 2025.

## Project Structure

```
.
├── Challenge_1a/                 # Solution for Challenge 1A - PDF Outline Extraction
│   ├── sample_dataset/          # Sample data for testing
│   │   ├── outputs/            # Generated output JSON files
│   │   ├── pdfs/               # Input PDF files
│   │   └── schema/             # Output schema definition
│   ├── Dockerfile              # Docker configuration
│   ├── process_pdfs.py         # Main processing script
│   ├── README.md               # Challenge 1A documentation
│   └── requirements.txt        # Python dependencies
│
└── Challenge_1b/                # Solution for Challenge 1B - Multi-Collection Analysis
    ├── Collection 1/           # First document collection
    │   ├── PDFs/              # PDF files for this collection
    │   ├── challenge1b_input.json  # Input configuration
    │   └── challenge1b_output.json # Generated output
    ├── Collection 2/           # Second document collection
    │   ├── PDFs/
    │   ├── challenge1b_input.json
    │   └── challenge1b_output.json
    ├── Collection 3/           # Third document collection
    │   ├── PDFs/
    │   ├── challenge1b_input.json
    │   └── challenge1b_output.json
    ├── Dockerfile              # Docker configuration
    ├── analyze_documents.py    # Main analysis script
    ├── README.md               # Challenge 1B documentation
    └── requirements.txt        # Python dependencies
```

## Prerequisites

- Docker
- Git (optional, for cloning the repository)
- At least 16GB RAM recommended for Challenge 1B

## Challenge 1A: PDF Outline Extraction

Extracts structured outlines from PDF documents, identifying titles and headings with their hierarchy.

### Quick Start

1. Build the Docker image:
   ```bash
   cd Challenge_1a
   docker build --platform linux/amd64 -t pdf-outline-extractor .
   ```

2. Place your PDFs in the `sample_dataset/pdfs` directory

3. Run the container:
   ```bash
   docker run --rm \
     -v $(pwd)/sample_dataset/pdfs:/app/input:ro \
     -v $(pwd)/sample_dataset/outputs:/app/output \
     --network none \
     pdf-outline-extractor
   ```

4. Find the results in `sample_dataset/outputs`

For detailed instructions, see [Challenge 1A README](Challenge_1a/README.md).

## Challenge 1B: Multi-Collection Document Analysis

Analyzes multiple collections of PDFs based on a specific persona and task, extracting relevant sections and generating structured analysis.

### Quick Start

1. Build the Docker image:
   ```bash
   cd Challenge_1b
   docker build --platform linux/amd64 -t pdf-collection-analyzer .
   ```

2. Prepare your collection directories (see `Collection 1` as an example)

3. Run the container:
   ```bash
   docker run --rm \
     -v $(pwd):/app/data \
     --network none \
     --memory="16g" \
     --cpus=8 \
     pdf-collection-analyzer
   ```

4. Find the results in each collection's directory

For detailed instructions, see [Challenge 1B README](Challenge_1b/README.md).

## Sample Collections

1. **Collection 1**: Travel Planning
   - Persona: Travel Planner
   - Task: Plan a 4-day trip for 10 college friends
   - Documents: Travel guides and recommendations

2. **Collection 2**: AI Research Papers
   - Persona: AI Research Scientist
   - Task: Analyze recent advances in efficient training of large language models
   - Documents: Research papers on AI and machine learning

3. **Collection 3**: Startup Business Plan
   - Persona: Startup Founder
   - Task: Develop a comprehensive business plan for a tech startup
   - Documents: Market analysis, financial projections, competitor analysis

## Performance

Both solutions are optimized to run within the challenge constraints:
- **Challenge 1A**: Processes a 50-page PDF in under 10 seconds
- **Challenge 1B**: Handles multiple document collections with efficient memory usage