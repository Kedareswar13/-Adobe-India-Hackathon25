# PDF Outline Extractor

This is a solution for Challenge 1A of the Adobe India Hackathon 2025. The application processes PDF files, extracts their outline structure, and generates JSON files with the extracted information.

## Features

- Extracts document title and headings from PDF files
- Identifies heading levels (H1-H6) based on font size and formatting
- Handles multi-page documents
- Processes multiple PDFs in batch mode
- Runs in a containerized environment with no internet access required
- Outputs structured JSON that conforms to the provided schema

## Prerequisites

- Docker
- Git (optional, for cloning the repository)

## Directory Structure

```
Challenge_1a/
├── sample_dataset/           # Sample data for testing
│   ├── outputs/             # Expected output JSON files
│   ├── pdfs/                # Input PDF files
│   └── schema/              # Output schema definition
├── Dockerfile               # Docker configuration
├── process_pdfs.py          # Main processing script
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

## Building the Docker Image

```bash
docker build --platform linux/amd64 -t pdf-outline-extractor .
```

## Running the Application

1. Place your PDF files in the `input` directory
2. Run the following command:

```bash
docker run --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  --network none \
  pdf-outline-extractor
```

### Parameters

- `-v $(pwd)/input:/app/input:ro`: Mounts the local `input` directory as read-only to `/app/input` in the container
- `-v $(pwd)/output:/app/output`: Mounts the local `output` directory to `/app/output` in the container
- `--network none`: Ensures no network access during execution

## Output Format

The application generates one JSON file per input PDF in the output directory. The JSON structure follows this schema:

```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Chapter 1",
      "page": 1,
      "confidence": 0.95,
      "is_bold": true,
      "is_italic": false
    }
  ],
  "metadata": {
    "processing_time": 1.23,
    "page_count": 10,
    "block_count": 45,
    "heading_count": 5,
    "version": "1.0.0"
  }
}
```

## Testing with Sample Data

1. Place test PDFs in the `sample_dataset/pdfs` directory
2. Run the application with the sample data:

```bash
docker run --rm \
  -v $(pwd)/sample_dataset/pdfs:/app/input:ro \
  -v $(pwd)/sample_dataset/outputs:/app/output \
  --network none \
  pdf-outline-extractor
```

## Performance

The solution is optimized to process a 50-page PDF in under 10 seconds on a system with 8 CPUs and 16GB RAM, as per the challenge requirements.

## License

This project is licensed under the MIT License.
