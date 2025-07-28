# Multi-Collection PDF Analyzer

This is a solution for Challenge 1B of the Adobe India Hackathon 2025. The application analyzes multiple collections of PDFs based on a specific persona and task, extracting relevant sections and generating structured analysis.

## Features

- Processes multiple collections of PDF documents
- Analyzes content based on specified persona and job-to-be-done
- Uses semantic search to find relevant sections
- Generates structured JSON output with relevance scores
- Runs in a containerized environment with no internet access required
- Handles large document collections efficiently

## Prerequisites

- Docker
- Git (optional, for cloning the repository)
- At least 8GB RAM and 20GB disk space (for model storage)

## Directory Structure

```
Challenge_1b/
├── Collection 1/
│   ├── PDFs/                     # Input PDF files for Collection 1
│   ├── challenge1b_input.json    # Input configuration
│   └── challenge1b_output.json   # Generated output
├── Collection 2/
│   ├── PDFs/
│   ├── challenge1b_input.json
│   └── challenge1b_output.json
├── Dockerfile                   # Docker configuration
├── analyze_documents.py         # Main analysis script
├── README.md                    # This file
└── requirements.txt             # Python dependencies
```

## Building the Docker Image

```bash
docker build --platform linux/amd64 -t pdf-collection-analyzer .
```

## Running the Application

1. Prepare your collection directories following the structure above
2. Place your input JSON and PDFs in the appropriate directories
3. Run the following command:

```bash
docker run --rm \
  -v $(pwd):/app/data \
  --network none \
  --memory="16g" \
  --cpus=8 \
  pdf-collection-analyzer
```

### Parameters

- `-v $(pwd):/app/data`: Mounts the current directory to `/app/data` in the container
- `--network none`: Ensures no network access during execution
- `--memory="16g"`: Limits container memory to 16GB
- `--cpus=8`: Limits container to use 8 CPU cores

## Input Format

The input JSON file should follow this structure:

```json
{
  "challenge_info": {
    "challenge_id": "round_1b_001",
    "test_case_name": "example_case"
  },
  "documents": [
    {
      "filename": "document1.pdf",
      "title": "Document Title"
    }
  ],
  "persona": {
    "role": "Role Name",
    "experience_level": "Intermediate",
    "interests": ["topic1", "topic2"],
    "group_size": 1,
    "trip_duration": 0
  },
  "job_to_be_done": {
    "task": "Task description",
    "objectives": ["Objective 1", "Objective 2"],
    "constraints": ["Constraint 1", "Constraint 2"]
  }
}
```

## Output Format

The application generates a JSON file with the analysis results:

```json
{
  "metadata": {
    "challenge_info": {
      "challenge_id": "round_1b_001",
      "test_case_name": "example_case"
    },
    "persona": {
      "role": "Role Name",
      "experience_level": "Intermediate",
      "interests": ["topic1", "topic2"],
      "group_size": 1,
      "trip_duration": 0
    },
    "job_to_be_done": {
      "task": "Task description",
      "objectives": ["Objective 1", "Objective 2"],
      "constraints": ["Constraint 1", "Constraint 2"]
    },
    "processing_time": 12.34,
    "document_count": 3,
    "section_count": 45,
    "relevant_sections": 12,
    "version": "1.0.0"
  },
  "extracted_sections": [
    {
      "document": "document1.pdf",
      "section_title": "Section title...",
      "importance_rank": 1,
      "page_number": 5,
      "relevance_score": 0.95,
      "char_count": 254,
      "word_count": 42
    }
  ],
  "subsection_analysis": [
    {
      "document": "document1.pdf",
      "refined_text": "Full text of the section...",
      "page_number": 5,
      "relevance_score": 0.95
    }
  ]
}
```

## Testing with Sample Data

1. Place test PDFs in the `Collection 1/PDFs` directory
2. Create a `challenge1b_input.json` file in the `Collection 1` directory
3. Run the application as described above
4. Check the generated `challenge1b_output.json` file for results

## Performance

The solution is optimized to process multiple PDFs within the challenge constraints:
- Uses efficient text extraction with PyMuPDF
- Leverages the lightweight `all-MiniLM-L6-v2` model for embeddings
- Implements efficient similarity search with scikit-learn
- Processes documents in parallel when possible
- Runs within the 16GB memory constraint