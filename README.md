# Invoice OCR
A OCR pipeline to extract information from invoices using OpenAI Vision API.

## Overview
This project provides a robust solution for extracting structured information from invoices and receipts. It utilizes OpenAI's Vision API for accurate text recognition and data extraction, handling various document formats including images (JPG, PNG) and PDFs.

## Project Structure
```
.
├── Dockerfile          # Container configuration
├── README.md          # Project documentation
├── data/              # Directory for input invoices
├── main.py            # Main script for batch processing
├── output/            # Directory for processed results
├── requirements.txt   # Python dependencies
└── src/              # Source code
    ├── __init__.py
    └── ocr_openai.py  # Core OCR implementation
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd InvoiceOCR
```

2. Build docker image
```bash
docker build -t invoice-ocr .
```

3. Run docker conatiner
```bash
docker run -it --rm -v $(pwd):/workspace invoice-ocr
```

## Configuration

Configure the openai key:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage
### Process Single Invoice
To process a single invoice file:
```bash
python src/ocr_openai.py
```

### Batch Process Multiple Invoices
To process all invoices in the data directory:
```bash
python main.py
```