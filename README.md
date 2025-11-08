# Project — Setup

Minimal, step-by-step setup instructions to get this project running locally.

## Prerequisites
- Git
- Python 3.8+ 

## Clone repository
```bash
git clone https://github.com/dairo95/sentiment_analysis_BERT.git .
cd sentiment_analysis_BERT
```

## Virtual environment (recommended)
Using venv:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

## Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Data 
Use the provided dataset or replace by yours in the 'data' folder

## Components 
 - data_extraction.py:  the data loader 
 - data_processing.py: this is where the text processing and tokenization operates. 
 - model.py: the model trainer, running time could differ depending on your setup. When running complete, create a folder with a saved model. 
 - inference.py: This is where you load and use your trained model. You can test with sentences to check the output and confidence. 

