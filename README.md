# TinyLlama-RAG-Qwen-2-VL — OCR → Embeddings → RAG Notebook
[Colab Notebook](https://colab.research.google.com/drive/1Fbw_rvjfWAUFFwpoyLkVgtmQ8Wlz3H5m?usp=sharing&authuser=1#scrollTo=JeG0cQsHMLNx)
Project: AI Engineering Intern Assessment — end-to-end pipeline that extracts text from scanned images/PDFs, converts text to embeddings, stores them in FAISS, and answers queries with a small LLM (TinyLlama) using retrieval-augmented generation (RAG).

This repository contains:

- `chatpdf.ipynb` — Jupyter Notebook demonstrating the full pipeline (OCR, preprocessing, embeddings, FAISS index, RAG queries).
- `chatpdf_documentation.md` — Documentation and reproduction notes (generated from the notebook).

## Quick overview

The notebook implements the following flow:
1. Upload or provide an image / PDF.
2. Preprocess image (grayscale + CLAHE) for better OCR results.
3. Run a vision-language OCR model (Qwen2-VL OCR variant) to extract text.
4. Split text into chunks using LangChain's `RecursiveCharacterTextSplitter`.
5. Embed chunks using `sentence-transformers/all-MiniLM-L6-v2` and normalize embeddings.
6. Store embeddings in FAISS (index written to `ocr_docs_index.faiss`) and cache mapping (`docs_mapping.pkl`).
7. Query the FAISS index and generate answers with `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (text-generation pipeline).

## Files created by the notebook

- `ocr_docs_index.faiss` — FAISS index with embeddings (created at runtime).
- `docs_mapping.pkl` — Pickled mapping of index rows to text chunks.

## Requirements

See `requirements.txt` for a pinned list of core Python packages. There are notes in the file about optional GPU-specific packages.

## Setup (recommended)

On Windows (local development):

1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install packages

```powershell
pip install -r requirements.txt
```

3. Install Poppler for Windows
- Download a Poppler binary from a trusted source (e.g., conda-forge or the official Poppler releases). Add the `bin` folder to your PATH or pass `poppler_path` to `pdf2image.convert_from_path`.

4. Open `chatpdf.ipynb` in Jupyter or VS Code and run cells in order. Replace `google.colab.files.upload()` with a local file picker or a path if needed.

On Google Colab

- Use the Colab notebook directly and run the `!apt-get install -y poppler-utils` cell to install Poppler. The notebook already uses `google.colab.files.upload()` for uploads.

## Usage


- Run the notebook cells sequentially. After creating the FAISS index, call `query_document("Your question")` to get answers or `query = "Summarize this document"` as in the example cell.
