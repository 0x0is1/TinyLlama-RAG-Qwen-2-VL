# AI Engineering Intern Assessment — Documentation

This document provides a professional, structured summary for TinyLlama-RAG-Qwen-2-VL. It explains the objectives, architecture, dependencies, step-by-step implementation details, and instructions to reproduce the end-to-end OCR → embedding → retrieval-augmented generation (RAG) pipeline demonstrated in the notebook.

## Goals and scope

- Demonstrate practical OCR for scanned documents using a fine-tuned vision-language model.
- Preprocess text for semantic search using sentence-transformers embeddings and FAISS.
- Provide a small LLM-driven retrieval-augmented generation (RAG) interface to answer queries over the processed document.
- Include bonus functionality for PDF/image upload handling and caching.

## High-level architecture

1. Input: image or PDF uploaded by user.
2. Preprocessing: image normalization + CLAHE to improve OCR quality.
3. OCR: apply a fine-tuned vision-language OCR model (Qwen-2-VL variant) to extract text.
4. Text splitting: split extracted text into semantically-sized chunks (RecursiveCharacterTextSplitter).
5. Embeddings: compute normalized sentence embeddings (all-MiniLM-L6-v2) for each chunk.
6. Storage: store embeddings in a FAISS index; cache index + mapping to disk.
7. Retrieval + LLM: embed query, retrieve top-k chunks, pass them to a small LLM (TinyLlama) to generate an answer.

## Dependencies and environment

The notebook uses the following major Python packages (examples and versions used or suggested in the notebook):

- Python 3.8+ (recommended)
- transformers (for vision-language OCR model and the TinyLlama pipeline)
- torch (PyTorch with CUDA support if available)
- sentence-transformers
- faiss-cpu (or faiss-gpu if CUDA enabled and desired)
- langchain (text splitter utility)
- pdf2image and poppler-utils (for converting PDFs to images)
- pillow (PIL)
- opencv-python (cv2)

Notes:
- The notebook includes platform-specific install commands (e.g., `apt-get install -y poppler-utils`) suitable for Linux/Colab environments. On Windows, install Poppler separately and point `pdf2image.convert_from_path` to the poppler bin folder.
- The notebook uses GPU acceleration (model.to("cuda"), device_map="cuda"). If CUDA is unavailable, adjust device settings and reduce model sizes or use CPU-only variants.

## Notebook walkthrough

### OCR model selection

- The notebook selects `JackChew/Qwen2-VL-2B-OCR` (a fine-tuned Qwen-2-VL variant) via `transformers.AutoProcessor` and `AutoModelForImageTextToText`.
- Rationale: open-source, lightweight relative to larger VL models, and tuned for OCR tasks.
- Code (not reproduced here): loading processor and model via `from_pretrained`.

### Input handling (PDF / images)

- The notebook demonstrates file upload patterns used in Google Colab (`google.colab.files.upload()`), accepts common image extensions, and supports PDF by converting the first page to an image with `pdf2image.convert_from_path`.
- On Windows or non-Colab environments, replace the `files.upload()` logic with a standard file-open dialog or command-line file path input.

### Image preprocessing

- The notebook converts images to grayscale and applies CLAHE (Contrast Limited Adaptive Histogram Equalization) via OpenCV to improve text contrast for OCR.
- This helps with poorly-scanned or low-contrast documents.

### OCR inference and text extraction

- The notebook constructs a chat-style prompt via `processor.apply_chat_template(...)` with an instruction to extract all data.
- Inputs are tokenized with the processor and sent to the OCR model to generate extracted text.
- The notebook expects `ocr_model.generate(...)` to return generated token IDs which are decoded back to text via `processor.batch_decode(...)`.

### Text preprocessing and chunking

- Extracted text is cleaned and split into chunks using LangChain's `RecursiveCharacterTextSplitter` with a chunk size of 500 tokens and 50-token overlap. Separators include double newlines, single newlines, periods, and spaces to preserve sentence boundaries.
- The result is a list of document chunks (`doc_texts`) suitable for semantic embedding.

### Embedding generation and storage (FAISS)

- The notebook uses `sentence-transformers/all-MiniLM-L6-v2` to generate embeddings for each chunk.
- Embeddings are normalized (L2) before being added to a FAISS index (IndexFlatIP used to compute cosine similarity via inner product on normalized vectors).
- The FAISS index and the mapping of index positions to original text chunks are cached to disk (`ocr_docs_index.faiss` and `docs_mapping.pkl`) to avoid recomputing embeddings.

### Retrieval and LLM-based RAG

- To answer queries, `query_document(query_text, top_k=2, max_tokens=200)`:
  - Loads or constructs the FAISS index and mapping.
  - Encodes and normalizes the query with the same embedder.
  - Performs a top-k search to retrieve the most relevant chunks.
  - Constructs a prompt containing the retrieved chunks and the user query.
  - Uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0` via `transformers.pipeline("text-generation")` to generate a final answer.

- The notebook demonstrates a summary query (`"Summarize this document"`) and a placeholder for continuing the chat.

## Files produced and caching

- `ocr_docs_index.faiss` — FAISS index storing embeddings.
- `docs_mapping.pkl` — Pickled list mapping index rows to document chunk text.

The notebook writes these files to the current working directory.

## Reproduction steps (recommended)

1. Prepare environment
   - Create and activate a Python virtual environment (venv or conda).
   - Install packages (example):

     pip install torch transformers sentence-transformers faiss-cpu langchain pdf2image pillow opencv-python

   - Install Poppler:
     - On Ubuntu/Debian: `sudo apt-get install -y poppler-utils`
     - On Windows: download Poppler for Windows and add to PATH, or pass `poppler_path` to `convert_from_path`.

2. Adjust device settings
   - If you have CUDA, ensure PyTorch with CUDA is installed and set models to use "cuda".
   - For CPU-only runs, set device to "cpu" or remove device_map arguments.

3. Update input handling (if not using Colab)
   - Replace `google.colab.files.upload()` with a local file path or file picker.

4. Run notebook cells in order
   - Load OCR processor and model.
   - Upload or select an image/PDF and run preprocessing.
   - Run OCR inference, chunking, embedding creation, and FAISS indexing.
   - Run `query_document` to query the processed document.

## Next steps and improvements

- Add automatic model fallback to CPU-safe or smaller models when CUDA unavailable.
- Deploy it on a fullstack backend and frontend with a heavy GPU and Gradio frontend