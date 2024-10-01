# Llama 3.2 RAG Chatbot

## Overview
"Llama-3.2-RAG-Chatbot" is a Retrieval-Augmented Generation (RAG) chatbot that integrates a powerful large language model (LLM) with document retrieval capabilities. The chatbot is built using the [Meta Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) model for text generation and leverages LangChain for document retrieval and vector database management. Users can upload PDF files, and the chatbot will use the content to generate responses grounded in the provided document context.

## Features
- **Retrieval-Augmented Generation (RAG)**: The chatbot answers user questions based on the content retrieved from the uploaded PDF documents.
- **Document Upload**: Users can upload a PDF file, and the document is indexed for efficient retrieval.
- **Contextual Responses**: The chatbot uses the relevant context from the PDF to generate accurate and informative responses.
- **Persistence**: The vector database persists across sessions, so the PDF content does not need to be re-uploaded every time.

## How It Works
1. **PDF Upload**: Users upload a PDF document that they want to interact with.
2. **Document Indexing**: The chatbot splits the document into manageable chunks and creates a vectorized database using the sentence-transformers embedding model.
3. **Question Answering**: The chatbot uses a combination of the uploaded document and a language model (Llama) to provide grounded responses to user questions.
4. **Interactive Chat**: Users can chat with the model through a simple Gradio interface.

## Technologies Used
- **LangChain** for document loading, splitting, and text retrieval.
- **Chroma** for vector database storage and retrieval.
- **HuggingFace Transformers** for loading the Llama language model.
- **Gradio** for the web interface.

## Demo
The file [demo.webm](demo.webm) is a vide demo of the project running. It can be accessed also from [this link](https://drive.google.com/file/d/1hShtlHCtHIrriE6j8hS2pSp0NFHFDpk_/view?usp=sharing).