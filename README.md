# Multimodal-Retrieval-Augmented-Generation-RAG-System
A complete Multimodal RAG pipeline capable of extracting, chunking, embedding, retrieving, and generating responses from PDF documents containing both text and images. The system enables semantic search, multimodal retrieval, and LLM-powered question answering through a web-based ChatGPT-style interface.

**Abstract**
This project presents a comprehensive Multimodal Retrieval-Augmented Generation (RAG) System designed for processing financial documents that contain both textual and visual information. The system integrates advanced NLP, computer vision, vector databases, and large language models to enable intelligent multimodal querying. It processes three financial-related PDF documents by extracting text, images, tables, charts, and visual elements, converting them into embeddings, and performing cross-modal retrieval. The system demonstrates high retrieval accuracy with a Precision@1 of 1.000 and Mean Average Precision (MAP) of 1.000, validating the effectiveness of the multimodal embedding and retrieval approach.

**Features**
**✔ Multimodal PDF Parsing**
Extracts text, tables, images, bar charts, and graphs
OCR support for images using EasyOCR

**✔ Text & Image Embedding**
Sentence Transformers for text
CLIP (OpenAI) for image embeddings
Stores embeddings with metadata in ChromaDB

**✔ Multimodal Semantic Search**
Text-to-text search
Text-to-image search
Image-to-text search
Ranked retrieval with cosine similarity

**✔ LLM Answer Generation**
Works with any LLM (GPT-4, LLaMA, Mistral, etc.). We used Gemini
Uses retrieved chunks as context
Supports:
Chain-of-Thought prompting
Zero-shot prompting
Few-shot prompting

**✔ Web-based Chat Interface**
ChatGPT-like UI
Upload image queries
Display retrieved text chunks + image previews
Show source PDF page numbers
Uses complete workflow and time taken to execute the query

UI:
<img width="296" height="285" alt="image" src="https://github.com/user-attachments/assets/67823917-cc00-4d2b-869a-fdcc70147b89" />


**✔ Evaluation & Visualization**
Precision@K
Recall@K
Mean Average Precision (MAP)
BLEU & ROUGE for generative evaluation
Embedding space visualization

**Technologies Used**
**NLP & CV**
PyMuPDF (PDF extraction)
EasyOCR (image OCR)
CLIP ViT-B/32 (image embeddings)
Sentence-BERT (text embeddings)
Vector Database
ChromaDB
Separate collections for text chunks & image chunks

**LLM Integration**
Gemini

**Evaluation Results**

<img width="215" height="116" alt="image" src="https://github.com/user-attachments/assets/76e4593c-99a4-4332-901d-2f7fb11653b9" />

Document Processing Volume
Document                     Pages               Text Chunks            Image Chunks
Annual Report                 85                     85                      66
Financial Statements          34                     34                      14
Handbook                      62                     26                      238

**Prompt Engineering Examples**
**Chain-of-Thought Prompt Example**
Let me analyze this step-by-step:
1. Identify relevant context
2. Extract financial performance indicators
3. Analyze trends
4. Provide final synthesized answer

**Running the System**
1. Install Dependencies
2. Run preprocessing
3. Start vector DB
4. Run RAG pipeline
5. Launch Web UI

**Conclusion**
This Multimodal RAG System successfully processes complex financial documents and retrieves both textual and visual information with outstanding accuracy. With CoT prompting, multimodal embeddings, and LLM-based generation, it provides a powerful foundation for intelligent document analysis applications. 
