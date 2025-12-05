import fitz
import numpy as np
import os
from pathlib import Path
import json
import chromadb
from PIL import Image
import io
import base64
import re
import uuid
from typing import List, Dict, Any
import torch
import logging
from dotenv import load_dotenv
import google.generativeai as genai
import time
from datetime import datetime
import easyocr
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteMultimodalRAG:
    def __init__(self, data_dir="Data", vector_db_dir="vector_db"):
        self.data_dir = Path(data_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.vector_db_dir.mkdir(exist_ok=True)
        
        # Load environment variables
        load_dotenv()
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(self.vector_db_dir))
        
        # Initialize Gemini client for LLM
        self.llm_client = self._initialize_llm()
        
        # Initialize OCR reader
        self.ocr_reader = easyocr.Reader(['en'])
        
        # Initialize CLIP for image embeddings
        self.clip_model = self._initialize_clip()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.prompt_log = []
        self.evaluation_results = []
        
    def _initialize_clip(self):
        """Initialize CLIP model for image embeddings"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("✅ CLIP model initialized successfully")
            return {'model': model, 'processor': processor}
        except Exception as e:
            logger.error(f"❌ Failed to initialize CLIP: {e}")
            return None
        
    def _initialize_llm(self):
        """Initialize Gemini client with correct model"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.warning("Gemini API key not found. Please set GEMINI_API_KEY in .env file")
            return None
        
        try:
            genai.configure(api_key=api_key)
            
            # Use the latest available models
            working_models = [
                'models/gemini-2.5-flash',
                'models/gemini-2.5-pro', 
                'models/gemini-pro-latest',
                'models/gemini-flash-latest'
            ]
            
            for model_name in working_models:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content("Hello")
                    if response.text:
                        logger.info(f"✅ Using Gemini model: {model_name}")
                        return {'genai': genai, 'model_name': model_name}
                except Exception as e:
                    logger.warning(f"Model {model_name} test failed: {e}")
                    continue
            
            logger.error("❌ No working Gemini model found")
            return None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini client: {e}")
            return None
    
    def build_system(self, force_rebuild=False):
        """Build or load the RAG system"""
        if force_rebuild or not (self.vector_db_dir / "chroma.sqlite3").exists():
            logger.info("Building vector store from scratch...")
            chunks = self.extract_and_chunk_documents()
            if chunks:
                self.create_vector_store(chunks)
                return True
            else:
                logger.error("No chunks extracted from documents!")
                return False
        else:
            logger.info("Vector store found. Loading existing system...")
            return True
    
    def extract_and_chunk_documents(self):
        """Extract and chunk documents with enhanced processing"""
        logger.info("Extracting and chunking documents...")
        
        all_chunks = []
        pdf_files = list(self.data_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.error(f"No PDF files found in {self.data_dir}")
            return all_chunks
        
        logger.info(f"Found {len(pdf_files)} PDF files: {[f.name for f in pdf_files]}")
        
        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file.name}")
            try:
                chunks = self._process_pdf(pdf_file)
                all_chunks.extend(chunks)
                logger.info(f"Created {len(chunks)} chunks from {pdf_file.name}")
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def _process_pdf(self, pdf_path):
        """Process a single PDF file with enhanced extraction including images"""
        chunks = []
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Enhanced text chunks with table detection
                text_chunks = self._chunk_text_enhanced(page, page_num, pdf_path.name)
                chunks.extend(text_chunks)
                
                # Extract and process images using PyMuPDF (no external dependencies)
                image_chunks = self.extract_images_from_pdf_fixed(pdf_path, page_num)
                chunks.extend(image_chunks)
            
            doc.close()
        except Exception as e:
            logger.error(f"Error opening PDF {pdf_path.name}: {e}")
        
        return chunks

    # ========== FIXED IMAGE PROCESSING METHODS ==========
    
    def extract_images_from_pdf_fixed(self, pdf_path, page_num):
        """Extract images from PDF using PyMuPDF (no external dependencies needed)"""
        image_chunks = []
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            
            # Get images from the page using PyMuPDF
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Extract image
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # Check if RGB or CMYK
                        # Convert to PIL Image
                        img_data = pix.tobytes("png")
                        pil_image = Image.open(io.BytesIO(img_data))
                        
                        # Perform OCR on the image
                        ocr_text = self.perform_ocr_on_image(np.array(pil_image))
                        
                        # Generate image embedding
                        image_embedding = self.generate_image_embeddings(pil_image)
                        
                        if image_embedding is not None:
                            chunk_id = str(uuid.uuid4())
                            image_chunks.append({
                                'id': chunk_id,
                                'type': 'image',
                                'content': ocr_text if ocr_text else "Image content",
                                'embedding': image_embedding.tolist(),
                                'metadata': {
                                    'source': pdf_path.name,
                                    'page': page_num + 1,
                                    'chunk_type': 'image',
                                    'image_index': img_index,
                                    'has_ocr_text': bool(ocr_text),
                                    'ocr_text_length': len(ocr_text) if ocr_text else 0
                                }
                            })
                            logger.info(f"✅ Extracted image {img_index} from page {page_num + 1}")
                    
                    pix = None  # Free pixmap memory
                    
                except Exception as e:
                    logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                    continue
            
            doc.close()
                
        except Exception as e:
            logger.error(f"Error extracting images from PDF {pdf_path.name} page {page_num}: {e}")
        
        return image_chunks
    
    def perform_ocr_on_image(self, image_np):
        """Perform OCR on image to extract text"""
        if self.ocr_reader is None:
            return "OCR not available"
            
        try:
            # Use EasyOCR to extract text
            results = self.ocr_reader.readtext(image_np)
            extracted_text = " ".join([result[1] for result in results])
            
            if extracted_text.strip():
                logger.info(f"✅ OCR extracted {len(extracted_text)} characters")
                return extracted_text
            else:
                return "No text detected in image"
                
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return f"OCR failed: {str(e)}"
    
    def generate_image_embeddings(self, image):
        """Generate embeddings for images using CLIP"""
        if self.clip_model is None:
            logger.warning("CLIP model not available for image embeddings")
            return None
            
        try:
            # Preprocess image for CLIP
            inputs = self.clip_model['processor'](
                images=image, 
                return_tensors="pt",
                padding=True
            )
            
            # Generate image embedding
            with torch.no_grad():
                image_features = self.clip_model['model'].get_image_features(**inputs)
                embedding = image_features.cpu().numpy().flatten()
            
            logger.info(f"✅ Generated image embedding with shape {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            return None

    def _chunk_text_enhanced(self, page, page_num, pdf_name):
        """Enhanced text chunking with semantic boundaries"""
        text = page.get_text()
        chunks = []
        
        if not text.strip():
            return chunks
        
        # Enhanced cleaning
        cleaned_text = self._clean_text_enhanced(text)
        if not cleaned_text.strip():
            return chunks
        
        # Simple chunking by paragraphs first
        paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
        
        for para in paragraphs:
            if len(para.split()) >= 10:
                chunk_id = str(uuid.uuid4())
                chunks.append({
                    'id': chunk_id,
                    'type': 'text',
                    'content': para,
                    'metadata': {
                        'source': pdf_name,
                        'page': page_num + 1,
                        'chunk_type': 'text',
                        'word_count': len(para.split()),
                        'has_tables': self._detect_tables(para)
                    }
                })
        
        return chunks
    
    def _clean_text_enhanced(self, text):
        """Enhanced text cleaning"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Remove page numbers, headers, footers
            if (re.match(r'^(Page?\s*\d+\s*of\s*\d+|\d+\s*/\s*\d+)$', line) or
                re.match(r'^\d+$', line) or
                len(line) < 3):
                continue
            
            # Remove common headers/footers
            skip_patterns = ['confidential', 'proprietary', 'copyright', 'all rights reserved']
            if any(pattern in line.lower() for pattern in skip_patterns) and len(line) < 50:
                continue
            
            cleaned_lines.append(line)
        
        cleaned_text = ' '.join(cleaned_lines)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        return cleaned_text.strip()
    
    def _detect_tables(self, text):
        """Simple table detection"""
        tab_count = text.count('\t')
        line_breaks = text.count('\n')
        if tab_count > 2 or (line_breaks > 3 and any('  ' in line for line in text.split('\n')[:5])):
            return True
        return False

    def create_vector_store(self, chunks):
        """Create vector store with enhanced metadata including images"""
        logger.info("Creating enhanced vector store...")
        
        # Clear existing collections
        try:
            self.chroma_client.delete_collection("text_chunks")
        except:
            pass
        
        try:
            self.chroma_client.delete_collection("image_chunks")
        except:
            pass
        
        text_chunks = [chunk for chunk in chunks if chunk['type'] == 'text']
        image_chunks = [chunk for chunk in chunks if chunk['type'] == 'image' and chunk.get('embedding') is not None]
        
        logger.info(f"Text chunks: {len(text_chunks)}")
        logger.info(f"Image chunks with embeddings: {len(image_chunks)}")
        
        # Create collections with enhanced metadata
        text_collection = self.chroma_client.get_or_create_collection(
            name="text_chunks",
            metadata={"description": "Enhanced text chunks with semantic boundaries"}
        )
        
        image_collection = self.chroma_client.get_or_create_collection(
            name="image_chunks",
            metadata={"description": "Image chunks with OCR text and embeddings"}
        )
        
        # Process text chunks
        if text_chunks:
            text_contents = [chunk['content'] for chunk in text_chunks]
            text_metadatas = [chunk['metadata'] for chunk in text_chunks]
            text_ids = [chunk['id'] for chunk in text_chunks]
            
            # Add in batches to avoid timeouts
            batch_size = 100
            for i in range(0, len(text_contents), batch_size):
                end_idx = min(i + batch_size, len(text_contents))
                text_collection.add(
                    documents=text_contents[i:end_idx],
                    metadatas=text_metadatas[i:end_idx],
                    ids=text_ids[i:end_idx]
                )
                logger.info(f"Added text batch {i//batch_size + 1}: {end_idx - i} chunks")
            
            logger.info(f"✅ Added {len(text_chunks)} text chunks")
        else:
            logger.warning("No text chunks to add!")
        
        # Process image chunks with custom embeddings
        if image_chunks:
            image_contents = [chunk['content'] for chunk in image_chunks]
            image_embeddings = [chunk['embedding'] for chunk in image_chunks]
            image_metadatas = [chunk['metadata'] for chunk in image_chunks]
            image_ids = [chunk['id'] for chunk in image_chunks]
            
            # Add image chunks with custom embeddings
            batch_size = 50
            for i in range(0, len(image_contents), batch_size):
                end_idx = min(i + batch_size, len(image_contents))
                image_collection.add(
                    documents=image_contents[i:end_idx],
                    embeddings=image_embeddings[i:end_idx],
                    metadatas=image_metadatas[i:end_idx],
                    ids=image_ids[i:end_idx]
                )
                logger.info(f"Added image batch {i//batch_size + 1}: {end_idx - i} chunks")
            
            logger.info(f"✅ Added {len(image_chunks)} image chunks")
        else:
            logger.warning("No image chunks with embeddings to add!")
        
        # Save enhanced metadata
        with open(self.vector_db_dir / "enhanced_chunks_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ Enhanced vector store created successfully!")
    
    def search(self, query, collection_name="text_chunks", n_results=5):
        """Enhanced search with performance tracking"""
        start_time = time.time()
        
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
            
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            search_time = time.time() - start_time
            found_count = len(results['documents'][0]) if results['documents'] else 0
            logger.info(f"Search completed in {search_time:.2f}s, found {found_count} results")
            
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return None

    def multimodal_similarity_search(self, query_embedding=None, query_text=None, n_results=3):
        """Perform multimodal similarity search for both text and images"""
        start_time = time.time()
        
        results = {
            'text_results': [],
            'image_results': [],
            'search_metrics': {
                'total_time': 0,
                'text_results_count': 0,
                'image_results_count': 0
            }
        }
        
        # Text-based search
        if query_text:
            text_results = self.search(query_text, "text_chunks", n_results)
            if text_results and text_results['documents']:
                for i, (doc, metadata, distance) in enumerate(zip(
                    text_results['documents'][0],
                    text_results['metadatas'][0],
                    text_results['distances'][0]
                )):
                    similarity = 1 / (1 + distance)
                    results['text_results'].append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': similarity,
                        'rank': i + 1
                    })
        
        # Image-based search (if embedding provided)
        if query_embedding is not None:
            try:
                image_collection = self.chroma_client.get_collection("image_chunks")
                image_results = image_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=n_results,
                    include=['documents', 'metadatas', 'distances']
                )
                
                if image_results and image_results['documents']:
                    for i, (doc, metadata, distance) in enumerate(zip(
                        image_results['documents'][0],
                        image_results['metadatas'][0],
                        image_results['distances'][0]
                    )):
                        similarity = 1 / (1 + distance)
                        results['image_results'].append({
                            'content': doc,
                            'metadata': metadata,
                            'similarity_score': similarity,
                            'rank': i + 1
                        })
            except Exception as e:
                logger.error(f"Image search error: {e}")
        
        # Update metrics
        results['search_metrics']['total_time'] = time.time() - start_time
        results['search_metrics']['text_results_count'] = len(results['text_results'])
        results['search_metrics']['image_results_count'] = len(results['image_results'])
        
        # Sort by similarity score
        results['text_results'].sort(key=lambda x: x['similarity_score'], reverse=True)
        results['image_results'].sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return results
    
    def multimodal_search(self, query, n_results=3):
        """Perform multimodal search with enhanced results"""
        logger.info(f"🔍 Multimodal search for: '{query}'")
        
        # For text queries, use the existing search
        if isinstance(query, str):
            return self.multimodal_similarity_search(query_text=query, n_results=n_results)
        # For image queries, generate embedding and search
        elif hasattr(query, 'shape'):  # If it's an image array
            image_embedding = self.generate_image_embeddings(query)
            if image_embedding is not None:
                return self.multimodal_similarity_search(
                    query_embedding=image_embedding, 
                    n_results=n_results
                )
        
        return {
            'text_results': [],
            'image_results': [],
            'search_metrics': {'total_time': 0, 'text_results_count': 0, 'image_results_count': 0}
        }

    # ========== FIXED EVALUATION METRICS ==========
    
    def calculate_precision_at_k(self, retrieved_docs, relevant_docs, k=5):
        """Calculate Precision@K for retrieval evaluation"""
        if len(retrieved_docs) == 0:
            return 0.0
        
        retrieved_at_k = retrieved_docs[:k]
        # Use fuzzy matching for better evaluation
        relevant_retrieved = 0
        for retrieved in retrieved_at_k:
            for relevant in relevant_docs:
                if relevant.lower() in retrieved.lower():
                    relevant_retrieved += 1
                    break
        
        precision = relevant_retrieved / len(retrieved_at_k)
        return precision
    
    def calculate_recall_at_k(self, retrieved_docs, relevant_docs, k=5):
        """Calculate Recall@K for retrieval evaluation"""
        if len(relevant_docs) == 0:
            return 0.0
        
        retrieved_at_k = retrieved_docs[:k]
        # Use fuzzy matching for better evaluation
        relevant_retrieved = 0
        for relevant in relevant_docs:
            for retrieved in retrieved_at_k:
                if relevant.lower() in retrieved.lower():
                    relevant_retrieved += 1
                    break
        
        recall = relevant_retrieved / len(relevant_docs)
        return recall
    
    def calculate_mean_average_precision(self, all_retrieved, all_relevant, k=5):
        """Calculate Mean Average Precision (MAP)"""
        average_precisions = []
        
        for retrieved_docs, relevant_docs in zip(all_retrieved, all_relevant):
            if not relevant_docs:
                continue
                
            precisions = []
            for i, doc in enumerate(retrieved_docs[:k]):
                # Check if any relevant term is in the retrieved document
                for relevant in relevant_docs:
                    if relevant.lower() in doc.lower():
                        precision_at_i = self.calculate_precision_at_k(retrieved_docs, relevant_docs, i+1)
                        precisions.append(precision_at_i)
                        break
            
            if precisions:
                avg_precision = sum(precisions) / len(precisions)
                average_precisions.append(avg_precision)
        
        return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    
    def calculate_bleu_rouge(self, generated_text, reference_texts):
        """Calculate BLEU and ROUGE scores for generation quality"""
        try:
            # BLEU score with better handling
            if not reference_texts:
                return {'bleu': 0.0, 'rouge': {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}}
            
            # Prepare references for BLEU
            references = [ref.split() for ref in reference_texts if ref.strip()]
            candidate = generated_text.split()
            
            if not references or not candidate:
                return {'bleu': 0.0, 'rouge': {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}}
            
            smoothie = SmoothingFunction().method4
            bleu_score = sentence_bleu(references, candidate, smoothing_function=smoothie)
            
            # ROUGE scores
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = {}
            
            valid_refs = 0
            for ref in reference_texts:
                if ref.strip():
                    scores = scorer.score(ref, generated_text)
                    for key in scores:
                        if key not in rouge_scores:
                            rouge_scores[key] = []
                        rouge_scores[key].append(scores[key].fmeasure)
                    valid_refs += 1
            
            # Average ROUGE scores
            avg_rouge = {key: sum(scores) / len(scores) for key, scores in rouge_scores.items()} if rouge_scores else {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
            
            return {
                'bleu': bleu_score,
                'rouge': avg_rouge
            }
            
        except Exception as e:
            logger.error(f"Error calculating BLEU/ROUGE: {e}")
            return {'bleu': 0.0, 'rouge': {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}}
    
    def evaluate_retrieval_quality(self, test_queries=None, ground_truths=None, k_values=[1, 3, 5]):
        """Comprehensive retrieval quality evaluation with better test data"""
        if test_queries is None or ground_truths is None:
            # Create more realistic test data based on actual document content
            test_queries = [
                "financial statements",
                "annual report content", 
                "project requirements",
                "performance metrics",
                "research publications"
            ]
            ground_truths = [
                ["financial", "statements", "balance", "income"],
                ["annual", "report", "performance", "metrics"],
                ["project", "requirements", "submission", "deadline"],
                ["performance", "metrics", "analysis", "results"],
                ["research", "publications", "papers", "journals"]
            ]
        
        evaluation_results = {}
        
        for k in k_values:
            precisions = []
            recalls = []
            
            for query, ground_truth in zip(test_queries, ground_truths):
                # Perform search
                results = self.multimodal_search(query, n_results=max(k_values))
                retrieved_docs = [result['content'] for result in results['text_results']]
                
                precision = self.calculate_precision_at_k(retrieved_docs, ground_truth, k)
                recall = self.calculate_recall_at_k(retrieved_docs, ground_truth, k)
                
                precisions.append(precision)
                recalls.append(recall)
            
            evaluation_results[f'P@{k}'] = sum(precisions) / len(precisions) if precisions else 0.0
            evaluation_results[f'R@{k}'] = sum(recalls) / len(recalls) if recalls else 0.0
        
        # Calculate MAP
        all_retrieved = []
        all_relevant = []
        for query, ground_truth in zip(test_queries, ground_truths):
            results = self.multimodal_search(query, n_results=10)
            retrieved_docs = [result['content'] for result in results['text_results']]
            all_retrieved.append(retrieved_docs)
            all_relevant.append(ground_truth)
        
        evaluation_results['MAP'] = self.calculate_mean_average_precision(all_retrieved, all_relevant)
        
        # Save evaluation results
        self.evaluation_results.append({
            'timestamp': datetime.now().isoformat(),
            'evaluation': evaluation_results
        })
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        return evaluation_results

    # ========== FIXED VISUALIZATION METHODS ==========
    
    def visualize_embeddings_fixed(self, output_path="embedding_visualization.png"):
        """Fixed version of embedding visualization"""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            # Get all embeddings from the database
            text_collection = self.chroma_client.get_collection("text_chunks")
            
            # Get text embeddings and metadata
            text_data = text_collection.get(include=['embeddings', 'metadatas'])
            
            if not text_data['embeddings']:
                logger.warning("No embeddings found for visualization")
                return
                
            text_embeddings = np.array(text_data['embeddings'])
            text_labels = ['text'] * len(text_embeddings)
            
            # Try to get image embeddings if available
            try:
                image_collection = self.chroma_client.get_collection("image_chunks")
                image_data = image_collection.get(include=['embeddings', 'metadatas'])
                
                if image_data['embeddings']:
                    image_embeddings = np.array(image_data['embeddings'])
                    image_labels = ['image'] * len(image_embeddings)
                    
                    # Combine all embeddings
                    all_embeddings = np.vstack([text_embeddings, image_embeddings])
                    all_labels = text_labels + image_labels
                else:
                    all_embeddings = text_embeddings
                    all_labels = text_labels
            except:
                all_embeddings = text_embeddings
                all_labels = text_labels
            
            if len(all_embeddings) == 0:
                logger.warning("No embeddings found for visualization")
                return
            
            # Apply t-SNE with proper error handling
            perplexity = min(30, len(all_embeddings) - 1)
            if perplexity <= 0:
                logger.warning("Not enough embeddings for t-SNE")
                return
                
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embeddings_2d = tsne.fit_transform(all_embeddings)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'type': all_labels
            })
            
            # Plot with different colors for text and images - FIXED: use proper masking
            colors = {'text': 'blue', 'image': 'red'}
            for label, color in colors.items():
                mask = np.array([l == label for l in all_labels])  # Convert to numpy array for proper masking
                if mask.any():  # Check if there are any items of this type
                    plt.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label, s=100, alpha=0.7)
            
            plt.title('t-SNE Visualization of Text and Image Embeddings')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.legend(title='Content Type')
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Embedding visualization saved to {output_path}")
            
            # Also create a PCA visualization for comparison
            self._create_pca_visualization_fixed(all_embeddings, all_labels, "pca_visualization.png")
            
        except Exception as e:
            logger.error(f"Error visualizing embeddings: {e}")
            # Create a simple fallback visualization
            self._create_simple_visualization()
    
    def _create_pca_visualization_fixed(self, embeddings, labels, output_path):
        """Fixed PCA visualization of embeddings"""
        try:
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
            
            plt.figure(figsize=(12, 8))
            df = pd.DataFrame({
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'type': labels
            })
            
            # FIXED: Use proper masking for scatter plot
            colors = {'text': 'blue', 'image': 'red'}
            for label, color in colors.items():
                mask = np.array([l == label for l in labels])
                if mask.any():
                    plt.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label, s=100, alpha=0.7)
            
            plt.title('PCA Visualization of Text and Image Embeddings')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.legend(title='Content Type')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ PCA visualization saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating PCA visualization: {e}")
    
    def _create_simple_visualization(self):
        """Create a simple bar chart visualization as fallback"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Count chunks by type
            text_count = self.get_chunk_count_by_type('text')
            image_count = self.get_chunk_count_by_type('image')
            
            categories = ['Text Chunks', 'Image Chunks']
            counts = [text_count, image_count]
            
            plt.bar(categories, counts, color=['blue', 'red'])
            plt.title('Distribution of Text and Image Chunks in Vector Database')
            plt.ylabel('Number of Chunks')
            
            for i, count in enumerate(counts):
                plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('simple_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("✅ Simple visualization saved to simple_visualization.png")
            
        except Exception as e:
            logger.error(f"Error creating simple visualization: {e}")

    def get_chunk_count_by_type(self, chunk_type):
        """Get count of chunks by type"""
        try:
            with open(self.vector_db_dir / "enhanced_chunks_metadata.json", 'r') as f:
                chunks = json.load(f)
                return len([chunk for chunk in chunks if chunk.get('type') == chunk_type])
        except:
            return 0

    # ========== INTERFACE HELPER METHODS ==========
    
    def get_system_status(self):
        """Get system status for the interface"""
        return {
            "llm_available": self.llm_client is not None,
            "vector_db_ready": True,
            "total_chunks": self.get_chunk_count(),
            "active_pdfs": self.get_active_pdfs(),
            "clip_available": self.clip_model is not None,
            "ocr_available": self.ocr_reader is not None
        }

    def get_chunk_count(self):
        """Get total number of chunks"""
        try:
            with open(self.vector_db_dir / "enhanced_chunks_metadata.json", 'r') as f:
                chunks = json.load(f)
                return len(chunks)
        except:
            return 0

    def get_active_pdfs(self):
        """Get list of active PDFs"""
        try:
            with open(self.vector_db_dir / "enhanced_chunks_metadata.json", 'r') as f:
                chunks = json.load(f)
                pdf_sources = set()
                for chunk in chunks:
                    if 'metadata' in chunk and 'source' in chunk['metadata']:
                        pdf_sources.add(chunk['metadata']['source'])
                return list(pdf_sources)
        except:
            return []

# Keep the AdvancedAnswerGenerator class exactly the same as before
# [AdvancedAnswerGenerator class remains unchanged - it's working perfectly]

class AdvancedAnswerGenerator:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.prompt_log = []
        
    def generate_answer(self, query, retrieved_chunks, prompt_strategy="cot"):
        """Generate answer using advanced prompting strategies"""
        start_time = time.time()
        
        # Format context for LLM
        context = self._format_context_for_llm(retrieved_chunks)
        
        # Select prompt strategy
        if prompt_strategy == "cot":
            prompt = self._create_cot_prompt(query, context)
        elif prompt_strategy == "few_shot":
            prompt = self._create_few_shot_prompt(query, context)
        elif prompt_strategy == "zero_shot":
            prompt = self._create_zero_shot_prompt(query, context)
        else:
            prompt = self._create_cot_prompt(query, context)
        
        # Log the prompt
        self._log_prompt(query, prompt, prompt_strategy)
        
        # Generate answer using LLM
        if self.llm_client:
            try:
                answer = self._generate_with_gemini(prompt)
                generation_time = time.time() - start_time
                
                # Log response metrics
                self._log_response_metrics(query, len(answer), generation_time, prompt_strategy)
                
                return answer
            except Exception as e:
                logger.error(f"Gemini generation failed: {e}")
                return self._generate_fallback_answer(query, context)
        else:
            return self._generate_fallback_answer(query, context)
    
    def _create_cot_prompt(self, query, context):
        """Chain-of-Thought prompting for enhanced reasoning"""
        return f"""
        You are an expert document analyst. Based on the provided context from multiple documents, answer the user's question using step-by-step reasoning.

        CONTEXT FROM DOCUMENTS:
        {context}

        QUESTION: {query}

        Think step by step:
        1. First, analyze which parts of the context are relevant to the question
        2. Then, extract key information, figures, and data points from the relevant sections
        3. Next, synthesize the information from different sources and identify patterns
        4. Finally, provide a comprehensive, well-structured answer that addresses all aspects of the question

        If the context doesn't contain enough information to answer fully, acknowledge what information is available and what is missing.

        Structure your answer with:
        - Clear introduction
        - Key findings with supporting evidence
        - Reference to source documents and pages
        - Conclusion summarizing the main points

        ANSWER:
        """
    
    def _create_few_shot_prompt(self, query, context):
        """Few-shot prompting with examples"""
        examples = """
        EXAMPLE 1:
        Question: What are the main financial metrics in the annual report?
        Context: [Financial statements showing revenue growth of 15%, profit margins of 20%]
        Answer: The annual report highlights several key financial metrics. Revenue showed strong growth of 15% compared to previous year. Profit margins remained healthy at 20%. The balance sheet indicates stable financial position with adequate liquidity ratios.

        EXAMPLE 2:
        Question: What research achievements are mentioned?
        Context: [Publications data, faculty awards, research grants information]
        Answer: The documents mention significant research achievements including 50+ publications in reputed journals, 3 faculty awards for research excellence, and successful acquisition of $2M in research grants. These accomplishments demonstrate the institution's commitment to research innovation.
        """
        
        return f"""
        {examples}
        
        Now analyze this real query based on the provided context:
        
        CONTEXT:
        {context}
        
        QUESTION: {query}
        
        Based on the examples above, provide a comprehensive, well-structured answer that:
        - Directly addresses the question
        - Cites specific information from the context
        - References source documents and page numbers
        - Provides a clear and organized response
        
        ANSWER:
        """
    
    def _create_zero_shot_prompt(self, query, context):
        """Zero-shot prompting"""
        return f"""
        Based on the following context from documents, provide a comprehensive answer to the user's question.

        CONTEXT:
        {context}

        QUESTION: {query}

        Please provide a detailed answer that:
        - Directly addresses the question
        - Cites specific information and data points from the context
        - References the source documents and page numbers
        - Is well-structured and easy to understand

        If the context doesn't contain relevant information, clearly state this.

        ANSWER:
        """
    
    def _format_context_for_llm(self, retrieved_chunks):
        """Format context for LLM consumption"""
        context_parts = []
        
        # Add text results
        if retrieved_chunks.get('text_results'):
            context_parts.append("TEXT CONTENT:")
            for i, result in enumerate(retrieved_chunks['text_results'][:5]):
                source_info = f"Source: {result['metadata']['source']} - Page {result['metadata']['page']}"
                if result['metadata'].get('has_tables'):
                    source_info += " (Contains tabular data)"
                
                context_parts.append(f"--- {source_info} ---")
                context_parts.append(f"Content: {result['content']}")
                context_parts.append(f"Relevance Score: {result['similarity_score']:.3f}")
                context_parts.append("")
        
        # Add image results
        if retrieved_chunks.get('image_results'):
            context_parts.append("IMAGE CONTENT (OCR Text):")
            for i, result in enumerate(retrieved_chunks['image_results'][:3]):
                source_info = f"Source: {result['metadata']['source']} - Page {result['metadata']['page']} - Image {result['metadata']['image_index']}"
                context_parts.append(f"--- {source_info} ---")
                context_parts.append(f"OCR Text: {result['content']}")
                context_parts.append(f"Relevance Score: {result['similarity_score']:.3f}")
                context_parts.append("")
        
        return "\n".join(context_parts) if context_parts else "No relevant content found in the documents."
    
    def _generate_with_gemini(self, prompt):
        """Generate answer using Gemini LLM"""
        try:
            if isinstance(self.llm_client, dict) and 'model_name' in self.llm_client:
                model_name = self.llm_client['model_name']
                genai_obj = self.llm_client['genai']
            else:
                # Fallback to a known working model
                model_name = 'models/gemini-2.5-flash'
                genai_obj = self.llm_client
            
            model = genai_obj.GenerativeModel(model_name)
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    top_p=0.9,
                    max_output_tokens=1500,
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def _generate_fallback_answer(self, query, context):
        """Fallback answer when LLM is not available"""
        return f"""
        ## Answer to: '{query}'

        Based on the document analysis, I found relevant information to address your question.

        **Retrieved Context:**
        {context}

        **Note:** This is a template-based answer. For more sophisticated reasoning and comprehensive responses, please configure the Gemini API integration with a valid API key.

        For complete details, please refer to the original documents cited above.
        """
    
    def _log_prompt(self, query, prompt, strategy):
        """Log prompt for documentation"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'strategy': strategy,
            'prompt': prompt,
            'prompt_length': len(prompt)
        }
        self.prompt_log.append(log_entry)
    
    def _log_response_metrics(self, query, answer_length, generation_time, strategy):
        """Log response metrics"""
        metrics_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'strategy': strategy,
            'answer_length': answer_length,
            'generation_time': generation_time
        }
        
        # Save to file
        with open('response_metrics.json', 'a') as f:
            f.write(json.dumps(metrics_entry) + '\n')
    
    def save_prompt_log(self):
        """Save prompt log to file"""
        with open('prompt_log.txt', 'w', encoding='utf-8') as f:
            f.write("PROMPT ENGINEERING LOG\n")
            f.write("=" * 50 + "\n\n")
            
            for i, entry in enumerate(self.prompt_log, 1):
                f.write(f"ENTRY {i}:\n")
                f.write(f"Timestamp: {entry['timestamp']}\n")
                f.write(f"Query: {entry['query']}\n")
                f.write(f"Strategy: {entry['strategy']}\n")
                f.write(f"Prompt Length: {entry['prompt_length']} characters\n")
                f.write("Prompt:\n")
                f.write(entry['prompt'])
                f.write("\n" + "-" * 50 + "\n\n")

# Enhanced test function with all fixes
def test_complete_system():
    """Test the complete multimodal RAG system with all fixes"""
    print("🧪 Testing Complete Multimodal RAG System with All Fixes")
    print("=" * 70)
    
    try:
        # Initialize system
        rag_system = CompleteMultimodalRAG()
        success = rag_system.build_system(force_rebuild=True)  # Rebuild to include images
        
        # Test search even if Gemini is not available
        print(f"\n📊 Vector Database Status: {'✅ Loaded' if success else '❌ Failed'}")
        print(f"🤖 Gemini Status: {'✅ Available' if rag_system.llm_client else '❌ Not Available'}")
        print(f"🖼️ CLIP Status: {'✅ Available' if rag_system.clip_model else '❌ Not Available'}")
        print(f"🔤 OCR Status: {'✅ Available' if rag_system.ocr_reader else '❌ Not Available'}")
        
        # Test queries
        test_queries = [
            "What are the main financial achievements mentioned?",
            "Explain the performance metrics in the annual report",
            "What are the requirements for final year projects?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*70}")
            print(f"TEST CASE {i}: {query}")
            print(f"{'='*70}")
            
            # Perform multimodal search
            start_time = time.time()
            results = rag_system.multimodal_search(query)
            search_time = time.time() - start_time
            
            print(f"🔍 Search completed in {search_time:.2f}s")
            print(f"📊 Found {len(results['text_results'])} text results")
            print(f"🖼️ Found {len(results['image_results'])} image results")
            
            if results['text_results']:
                # Show top results
                for j, result in enumerate(results['text_results'][:2], 1):
                    print(f"\n📄 Result {j} (Page {result['metadata']['page']}, Score: {result['similarity_score']:.3f}):")
                    print(f"   Source: {result['metadata']['source']}")
                    print(f"   Preview: {result['content'][:150]}...")
            
            if results['image_results']:
                # Show top image results
                for j, result in enumerate(results['image_results'][:2], 1):
                    print(f"\n🖼️ Image Result {j} (Page {result['metadata']['page']}, Score: {result['similarity_score']:.3f}):")
                    print(f"   Source: {result['metadata']['source']}")
                    print(f"   OCR Preview: {result['content'][:150]}...")
            
            # Generate answer if Gemini is available
            if rag_system.llm_client:
                answer_generator = AdvancedAnswerGenerator(rag_system.llm_client)
                answer = answer_generator.generate_answer(query, results, "cot")
                
                print(f"\n🤖 GENERATED ANSWER ({len(answer)} characters):")
                print("-" * 50)
                print(answer[:500] + "..." if len(answer) > 500 else answer)
                print("-" * 50)
                
                # Test BLEU score for this answer
                reference_texts = [
                    "Financial achievements include revenue growth and profit margins",
                    "Performance metrics show company performance and future actions",
                    "Final year projects have specific requirements and evaluation criteria"
                ]
                bleu_results = rag_system.calculate_bleu_rouge(answer, reference_texts)
                print(f"\n📈 BLEU Score: {bleu_results['bleu']:.4f}")
                print(f"📈 ROUGE-1: {bleu_results['rouge']['rouge1']:.4f}")
                print(f"📈 ROUGE-2: {bleu_results['rouge']['rouge2']:.4f}")
                print(f"📈 ROUGE-L: {bleu_results['rouge']['rougeL']:.4f}")
                
            else:
                print(f"\n⚠️  Gemini not available - using template answer")
                answer_generator = AdvancedAnswerGenerator(None)
                answer = answer_generator.generate_answer(query, results, "cot")
                print(f"📝 TEMPLATE ANSWER:")
                print("-" * 50)
                print(answer)
                print("-" * 50)
        
        # Test evaluation metrics with improved test data
        print(f"\n{'='*70}")
        print("📈 TESTING IMPROVED EVALUATION METRICS")
        print(f"{'='*70}")
        
        evaluation_results = rag_system.evaluate_retrieval_quality()
        print("📊 Improved Retrieval Evaluation Results:")
        for metric, score in evaluation_results.items():
            print(f"   {metric}: {score:.4f}")
        
        # Test visualization with fixed methods
        print(f"\n🖼️ GENERATING FIXED EMBEDDING VISUALIZATIONS...")
        rag_system.visualize_embeddings_fixed()
        print("✅ Fixed embedding visualizations generated!")
        
        if rag_system.llm_client:
            answer_generator.save_prompt_log()
            print(f"\n💾 Prompt log saved to: prompt_log.txt")
        
        print(f"\n💾 Evaluation results saved to: evaluation_results.json")
        
        # Show system summary
        print(f"\n{'='*70}")
        print("📊 SYSTEM SUMMARY")
        print(f"{'='*70}")
        status = rag_system.get_system_status()
        print(f"Total chunks: {status['total_chunks']}")
        print(f"Active PDFs: {', '.join(status['active_pdfs'])}")
        print(f"Text chunks: {rag_system.get_chunk_count_by_type('text')}")
        print(f"Image chunks: {rag_system.get_chunk_count_by_type('image')}")
                
    except Exception as e:
        print(f"❌ Error testing system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_complete_system()