from flask import Flask, render_template, request, jsonify, session
import json
import time
from datetime import datetime
import sys
import os

# Add the current directory to path to import our RAG system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multimodal_rag import CompleteMultimodalRAG, AdvancedAnswerGenerator

app = Flask(__name__)
app.secret_key = 'multimodal_rag_secret_key_2024'

class RAGWebInterface:
    def __init__(self):
        self.rag_system = None
        self.answer_generator = None
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the RAG system"""
        print("🚀 Initializing Multimodal RAG System...")
        self.rag_system = CompleteMultimodalRAG()
        success = self.rag_system.build_system(force_rebuild=False)
        if success and self.rag_system.llm_client:
            self.answer_generator = AdvancedAnswerGenerator(self.rag_system.llm_client)
        print("✅ RAG System Initialized!")
    
    def get_system_status(self):
        """Get current system status"""
        if not self.rag_system:
            return {
                "llm_model": "Initializing...",
                "embedding_model": "Initializing...",
                "vector_db": "Initializing...",
                "pdf_parser": "Initializing...",
                "ocr_engine": "Initializing...",
                "total_chunks": 0,
                "active_pdfs": []
            }
        
        status = self.rag_system.get_system_status()
        
        return {
            "llm_model": "Gemini 2.5 Flash" if status["llm_available"] else "Not Available",
            "embedding_model": "all-MiniLM-L6-v2 (Sentence-BERT)",
            "vector_db": "ChromaDB",
            "pdf_parser": "PyMuPDF",
            "ocr_engine": "EasyOCR",
            "total_chunks": status["total_chunks"],
            "active_pdfs": status["active_pdfs"]
        }

# Initialize the RAG interface
rag_interface = RAGWebInterface()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get system status"""
    status = rag_interface.get_system_status()
    return jsonify(status)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat queries"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        # Perform search
        search_start = time.time()
        search_results = rag_interface.rag_system.multimodal_search(query)
        search_time = time.time() - search_start
        
        # Generate answer
        generation_start = time.time()
        if rag_interface.answer_generator:
            answer = rag_interface.answer_generator.generate_answer(query, search_results, "cot")
        else:
            answer = "LLM not available - using template response"
        generation_time = time.time() - generation_start
        
        total_time = time.time() - search_start
        
        # Prepare response
        response_data = {
            'query': query,
            'answer': answer,
            'search_results': {
                'text_results': [
                    {
                        'content': result['content'][:200] + '...' if len(result['content']) > 200 else result['content'],
                        'metadata': result['metadata'],
                        'similarity_score': round(result['similarity_score'], 3)
                    }
                    for result in search_results.get('text_results', [])[:3]
                ],
                'total_results': len(search_results.get('text_results', []))
            },
            'processing_times': {
                'search': round(search_time, 2),
                'generation': round(generation_time, 2),
                'total': round(total_time, 2)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/clear_chat', methods=['POST'])
def clear_chat():
    """Clear chat history"""
    if 'chat_history' in session:
        session.pop('chat_history')
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)