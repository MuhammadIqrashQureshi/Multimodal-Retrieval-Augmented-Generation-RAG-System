class RAGInterface {
    constructor() {
        this.chatHistory = [];
        this.systemStatus = {};
        this.initializeEventListeners();
        this.loadSystemStatus();
        this.updateSystemConfig();
    }

    initializeEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Search button
        document.getElementById('search-btn').addEventListener('click', () => {
            this.processQuery();
        });

        // Enter key in query input
        document.getElementById('query-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.processQuery();
            }
        });

        // Clear chat
        document.getElementById('clear-chat').addEventListener('click', () => {
            this.clearChat();
        });

        // Image upload
        document.getElementById('image-upload').addEventListener('change', (e) => {
            this.handleImageUpload(e);
        });
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-button').forEach(button => {
            button.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');

        // Refresh analytics if switching to analytics tab
        if (tabName === 'analytics') {
            this.updateAnalytics();
        }
    }

    async loadSystemStatus() {
        try {
            const response = await fetch('/api/status');
            this.systemStatus = await response.json();
            
            // Update UI
            document.getElementById('llm-status').textContent = this.systemStatus.llm_model;
            document.getElementById('embedding-status').textContent = this.systemStatus.embedding_model;
            document.getElementById('vector-db-status').textContent = this.systemStatus.vector_db;
            document.getElementById('chunk-count').textContent = this.systemStatus.total_chunks;
            
            // Update active PDFs
            const pdfsList = document.getElementById('active-pdfs');
            pdfsList.innerHTML = '';
            this.systemStatus.active_pdfs.forEach(pdf => {
                const li = document.createElement('li');
                li.textContent = pdf;
                pdfsList.appendChild(li);
            });
        } catch (error) {
            console.error('Failed to load system status:', error);
        }
    }

    async processQuery() {
        const queryInput = document.getElementById('query-input');
        const query = queryInput.value.trim();
        
        if (!query) {
            alert('Please enter a question');
            return;
        }

        // Show loading spinner
        this.showLoading(true);

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query })
            });

            const data = await response.json();

            if (response.ok) {
                this.addToChatHistory(data);
                this.updateAnalytics();
                this.updateRawResults(data);
            } else {
                throw new Error(data.error || 'Failed to process query');
            }
        } catch (error) {
            console.error('Error processing query:', error);
            alert('Error processing query: ' + error.message);
        } finally {
            this.showLoading(false);
            queryInput.value = '';
        }
    }

    addToChatHistory(chatData) {
        this.chatHistory.push(chatData);
        this.renderChatHistory();
    }

    renderChatHistory() {
        const chatHistoryElement = document.getElementById('chat-history');
        chatHistoryElement.innerHTML = '';

        this.chatHistory.slice().reverse().forEach((chat, index) => {
            const messageElement = document.createElement('div');
            messageElement.className = `chat-message ${index === 0 ? 'answer' : 'question'}`;
            
            const sourcesHtml = chat.search_results.text_results.map(result => 
                `<div class="source-item">
                    ${result.metadata.source} (Page ${result.metadata.page}) - Score: ${result.similarity_score}
                </div>`
            ).join('');

            messageElement.innerHTML = `
                <div class="message-header">
                    <div class="message-query">${chat.query}</div>
                    <div class="message-time">${new Date(chat.timestamp).toLocaleTimeString()}</div>
                </div>
                <div class="message-content">${chat.answer}</div>
                ${sourcesHtml ? `
                <div class="sources-list">
                    <strong>📄 Retrieved Sources:</strong>
                    ${sourcesHtml}
                </div>
                ` : ''}
                <div class="message-time" style="margin-top: 1rem; font-size: 0.8rem;">
                    ⏱️ Processing time: ${chat.processing_times.total}s
                </div>
            `;

            chatHistoryElement.appendChild(messageElement);
        });
    }

    updateAnalytics() {
        if (this.chatHistory.length === 0) return;

        // Update metrics
        const avgSearchTime = this.chatHistory.reduce((sum, chat) => 
            sum + chat.processing_times.search, 0) / this.chatHistory.length;
        const avgGenTime = this.chatHistory.reduce((sum, chat) => 
            sum + chat.processing_times.generation, 0) / this.chatHistory.length;

        document.getElementById('avg-search-time').textContent = avgSearchTime.toFixed(2) + 's';
        document.getElementById('avg-gen-time').textContent = avgGenTime.toFixed(2) + 's';
        document.getElementById('total-queries').textContent = this.chatHistory.length;

        // Update pipeline visualization
        this.updatePipelineVisualization();
        
        // Update charts with latest results
        const latestResults = this.chatHistory[this.chatHistory.length - 1].search_results;
        this.updateCharts(latestResults);
    }

    updatePipelineVisualization() {
        const pipelineElement = document.getElementById('pipeline-visualization');
        const latestChat = this.chatHistory[this.chatHistory.length - 1];
        
        pipelineElement.innerHTML = `
            <div class="pipeline-step">
                <div class="step-status">✅</div>
                <div class="step-name">Query Input</div>
                <div class="step-time">User</div>
            </div>
            <div class="pipeline-step">
                <div class="step-status">✅</div>
                <div class="step-name">Text Processing</div>
                <div class="step-time">Instant</div>
            </div>
            <div class="pipeline-step">
                <div class="step-status">✅</div>
                <div class="step-name">Vector Search</div>
                <div class="step-time">${latestChat.processing_times.search}s</div>
            </div>
            <div class="pipeline-step">
                <div class="step-status">✅</div>
                <div class="step-name">LLM Generation</div>
                <div class="step-time">${latestChat.processing_times.generation}s</div>
            </div>
            <div class="pipeline-step">
                <div class="step-status">✅</div>
                <div class="step-name">Response Output</div>
                <div class="step-time">Instant</div>
            </div>
        `;
    }

    updateCharts(searchResults) {
        if (!searchResults.text_results.length) return;

        // Similarity scores chart
        const similarityData = [{
            x: searchResults.text_results.map((_, i) => i + 1),
            y: searchResults.text_results.map(r => r.similarity_score),
            type: 'bar',
            marker: {
                color: searchResults.text_results.map(r => r.similarity_score),
                colorscale: 'Viridis'
            }
        }];

        Plotly.newPlot('similarity-chart', similarityData, {
            title: 'Similarity Scores by Rank',
            xaxis: { title: 'Rank' },
            yaxis: { title: 'Similarity Score' }
        });

        // Source distribution chart
        const sourceCounts = {};
        searchResults.text_results.forEach(result => {
            const source = result.metadata.source;
            sourceCounts[source] = (sourceCounts[source] || 0) + 1;
        });

        const sourceData = [{
            values: Object.values(sourceCounts),
            labels: Object.keys(sourceCounts),
            type: 'pie'
        }];

        Plotly.newPlot('source-chart', sourceData, {
            title: 'Document Source Distribution'
        });
    }

    updateRawResults(chatData) {
        document.getElementById('raw-results').textContent = 
            JSON.stringify(chatData, null, 2);
    }

    updateSystemConfig() {
        const config = {
            "llm_integration": "Google Gemini API",
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_database": "ChromaDB (Persistent)",
            "pdf_processing": "PyMuPDF",
            "ocr_capability": "EasyOCR",
            "chunking_strategy": "Semantic paragraph-based",
            "retrieval_method": "Cosine similarity",
            "prompting_strategies": ["Chain-of-Thought", "Few-shot", "Zero-shot"]
        };
        
        document.getElementById('system-config').textContent = 
            JSON.stringify(config, null, 2);
    }

    showLoading(show) {
        const spinner = document.getElementById('loading-spinner');
        const searchBtn = document.getElementById('search-btn');
        
        if (show) {
            spinner.style.display = 'flex';
            searchBtn.disabled = true;
            searchBtn.textContent = 'Processing...';
        } else {
            spinner.style.display = 'none';
            searchBtn.disabled = false;
            searchBtn.textContent = '🔍 Search';
        }
    }

    handleImageUpload(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const preview = document.getElementById('image-preview');
                preview.innerHTML = `<img src="${e.target.result}" alt="Uploaded Image">`;
            };
            reader.readAsDataURL(file);
        }
    }

    clearChat() {
        this.chatHistory = [];
        this.renderChatHistory();
        fetch('/api/clear_chat', { method: 'POST' });
    }
}

// Initialize the interface when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.ragInterface = new RAGInterface();
});