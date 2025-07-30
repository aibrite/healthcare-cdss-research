// Medical Intake Assistant - Frontend Application
    constructor() {
        this.currentChatId = null;
        this.currentSymptoms = [];
        this.clarifyingQuestions = [];
        this.conversationStep = 0;
        this.isLoading = false;
        this.currentEventSource = null;
        this.showThinking = false;
        this.currentThinkingSection = null;
        
        this.initializeElements();
        this.bindEvents();
        this.updateConnectionStatus();
    }

    initializeElements() {
        // Main screens
        this.welcomeScreen = document.getElementById('welcomeScreen');
        this.chatArea = document.getElementById('chatArea');
        this.loadingIndicator = document.getElementById('loadingIndicator');
        
        // Input elements
        this.initialSymptomsInput = document.getElementById('initialSymptoms');
        this.startChatBtn = document.getElementById('startChatBtn');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.thinkingToggle = document.getElementById('thinkingToggle');
        
        // Chat elements
        this.messagesContainer = document.getElementById('messagesContainer');
        this.inputArea = document.getElementById('inputArea');
        
        // Sidebar elements
        this.symptomsList = document.getElementById('symptomsList');
        this.sessionInfo = document.getElementById('sessionInfo');
        
        // Modal elements
        this.errorModal = document.getElementById('errorModal');
        this.errorMessage = document.getElementById('errorMessage');
        this.closeErrorModal = document.getElementById('closeErrorModal');
        
        // Connection status
        this.connectionStatus = document.getElementById('connectionStatus');
    }

    bindEvents() {
        // Button events
        this.startChatBtn.addEventListener('click', () => this.startChat());
        this.newChatBtn.addEventListener('click', () => this.resetChat());
        this.closeErrorModal.addEventListener('click', () => this.hideError());
        
        // Thinking toggle
        this.thinkingToggle.addEventListener('change', (e) => {
            this.showThinking = e.target.checked;
            this.updateThinkingVisibility();
        });
        
        // Keyboard events
        this.initialSymptomsInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.startChat();
            }
        });
        
        // Modal events
        this.errorModal.addEventListener('click', (e) => {
            if (e.target === this.errorModal) {
                this.hideError();
            }
        });
        
        // Auto-resize textarea
        this.initialSymptomsInput.addEventListener('input', this.autoResizeTextarea);
    }

    autoResizeTextarea(e) {
        const textarea = e.target;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }

    async startChat() {
        const symptoms = this.initialSymptomsInput.value.trim();
        
        if (!symptoms) {
            this.showError('Please enter your symptoms before starting the intake.');
            return;
        }

        this.setLoading(true);
        this.showChatArea();
        
        try {
            // Use streaming endpoint
            await this.startChatStream(symptoms);
        } catch (error) {
            console.error('Error starting chat:', error);
            this.showError(error.message);
            this.setLoading(false);
        }
    }

    async startChatStream(symptoms) {
        return new Promise((resolve, reject) => {
            // Close any existing event source
            if (this.currentEventSource) {
                this.currentEventSource.close();
            }

            fetch('/start_chat_stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ symptoms }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to start streaming');
                }
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                
                let buffer = '';
                
                const processStream = () => {
                    reader.read().then(({ done, value }) => {
                        if (done) {
                            this.setLoading(false);
                            resolve();
                            return;
                        }
                        
                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n');
                        buffer = lines.pop(); // Keep incomplete line in buffer
                        
                        for (const line of lines) {
                            if (line.startsWith('event: ')) {
                                const eventType = line.slice(7);
                                continue;
                            }
                            if (line.startsWith('data: ')) {
                                const data = line.slice(6);
                                if (data.trim()) {
                                    try {
                                        this.handleStreamEvent(JSON.parse(data));
                                    } catch (e) {
                                        console.error('Error parsing stream data:', e);
                                    }
                                }
                            }
                        }
                        
                        processStream();
                    }).catch(reject);
                };
                
                processStream();
            })
            .catch(reject);
        });
    }

    handleStreamEvent(data) {
        if (data.error) {
            this.showError(data.error);
            this.setLoading(false);
            return;
        }

        if (data.status) {
            this.showStatusIndicator(data.status);
        }

        if (data.thinking !== undefined || data.response !== undefined) {
            this.updateThinkingDisplay(data.thinking || '', data.response || '');
        }

        if (data.chat_id) {
            this.currentChatId = data.chat_id;
        }

        if (data.clarifying_questions) {
            this.currentSymptoms = data.current_symptoms || [];
            this.clarifyingQuestions = data.clarifying_questions;
            this.conversationStep = data.conversation_step || 1;

            this.addAIMessage(data);
            this.updateSidebar();
            this.setupFollowUpInput();
            this.setLoading(false);
        }

        if (data.intake_complete) {
            this.showFinalReport(data.final_report, data.thinking);
            this.setLoading(false);
        }

        // Handle session updates
        if (data.session_update) {
            this.updateSessionState(data.session_update);
        }
    }

    async updateSessionState(sessionData) {
        try {
            await fetch('/update_session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(sessionData),
            });
        } catch (error) {
            console.warn('Failed to update session state:', error);
        }
    }

    createThinkingSection() {
        const thinkingSection = document.createElement('div');
        thinkingSection.className = 'thinking-section';
        thinkingSection.style.display = this.showThinking ? 'block' : 'none';
        
        thinkingSection.innerHTML = `
            <div class="thinking-header" onclick="app.toggleThinkingContent(this)">
                <div class="thinking-title">
                    <i class="fas fa-brain"></i>
                    AI Thinking Process
                </div>
                <i class="fas fa-chevron-down thinking-toggle-icon"></i>
            </div>
            <div class="thinking-content">
                <div class="thinking-text"></div>
            </div>
        `;
        
        return thinkingSection;
    }

    updateThinkingDisplay(thinking, response) {
        if (!this.currentThinkingSection) {
            this.currentThinkingSection = this.createThinkingSection();
            this.messagesContainer.appendChild(this.currentThinkingSection);
        }

        const thinkingText = this.currentThinkingSection.querySelector('.thinking-text');
        
        let displayText = '';
        if (thinking) {
            displayText += `THINKING:\n${thinking}\n\n`;
        }
        if (response) {
            displayText += `RESPONSE:\n${response}`;
        }
        
        thinkingText.textContent = displayText;
        this.scrollToBottom();
    }

    toggleThinkingContent(header) {
        const content = header.nextElementSibling;
        const icon = header.querySelector('.thinking-toggle-icon');
        
        content.classList.toggle('collapsed');
        icon.classList.toggle('expanded');
    }

    updateThinkingVisibility() {
        const thinkingSections = document.querySelectorAll('.thinking-section');
        thinkingSections.forEach(section => {
            section.style.display = this.showThinking ? 'block' : 'none';
        });
    }

    showStatusIndicator(status) {
        // Remove any existing status indicators
        const existingIndicators = document.querySelectorAll('.streaming-indicator, .status-indicator');
        existingIndicators.forEach(indicator => indicator.remove());

        const statusMessages = {
            'generating_question': 'Generating intake question...',
            'processing_response': 'Processing your response...',
            'checking_completion': 'Checking if intake is complete...',
            'generating_report': 'Generating final report...'
        };

        const indicator = document.createElement('div');
        indicator.className = 'streaming-indicator';
        indicator.innerHTML = `
            <i class="fas fa-cog"></i>
            <span>${statusMessages[status] || status}</span>
            <div class="streaming-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;

        this.messagesContainer.appendChild(indicator);
        this.scrollToBottom();
    }

    async respondToQuestion(response) {
        this.setLoading(true);
        this.currentThinkingSection = null; // Reset for new response
        
        try {
            await this.respondToQuestionStream(response);
        } catch (error) {
            console.error('Error responding to question:', error);
            this.showError(error.message);
            this.setLoading(false);
        }
    }

    async respondToQuestionStream(response) {
        return new Promise((resolve, reject) => {
            // Add user message immediately
            this.addUserMessage(response);

            fetch('/respond_to_question_stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ response }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to process response');
                }
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                
                let buffer = '';
                
                const processStream = () => {
                    reader.read().then(({ done, value }) => {
                        if (done) {
                            this.setLoading(false);
                            resolve();
                            return;
                        }
                        
                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n');
                        buffer = lines.pop();
                        
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const data = line.slice(6);
                                if (data.trim()) {
                                    try {
                                        this.handleStreamEvent(JSON.parse(data));
                                    } catch (e) {
                                        console.error('Error parsing stream data:', e);
                                    }
                                }
                            }
                        }
                        
                        processStream();
                    }).catch(reject);
                };
                
                processStream();
            })
            .catch(reject);
        });
    }

    addAIMessage(data) {
        // Remove any status indicators
        const indicators = document.querySelectorAll('.streaming-indicator, .status-indicator');
        indicators.forEach(indicator => indicator.remove());

        const messageDiv = document.createElement('div');
        messageDiv.className = 'message message-ai';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        
        const header = document.createElement('div');
        header.className = 'message-header';
        header.innerHTML = '<i class="fas fa-robot"></i> AI Intake Assistant';
        
        content.appendChild(header);

        // Questions section
        if (data.clarifying_questions && data.clarifying_questions.length > 0) {
            const questionsSection = document.createElement('div');
            questionsSection.className = 'questions-section';
            
            const questionsHTML = data.clarifying_questions.map((question, index) => 
                `<div class="question-item">
                    <i class="fas fa-question-circle"></i>
                    <span>${question}</span>
                </div>`
            ).join('');
            
            questionsSection.innerHTML = `
                <div class="questions-title">
                    <i class="fas fa-clipboard-question"></i>
                    Intake Questions
                </div>
                ${questionsHTML}
            `;
            content.appendChild(questionsSection);
        }

        messageDiv.appendChild(content);
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    showFinalReport(report, thinking) {
        // Remove any status indicators
        const indicators = document.querySelectorAll('.streaming-indicator, .status-indicator');
        indicators.forEach(indicator => indicator.remove());

        const messageDiv = document.createElement('div');
        messageDiv.className = 'message message-ai final-report';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        
        const header = document.createElement('div');
        header.className = 'message-header';
        header.innerHTML = '<i class="fas fa-file-medical"></i> Patient Intake Report';
        
        content.appendChild(header);

        const reportSection = document.createElement('div');
        reportSection.className = 'report-section';
        reportSection.innerHTML = `
            <div class="report-content">
                <pre style="white-space: pre-wrap; font-family: inherit;">${report}</pre>
            </div>
        `;
        content.appendChild(reportSection);

        messageDiv.appendChild(content);
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();

        this.showFinalMessage();
    }

    addUserMessage(response) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message message-user';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        
        const header = document.createElement('div');
        header.className = 'message-header';
        header.innerHTML = '<i class="fas fa-user"></i> You';
        
        content.appendChild(header);

        const messageP = document.createElement('p');
        messageP.textContent = response;
        content.appendChild(messageP);

        messageDiv.appendChild(content);
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    setupFollowUpInput() {
        if (this.clarifyingQuestions.length === 0) {
            this.showCompletionOptions();
            return;
        }

        this.inputArea.innerHTML = `
            <div class="text-input-area">
                <textarea id="responseInput" placeholder="Please answer the questions above..." rows="3"></textarea>
                <div class="input-buttons">
                    <button class="btn btn-primary" onclick="app.handleTextResponse()">
                        <i class="fas fa-paper-plane"></i> Send Answer
                    </button>
                    <button class="btn btn-success" onclick="app.handleCompletion()">
                        <i class="fas fa-check"></i> I'm Done
                    </button>
                </div>
            </div>
        `;

        // Add enter key support (Ctrl+Enter to send)
        const responseInput = document.getElementById('responseInput');
        responseInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.handleTextResponse();
            }
        });
        
        // Auto-resize textarea
        responseInput.addEventListener('input', this.autoResizeTextarea);
        
        // Focus on input
        responseInput.focus();
    }

    handleTextResponse() {
        const responseInput = document.getElementById('responseInput');
        const response = responseInput.value.trim();
        
        if (!response) {
            this.showError('Please enter a response.');
            return;
        }

        this.respondToQuestion(response);
    }

    handleCompletion() {
        this.respondToQuestion('done');
    }

    showCompletionOptions() {
        this.inputArea.innerHTML = `
            <div class="completion-options">
                <p style="color: #666; margin-bottom: 1rem;">
                    <i class="fas fa-info-circle"></i>
                    Do you have any additional information to share, or are you ready to complete the intake?
                </p>
                <div class="input-buttons">
                    <button class="btn btn-success" onclick="app.handleCompletion()">
                        <i class="fas fa-check"></i> Complete Intake
                    </button>
                    <button class="btn btn-secondary" onclick="app.setupFollowUpInput()">
                        <i class="fas fa-plus"></i> Add More Information
                    </button>
                </div>
            </div>
        `;
    }

    showFinalMessage() {
        this.inputArea.innerHTML = `
            <div class="text-center">
                <p style="color: #666; margin-bottom: 1rem;">
                    <i class="fas fa-check-circle" style="color: #27ae60;"></i>
                    Patient intake completed successfully.
                </p>
                <button class="btn btn-primary" onclick="app.resetChat()">
                    <i class="fas fa-plus"></i> Start New Intake
                </button>
            </div>
        `;
    }

    updateSidebar() {
        // Update symptoms list
        if (this.currentSymptoms.length === 0) {
            this.symptomsList.innerHTML = '<p class="no-symptoms">No symptoms recorded yet</p>';
        } else {
            this.symptomsList.innerHTML = this.currentSymptoms.map(symptom => 
                `<div class="symptom-item">${symptom}</div>`
            ).join('');
        }

        // Update session info
        const startTime = new Date().toLocaleTimeString();
        this.sessionInfo.innerHTML = `
            <p><strong>Session ID:</strong><br>${this.currentChatId ? this.currentChatId.slice(0, 8) : 'N/A'}</p>
            <p><strong>Symptoms Count:</strong><br>${this.currentSymptoms.length}</p>
            <p><strong>Conversation Step:</strong><br>${this.conversationStep}</p>
            <p><strong>Started:</strong><br>${startTime}</p>
            <p><strong>Streaming:</strong><br>Enabled</p>
            <p><strong>Thinking:</strong><br>${this.showThinking ? 'Visible' : 'Hidden'}</p>
        `;
    }

    showChatArea() {
        this.welcomeScreen.classList.add('hidden');
        this.chatArea.classList.remove('hidden');
    }

    showWelcomeScreen() {
        this.chatArea.classList.add('hidden');
        this.welcomeScreen.classList.remove('hidden');
        this.messagesContainer.innerHTML = '';
        this.inputArea.innerHTML = '';
    }

    async resetChat() {
        this.setLoading(true);
        
        // Close any active event source
        if (this.currentEventSource) {
            this.currentEventSource.close();
            this.currentEventSource = null;
        }
        
        try {
            await fetch('/reset_chat', { method: 'POST' });
            
            this.currentChatId = null;
            this.currentSymptoms = [];
            this.clarifyingQuestions = [];
            this.conversationStep = 0;
            this.currentThinkingSection = null;
            this.initialSymptomsInput.value = '';
            
            this.showWelcomeScreen();
            this.updateSidebar();
            
        } catch (error) {
            console.error('Error resetting chat:', error);
            this.showError('Failed to reset chat session.');
        } finally {
            this.setLoading(false);
        }
    }

    setLoading(loading) {
        this.isLoading = loading;
        
        if (loading) {
            this.loadingIndicator.classList.remove('hidden');
            this.startChatBtn.disabled = true;
            this.newChatBtn.disabled = true;
        } else {
            this.loadingIndicator.classList.add('hidden');
            this.startChatBtn.disabled = false;
            this.newChatBtn.disabled = false;
        }
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorModal.classList.remove('hidden');
    }

    hideError() {
        this.errorModal.classList.add('hidden');
    }

    scrollToBottom() {
        setTimeout(() => {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }, 100);
    }

    async updateConnectionStatus() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            if (response.ok && data.status === 'healthy') {
                this.connectionStatus.innerHTML = `
                    <i class="fas fa-circle" style="color: #27ae60;"></i>
                    <span>Connected</span>
                `;
            } else {
                throw new Error('Unhealthy response');
            }
        } catch (error) {
            this.connectionStatus.innerHTML = `
                <i class="fas fa-circle" style="color: #e74c3c;"></i>
                <span>Disconnected</span>
            `;
        }
        
        // Check connection every 30 seconds
        setTimeout(() => this.updateConnectionStatus(), 30000);
    }
}

// Initialize the application
const app = new MedicalIntakeApp();

// Add some utility functions for global access
window.app = app;

// Loading screen fade out
document.addEventListener('DOMContentLoaded', () => {
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.5s ease-in-out';
    
    setTimeout(() => {
        document.body.style.opacity = '1';
    }, 100);
});

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl+N for new chat
    if (e.ctrlKey && e.key === 'n') {
        e.preventDefault();
        app.resetChat();
    }
    
    // Escape to close modal
    if (e.key === 'Escape') {
        app.hideError();
    }

    // Toggle thinking with Ctrl+T
    if (e.ctrlKey && e.key === 't') {
        e.preventDefault();
        app.thinkingToggle.checked = !app.thinkingToggle.checked;
        app.showThinking = app.thinkingToggle.checked;
        app.updateThinkingVisibility();
    }
});

// Add page visibility change handler
document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
        app.updateConnectionStatus();
    }
});

// Prevent accidental page refresh during active chat
window.addEventListener('beforeunload', (e) => {
    if (app.currentChatId) {
        e.preventDefault();
        e.returnValue = 'You have an active intake session. Are you sure you want to leave?';
    }
}); 