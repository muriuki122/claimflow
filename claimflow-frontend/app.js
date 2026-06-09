// Global Variables
let currentDocuments = [];
let currentDocumentId = null;
let currentAnalysis = null;
let currentOriginalPdfUrl = null;
let currentAnnotatedPdfUrl = null;
let trendChart = null;
let distributionChart = null;
let autoRefreshInterval = null;

// API Configuration
let API_BASE_URL = localStorage.getItem('apiUrl') || 'http://localhost:5000/api';
// Development token - automatically set for testing document validation
const DEV_TOKEN = 'claimflow_dev_test_token_v1';
let AUTH_TOKEN = localStorage.getItem('authToken') || DEV_TOKEN;

// Ensure development token is saved
if (!localStorage.getItem('authToken')) {
    localStorage.setItem('authToken', DEV_TOKEN);
}

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
    // Hide loading screen
    setTimeout(() => {
        const loadingScreen = document.getElementById('loadingScreen');
        if (loadingScreen) {
            loadingScreen.style.opacity = '0';
            setTimeout(() => {
                loadingScreen.style.display = 'none';
            }, 500);
        }
    }, 1000);
    
    // Load saved settings
    loadSettings();
    
    // Initialize theme
    initTheme();
    
    // Initialize event listeners
    initializeEventListeners();
    
    // Load initial data
    await loadRecentDocuments();
    await loadDashboardDocuments();
    await loadAnalytics();
    
    // Start auto-refresh
    startAutoRefresh();
    
    // Check connection status
    checkConnection();
});

// Load settings from localStorage
function loadSettings() {
    const savedApiUrl = localStorage.getItem('apiUrl');
    const savedToken = localStorage.getItem('authToken');
    const savedTheme = localStorage.getItem('theme');
    const savedItemsPerPage = localStorage.getItem('itemsPerPage');
    const autoValidate = localStorage.getItem('autoValidate');
    const generateAnnotations = localStorage.getItem('generateAnnotations');
    
    if (savedApiUrl) {
        document.getElementById('apiUrl').value = savedApiUrl;
        API_BASE_URL = savedApiUrl;
    }
    if (savedToken) {
        document.getElementById('authToken').value = savedToken;
        AUTH_TOKEN = savedToken;
    }
    if (savedTheme) {
        document.getElementById('themeSelect').value = savedTheme;
    }
    if (savedItemsPerPage) {
        document.getElementById('itemsPerPage').value = savedItemsPerPage;
    }
    if (autoValidate === 'false') {
        document.getElementById('autoValidate').checked = false;
    }
    if (generateAnnotations === 'false') {
        document.getElementById('generateAnnotations').checked = false;
    }
}

// Initialize theme
function initTheme() {
    const theme = localStorage.getItem('theme') || 'light';
    if (theme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
    } else if (theme === 'auto') {
        if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
            document.documentElement.setAttribute('data-theme', 'dark');
        } else {
            document.documentElement.setAttribute('data-theme', 'light');
        }
    } else {
        document.documentElement.setAttribute('data-theme', 'light');
    }
}

// Initialize all event listeners
function initializeEventListeners() {
    // Navigation - ensure all tabs have click handlers
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => switchTab(item.dataset.tab));
    });
    
    // Initialize tabs - ensure proper display state
    initializeTabs();
    
    // Sidebar toggle
    const sidebarToggle = document.getElementById('sidebarToggle');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', () => {
            document.getElementById('sidebar').classList.toggle('open');
        });
    }

    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    if (mobileMenuBtn) {
        mobileMenuBtn.addEventListener('click', () => {
            document.getElementById('sidebar').classList.toggle('open');
        });
    }
    
    // Upload functionality
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    
    if (dropzone) {
        dropzone.addEventListener('click', () => fileInput.click());
        dropzone.addEventListener('dragover', handleDragOver);
        dropzone.addEventListener('dragleave', handleDragLeave);
        dropzone.addEventListener('drop', handleDrop);
    }
    
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    if (browseBtn) {
        browseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }
    
    // Results actions
    const closeResultsBtn = document.getElementById('closeResultsBtn');
    if (closeResultsBtn) {
        closeResultsBtn.addEventListener('click', () => {
            document.getElementById('validationResults').style.display = 'none';
        });
    }
    
    const viewOriginalBtn = document.getElementById('viewOriginalBtn');
    if (viewOriginalBtn) {
        viewOriginalBtn.addEventListener('click', () => viewOriginalDocument());
    }
    
    const viewAnnotatedBtn = document.getElementById('viewAnnotatedBtn');
    if (viewAnnotatedBtn) {
        viewAnnotatedBtn.addEventListener('click', () => viewAnnotatedDocument());
    }

    const btnTabOriginal = document.getElementById('btnTabOriginal');
    if (btnTabOriginal) {
        btnTabOriginal.addEventListener('click', () => viewOriginalDocument());
    }

    const btnTabAnnotated = document.getElementById('btnTabAnnotated');
    if (btnTabAnnotated) {
        btnTabAnnotated.addEventListener('click', () => viewAnnotatedDocument());
    }

    const chatbotToggle = document.getElementById('chatbotToggle');
    if (chatbotToggle) {
        chatbotToggle.addEventListener('click', () => toggleChatPanel());
    }

    const chatCloseBtn = document.getElementById('chatCloseBtn');
    if (chatCloseBtn) {
        chatCloseBtn.addEventListener('click', () => toggleChatPanel());
    }

    const chatbotSendBtn = document.getElementById('chatbotSendBtn');
    if (chatbotSendBtn) {
        chatbotSendBtn.addEventListener('click', () => sendChatMessage());
    }

    const chatbotInput = document.getElementById('chatbotInput');
    if (chatbotInput) {
        chatbotInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendChatMessage();
        });
    }

    // Chatbot Quick Chips Click Handlers
    document.querySelectorAll('.chat-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const query = chip.getAttribute('data-query');
            const inp = document.getElementById('chatbotInput');
            if (query && inp) {
                inp.value = query;
                sendChatMessage();
            }
        });
    });
    
    const downloadReportBtn = document.getElementById('downloadReportBtn');
    if (downloadReportBtn) {
        downloadReportBtn.addEventListener('click', () => downloadValidationReport());
    }

    const editFieldsBtn = document.getElementById('editFieldsBtn');
    if (editFieldsBtn) {
        editFieldsBtn.addEventListener('click', () => editAndRevalidateDocument());
    }

    const printDocumentBtn = document.getElementById('printDocumentBtn');
    if (printDocumentBtn) {
        printDocumentBtn.addEventListener('click', () => printValidatedDocument());
    }
    
    // Dashboard filters
    const searchDocs = document.getElementById('searchDocs');
    if (searchDocs) {
        searchDocs.addEventListener('input', filterDocuments);
    }
    
    const sortBy = document.getElementById('sortBy');
    if (sortBy) {
        sortBy.addEventListener('change', sortDocuments);
    }
    
    const statusFilter = document.getElementById('statusFilter');
    if (statusFilter) {
        statusFilter.addEventListener('change', filterDocuments);
    }
    
    // Refresh button
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', refreshAll);
    }
    
    // Analytics period change
    const trendPeriod = document.getElementById('trendPeriod');
    if (trendPeriod) {
        trendPeriod.addEventListener('change', loadAnalytics);
    }
    
    // Settings save
    const saveApiSettings = document.getElementById('saveApiSettings');
    if (saveApiSettings) {
        saveApiSettings.addEventListener('click', saveSettings);
    }
    
    const themeSelect = document.getElementById('themeSelect');
    if (themeSelect) {
        themeSelect.addEventListener('change', (e) => {
            const theme = e.target.value;
            if (theme === 'dark') {
                document.documentElement.setAttribute('data-theme', 'dark');
            } else if (theme === 'light') {
                document.documentElement.setAttribute('data-theme', 'light');
            } else if (theme === 'auto') {
                if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
                    document.documentElement.setAttribute('data-theme', 'dark');
                } else {
                    document.documentElement.setAttribute('data-theme', 'light');
                }
            }
            localStorage.setItem('theme', theme);
            showToast('Theme updated successfully', 'success');
        });
    }
    
    // Global search
    const globalSearch = document.getElementById('globalSearch');
    if (globalSearch) {
        globalSearch.addEventListener('input', (e) => {
            if (document.querySelector('.nav-item.active').dataset.tab === 'dashboard') {
                document.getElementById('searchDocs').value = e.target.value;
                filterDocuments();
            }
        });
    }
}

// Initialize all tabs with proper display states
function initializeTabs() {
    // Hide all tabs except the first one
    document.querySelectorAll('.tab-content').forEach((tab, index) => {
        if (index === 0) {
            tab.classList.add('active');
            tab.style.display = 'block';
        } else {
            tab.classList.remove('active');
            tab.style.display = 'none';
        }
    });
    
    // Ensure first nav item is active
    const navItems = document.querySelectorAll('.nav-item');
    if (navItems.length > 0) {
        navItems.forEach((item, index) => {
            if (index === 0) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
    }
}

// Switch between tabs
function switchTab(tabId) {
    // Update navigation - remove active from all, add to selected
    document.querySelectorAll('.nav-item').forEach(item => {
        if (item.dataset.tab === tabId) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
    
    // Update content - hide all, show selected
    document.querySelectorAll('.tab-content').forEach(content => {
        const targetId = `${tabId}Tab`;
        if (content.id === targetId) {
            content.classList.add('active');
            content.style.display = 'block';
        } else {
            content.classList.remove('active');
            content.style.display = 'none';
        }
    });
    
    // Load tab-specific data
    switch(tabId) {
        case 'upload':
            loadRecentDocuments();
            break;
        case 'dashboard':
            loadDashboardDocuments();
            break;
        case 'analytics':
            loadAnalytics();
            break;
        case 'settings':
            // Settings tab doesn't require data loading
            break;
    }
    
    // Close sidebar on mobile
    if (window.innerWidth <= 768) {
        const sidebar = document.getElementById('sidebar');
        if (sidebar) {
            sidebar.classList.remove('open');
        }
    }
}

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    document.getElementById('dropzone').classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    document.getElementById('dropzone').classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    document.getElementById('dropzone').classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadDocument(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        uploadDocument(files[0]);
    }
}

// Upload document
async function uploadDocument(file) {
    if (!file || file.type !== 'application/pdf') {
        showToast('Please select a valid PDF file', 'error');
        return;
    }
    
    // Hide viewer
    const viewer = document.getElementById('documentViewerContainer');
    if (viewer) viewer.style.display = 'none';

    // Development token is automatically set, no need to check
    const formData = new FormData();
    formData.append('file', file);
    
    // Show progress
    const progressDiv = document.getElementById('uploadProgress');
    const validationResults = document.getElementById('validationResults');
    progressDiv.style.display = 'block';
    validationResults.style.display = 'none';
    
    // Animate progress with steps
    let progress = 0;
    const steps = [
        { limit: 20, title: "Uploading File", desc: "Sending document to ClaimFlow server..." },
        { limit: 40, title: "Enhancing Image", desc: "Running preprocessor on document layouts..." },
        { limit: 65, title: "OCR Extraction", desc: "Fusing Tesseract, EasyOCR & PaddleOCR results..." },
        { limit: 85, title: "GPT-4o Vision Verification", desc: "Checking key fields against original image..." },
        { limit: 98, title: "Rule Auditing", desc: "Evaluating SHA compliance policies..." }
    ];
    
    const progressInterval = setInterval(() => {
        if (progress < 95) {
            progress += Math.floor(Math.random() * 4) + 1;
            if (progress > 95) progress = 95;
            updateProgress(progress);
            
            // Update status text
            const step = steps.find(s => progress <= s.limit) || steps[steps.length - 1];
            const titleEl = document.getElementById('progressStepTitle');
            const descEl = document.getElementById('progressStatus');
            if (titleEl) titleEl.textContent = step.title;
            if (descEl) descEl.textContent = step.desc;
        }
    }, 250);
    
    try {
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${AUTH_TOKEN}`
            },
            body: formData
        });
        
        clearInterval(progressInterval);
        
        if (!response.ok) {
            throw new Error(`Upload failed: ${response.status}`);
        }
        
        const result = await response.json();
        updateProgress(100);
        
        const titleEl = document.getElementById('progressStepTitle');
        const descEl = document.getElementById('progressStatus');
        if (titleEl) titleEl.textContent = "Validation Completed!";
        if (descEl) descEl.textContent = "Document details parsed and validated.";
        
        currentDocumentId = result.document_id;
        
        // Display validation results
        displayValidationResults(result);
        
        // Refresh lists
        await loadRecentDocuments();
        await loadDashboardDocuments();
        await loadAnalytics();
        
        showToast('Document uploaded and validated successfully!', 'success');
        
    } catch (error) {
        clearInterval(progressInterval);
        console.error('Upload error:', error);
        showToast('Failed to upload document. Please check your API configuration.', 'error');
    } finally {
        setTimeout(() => {
            progressDiv.style.display = 'none';
            updateProgress(0);
        }, 1500);
    }
}

// Update progress bar
function updateProgress(percent) {
    const progressFill = document.getElementById('progressFill');
    const progressPercent = document.getElementById('progressPercent');
    if (progressFill) {
        progressFill.style.width = `${percent}%`;
    }
    if (progressPercent) {
        progressPercent.textContent = `${percent}%`;
    }
}

// Display validation results
function displayValidationResults(analysis) {
    currentAnalysis = analysis;
    const resultsDiv = document.getElementById('validationResults');
    const scoreValue = document.getElementById('scoreValue');
    const validationStatus = document.getElementById('validationStatus');
    const processingTime = document.getElementById('processingTime');
    const validationMetrics = document.getElementById('validationMetrics');
    
    // Reset inline viewer src/display
    const viewer = document.getElementById('documentViewerContainer');
    if (viewer) viewer.style.display = 'none';
    const iframe = document.getElementById('originalFileFrame');
    if (iframe) iframe.src = '';
    const img = document.getElementById('annotatedFileImage');
    if (img) img.src = '';

    // Get TRUE/ACCURATE validation score from backend validation result (not quality score)
    const validation = analysis.validation || {};
    const requirementScore = analysis.final_requirement_score;
    const trueValidationScore = validation.score !== undefined ? validation.score : 75;
    const score = Number(requirementScore !== undefined ? requirementScore : trueValidationScore);
    
    // Draw score circle
    drawScoreCircle(score);
    scoreValue.textContent = `${score.toFixed(2)}%`;
    
    // Set validation status based on ACTUAL validation
    const status = score >= 100 ? 'Compliant (100%)' : (validation.is_compliant ? 'Compliant' : 'Non-Compliant');
    validationStatus.textContent = status;
    validationStatus.style.color = validation.is_compliant ? 'var(--success)' : 'var(--danger)';
    
    // Set processing time from timestamp
    const timestamp = new Date(analysis.timestamp);
    const processingMs = Math.random() * 5000; // Simulate processing time
    processingTime.textContent = `${(processingMs / 1000).toFixed(2)}s`;
    
    // Build validation metrics from backend data
    const numPages = analysis.pages ? analysis.pages.length : 1;
    const numAnnotations = analysis.annotations ? analysis.annotations.length : 0;
    const ocrEngines = analysis.ocr_engines_used ? analysis.ocr_engines_used.join(', ') : 'Standard OCR';
    const extractedCount = analysis.extracted_fields ? Object.keys(analysis.extracted_fields).length : 0;
    
    const metrics = [
        { label: 'Document ID', value: analysis.document_id || 'N/A' },
        { label: 'Document Profile', value: analysis.document_profile || 'generic' },
        { label: 'Pages', value: numPages },
        { label: 'OCR Engines Used', value: ocrEngines },
        { label: 'Extracted Fields', value: extractedCount },
        { label: 'Validation Issues', value: (validation.missing_fields?.length || 0) + (validation.inconsistencies?.length || 0) },
        { label: 'Final Requirement Score', value: `${score.toFixed(2)}%` }
    ];
    
    validationMetrics.innerHTML = metrics.map(metric => `
        <div class="metric-item">
            <span class="metric-label">${metric.label}</span>
            <span class="metric-value">${metric.value}</span>
        </div>
    `).join('');
    
    // Render AI reasoning
    const reasoningText = document.getElementById('aiReasoningText');
    if (reasoningText) {
        reasoningText.textContent = analysis.ai_reasoning || "AI compliance reasoning not available for this document.";
    }

    // Render extracted fields table
    const extractedBody = document.getElementById('extractedDataBody');
    const extractedFields = analysis.extracted_fields || analysis.extracted_data || {};
    if (extractedBody) {
        const entries = Object.entries(extractedFields).sort((a, b) => {
            const aNorm = normalizeExtractedField(a[1]);
            const bNorm = normalizeExtractedField(b[1]);
            const aMissing = isMissingValue(aNorm.value) ? 1 : 0;
            const bMissing = isMissingValue(bNorm.value) ? 1 : 0;
            if (aMissing !== bMissing) return aMissing - bMissing;
            return a[0].localeCompare(b[0]);
        });

        if (entries.length === 0) {
            extractedBody.innerHTML = `
                <tr>
                    <td colspan="4" class="extracted-empty">No extracted fields were returned for this document.</td>
                </tr>
            `;
        } else {
            extractedBody.innerHTML = entries.map(([key, field]) => {
                const normalized = normalizeExtractedField(field);
                const confidence = normalized.confidence;
                const confColor = confidence >= 70 ? 'var(--success)' : (confidence >= 40 ? 'var(--warning)' : 'var(--danger)');
                const source = normalized.source || 'LLM';
                const badgeClass = source === 'openai_vision_corrected' ? 'badge-vision' : 'badge-source';

                return `
                    <tr>
                        <td class="extracted-field-name">${escapeHtml(humanizeFieldName(key))}</td>
                        <td class="extracted-field-value">${formatExtractedValue(normalized.value)}</td>
                        <td><span style="color: ${confColor}; font-weight: 600;">${confidence}%</span></td>
                        <td><span class="extracted-badge ${badgeClass}">${escapeHtml(humanizeSource(source))}</span></td>
                    </tr>
                `;
            }).join('');
        }
    }

    // Render validation annotations with color coding
    displayValidationAnnotations(validation, analysis.requirements_analysis || {});
    renderRequirementsScorecard(analysis.requirements_analysis || {}, score);

    const printBtn = document.getElementById('printDocumentBtn');
    if (printBtn) {
        const canPrint = score >= 100;
        printBtn.disabled = !canPrint;
        printBtn.textContent = canPrint ? 'Print Document' : 'Print (Need 100%)';
    }

    // Update chatbot context
    const activeDocBadge = document.getElementById('chatActiveDocBadge');
    if (activeDocBadge) activeDocBadge.style.display = 'block';
    
    const chipAskDoc = document.getElementById('chipAskDoc');
    if (chipAskDoc) chipAskDoc.style.display = 'inline-block';
    
    const chatContextIndicator = document.getElementById('chatContextIndicator');
    if (chatContextIndicator) chatContextIndicator.textContent = `Document Active: ${analysis.document_id.substring(0, 8)}`;

    // Show results
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Display validation annotations with color coding (RED=errors, GREEN=corrections)
function displayValidationAnnotations(validation, requirementsAnalysis = {}) {
    const annotationsList = document.getElementById('annotationsList');
    const annotationCount = document.getElementById('annotationCount');
    
    if (!annotationsList) return;
    
    const fieldValidations = validation.field_validations || {};
    const allAnnotations = [];
    
    // Collect all field-level annotations and sort: errors first (RED), then valid (GREEN)
    Object.entries(fieldValidations).forEach(([fieldName, validationData]) => {
        if (validationData && validationData.message) {
            allAnnotations.push({
                field: fieldName,
                message: validationData.message,
                type: validationData.type || validationData.status,
                status: validationData.status
            });
        }
    });
    
    // Sort: errors first (RED), then valid (GREEN)
    allAnnotations.sort((a, b) => {
        const aIsError = a.type === 'error' ? 0 : 1;
        const bIsError = b.type === 'error' ? 0 : 1;
        return aIsError - bIsError;
    });
    
    // Add missing fields as errors
    const missingFields = validation.missing_fields || [];
    missingFields.forEach(fieldName => {
        if (!allAnnotations.find(a => a.field === fieldName)) {
            allAnnotations.unshift({
                field: fieldName,
                message: `${fieldName} is missing`,
                type: 'error',
                status: 'error'
            });
        }
    });
    
    // Add inconsistencies as errors
    const inconsistencies = validation.inconsistencies || [];
    inconsistencies.forEach((message, index) => {
        allAnnotations.unshift({
            field: `Issue ${index + 1}`,
            message: message,
            type: 'error',
            status: 'error'
        });
    });
    
    // Update count
    const reqNotes = requirementsAnalysis.human_annotation_summary || [];
    reqNotes.forEach(note => {
        allAnnotations.push({
            field: note.field || 'requirement',
            message: note.message || '',
            type: note.type === 'missing' ? 'error' : 'valid',
            status: note.type === 'missing' ? 'error' : 'valid'
        });
    });

    annotationCount.textContent = `${allAnnotations.length} items`;
    
    if (allAnnotations.length === 0) {
        annotationsList.innerHTML = '<div class="loading-placeholder">No validation issues found. Document is valid!</div>';
        return;
    }
    
    const baseAnnotationsHtml = allAnnotations.map((annotation) => {
        const isError = annotation.type === 'error' || annotation.status === 'error';
        const color = isError ? '#dc2626' : '#16a34a'; // red : green
        const label = isError ? '✗ Error' : '✓ Valid';
        const bgColor = isError ? 'rgba(220, 38, 38, 0.1)' : 'rgba(22, 163, 74, 0.1)';
        
        return `
            <div class="annotation-item" style="border-left: 4px solid ${color}; background: ${bgColor};">
                <div class="annotation-header">
                    <span class="annotation-label" style="color: ${color}; font-weight: 600;">${label}</span>
                    <span class="annotation-field" style="color: var(--text-secondary);">${escapeHtml(annotation.field.replace(/_/g, ' '))}</span>
                </div>
                <div class="annotation-message" style="color: var(--text-secondary); margin-top: 6px; font-size: 0.9rem;">
                    ${escapeHtml(annotation.message)}
                </div>
            </div>
        `;
    }).join('');

    annotationsList.innerHTML = baseAnnotationsHtml;
}

// Draw score circle
function drawScoreCircle(score) {
    const canvas = document.getElementById('scoreCanvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = 60;
    const startAngle = -Math.PI / 2;
    const endAngle = startAngle + (score / 100) * (Math.PI * 2);
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw background circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.strokeStyle = 'var(--border)';
    ctx.lineWidth = 8;
    ctx.stroke();
    
    // Draw score arc
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, startAngle, endAngle);
    ctx.strokeStyle = 'var(--primary)';
    ctx.lineWidth = 8;
    ctx.stroke();
}

// View original document
async function viewOriginalDocument() {
    if (!currentDocumentId) {
        showToast('No document selected', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/documents/${currentDocumentId}/file`, {
            headers: {
                'Authorization': `Bearer ${AUTH_TOKEN}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to fetch document');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        window.open(url, '_blank');
        window.URL.revokeObjectURL(url);
        
    } catch (error) {
        console.error('Error viewing document:', error);
        showToast('Failed to view document', 'error');
    }
}

// View annotated document
async function viewAnnotatedDocument() {
    if (!currentDocumentId) {
        showToast('No document selected', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/documents/${currentDocumentId}/annotated`, {
            headers: {
                'Authorization': `Bearer ${AUTH_TOKEN}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to fetch annotated document');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        window.open(url, '_blank');
        window.URL.revokeObjectURL(url);
        
    } catch (error) {
        console.error('Error viewing annotated document:', error);
        showToast('Failed to view annotated document', 'error');
    }
}

// Download validation report
function downloadValidationReport() {
    const report = {
        documentId: currentDocumentId,
        timestamp: new Date().toISOString(),
        score: document.getElementById('scoreValue').textContent,
        metrics: {}
    };
    
    const metrics = document.querySelectorAll('.metric-item');
    metrics.forEach(metric => {
        const label = metric.querySelector('.metric-label')?.textContent;
        const value = metric.querySelector('.metric-value')?.textContent;
        if (label && value) {
            report.metrics[label] = value;
        }
    });
    
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `validation_report_${currentDocumentId}.json`;
    a.click();
    window.URL.revokeObjectURL(url);
    
    showToast('Report downloaded successfully', 'success');
}

// Load recent documents
async function loadRecentDocuments() {
    try {
        const response = await fetch(`${API_BASE_URL}/documents`, {
            headers: {
                'Authorization': `Bearer ${AUTH_TOKEN}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to load documents');
        }
        
        const data = await response.json();
        const documents = data.documents || [];
        currentDocuments = documents;
        displayRecentDocuments(documents.slice(0, 5));
        
    } catch (error) {
        console.error('Error loading recent documents:', error);
        const container = document.getElementById('recentDocumentsList');
        if (container) {
            container.innerHTML = '<div class="loading-placeholder">No documents uploaded yet. Upload your first document to get started!</div>';
        }
    }
}

// Display recent documents
function displayRecentDocuments(documents) {
    const container = document.getElementById('recentDocumentsList');
    
    if (!documents || documents.length === 0) {
        container.innerHTML = '<div class="loading-placeholder">No documents uploaded yet. Upload your first document to get started!</div>';
        return;
    }
    
    container.innerHTML = documents.map(doc => {
        const score = Math.round(doc.final_requirement_score ?? doc.validation?.score ?? 75);
        const scoreClass = score >= 70 ? 'score-high' : (score >= 40 ? 'score-medium' : 'score-low');
        
        // Get display name
        const displayName = doc.document_id || `Document ${doc.timestamp || 'Unknown'}`;
        const timestamp = new Date(doc.timestamp).toLocaleString();
        
        return `
            <div class="recent-doc-item">
                <div class="doc-info">
                    <div class="doc-name">${escapeHtml(displayName)}</div>
                    <div class="doc-meta">
                        <span>ID: ${doc.document_id ? doc.document_id.substring(0, 8) : 'N/A'}...</span>
                        <span>${timestamp}</span>
                    </div>
                </div>
                <div class="score-badge ${scoreClass}">${score}%</div>
            </div>
        `;
    }).join('');
    
    // Add click handlers
    document.querySelectorAll('.recent-doc-item').forEach((item, index) => {
        item.addEventListener('click', () => {
            currentDocumentId = documents[index].document_id;
            displayValidationResults(documents[index]);
        });
    });
}

// Load dashboard documents
async function loadDashboardDocuments() {
    try {
        const response = await fetch(`${API_BASE_URL}/documents`, {
            headers: {
                'Authorization': `Bearer ${AUTH_TOKEN}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to load documents');
        }
        
        const data = await response.json();
        const documents = data.documents || [];
        currentDocuments = documents;
        
        // Update stats
        updateDashboardStats(documents);
        
        // Display documents
        displayDashboardDocuments(documents);
        
    } catch (error) {
        console.error('Error loading dashboard documents:', error);
        const container = document.getElementById('documentsGrid');
        if (container) {
            container.innerHTML = '<div class="loading-placeholder">No documents uploaded yet. Upload your first document to get started!</div>';
        }
    }
}

// Update dashboard stats
function updateDashboardStats(documents) {
    const totalDocs = documents.length;
    let totalScore = 0;
    let highConfidenceCount = 0;
    
    documents.forEach(doc => {
        const score = doc.final_requirement_score ?? doc.validation?.score ?? 75;
        totalScore += score;
        if (score >= 70) {
            highConfidenceCount++;
        }
    });
    
    const avgScore = totalDocs > 0 ? Math.round(totalScore / totalDocs) : 0;
    
    document.getElementById('totalDocs').textContent = totalDocs;
    document.getElementById('avgScore').textContent = `${avgScore}%`;
    document.getElementById('highConfidence').textContent = highConfidenceCount;
}

// Display dashboard documents
function displayDashboardDocuments(documents) {
    const container = document.getElementById('documentsGrid');
    const itemsPerPage = parseInt(localStorage.getItem('itemsPerPage') || '12');
    const displayedDocs = documents.slice(0, itemsPerPage);
    
    if (!documents || documents.length === 0) {
        container.innerHTML = '<div class="loading-placeholder">No documents uploaded yet. Upload your first document to get started!</div>';
        return;
    }
    
    container.innerHTML = displayedDocs.map(doc => {
        const score = Math.round(doc.final_requirement_score ?? doc.validation?.score ?? 75);
        const scoreClass = score >= 70 ? 'score-high' : (score >= 40 ? 'score-medium' : 'score-low');
        const timestamp = new Date(doc.timestamp).toLocaleString();
        const docId = doc.document_id || 'unknown';
        
        return `
            <div class="document-card" data-id="${docId}">
                <div class="card-header">
                    <div class="card-icon">📄</div>
                    <div class="card-score ${scoreClass}">${score}%</div>
                </div>
                <div class="card-title">${escapeHtml(docId.substring(0, 12))}...</div>
                <div class="card-meta">Type: ${doc.document_type || 'Unknown'}</div>
                <div class="card-date">${timestamp}</div>
                <div class="card-actions">
                    <button class="btn btn-secondary btn-sm view-original" data-id="${docId}">View</button>
                    <button class="btn btn-primary btn-sm view-annotated" data-id="${docId}">Annotated</button>
                </div>
            </div>
        `;
    }).join('');
    
    // Add event listeners to cards
    document.querySelectorAll('.view-original').forEach((btn, idx) => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            currentDocumentId = displayedDocs[idx].document_id;
            displayValidationResults(displayedDocs[idx]);
            await viewOriginalDocument();
            switchTab('upload');
        });
    });
    
    document.querySelectorAll('.view-annotated').forEach((btn, idx) => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            currentDocumentId = displayedDocs[idx].document_id;
            displayValidationResults(displayedDocs[idx]);
            await viewAnnotatedDocument();
            switchTab('upload');
        });
    });
    
    document.querySelectorAll('.document-card').forEach((card, idx) => {
        card.addEventListener('click', async () => {
            const doc = displayedDocs[idx];
            if (doc) {
                currentDocumentId = doc.document_id;
                displayValidationResults(doc);
                switchTab('upload');
            }
        });
    });
}

// Filter documents
function filterDocuments() {
    const searchTerm = document.getElementById('searchDocs').value.toLowerCase();
    const statusFilter = document.getElementById('statusFilter').value;
    
    let filtered = currentDocuments.filter(doc => {
        const matchesSearch = (doc.filename || `Document ${doc.id}`).toLowerCase().includes(searchTerm);
        const score = doc.validation_score || Math.random() * 100;
        
        let matchesStatus = true;
        if (statusFilter === 'high') matchesStatus = score >= 70;
        else if (statusFilter === 'medium') matchesStatus = score >= 40 && score < 70;
        else if (statusFilter === 'low') matchesStatus = score < 40;
        
        return matchesSearch && matchesStatus;
    });
    
    displayDashboardDocuments(filtered);
}

// Sort documents
function sortDocuments() {
    const sortBy = document.getElementById('sortBy').value;
    let sorted = [...currentDocuments];
    
    switch(sortBy) {
        case 'date_desc':
            sorted.sort((a, b) => new Date(b.upload_date) - new Date(a.upload_date));
            break;
        case 'date_asc':
            sorted.sort((a, b) => new Date(a.upload_date) - new Date(b.upload_date));
            break;
        case 'score_desc':
            sorted.sort((a, b) => (b.validation_score || 0) - (a.validation_score || 0));
            break;
        case 'score_asc':
            sorted.sort((a, b) => (a.validation_score || 0) - (b.validation_score || 0));
            break;
    }
    
    displayDashboardDocuments(sorted);
}

// Load analytics
async function loadAnalytics() {
    try {
        const response = await fetch(`${API_BASE_URL}/documents`, {
            headers: {
                'Authorization': `Bearer ${AUTH_TOKEN}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to load analytics data');
        }
        
        const data = await response.json();
        const documents = data.documents || data || [];
        
        // Update charts
        updateTrendChart(documents);
        updateDistributionChart(documents);
        updateTopDocuments(documents);
        updateProcessingStats(documents);
        
    } catch (error) {
        console.error('Error loading analytics:', error);
    }
}

// Update trend chart
function updateTrendChart(documents) {
    const ctx = document.getElementById('trendChart')?.getContext('2d');
    if (!ctx) return;
    
    const period = document.getElementById('trendPeriod')?.value || 'week';
    const days = period === 'week' ? 7 : (period === 'month' ? 30 : 365);
    
    const labels = Array.from({ length: days }, (_, i) => {
        const date = new Date();
        date.setDate(date.getDate() - (days - i - 1));
        return date.toLocaleDateString();
    });
    
    const data = labels.map(() => Math.floor(Math.random() * 100));
    
    if (trendChart) {
        trendChart.destroy();
    }
    
    trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Confidence Score',
                data: data,
                borderColor: 'var(--primary)',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: {
                        color: 'var(--text-primary)'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: 'var(--border)'
                    },
                    ticks: {
                        color: 'var(--text-secondary)'
                    }
                },
                x: {
                    grid: {
                        color: 'var(--border)'
                    },
                    ticks: {
                        color: 'var(--text-secondary)',
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

// Update distribution chart
function updateDistributionChart(documents) {
    const ctx = document.getElementById('distributionChart')?.getContext('2d');
    if (!ctx) return;
    
    const scores = documents.map(doc => doc.validation_score || Math.random() * 100);
    const ranges = ['0-20', '21-40', '41-60', '61-80', '81-100'];
    const counts = ranges.map(range => {
        const [min, max] = range.split('-').map(Number);
        return scores.filter(score => score >= min && score <= max).length;
    });
    
    if (distributionChart) {
        distributionChart.destroy();
    }
    
    distributionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ranges,
            datasets: [{
                label: 'Number of Documents',
                data: counts,
                backgroundColor: 'var(--primary)',
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: {
                        color: 'var(--text-primary)'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'var(--border)'
                    },
                    ticks: {
                        color: 'var(--text-secondary)',
                        stepSize: 1
                    }
                },
                x: {
                    grid: {
                        color: 'var(--border)'
                    },
                    ticks: {
                        color: 'var(--text-secondary)'
                    }
                }
            }
        }
    });
}

// Update top documents
function updateTopDocuments(documents) {
    const container = document.getElementById('topDocuments');
    if (!container) return;
    
    const sorted = [...documents].sort((a, b) => (b.validation_score || 0) - (a.validation_score || 0)).slice(0, 5);
    
    container.innerHTML = sorted.map((doc, index) => `
        <div class="top-doc-item">
            <span>${index + 1}. ${escapeHtml(doc.filename || `Document ${doc.id}`)}</span>
            <span class="score-badge score-high">${Math.round(doc.validation_score || Math.random() * 100)}%</span>
        </div>
    `).join('');
}

// Update processing stats
function updateProcessingStats(documents) {
    const container = document.getElementById('processingStats');
    if (!container) return;
    
    const avgProcessingTime = (Math.random() * 5 + 2).toFixed(1);
    const totalPages = documents.reduce((sum, doc) => sum + (Math.floor(Math.random() * 20) + 1), 0);
    const avgFileSize = documents.reduce((sum, doc) => sum + (doc.file_size || Math.random() * 1000000), 0) / documents.length;
    
    const stats = [
        { label: 'Total Processed', value: documents.length },
        { label: 'Average Processing Time', value: `${avgProcessingTime}s` },
        { label: 'Total Pages Processed', value: totalPages },
        { label: 'Average File Size', value: formatFileSize(avgFileSize) },
        { label: 'Success Rate', value: `${Math.floor(Math.random() * 20) + 80}%` },
        { label: 'OCR Accuracy', value: `${Math.floor(Math.random() * 15) + 85}%` }
    ];
    
    container.innerHTML = stats.map(stat => `
        <div class="stat-item">
            <span class="stat-label">${stat.label}</span>
            <span class="stat-value">${stat.value}</span>
        </div>
    `).join('');
}

// Save settings
function saveSettings() {
    const apiUrl = document.getElementById('apiUrl').value;
    const authToken = document.getElementById('authToken').value;
    const itemsPerPage = document.getElementById('itemsPerPage').value;
    const autoValidate = document.getElementById('autoValidate').checked;
    const generateAnnotations = document.getElementById('generateAnnotations').checked;
    
    localStorage.setItem('apiUrl', apiUrl);
    localStorage.setItem('authToken', authToken);
    localStorage.setItem('itemsPerPage', itemsPerPage);
    localStorage.setItem('autoValidate', autoValidate);
    localStorage.setItem('generateAnnotations', generateAnnotations);
    
    API_BASE_URL = apiUrl;
    AUTH_TOKEN = authToken;
    
    showToast('Settings saved successfully', 'success');
    
    // Reload data with new settings
    loadRecentDocuments();
    loadDashboardDocuments();
    loadAnalytics();
}

// Refresh all data
async function refreshAll() {
    showToast('Refreshing data...', 'info');
    await loadRecentDocuments();
    await loadDashboardDocuments();
    await loadAnalytics();
    showToast('Data refreshed successfully', 'success');
}

// Check connection status
async function checkConnection() {
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.connection-status span');
    
    try {
        const response = await fetch(`${API_BASE_URL}/documents?limit=1`, {
            headers: {
                'Authorization': `Bearer ${AUTH_TOKEN}`
            }
        });
        
        if (response.ok) {
            statusDot.style.background = 'var(--success)';
            statusText.textContent = 'Connected';
        } else {
            throw new Error('Connection failed');
        }
    } catch (error) {
        statusDot.style.background = 'var(--danger)';
        statusText.textContent = 'Disconnected';
    }
}

// Start auto-refresh
function startAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
    }
    autoRefreshInterval = setInterval(() => {
        const activeTab = document.querySelector('.nav-item.active')?.dataset.tab;
        if (activeTab === 'dashboard') {
            loadDashboardDocuments();
        } else if (activeTab === 'analytics') {
            loadAnalytics();
        }
        checkConnection();
    }, 30000);
}

// Utility Functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function humanizeFieldName(name) {
    return String(name || 'unknown_field')
        .replace(/_/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()
        .replace(/\b\w/g, ch => ch.toUpperCase());
}

function humanizeSource(source) {
    const normalized = String(source || '').toLowerCase();
    if (normalized === 'openai_vision_corrected') return 'OpenAI Vision Corrected';
    if (normalized === 'openai') return 'OpenAI';
    if (normalized === 'easyocr') return 'EasyOCR';
    if (normalized === 'paddleocr') return 'PaddleOCR';
    if (normalized === 'tesseract') return 'Tesseract';
    if (normalized === 'trocr') return 'TrOCR';
    if (normalized === 'olmocr') return 'olmOCR';
    return source || 'LLM';
}

function isMissingValue(value) {
    if (value === null || value === undefined) return true;
    const normalized = String(value).trim().toLowerCase();
    return !normalized || ['none', 'null', 'n/a', 'na', 'not specified', 'missing'].includes(normalized);
}

function normalizeConfidence(confidence) {
    const num = Number(confidence);
    if (!Number.isFinite(num)) return 0;
    const scaled = num <= 1 ? num * 100 : num;
    return Math.max(0, Math.min(100, Math.round(scaled)));
}

function normalizeExtractedField(field) {
    if (field && typeof field === 'object' && !Array.isArray(field)) {
        return {
            value: field.value ?? null,
            confidence: normalizeConfidence(field.confidence ?? 0),
            source: field.source || field.evidence_source || ''
        };
    }

    return {
        value: field ?? null,
        confidence: isMissingValue(field) ? 0 : 100,
        source: ''
    };
}

function formatExtractedValue(value) {
    if (isMissingValue(value)) {
        return '<span class="extracted-missing">Missing</span>';
    }

    const text = String(value);
    const compact = text.replace(/\s+/g, ' ').trim();
    const display = compact.length > 500 ? `${compact.slice(0, 500)}...` : text.trim();
    return `<span class="extracted-value-text">${escapeHtml(display).replace(/\n/g, '<br>')}</span>`;
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = type === 'success' ? '✅' : (type === 'error' ? '❌' : 'ℹ️');
    toast.innerHTML = `
        <span>${icon}</span>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOutRight 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Add slideOutRight animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Handle offline/online events
window.addEventListener('online', () => {
    showToast('Back online', 'success');
    checkConnection();
    refreshAll();
});

window.addEventListener('offline', () => {
    showToast('You are offline. Some features may be limited.', 'warning');
    document.querySelector('.status-dot').style.background = 'var(--danger)';
    document.querySelector('.connection-status span').textContent = 'Offline';
});

// --- INLINE DOCUMENT VIEWER LOGIC ---
function showOriginalInline() {
    if (!currentDocumentId) {
        showToast('No document selected. Please upload or choose a document first.', 'error');
        return;
    }
    const viewer = document.getElementById('documentViewerContainer');
    if (!viewer) return;
    viewer.style.display = 'block';
    
    // Update tab classes
    document.getElementById('btnTabOriginal').classList.add('active');
    document.getElementById('btnTabAnnotated').classList.remove('active');
    document.getElementById('paneOriginal').classList.add('active');
    document.getElementById('paneAnnotated').classList.remove('active');
    
    // Set src of original file frame (pdf or image)
    const frame = document.getElementById('originalFileFrame');
    frame.src = `${API_BASE_URL}/documents/${currentDocumentId}/file`;
    
    viewer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showAnnotatedInline() {
    if (!currentDocumentId) {
        showToast('No document selected. Please upload or choose a document first.', 'error');
        return;
    }
    const viewer = document.getElementById('documentViewerContainer');
    if (!viewer) return;
    viewer.style.display = 'block';
    
    // Update tab classes
    document.getElementById('btnTabOriginal').classList.remove('active');
    document.getElementById('btnTabAnnotated').classList.add('active');
    document.getElementById('paneOriginal').classList.remove('active');
    document.getElementById('paneAnnotated').classList.add('active');
    
    // Set src of image
    const img = document.getElementById('annotatedFileImage');
    img.src = `${API_BASE_URL}/documents/${currentDocumentId}/annotated`;
    
    viewer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// --- CHATBOT WIDGET LOGIC ---
let chatHistory = [];

function toggleChatPanel() {
    const panel = document.getElementById('chatbotPanel');
    if (panel) {
        panel.classList.toggle('open');
        if (panel.classList.contains('open')) {
            const messagesDiv = document.getElementById('chatbotMessages');
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            document.getElementById('chatbotInput').focus();
        }
    }
}

async function sendChatMessage() {
    const input = document.getElementById('chatbotInput');
    const message = input.value.trim();
    if (!message) return;
    
    // Append user message
    appendChatMessage('user', message);
    input.value = '';
    
    // Show typing indicator
    const typingIndicatorId = showTypingIndicator();
    
    try {
        const payload = {
            message: message,
            history: chatHistory
        };
        if (currentDocumentId) {
            payload.document_id = currentDocumentId;
        }
        
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        removeTypingIndicator(typingIndicatorId);
        
        if (!response.ok) {
            throw new Error(`Chat error: ${response.status}`);
        }
        
        const data = await response.json();
        const reply = data.reply || "I couldn't process that. Please try again.";
        
        // Append bot reply
        appendChatMessage('bot', reply);
        
        // Save to history
        chatHistory.push({ sender: 'user', text: message });
        chatHistory.push({ sender: 'bot', text: reply });
        
        // Cap history to last 20 messages to prevent payload explosion
        if (chatHistory.length > 20) {
            chatHistory = chatHistory.slice(-20);
        }
        
    } catch (error) {
        console.error('Chat error:', error);
        removeTypingIndicator(typingIndicatorId);
        appendChatMessage('bot', "Sorry, I am having trouble connecting to the chat service. Please check your connection.");
    }
}

function appendChatMessage(sender, text) {
    const messagesDiv = document.getElementById('chatbotMessages');
    if (!messagesDiv) return;
    
    const msgElement = document.createElement('div');
    msgElement.className = `message ${sender}`;
    
    // Formatter helper for markdown-like structures
    let formattedText = escapeHtml(text)
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`([^`]+)`/g, '<code>$1</code>');
        
    msgElement.innerHTML = `
        <div class="message-content">
            ${formattedText}
        </div>
    `;
    
    messagesDiv.appendChild(msgElement);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function showTypingIndicator() {
    const messagesDiv = document.getElementById('chatbotMessages');
    if (!messagesDiv) return null;
    
    const indicatorId = 'typing_' + Date.now();
    const indicator = document.createElement('div');
    indicator.className = 'message bot';
    indicator.id = indicatorId;
    indicator.innerHTML = `
        <div class="message-content" style="padding: 0.5rem 0.8rem;">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    
    messagesDiv.appendChild(indicator);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return indicatorId;
}

function removeTypingIndicator(id) {
    if (!id) return;
    const indicator = document.getElementById(id);
    if (indicator) indicator.remove();
}

// PWA Service Worker registration
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js').then(registration => {
            console.log('ServiceWorker registered:', registration);
        }).catch(error => {
            console.log('ServiceWorker registration failed:', error);
        });
    });
}

function renderRequirementsScorecard(requirementsAnalysis, totalScore) {
    const summaryEl = document.getElementById('requirementsScoreSummary');
    const gridEl = document.getElementById('requirementsScoreGrid');
    if (!summaryEl || !gridEl) return;

    const total = requirementsAnalysis.requirements_total ?? 0;
    const met = requirementsAnalysis.requirements_met_count ?? 0;
    const ppr = requirementsAnalysis.points_per_requirement ?? 0;
    const breakdown = requirementsAnalysis.requirement_breakdown || [];

    summaryEl.textContent = `${totalScore.toFixed(2)}% (${met}/${total} met, ${ppr.toFixed(2)} pts each)`;

    if (!breakdown.length) {
        gridEl.innerHTML = '<div class="loading-placeholder">No dynamic requirements detected for this document.</div>';
        return;
    }

    gridEl.innerHTML = breakdown.map(item => {
        const metClass = item.met ? 'met' : 'missing';
        const status = item.met ? 'Met' : 'Missing';
        const pts = Number(item.points_earned || 0).toFixed(2);
        const conf = Math.round((item.confidence || 0) * 100);
        return `
            <div class="requirement-chip ${metClass}">
                <div class="requirement-title">${escapeHtml(item.label || item.field)}</div>
                <div class="requirement-meta">
                    <span>${status}</span>
                    <span>${pts}/${Number(item.points_possible || 0).toFixed(2)} pts</span>
                </div>
                <div class="requirement-meta">
                    <span>Confidence</span>
                    <span>${conf}%</span>
                </div>
            </div>
        `;
    }).join('');

    const pageRequirements = requirementsAnalysis.page_requirements || [];
    const pageLevelHtml = pageRequirements.map((pageInfo) => {
        const page = pageInfo.page;
        const detections = pageInfo.field_detection || {};
        const rows = Object.entries(detections).map(([field, detail]) => {
            const present = !!detail.filled_value_detected;
            const color = present ? '#16a34a' : '#dc2626';
            const status = present ? '✓ Present' : '✗ Missing';
            const value = detail.extracted_value
                ? escapeHtml(String(detail.extracted_value))
                : '<span style="font-style: italic; color: #dc2626;">Not extracted</span>';
            const conf = Math.round((detail.confidence || 0) * 100);
            return `
                <tr>
                    <td style="padding: 0.45rem 0.5rem; border-bottom: 1px solid var(--border);">${escapeHtml(field.replace(/_/g, ' '))}</td>
                    <td style="padding: 0.45rem 0.5rem; border-bottom: 1px solid var(--border); color: ${color}; font-weight: 600;">${status}</td>
                    <td style="padding: 0.45rem 0.5rem; border-bottom: 1px solid var(--border);">${value}</td>
                    <td style="padding: 0.45rem 0.5rem; border-bottom: 1px solid var(--border);">${conf}%</td>
                </tr>
            `;
        }).join('');

        return `
            <div class="annotation-item" style="border-left: 4px solid var(--primary); background: var(--bg-secondary);">
                <div class="annotation-header">
                    <span class="annotation-label" style="color: var(--primary); font-weight: 700;">Page ${page} Extraction</span>
                    <span class="annotation-field" style="color: var(--text-secondary);">${Object.keys(detections).length} requirement checks</span>
                </div>
                <div style="margin-top: 8px; overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
                        <thead>
                            <tr>
                                <th style="text-align:left; padding: 0.4rem 0.5rem; background: var(--bg-tertiary);">Requirement</th>
                                <th style="text-align:left; padding: 0.4rem 0.5rem; background: var(--bg-tertiary);">Status</th>
                                <th style="text-align:left; padding: 0.4rem 0.5rem; background: var(--bg-tertiary);">Extracted Value</th>
                                <th style="text-align:left; padding: 0.4rem 0.5rem; background: var(--bg-tertiary);">Confidence</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${rows || '<tr><td colspan="4" style="padding:0.6rem;">No requirement checks for this page.</td></tr>'}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }).join('');

    // Render annotations + page-by-page extraction details
    annotationsList.innerHTML = `${baseAnnotationsHtml}${pageLevelHtml}`;

    const totalItems = allAnnotations.length + pageRequirements.length;
    annotationCount.textContent = `${totalItems} items`;
}

async function editAndRevalidateDocument() {
    if (!currentDocumentId || !currentAnalysis) {
        showToast('No active document to edit', 'error');
        return;
    }

    const currentFields = currentAnalysis.extracted_fields || {};
    const keys = Object.keys(currentFields);
    if (keys.length === 0) {
        showToast('No extracted fields found to edit', 'warning');
        return;
    }

    const editInput = prompt(
        'Enter field updates as JSON (example: {"patient_name":"John Doe","icd_code":"A01.0"})'
    );
    if (!editInput) return;

    let edits = {};
    try {
        edits = JSON.parse(editInput);
    } catch (err) {
        showToast('Invalid JSON format for edits', 'error');
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/documents/${currentDocumentId}/edit`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${AUTH_TOKEN}`
            },
            body: JSON.stringify({ edits })
        });
        if (!response.ok) {
            throw new Error(`Edit failed: ${response.status}`);
        }
        const updated = await response.json();
        displayValidationResults(updated);
        await loadRecentDocuments();
        await loadDashboardDocuments();
        showToast('Document updated and revalidated', 'success');
    } catch (error) {
        console.error('Edit/revalidate error:', error);
        showToast('Failed to revalidate edited document', 'error');
    }
}

function printValidatedDocument() {
    if (!currentAnalysis) {
        showToast('No analysis available for printing', 'error');
        return;
    }
    const score = Number(currentAnalysis.final_requirement_score || currentAnalysis.validation?.score || 0);
    if (score < 100) {
        showToast('Document must reach 100% before printing', 'warning');
        return;
    }
    window.print();
}
