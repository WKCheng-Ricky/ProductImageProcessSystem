/**
 * Common JavaScript Utilities
 *
 * Author: W Cheng
 * Co-Author: Claude (Anthropic AI Assistant)
 * Created: 2026-01-09
 * Project: Product Image Processing System
 */

/**
 * API Client for making requests to the backend
 */
class APIClient {
    constructor(baseURL = '/api') {
        this.baseURL = baseURL;
    }

    /**
     * Make a generic HTTP request
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;

        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
            ...options
        };

        try {
            const response = await fetch(url, defaultOptions);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API Request Error:', error);
            throw error;
        }
    }

    /**
     * Upload a file to the specified endpoint
     */
    async uploadFile(endpoint, file, additionalData = {}) {
        const formData = new FormData();
        formData.append('file', file);

        // Add any additional form data
        for (const [key, value] of Object.entries(additionalData)) {
            formData.append(key, value);
        }

        const url = `${this.baseURL}${endpoint}`;

        try {
            const response = await fetch(url, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Upload failed: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('File Upload Error:', error);
            throw error;
        }
    }

    /**
     * GET request
     */
    async get(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    }

    /**
     * POST request
     */
    async post(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    /**
     * DELETE request
     */
    async delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }
}

/**
 * Progress Tracker for visual feedback
 */
class ProgressTracker {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container ${containerId} not found`);
            return;
        }
        this.progressBar = this.container.querySelector('.progress-bar');
        this.statusText = this.container.querySelector('.status-text');
    }

    /**
     * Update progress bar
     */
    update(percentage, message = null) {
        if (this.progressBar) {
            this.progressBar.style.width = `${percentage}%`;
            this.progressBar.setAttribute('aria-valuenow', percentage);
            this.progressBar.textContent = `${percentage}%`;
        }

        if (message && this.statusText) {
            this.statusText.textContent = message;
        }
    }

    /**
     * Mark as complete
     */
    complete(message = 'Complete!') {
        this.update(100, message);
        if (this.progressBar) {
            this.progressBar.classList.remove('progress-bar-striped', 'progress-bar-animated');
            this.progressBar.classList.add('bg-success');
        }
    }

    /**
     * Mark as error
     */
    error(message = 'Error occurred') {
        if (this.statusText) {
            this.statusText.textContent = message;
        }
        if (this.progressBar) {
            this.progressBar.classList.remove('progress-bar-striped', 'progress-bar-animated');
            this.progressBar.classList.add('bg-danger');
        }
    }

    /**
     * Reset progress bar
     */
    reset() {
        this.update(0, '');
        if (this.progressBar) {
            this.progressBar.className = 'progress-bar progress-bar-striped progress-bar-animated';
        }
    }
}

/**
 * Utility Functions
 */

/**
 * Format bytes to human-readable size
 */
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];

    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

/**
 * Validate image file
 */
function validateImageFile(file, maxSizeMB = 50) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];

    if (!validTypes.includes(file.type)) {
        throw new Error('Invalid file type. Please upload JPG, PNG, or WEBP image.');
    }

    const maxSizeBytes = maxSizeMB * 1024 * 1024;
    if (file.size > maxSizeBytes) {
        throw new Error(`File size exceeds ${maxSizeMB}MB limit.`);
    }

    return true;
}

/**
 * Show loading spinner
 */
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.remove('d-none');
    }
}

/**
 * Hide loading spinner
 */
function hideLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.add('d-none');
    }
}

/**
 * Show alert message
 */
function showAlert(message, type = 'info', containerId = 'alert-container') {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Alert container ${containerId} not found`);
        return;
    }

    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    container.appendChild(alertDiv);

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.classList.remove('show');
        setTimeout(() => alertDiv.remove(), 150);
    }, 5000);
}

/**
 * Debounce function for input handlers
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Export for use in templates
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        APIClient,
        ProgressTracker,
        formatBytes,
        validateImageFile,
        showLoading,
        hideLoading,
        showAlert,
        debounce
    };
}
