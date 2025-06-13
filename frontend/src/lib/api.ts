// src/lib/api.ts
import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_BACKEND_BASE_URL
});

api.interceptors.request.use(
  (config) => {
    const accessToken = localStorage.getItem('access_token');
    if (accessToken) {
      config.headers.Authorization = `Bearer ${accessToken}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

api.interceptors.response.use(
  response => response,
  error => {
    if (error.response?.status === 401) {
      console.error('Authentication error: Token expired or invalid.');
      logoutUser();
    }
    return Promise.reject(error);
  }
);

export function logoutUser() {
  localStorage.removeItem('access_token');
  localStorage.removeItem('refresh_token');
  localStorage.removeItem('user_type');
  window.location.href = '/login';
}

export const registerUser = (data) => api.post('auth/register/', data);
export const loginUser = (data) => api.post('auth/login/', data);

// --- Document Analysis API ---

/**
 * @param {File} file 
 * @param {boolean} sendReportToOther 
 * @param {string} [recipientEmail] 
 * @returns {Promise<any>} 
 */
export const analyzeDocument = async (file, sendReportToOther = false, recipientEmail = null) => {
  const formData = new FormData();
  formData.append('file', file);

  if (sendReportToOther) {
    formData.append('send_report_to_other', 'true');
    if (recipientEmail) {
      formData.append('recipient_email', recipientEmail);
    }
  } else {
    formData.append('send_report_to_other', 'false');
  }

  const response = await api.post('documents/', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    }
  });
  return response.data;
};

// --- Document History API ---

/**
 * Retrieves the history of a document using its unique document code.
 * This can be used by both authenticated and unauthenticated users.
 * @param {string} documentCode - The unique code of the document.
 * @returns {Promise<any>} The document details including its history records.
 */
export const checkDocumentHistory = async (documentCode) => {
  const response = await api.get(`documents/check_document_history/?document_code=${documentCode}`);
  return response.data;
};

/**
 * Retrieves all documents uploaded by the authenticated user.
 * @returns {Promise<any[]>} An array of document objects.
 */
export const fetchMyDocuments = async () => {
  const response = await api.get('documents/my_documents/');
  return response.data;
};


export default api;