import axios from 'axios'


const api = axios.create({
  baseURL: import.meta.env.VITE_BACKEND_BASE_URL
})

export function logoutUser() {
  localStorage.removeItem('access_token')
  localStorage.removeItem('refresh_token')
  localStorage.removeItem('user_type')
  window.location.href = '/login'

};
export const backendAPI = (data) => api.get('documents/test-csrf/', data)
// Auth API
export const registerUser = (data) => api.post('auth/register/', data)
export const loginUser = (data) => api.post('auth/login/', data)
export const documentHistory = (data) => api.get('/history/', data)
// Add interceptors for JWT handling
api.interceptors.response.use(
  response => response,
  error => {
    if (error.response?.status === 401) {
      //
    }
    return Promise.reject(error)
  }
)

export const analyzeDocument = async (file: File) => {
  const formData = new FormData();
  formData.append('document', file);

  const response = await api.post('analyze/', formData, {
    headers: {
      'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
    }
  });
  return response.data;
};


export default api