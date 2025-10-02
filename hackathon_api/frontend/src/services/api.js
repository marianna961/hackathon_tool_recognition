import axios from 'axios';

// Для разработки используем localhost, для продакшена - имя сервиса
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Добавь обработку CORS ошибок
api.interceptors.response.use(
  response => response,
  error => {
    if (error.code === 'ECONNREFUSED') {
      console.error('Backend недоступен. Запусти backend на порту 8000');
    }
    return Promise.reject(error);
  }
);

// Остальные функции остаются такими же...
export const uploadSingleImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('event_type', 'hand_out');
  formData.append('camera_id', 'test_camera');
  
  return api.post('/predict/single', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

// Upload multiple images
export const uploadBatchImages = async (files) => {
  const formData = new FormData();
  files.forEach(file => {
    formData.append('files', file);
  });
  formData.append('event_type', 'hand_out');
  formData.append('camera_id', 'test_camera');
  
  return api.post('/predict/batch', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

// Upload ZIP folder
export const uploadZipFolder = async (zipFile) => {
  const formData = new FormData();
  formData.append('zip_file', zipFile);
  formData.append('event_type', 'hand_out');
  formData.append('camera_id', 'test_camera');
  
  return api.post('/predict/zip', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

// api.js
export const predictSingle = async (formData) => {
  return api.post('/predict/single', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const predictBatch = async (formData) => {
  return api.post('/predict/batch', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const predictZip = async (formData) => {
  return api.post('/predict/zip', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const getModelInfo = async () => {
  return api.get('/model/info');
};

// // Get model information
// export const getModelInfo = async () => {
//   return api.get('/model/info');
// };

// // Get transaction history
// export const getTransactions = async (limit = 10) => {
//   return api.get(`/transactions?limit=${limit}`);
// };

// // Get specific transaction
// export const getTransaction = async (transactionId) => {
//   return api.get(`/transactions/${transactionId}`);
// };

export default api;