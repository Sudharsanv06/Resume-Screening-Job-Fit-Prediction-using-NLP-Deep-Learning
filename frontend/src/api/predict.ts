import axios from 'axios';
import type { PredictionResponse } from '../types';

export const predictText = async (text: string): Promise<PredictionResponse> => {
  try {
    const response = await axios.post<PredictionResponse>('/api/predict/text', { text });
    return response.data;
  } catch (error) {
    throw new Error(handleApiError(error));
  }
};

export const predictFile = async (file: File): Promise<PredictionResponse> => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post<PredictionResponse>('/api/predict/file', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    throw new Error(handleApiError(error));
  }
};

const handleApiError = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    const status = error.response?.status;
    const data = error.response?.data;
    
    // Attempt to extract detail from FastAPI validation structure
    let detail = '';
    if (data && typeof data === 'object') {
      if ('detail' in data) {
        const d = (data as { detail: unknown }).detail;
        if (typeof d === 'string') {
          detail = d;
        } else if (Array.isArray(d)) {
          // Validation list errors format
          detail = d.map((err: unknown) => {
            if (err && typeof err === 'object' && 'msg' in err) {
              return (err as { msg: string }).msg;
            }
            return JSON.stringify(err);
          }).join(', ');
        } else {
          detail = JSON.stringify(d);
        }
      }
    }

    if (status === 413) {
      return 'File size is too large (maximum size is 5MB).';
    }
    if (status === 415) {
      return 'Unsupported file format. Only PDF and DOCX files are allowed.';
    }
    if (status === 422) {
      return detail || 'Unprocessable entity. Validation error in data submitted.';
    }
    if (status === 400) {
      return detail || 'Invalid request parameters.';
    }
    if (status === 500) {
      return 'Internal server error occurred on the prediction service.';
    }
    
    if (error.code === 'ERR_NETWORK') {
      return 'Network error: Cannot reach the backend. Check if the server is running.';
    }
    
    return detail || error.message || `Server returned error status ${status}`;
  }
  
  if (error instanceof Error) {
    return error.message;
  }
  
  return 'An unexpected error occurred.';
};
