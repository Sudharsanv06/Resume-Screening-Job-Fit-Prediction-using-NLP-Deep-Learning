import axios, { AxiosError } from "axios";
import type { PredictionResponse, BatchResponse, ApiError } from "../types";

// Reads from .env.local in dev, from Vercel dashboard env in prod.
// Falls back to localhost so the app still works without the env file.
const BASE_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30_000, // 30s — model inference can be slow on free Render tier
});

// ── Helpers ───────────────────────────────────────────────────────────────────

function extractErrorMessage(err: unknown): string {
  const axiosErr = err as AxiosError<ApiError>;
  const detail   = axiosErr?.response?.data?.detail;

  if (typeof detail === "string")  return detail;
  if (Array.isArray(detail))       return detail.map((d) => d.msg ?? JSON.stringify(d)).join(", ");

  if (axiosErr?.code === "ECONNABORTED") return "Request timed out. The server may be waking up — please try again.";
  if (axiosErr?.code === "ERR_NETWORK")  return "Cannot reach the server. Check your connection or try again shortly.";

  return "Something went wrong. Please try again.";
}

// ── API calls ─────────────────────────────────────────────────────────────────

export async function predictFromText(text: string): Promise<PredictionResponse> {
  try {
    const { data } = await api.post<PredictionResponse>("/predict/text", { text });
    return data;
  } catch (err) {
    throw new Error(extractErrorMessage(err));
  }
}

export async function predictFromFile(file: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);
  try {
    const { data } = await api.post<PredictionResponse>("/predict/file", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return data;
  } catch (err) {
    throw new Error(extractErrorMessage(err));
  }
}

export async function predictBatch(texts: string[]): Promise<BatchResponse> {
  try {
    const { data } = await api.post<BatchResponse>("/predict/batch", { texts });
    return data;
  } catch (err) {
    throw new Error(extractErrorMessage(err));
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const { data } = await api.get("/health");
    return data?.status === "ok";
  } catch {
    return false;
  }
}
