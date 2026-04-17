import { PredictionInput, PredictionResult, HistoryRow } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000";

export async function checkHealth(): Promise<{ status: string; model_version: string }> {
  const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error("API health check failed");
  return res.json();
}

export async function submitPrediction(input: PredictionInput): Promise<PredictionResult> {
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Unknown error" }));
    throw new Error(err.error ?? "Prediction failed");
  }
  return res.json();
}

export async function fetchHistory(): Promise<HistoryRow[]> {
  const res = await fetch(`${API_URL}/history`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch history");
  return res.json();
}
