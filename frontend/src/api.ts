const API_BASE = import.meta.env.VITE_API_URL ?? "/api";

export interface PredictResponse {
  text: string;
  lacuna_mask: boolean[];
}

export interface PredictKResponse {
  texts: string[];
  lacuna_mask: boolean[];
}

export interface RankResponse {
  ranked: [string, number][];
}

export interface FileResultItem {
  text: string;
  reconstruction: string;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function predict(text: string): Promise<PredictResponse> {
  return post("/predict/", { text });
}

export function predictK(text: string, k: number): Promise<PredictKResponse> {
  return post("/predict-k/", { text, k });
}

export function rank(text: string, options: string[]): Promise<RankResponse> {
  return post("/rank/", { text, options });
}

export async function* predictFile(file: File): AsyncGenerator<FileResultItem> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/predict-file/`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop()!;
    for (const line of lines) {
      if (line.trim()) yield JSON.parse(line) as FileResultItem;
    }
  }
}

export async function getModels(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/models/`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getDefaultModel(): Promise<string> {
  const res = await fetch(`${API_BASE}/default-model/`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.model;
}

export async function changeModel(modelName: string): Promise<void> {
  const res = await fetch(`${API_BASE}/change-model/${modelName}`, {
    method: "PATCH",
  });
  if (!res.ok) throw new Error(await res.text());
}
