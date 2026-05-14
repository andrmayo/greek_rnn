import type { ChangeEvent, SyntheticEvent } from "react";
import { useState, useRef } from "react";

import type { FileResultItem } from "../api";
import { predictFile } from "../api";

export function FileUploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<FileResultItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function handleSubmit(e: SyntheticEvent) {
    e.preventDefault();
    if (!file) return;
    setError(null);
    setResults([]);
    setLoading(true);

    try {
      for await (const item of predictFile(file)) {
        setResults((prev) => [...prev, item]);
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    setFile(e.target.files?.[0] ?? null);
    setResults([]);
    setError(null);
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="px-4 py-2 border border-[var(--color-border)] rounded font-['Noto_Sans'] text-sm text-[var(--color-text-secondary)] hover:border-[var(--color-brand)] hover:text-[var(--color-brand)]"
        >
          Choose file
        </button>
        <span className="text-sm font-['Noto_Sans'] text-[var(--color-text-secondary)]">
          {file ? file.name : "No file chosen - .json or .jsonl"}
        </span>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json,.jsonl"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>
      <button
        type="submit"
        disabled={loading || !file}
        className="self-start px-4 py-2 bg-[var(--color-brand)] text-white rounded hover:opacity-90 disabled:opacity-50"
      >
        {loading ? "Processing..." : "Run batch prediction"}
      </button>
      {error && <p className="text-red-600 text-sm">{error}</p>}
      {results.length > 0 && (
        <ol className="flex flex-col gap-2">
          {results.map((item, i) => (
            <li
              key={i}
              className="border border-[var(--color-border)] rounded p-4 flex flex-col gap-1"
            >
              <p className="font-['Noto_Serif'] text-sm text-[var(--color-text-secondary)]">
                {item.text}
              </p>
              <p className="font-['Noto_Serif'] text-base text-[var(--color-brand)]">
                {item.reconstruction}
              </p>
            </li>
          ))}
        </ol>
      )}
    </form>
  );
}
