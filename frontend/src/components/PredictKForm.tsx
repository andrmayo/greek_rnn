import { useState, type SyntheticEvent } from "react";

import type { PredictKResponse } from "../api";

import { predictK } from "../api";
import { ReconstructionDisplay } from "./ReconstructionDisplay";

export function PredictKForm() {
  const [text, setText] = useState("");
  const [k, setK] = useState(5);
  const [result, setResult] = useState<PredictKResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: SyntheticEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      setResult(await predictK(text, k));
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter Greek text with lacunae, e.g. κα[..]α"
        rows={4}
        className="w-full border border-[var(--color-border)] rounded p-2 font-['Noto_Serif'] text-base resize-y focus:outline-none focus:border-[var(--color-brand)]"
      />
      <div className="flex items-center gap-3">
        <label className="font-['Noto_Sans'] text-sm text-[var(--color-text-secondary)]">
          k
        </label>
        <input
          type="number"
          min={1}
          value={k}
          onChange={(e) => setK(Number(e.target.value))}
          className="w-20 border border-[var(--color-border)] rounded p-2 text-sm focus:outline-none focus:border-[var(--color-brand)]"
        />
      </div>
      <button
        type="submit"
        disabled={loading || !text.trim() || k < 1}
        className="self-start px-4 py-2 bg-[var(--color-brand)] text-white rounded hover:opacity-90 disabled:opacity-50"
      >
        {loading ? "Predicting..." : `Predict top ${k}`}
      </button>
      {error && <p className="text-red-600 text-sm">{error}</p>}
      {result && (
        <ol className="flex flex-col gap-2">
          {result.texts.map((t, i) => (
            <li
              key={i}
              className="border border-[var(--color-border)] rounded p-4"
            >
              <span className="font-['Noto_Sans'] text-xs text-[var(--color-text-secondary)] mr-2">
                {i + 1}.
              </span>
              <ReconstructionDisplay text={t} lacunaMask={result.lacuna_mask} />
            </li>
          ))}
        </ol>
      )}
    </form>
  );
}
