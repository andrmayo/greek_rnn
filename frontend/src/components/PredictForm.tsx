import type { SyntheticEvent } from "react";
import { useState } from "react";
import { predict } from "../api";

import type { PredictResponse } from "../api";
import { ReconstructionDisplay } from "./ReconstructionDisplay";

export function PredictForm({
  onResult,
}: {
  onResult?: (input: string, result: PredictResponse) => void;
}) {
  const [text, setText] = useState("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: SyntheticEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const data = await predict(text);
      setResult(data);
      onResult?.(text, data);
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
      <button
        type="submit"
        disabled={loading || !text.trim()}
        className="self-start px-4 py-2 bg-[var(--color-brand)] text-white rounded hover:opacity-90 disabled:opacity-50"
      >
        {loading ? "Predicting..." : "Predict"}
      </button>
      {error && <p className="text-red-600 text-sm">{error}</p>}
      {result && (
        <div className="border border-[var(--color-border)] rounded p-4">
          <ReconstructionDisplay
            text={result.text}
            lacunaMask={result.lacuna_mask}
          />
        </div>
      )}
    </form>
  );
}
