import type { SyntheticEvent } from "react";
import { useState } from "react";

import type { RankResponse } from "../api";
import { rank } from "../api";

export function RankForm() {
  const [text, setText] = useState("");
  const [options, setOptions] = useState<string[]>([""]);
  const [simpleOptions, setSimpleOptions] = useState("");
  const [dynamicMode, setDynamicMode] = useState(true);
  const [result, setResult] = useState<RankResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function getOptions(): string[] {
    return dynamicMode
      ? options.filter((o) => o.trim())
      : simpleOptions.split(" ").filter((o) => o.trim());
  }

  async function handleSubmit(e: SyntheticEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      setResult(await rank(text, getOptions()));
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  function addOption() {
    setOptions([...options, ""]);
  }

  function removeOption(i: number) {
    setOptions(options.filter((_, idx) => idx !== i));
  }

  function updateOption(i: number, value: string) {
    setOptions(options.map((o, idx) => (idx === i ? value : o)));
  }

  const validOptions = getOptions();
  const canSubmit = !loading && text.trim() && validOptions.length > 0;

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter Greek text with lacunae, e.g. κα[..]α"
        rows={4}
        className="w-full border border-[var(--color-border)] rounded p-2 font-['Noto_Serif'] text-base resize-y focus:outline-none focus:border-[var(--color-brand)]"
      />
      <div className="flex items-center justify-between">
        <span className="font-['Noto_Sans'] text-sm text-[var(--color-text-secondary)]">
          Options
        </span>
        <button
          type="button"
          onClick={() => setDynamicMode(!dynamicMode)}
          className="text-xs font-['Noto_Sans'] text-[var(--color-text-secondary)] hover:text-[var(--color-text)] underline"
        >
          {dynamicMode ? "Switch to simple input" : "Switch to dynamic input"}
        </button>
      </div>
      {dynamicMode ? (
        <div className="flex flex-col gap-2">
          {options.map((opt, i) => (
            <div key={i} className="flex gap-2">
              <input
                type="text"
                value={opt}
                onChange={(e) => updateOption(i, e.target.value)}
                placeholder={`Option ${i + 1}`}
                className="flex-1 border border-[var(--color-border)] rounded p-2 font-['Noto_Serif'] text-sm focus:outline-none focus:border-[var(--color-brand)]"
              />
              {options.length > 1 && (
                <button
                  type="button"
                  onClick={() => removeOption(i)}
                  className="text-[var(--color-text-secondary)] hover:text-red-600 text-sm px-2"
                >
                  x
                </button>
              )}
            </div>
          ))}
          <button
            type="button"
            onClick={addOption}
            className="self-start text-sm font-['Noto_Sans'] text-[var(--color-brand)] hover:opacity-75"
          >
            + Add option
          </button>
        </div>
      ) : (
        <input
          type="text"
          value={simpleOptions}
          onChange={(e) => setSimpleOptions(e.target.value)}
          placeholder="Space-separated options, e.g. κατα παρα"
          className="w-full border border-[var(--color-border)] rounded p-2 font-['Noto_Serif'] text-sm focus:outline-none focus:border-[var(--color-brand)]"
        />
      )}
      <button
        type="submit"
        disabled={!canSubmit}
        className="self-start px-4 py-2 bg-[var(--color-brand)] text-white rounded hover:opacity-90 disabled:opacity-50"
      >
        {loading ? "Ranking..." : "Rank"}
      </button>
      {error && <p className="text-red-600 text-sm">{error}</p>}
      {result && (
        <ol className="flex flex-col gap-2">
          {result.ranked.map(([option, score], i) => (
            <li
              key={i}
              className="flex justify-between items-center border border-[var(--color-border)] rounded p-4"
            >
              <span className="font-['Noto_Serif'] text-base">
                <span className="text-xs text-[var(--color-text-secondary)] mr-2">
                  {i + 1}.
                </span>
                {option}
              </span>
              <span className="font-['Noto_Sans'] text-xs text-[var(--color-text-secondary)]">
                {score.toFixed(4)}
              </span>
            </li>
          ))}
        </ol>
      )}
    </form>
  );
}
