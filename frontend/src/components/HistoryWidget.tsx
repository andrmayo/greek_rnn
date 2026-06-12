import type { HistoryEntry } from "../types";
import { ReconstructionDisplay } from "./ReconstructionDisplay";

const TYPE_LABELS: Record<HistoryEntry["type"], string> = {
  predict: "Predict",
  "predict-k": "Top K",
  rank: "Rank",
};

function EntryCard({ entry }: { entry: HistoryEntry }) {
  const truncated =
    entry.input.length > 40 ? entry.input.slice(0, 37) + "…" : entry.input;

  return (
    <div className="border-b border-[var(--color-border)] p-3 flex flex-col gap-2">
      <div className="flex items-center gap-2 min-w-0">
        <span className="shrink-0 text-xs font-['Noto_Sans'] text-[var(--color-text-secondary)] bg-[var(--color-bg)] border border-[var(--color-border)] px-1.5 py-0.5 rounded">
          {TYPE_LABELS[entry.type]}
        </span>
        <span className="text-xs font-['Noto_Serif'] text-[var(--color-text-secondary)] truncate">
          {truncated}
        </span>
      </div>
      <span className="shrink-0 text-xs font-['Noto_Sans'] text-[var(--color-text-secondary)] bg-[var(--color-bg)] border border-[var(--color-border)] px-1.5 py-0.5">
        MODEL
      </span>
      <span className="text-xs font-['Noto_Serif'] text-[var(--color-text-secondary)]">
        {entry.model}
      </span>
      {entry.type === "predict" && (
        <ReconstructionDisplay
          text={entry.result.text}
          lacunaMask={entry.result.lacuna_mask}
        />
      )}
      {entry.type === "predict-k" && (
        <div className="flex flex-col gap-1">
          {entry.result.texts.length > 0 && (
            <ReconstructionDisplay
              text={entry.result.texts[0]}
              lacunaMask={entry.result.lacuna_mask}
            />
          )}
          {entry.result.texts.length > 1 && (
            <span className="text-xs text-[var(--color-text-secondary)] font-['Noto_Sans']">
              +{entry.result.texts.length - 1} more
            </span>
          )}
        </div>
      )}
      {entry.type === "rank" && (
        <ol className="flex flex-col gap-1">
          {entry.result.ranked.map(([opt, score], i) => (
            <li key={i} className="flex justify-between items-baseline text-sm">
              <span className="font-['Noto_Serif']">
                <span className="text-xs text-[var(--color-text-secondary)] mr-1">
                  {i + 1}.
                </span>
                {opt}
              </span>
              <span className="text-xs text-[var(--color-text-secondary)] font-['Noto_Sans'] ml-2 shrink-0">
                {score.toFixed(4)}
              </span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

export function HistoryWidget({ history }: { history: HistoryEntry[] }) {
  if (history.length === 0) {
    return (
      <p className="p-3 text-sm text-[var(--color-text-secondary)] font-['Noto_Sans']">
        No results yet.
      </p>
    );
  }
  return (
    <div>
      {history.map((entry) => (
        <EntryCard key={entry.id} entry={entry} />
      ))}
    </div>
  );
}
