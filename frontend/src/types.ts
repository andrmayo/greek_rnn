import type { PredictResponse, PredictKResponse, RankResponse } from "./api";

export type HistoryEntryData =
  | { type: "predict"; input: string; result: PredictResponse; model: string }
  | {
      type: "predict-k";
      input: string;
      k: number;
      result: PredictKResponse;
      model: string;
    }
  | { type: "rank"; input: string; result: RankResponse; model: string };

export type HistoryEntry = HistoryEntryData & { id: number };
