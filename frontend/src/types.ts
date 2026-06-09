import type { PredictResponse, PredictKResponse, RankResponse } from "./api";

export type HistoryEntryData =
  | { type: "predict"; input: string; result: PredictResponse }
  | {
      type: "predict-k";
      input: string;
      k: number;
      result: PredictKResponse;
    }
  | { type: "rank"; input: string; result: RankResponse };

export type HistoryEntry = HistoryEntryData & { id: number };
