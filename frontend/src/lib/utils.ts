import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNSS(v: number): string {
  return `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`;
}

export function formatPct(v: number): string {
  return `${v.toFixed(1)}%`;
}

export function formatNum(v: number): string {
  return v.toLocaleString();
}

export function nssColor(v: number): string {
  if (v < 0) return "#F44336";
  if (v < 30) return "#FF9800";
  return "#4CAF50";
}

export function filtersToQuery(filters: FilterState): string {
  const p = new URLSearchParams();
  if (filters.dateStart) p.set("date_start", filters.dateStart);
  if (filters.dateEnd) p.set("date_end", filters.dateEnd);
  filters.aspects.forEach((a) => p.append("aspects", a));
  filters.sources.forEach((s) => p.append("sources", s));
  filters.btsLines.forEach((b) => p.append("bts_lines", b));
  filters.sentiments.forEach((s) => p.append("sentiments", s));
  if (filters.minConfidence > 0) p.set("min_confidence", String(filters.minConfidence));
  p.set("label_source", filters.labelSource);
  return p.toString();
}

export interface FilterState {
  dateStart: string;
  dateEnd: string;
  aspects: string[];
  sources: string[];
  btsLines: string[];
  sentiments: string[];
  minConfidence: number;
  labelSource: "ground_truth" | "predicted";
}

export const DEFAULT_FILTERS: FilterState = {
  dateStart: "",
  dateEnd: "",
  aspects: [],
  sources: [],
  btsLines: [],
  sentiments: [],
  minConfidence: 0,
  labelSource: "predicted",
};
