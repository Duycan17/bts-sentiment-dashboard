import { apiFetch } from "./client";
import type {
  AspectDetailResponse,
  AspectNSSResponse,
  AspectReviewsResponse,
  BusinessResponse,
  ErrorAnalysisResponse,
  ExplainResponse,
  InsightsResponse,
  KPIsResponse,
  MetaResponse,
  PerformanceResponse,
  PredictResponse,
  TrendResponse,
  VoiceResponse,
} from "./types";

const q = (qs: string) => (qs ? `?${qs}` : "");

export const api = {
  meta: () => apiFetch<MetaResponse>("/api/meta"),
  kpis: (qs: string) => apiFetch<KPIsResponse>(`/api/kpis${q(qs)}`),
  aspects: (qs: string) => apiFetch<AspectNSSResponse>(`/api/aspects${q(qs)}`),
  trends: (qs: string) => apiFetch<TrendResponse>(`/api/trends${q(qs)}`),
  voice: (qs: string) => apiFetch<VoiceResponse>(`/api/voice${q(qs)}`),
  wordcloudUrl: (polarity: "positive" | "negative", qs: string) =>
    `/api/wordcloud/${polarity}${q(qs)}`,
  performance: (qs: string) => apiFetch<PerformanceResponse>(`/api/performance${q(qs)}`),
  insights: (qs: string) => apiFetch<InsightsResponse>(`/api/insights${q(qs)}`),
  errorAnalysis: (qs: string) => apiFetch<ErrorAnalysisResponse>(`/api/error-analysis${q(qs)}`),
  business: (qs: string) => apiFetch<BusinessResponse>(`/api/business${q(qs)}`),
  aspectReviews: (aspect: string, qs: string) =>
    apiFetch<AspectReviewsResponse>(`/api/aspect-reviews/${encodeURIComponent(aspect)}${q(qs)}`),
  aspectDetail: (aspect: string, qs: string) =>
    apiFetch<AspectDetailResponse>(`/api/aspect-detail/${encodeURIComponent(aspect)}${q(qs)}`),
  predict: (text: string, aspect: string) =>
    apiFetch<PredictResponse>("/api/predict", {
      method: "POST",
      body: JSON.stringify({ text, aspect }),
    }),
  explain: (text: string, aspect: string) =>
    apiFetch<ExplainResponse>("/api/explain", {
      method: "POST",
      body: JSON.stringify({ text, aspect }),
    }),
  aspectRecommendation: (aspect: string) =>
    apiFetch<{ aspect: string; content: string; context: Record<string, unknown> }>(
      `/api/aspect-recommendation/${encodeURIComponent(aspect)}`
    ),
  chat: (message: string, history: { role: string; content: string }[]) =>
    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history }),
    }),
};
