export interface MetaResponse {
  sources: string[];
  bts_lines: string[];
  aspects: string[];
  date_min: string;
  date_max: string;
}

export interface KPIsResponse {
  rows: number;
  nss: number;
  pct_positive: number;
  pct_neutral: number;
  pct_negative: number;
  n_aspects: number;
  n_sources: number;
  date_min: string;
  date_max: string;
  avg_confidence: number;
}

export interface AspectRow {
  aspect: string;
  rows: number;
  pos: number;
  neg: number;
  neu: number;
  nss: number;
  pct_negative: number;
}

export interface AspectNSSResponse {
  aspects: AspectRow[];
}

export interface MonthlyRow {
  year_month: string;
  rows: number;
  nss: number;
}

export interface SeasonalityCell {
  year: number;
  month: number;
  rows: number;
}

export interface TrendResponse {
  monthly: MonthlyRow[];
  seasonality: SeasonalityCell[];
}

export interface NgramRow {
  ngram: string;
  count: number;
}

export interface VoiceResponse {
  positive_ngrams: NgramRow[];
  negative_ngrams: NgramRow[];
}

export interface ConfusionMatrixData {
  labels: string[];
  counts: number[][];
  normalized: number[][];
}

export interface ClassMetrics {
  label: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface CalibrationBin {
  avg_conf: number;
  accuracy: number;
  rows: number;
}

export interface AspectF1Row {
  aspect: string;
  rows: number;
  accuracy: number;
  macro_f1: number;
  weighted_f1: number;
}

export interface PerformanceResponse {
  accuracy: number;
  macro_f1: number;
  weighted_f1: number;
  confusion_matrix: ConfusionMatrixData;
  per_class: ClassMetrics[];
  calibration: CalibrationBin[];
  aspect_agreement: number;
  aspect_confusion: ConfusionMatrixData;
  per_aspect_f1: AspectF1Row[];
}

export interface InsightCard {
  title: string;
  finding: string;
  recommendation: string;
  severity: "critical" | "warning" | "good" | "info";
  metric: string;
}

export interface InsightsResponse {
  cards: InsightCard[];
}

export interface ErrorSample {
  text: string;
  rating: number;
  source: string;
  sentiment: string;
  date: string;
  aspect: string;
  predicted: string;
  confidence: number;
}

export interface AspectErrorRow {
  aspect: string;
  rows: number;
  errors: number;
  error_rate: number;
  fp: number;
  fn: number;
  false_neutral: number;
}

export interface ErrorAnalysisResponse {
  total: number;
  total_errors: number;
  error_rate: number;
  fp_count: number;
  fn_count: number;
  false_neutral_count: number;
  missed_neutral_count: number;
  low_conf_error_count: number;
  confusion_matrix: { labels: string[]; counts: number[][] };
  fp_samples: ErrorSample[];
  fn_samples: ErrorSample[];
  false_neutral_samples: ErrorSample[];
  missed_neutral_samples: ErrorSample[];
  low_conf_samples: ErrorSample[];
  aspect_errors: AspectErrorRow[];
}

export interface YearlyNSS {
  year: number;
  rows: number;
  nss: number;
  pos: number;
  neg: number;
  neu: number;
}

export interface AspectDetailResponse {
  aspect: string;
  nss: number;
  rows: number;
  pct_positive: number;
  pct_neutral: number;
  pct_negative: number;
  priority: string;
  trend: TrendDirection;
  action: string;
  top_complaints: string[];
  top_praises: string[];
  yearly_nss: YearlyNSS[];
  rating_dist: Record<string, number>;
  source_dist: SourceBreakdown[];
  sample_negative: SampleReview[];
  sample_positive: SampleReview[];
}

export interface SampleReview {
  text: string;
  rating: number;
  source: string;
  sentiment: string;
  date: string;
}

export interface AspectReviewsResponse {
  aspect: string;
  negative: SampleReview[];
  positive: SampleReview[];
}

export interface TrendDirection {
  direction: "improving" | "declining" | "stable";
  recent_nss: number;
  prior_nss: number;
  delta: number;
}

export interface AspectDeepDive {
  aspect: string;
  nss: number;
  rows: number;
  pct_negative: number;
  pct_positive: number;
  top_complaints: string[];
  top_praises: string[];
  trend: TrendDirection;
  priority: "urgent" | "monitor" | "leverage";
  action: string;
}

export interface SourceBreakdown {
  source: string;
  rows: number;
  nss: number;
  pct: number;
}

export interface ExecutiveSummary {
  overall_nss: number;
  total_reviews: number;
  date_range: string;
  top_pain: string;
  top_strength: string;
  trend_direction: "improving" | "declining" | "stable";
  headline: string;
}

export interface YoYRow {
  year: number;
  nss: number;
  rows: number;
}

export interface BusinessResponse {
  summary: ExecutiveSummary;
  aspect_deep_dives: AspectDeepDive[];
  source_breakdown: SourceBreakdown[];
  yoy_nss: YoYRow[];
}

export interface ReviewSource {
  text: string;
  aspect: string;
  sentiment: string;
  date: string;
  source: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: ReviewSource[];
}

export interface PredictResponse {
  label: string;
  confidence: number;
  proba: Record<string, number>;
}

export interface TokenContribution {
  token: string;
  value: number;
}

export interface ExplainResponse extends PredictResponse {
  contributions: TokenContribution[];
  base_value: number;
}
