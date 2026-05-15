import { useState } from "react";
import { api } from "../api/endpoints";
import { ChartCard } from "../components/ui/ChartCard";
import { ShapBar } from "../components/charts/Charts";
import { SENTIMENT_COLORS } from "../lib/colors";
import type { ExplainResponse, PredictResponse } from "../api/types";

const ASPECTS = [
  "Accessibility", "Cleanliness", "Crowding & Comfort", "Fare & Payment System",
  "Information & Navigation", "Overall Experience & Convenience", "Route Coverage & Connectivity",
  "Safety & Security", "Staff & Service", "Train Frequency & Waiting Time",
];

const SAMPLES: Record<string, string> = {
  "Strength — fare": "BTS is super convenient and cheap, the rabbit card top-up takes seconds.",
  "Pain — accessibility": "The elevator was broken again and there is no ramp, useless for wheelchair users.",
  "Pain — navigation": "The signage is so confusing, I got off at the wrong station because the announcements are too quiet.",
  "Mixed — crowding": "Trains run on time but the carriages are unbearably packed at rush hour.",
};

export default function LivePredictor() {
  const [text, setText] = useState("");
  const [aspect, setAspect] = useState(ASPECTS[3]);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [explain, setExplain] = useState<ExplainResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handlePredict(withExplain: boolean) {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setExplain(null);
    try {
      if (withExplain) {
        const r = await api.explain(text, aspect);
        setResult(r);
        setExplain(r);
      } else {
        const r = await api.predict(text, aspect);
        setResult(r);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  const labelColor = result ? SENTIMENT_COLORS[result.label] ?? "#3F51B5" : "#3F51B5";

  return (
    <div className="space-y-6 max-w-3xl">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Live Predictor + SHAP</h1>
        <p className="text-sm text-gray-500 mt-1">Paste a review, pick an aspect, and get a prediction with token-level explanations.</p>
      </div>

      {/* Sample selector */}
      <div>
        <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Load a sample</label>
        <div className="flex flex-wrap gap-2">
          {Object.entries(SAMPLES).map(([label, sample]) => (
            <button key={label} onClick={() => setText(sample)}
              className="px-3 py-1 rounded-full border border-gray-200 text-xs text-gray-600 hover:border-brand hover:text-brand transition-colors">
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Input */}
      <div className="space-y-3">
        <div>
          <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Review text</label>
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={4}
            placeholder="Type or paste a review…"
            className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand resize-none" />
        </div>
        <div>
          <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Aspect</label>
          <select value={aspect} onChange={(e) => setAspect(e.target.value)}
            className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand bg-white">
            {ASPECTS.map((a) => <option key={a}>{a}</option>)}
          </select>
        </div>
        <div className="flex gap-2">
          <button onClick={() => handlePredict(false)} disabled={loading || !text.trim()}
            className="px-5 py-2 rounded-lg bg-brand text-white text-sm font-semibold hover:bg-indigo-700 disabled:opacity-40 transition-colors">
            {loading ? "Predicting…" : "Predict"}
          </button>
          <button onClick={() => handlePredict(true)} disabled={loading || !text.trim()}
            className="px-5 py-2 rounded-lg border border-brand text-brand text-sm font-semibold hover:bg-indigo-50 disabled:opacity-40 transition-colors">
            {loading ? "Explaining…" : "Predict + Explain (SHAP)"}
          </button>
        </div>
      </div>

      {error && <div className="rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700">{error}</div>}

      {/* Result */}
      {result && (
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-3">
            <div className="rounded-xl border bg-white p-4 shadow-sm">
              <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Predicted</p>
              <p className="text-2xl font-bold mt-1" style={{ color: labelColor }}>{result.label}</p>
            </div>
            <div className="rounded-xl border bg-white p-4 shadow-sm">
              <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Confidence</p>
              <p className="text-2xl font-bold mt-1 text-gray-900">{(result.confidence * 100).toFixed(1)}%</p>
            </div>
            <div className="rounded-xl border bg-white p-4 shadow-sm">
              <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Aspect</p>
              <p className="text-sm font-bold mt-1 text-gray-900 leading-tight">{aspect}</p>
            </div>
          </div>

          {/* Proba bars */}
          <ChartCard title="Class probabilities">
            <div className="space-y-2 pt-1">
              {Object.entries(result.proba).sort((a, b) => b[1] - a[1]).map(([cls, prob]) => (
                <div key={cls}>
                  <div className="flex justify-between text-xs mb-0.5">
                    <span className="font-medium text-gray-700">{cls}</span>
                    <span className="text-gray-500">{(prob * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-3 rounded-full bg-gray-100 overflow-hidden">
                    <div className="h-full rounded-full transition-all" style={{ width: `${prob * 100}%`, backgroundColor: SENTIMENT_COLORS[cls] ?? "#3F51B5" }} />
                  </div>
                </div>
              ))}
            </div>
          </ChartCard>
        </div>
      )}

      {/* SHAP */}
      {explain && explain.contributions.length > 0 && (
        <ChartCard title={`SHAP — top tokens driving "${explain.label}"`}>
          <ShapBar contributions={explain.contributions} label={explain.label} />
          <p className="text-xs text-gray-400 mt-2">
            Coloured bars push toward the predicted class; grey bars push away.
            Base log-odds = {explain.base_value >= 0 ? "+" : ""}{explain.base_value.toFixed(3)}.
          </p>
        </ChartCard>
      )}
    </div>
  );
}
