import { useState } from "react";
import { useFilters } from "../hooks/useFilters";
import { useApi } from "../hooks/useApi";
import { api } from "../api/endpoints";
import { filtersToQuery } from "../lib/utils";
import { ChartCard } from "../components/ui/ChartCard";
import { Skeleton } from "../components/ui/ChartCard";
import { ConfusionMatrix } from "../components/charts/Charts";
import type { ErrorAnalysisResponse, ErrorSample, AspectErrorRow } from "../api/types";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";

// ─── Error type config ───────────────────────────────────────────────────────

const ERROR_TYPES = [
  {
    key: "fp",
    label: "False Positive",
    short: "FP",
    color: "#F44336",
    bg: "#FFF5F5",
    icon: "🔴",
    desc: "Predicted Positive — actually Negative",
    why: "Model picks up positive keywords (e.g. 'convenient', 'easy') but misses the negative context around them.",
    impact: "Inflates NSS. Hides real pain points from the C-suite.",
    samplesKey: "fp_samples" as const,
    countKey: "fp_count" as const,
  },
  {
    key: "fn",
    label: "False Negative",
    short: "FN",
    color: "#FF5722",
    bg: "#FFF3F0",
    icon: "🟠",
    desc: "Predicted Negative — actually Positive",
    why: "Sarcasm, irony, or comparative praise ('better than taxis') confuses the model.",
    impact: "Deflates NSS. Makes strong aspects look weaker than they are.",
    samplesKey: "fn_samples" as const,
    countKey: "fn_count" as const,
  },
  {
    key: "false_neutral",
    label: "False Neutral",
    short: "FNeu",
    color: "#9E9E9E",
    bg: "#F5F5F5",
    icon: "⚪",
    desc: "Predicted Neutral — actually Positive or Negative",
    why: "Mixed-sentiment reviews or low-confidence predictions collapse to Neutral as a safe default.",
    impact: "Largest error class (876 cases). Suppresses both positive and negative signal.",
    samplesKey: "false_neutral_samples" as const,
    countKey: "false_neutral_count" as const,
  },
  {
    key: "missed_neutral",
    label: "Missed Neutral",
    short: "MNeu",
    color: "#607D8B",
    bg: "#F0F4F8",
    icon: "🔵",
    desc: "Predicted Positive/Negative — actually Neutral",
    why: "Factual or descriptive reviews get pushed to a polarity class by dominant keywords.",
    impact: "Adds noise to NSS calculations.",
    samplesKey: "missed_neutral_samples" as const,
    countKey: "missed_neutral_count" as const,
  },
  {
    key: "low_conf",
    label: "Low-confidence Errors",
    short: "LCE",
    color: "#795548",
    bg: "#FBF8F6",
    icon: "🟤",
    desc: "Wrong prediction with confidence < 60%",
    why: "Model is uncertain — these are the most fixable errors via threshold tuning or retraining.",
    impact: "745 cases. Routing these to manual review would cut error rate significantly.",
    samplesKey: "low_conf_samples" as const,
    countKey: "low_conf_error_count" as const,
  },
];

// ─── KPI strip ───────────────────────────────────────────────────────────────

function ErrorKPIs({ d }: { d: ErrorAnalysisResponse }) {
  const items = [
    { label: "Total reviews", value: d.total.toLocaleString(), color: undefined },
    { label: "Total errors", value: d.total_errors.toLocaleString(), color: "#F44336" },
    { label: "Error rate", value: `${d.error_rate.toFixed(1)}%`, color: d.error_rate > 20 ? "#F44336" : "#FF9800" },
    { label: "False Positive", value: d.fp_count.toString(), color: "#F44336" },
    { label: "False Negative", value: d.fn_count.toString(), color: "#FF5722" },
    { label: "False Neutral", value: d.false_neutral_count.toString(), color: "#9E9E9E" },
  ];
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
      {items.map(({ label, value, color }) => (
        <div key={label} className="rounded-xl border border-gray-100 bg-white px-4 py-3 shadow-sm">
          <p className="text-xs font-semibold uppercase tracking-wide text-gray-400">{label}</p>
          <p className="text-2xl font-black mt-0.5" style={color ? { color } : undefined}>{value}</p>
        </div>
      ))}
    </div>
  );
}

// ─── Error distribution bar ───────────────────────────────────────────────────

function ErrorDistBar({ d }: { d: ErrorAnalysisResponse }) {
  const data = ERROR_TYPES.map((t) => ({ name: t.short, count: d[t.countKey], color: t.color, label: t.label }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 20, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" tick={{ fontSize: 12 }} />
        <YAxis tick={{ fontSize: 11 }} />
        <Tooltip formatter={(v, _, p) => [Number(v).toLocaleString(), p.payload.label]} />
        <Bar dataKey="count" radius={[4, 4, 0, 0]}>
          {data.map((d) => <Cell key={d.name} fill={d.color} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ─── Aspect error table ───────────────────────────────────────────────────────

function AspectErrorTable({ rows }: { rows: AspectErrorRow[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-gray-100">
            {["Aspect", "Reviews", "Errors", "Error %", "FP", "FN", "False Neutral"].map((h) => (
              <th key={h} className="text-left py-2 px-3 text-gray-500 font-semibold uppercase tracking-wide">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.aspect} className="border-b border-gray-50 hover:bg-gray-50">
              <td className="py-2 px-3 font-medium text-gray-800">{r.aspect}</td>
              <td className="py-2 px-3 text-gray-600">{r.rows.toLocaleString()}</td>
              <td className="py-2 px-3 text-gray-600">{r.errors}</td>
              <td className="py-2 px-3 font-mono font-semibold" style={{ color: r.error_rate > 25 ? "#F44336" : r.error_rate > 15 ? "#FF9800" : "#4CAF50" }}>
                {r.error_rate.toFixed(1)}%
              </td>
              <td className="py-2 px-3 text-red-500">{r.fp}</td>
              <td className="py-2 px-3 text-orange-500">{r.fn}</td>
              <td className="py-2 px-3 text-gray-500">{r.false_neutral}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Sample card ─────────────────────────────────────────────────────────────

function SampleCard({ s, trueColor, predColor }: { s: ErrorSample; trueColor: string; predColor: string }) {
  return (
    <div className="rounded-lg border border-gray-100 bg-white p-3 space-y-2 hover:shadow-sm transition-shadow">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs font-semibold px-2 py-0.5 rounded-full text-white shrink-0" style={{ backgroundColor: trueColor }}>
          True: {s.sentiment}
        </span>
        <span className="text-xs font-semibold px-2 py-0.5 rounded-full text-white shrink-0" style={{ backgroundColor: predColor }}>
          Pred: {s.predicted}
        </span>
        <span className="text-xs text-gray-400 shrink-0">conf: {(s.confidence * 100).toFixed(0)}%</span>
        <span className="text-xs text-gray-400 truncate">{s.aspect}</span>
        <span className="text-xs text-gray-400 ml-auto shrink-0">{s.source}</span>
      </div>
      <p className="text-xs text-gray-700 leading-relaxed">{s.text}</p>
    </div>
  );
}

// ─── Error type section ───────────────────────────────────────────────────────

const SENT_COLORS: Record<string, string> = {
  Positive: "#4CAF50", Neutral: "#9E9E9E", Negative: "#F44336",
};

function ErrorSection({ type, samples, count }: {
  type: typeof ERROR_TYPES[0]; samples: ErrorSample[]; count: number;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl border bg-white shadow-sm overflow-hidden" style={{ borderLeftWidth: 4, borderLeftColor: type.color }}>
      <button
        className="w-full px-4 py-3 flex items-start justify-between gap-3 hover:bg-gray-50 transition-colors text-left"
        style={{ backgroundColor: type.bg }}
        onClick={() => setOpen((v) => !v)}
      >
        <div className="min-w-0">
          <div className="flex items-center gap-2 mb-0.5 flex-wrap">
            <span className="text-xs font-bold uppercase tracking-wide" style={{ color: type.color }}>{type.icon} {type.label}</span>
            <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-white border font-semibold" style={{ color: type.color }}>{count.toLocaleString()} cases</span>
          </div>
          <p className="text-xs text-gray-600 font-medium">{type.desc}</p>
        </div>
        <span className="text-gray-400 text-lg shrink-0">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="px-4 py-4 space-y-4 border-t border-gray-100">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="rounded-lg bg-amber-50 border border-amber-100 px-3 py-2">
              <p className="text-xs font-semibold text-amber-700 mb-0.5">Why does this happen?</p>
              <p className="text-xs text-amber-800">{type.why}</p>
            </div>
            <div className="rounded-lg bg-red-50 border border-red-100 px-3 py-2">
              <p className="text-xs font-semibold text-red-700 mb-0.5">Business impact</p>
              <p className="text-xs text-red-800">{type.impact}</p>
            </div>
          </div>
          {samples.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Sample cases</p>
              <div className="space-y-2">
                {samples.map((s, i) => (
                  <SampleCard
                    key={i} s={s}
                    trueColor={SENT_COLORS[s.sentiment] ?? "#9E9E9E"}
                    predColor={SENT_COLORS[s.predicted] ?? "#9E9E9E"}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function ErrorAnalysis() {
  const { filters } = useFilters();
  const qs = filtersToQuery(filters);
  const ea = useApi<ErrorAnalysisResponse>(() => api.errorAnalysis(qs), [qs]);

  if (ea.loading) {
    return (
      <div className="space-y-4 max-w-5xl">
        <Skeleton className="h-8 w-48 rounded" />
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-20 rounded-xl" />)}
        </div>
        <Skeleton className="h-52 rounded-xl" />
        <div className="space-y-3">
          {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-16 rounded-xl" />)}
        </div>
      </div>
    );
  }

  if (!ea.data) return null;
  const d = ea.data;

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Error Analysis</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          Why the model fails — FP, FN, False Neutral, Missed Neutral, low-confidence errors · {d.total.toLocaleString()} reviews
        </p>
      </div>

      <ErrorKPIs d={d} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartCard title="Error distribution by type">
          <ErrorDistBar d={d} />
        </ChartCard>
        <ChartCard title="Confusion matrix">
          <ConfusionMatrix labels={d.confusion_matrix.labels} matrix={d.confusion_matrix.counts} />
        </ChartCard>
      </div>

      <ChartCard title="Error rate by aspect">
        <AspectErrorTable rows={d.aspect_errors} />
      </ChartCard>

      <div>
        <h2 className="text-sm font-bold text-gray-700 uppercase tracking-wide mb-3">Error type deep-dives — click to expand</h2>
        <div className="space-y-3">
          {ERROR_TYPES.map((type) => (
            <ErrorSection
              key={type.key}
              type={type}
              samples={d[type.samplesKey]}
              count={d[type.countKey]}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
