import { useParams, useNavigate } from "react-router-dom";
import { useFilters } from "../hooks/useFilters";
import { useApi } from "../hooks/useApi";
import { api } from "../api/endpoints";
import { filtersToQuery, formatNSS, formatNum, nssColor } from "../lib/utils";
import { ChartCard } from "../components/ui/ChartCard";
import { Skeleton } from "../components/ui/ChartCard";
import type { AspectDetailResponse, SampleReview, YearlyNSS } from "../api/types";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Cell, BarChart, PieChart, Pie,
} from "recharts";
import { BRAND, SENTIMENT_COLORS } from "../lib/colors";

// ─── Palette ─────────────────────────────────────────────────────────────────

const PRIORITY_META: Record<string, { color: string; bg: string; label: string; icon: string }> = {
  urgent:   { color: "#F44336", bg: "#FFF5F5", label: "Urgent",   icon: "🔴" },
  monitor:  { color: "#FF9800", bg: "#FFFBF0", label: "Monitor",  icon: "🟠" },
  leverage: { color: "#4CAF50", bg: "#F5FFF5", label: "Leverage", icon: "🟢" },
};

const TREND_META: Record<string, { icon: string; color: string; label: string }> = {
  improving: { icon: "↑", color: "#4CAF50", label: "Improving" },
  declining: { icon: "↓", color: "#F44336", label: "Declining" },
  stable:    { icon: "→", color: "#9E9E9E", label: "Stable" },
};

// ─── KPI tile ────────────────────────────────────────────────────────────────

function Tile({ label, value, color, sub }: { label: string; value: string; color?: string; sub?: string }) {
  return (
    <div className="rounded-xl border border-gray-100 bg-white px-4 py-3 shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-wide text-gray-400">{label}</p>
      <p className="text-2xl font-black mt-0.5 leading-tight" style={color ? { color } : undefined}>{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-0.5">{sub}</p>}
    </div>
  );
}

// ─── NSS trend chart (bars = volume, line = NSS) ─────────────────────────────

function NSSTimeline({ data }: { data: YearlyNSS[] }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <ComposedChart data={data} margin={{ top: 8, right: 40, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="year" tick={{ fontSize: 11 }} />
        <YAxis yAxisId="vol" orientation="left" tickFormatter={(v) => v.toLocaleString()} tick={{ fontSize: 10 }} />
        <YAxis yAxisId="nss" orientation="right" tickFormatter={(v) => `${v >= 0 ? "+" : ""}${Number(v).toFixed(0)}%`} tick={{ fontSize: 10 }} />
        <Tooltip
          formatter={(v, name) => name === "NSS (%)" ? `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(1)}%` : Number(v).toLocaleString()}
          labelFormatter={(l) => `Year: ${l}`}
        />
        <ReferenceLine yAxisId="nss" y={0} stroke="#F44336" strokeDasharray="4 2" />
        <Bar yAxisId="vol" dataKey="rows" name="Reviews" fill="#E2E8F0" radius={[2, 2, 0, 0]} />
        <Line yAxisId="nss" type="monotone" dataKey="nss" name="NSS (%)" stroke={BRAND} strokeWidth={2.5} dot={{ r: 4, fill: BRAND }} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// ─── Sentiment donut ─────────────────────────────────────────────────────────

function SentimentDonut({ pos, neu, neg }: { pos: number; neu: number; neg: number }) {
  const data = [
    { name: "Positive", value: pos },
    { name: "Neutral",  value: neu },
    { name: "Negative", value: neg },
  ];
  return (
    <ResponsiveContainer width="100%" height={200}>
      <PieChart>
        <Pie data={data} cx="50%" cy="50%" innerRadius={55} outerRadius={85} dataKey="value"
          label={({ name, percent }) => `${name} ${((percent ?? 0) * 100).toFixed(0)}%`} labelLine={false}>
          {data.map((d) => <Cell key={d.name} fill={SENTIMENT_COLORS[d.name] ?? "#9E9E9E"} />)}
        </Pie>
        <Tooltip formatter={(v) => Number(v).toLocaleString()} />
      </PieChart>
    </ResponsiveContainer>
  );
}

// ─── Rating distribution ─────────────────────────────────────────────────────

function RatingBar({ dist }: { dist: Record<string, number> }) {
  const data = [1, 2, 3, 4, 5].map((r) => ({ star: `★${r}`, count: dist[String(r)] ?? 0 }));
  const max = Math.max(...data.map((d) => d.count), 1);
  return (
    <div className="space-y-1.5 pt-1">
      {[...data].reverse().map(({ star, count }) => (
        <div key={star} className="flex items-center gap-2">
          <span className="text-xs text-yellow-500 w-6 shrink-0">{star}</span>
          <div className="flex-1 h-4 rounded bg-gray-100 overflow-hidden">
            <div className="h-full rounded transition-all" style={{ width: `${count / max * 100}%`, backgroundColor: "#FBBF24" }} />
          </div>
          <span className="text-xs text-gray-500 w-10 text-right">{count.toLocaleString()}</span>
        </div>
      ))}
    </div>
  );
}

// ─── Source breakdown ─────────────────────────────────────────────────────────

function SourceBreakdownChart({ data }: { data: AspectDetailResponse["source_dist"] }) {
  const sorted = [...data].sort((a, b) => a.nss - b.nss);
  return (
    <ResponsiveContainer width="100%" height={Math.max(160, sorted.length * 32)}>
      <BarChart data={sorted} layout="vertical" margin={{ left: 90, right: 50, top: 4, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" tickFormatter={(v) => `${v >= 0 ? "+" : ""}${Number(v).toFixed(0)}%`} tick={{ fontSize: 10 }} />
        <YAxis type="category" dataKey="source" width={85} tick={{ fontSize: 11 }} />
        <Tooltip formatter={(v) => `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(1)}%`} />
        <ReferenceLine x={0} stroke="#333" strokeWidth={1} />
        <Bar dataKey="nss" name="NSS" radius={[0, 4, 4, 0]}>
          {sorted.map((d) => <Cell key={d.source} fill={nssColor(d.nss)} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ─── Sample review card ───────────────────────────────────────────────────────

function ReviewCard({ r }: { r: SampleReview }) {
  const color = SENTIMENT_COLORS[r.sentiment] ?? "#9E9E9E";
  return (
    <div className="rounded-lg border border-gray-100 bg-white p-3 space-y-2 hover:shadow-sm transition-shadow">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs font-semibold px-2 py-0.5 rounded-full text-white shrink-0" style={{ backgroundColor: color }}>{r.sentiment}</span>
        <span className="text-xs text-yellow-500">{"★".repeat(Math.min(r.rating, 5))}</span>
        <span className="text-xs text-gray-400 truncate">{r.source}</span>
        <span className="text-xs text-gray-400 ml-auto shrink-0">{r.date}</span>
      </div>
      <p className="text-xs text-gray-700 leading-relaxed">{r.text}</p>
    </div>
  );
}

// ─── Bigram pills ─────────────────────────────────────────────────────────────

function BigramList({ items, color }: { items: string[]; color: string }) {
  return (
    <div className="flex flex-wrap gap-1.5 pt-1">
      {items.map((b) => (
        <span key={b} className="px-2 py-0.5 rounded-full text-xs font-medium border"
          style={{ borderColor: color, color, backgroundColor: `${color}15` }}>
          {b}
        </span>
      ))}
    </div>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function AspectDetail() {
  const { aspect } = useParams<{ aspect: string }>();
  const navigate = useNavigate();
  const { filters } = useFilters();
  const qs = filtersToQuery(filters);
  const detail = useApi<AspectDetailResponse>(
    () => api.aspectDetail(aspect ?? "", qs),
    [aspect, qs]
  );
  const rec = useApi<{ aspect: string; content: string; context: Record<string, unknown> }>(
    () => api.aspectRecommendation(aspect ?? ""),
    [aspect]
  );

  if (detail.loading) {
    return (
      <div className="space-y-4 max-w-5xl">
        <Skeleton className="h-8 w-48 rounded" />
        <Skeleton className="h-28 rounded-2xl" />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-20 rounded-xl" />)}
        </div>
        <Skeleton className="h-64 rounded-xl" />
      </div>
    );
  }

  if (!detail.data) return null;
  const d = detail.data;
  const pm = PRIORITY_META[d.priority] ?? PRIORITY_META.monitor;
  const tm = TREND_META[d.trend.direction];

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Back + header */}
      <div>
        <button onClick={() => navigate("/business")}
          className="text-xs text-brand hover:underline flex items-center gap-1 mb-3">
          ← Back to Business Insights
        </button>
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-bold uppercase tracking-wide px-2 py-0.5 rounded-full text-white" style={{ backgroundColor: pm.color }}>
                {pm.icon} {pm.label}
              </span>
              <span className="text-xs font-semibold" style={{ color: tm.color }}>{tm.icon} {tm.label} ({d.trend.delta >= 0 ? "+" : ""}{d.trend.delta.toFixed(1)}pp)</span>
            </div>
            <h1 className="text-2xl font-bold text-gray-900">{d.aspect}</h1>
            <p className="text-sm text-gray-500 mt-0.5">{formatNum(d.rows)} reviews · {d.pct_negative.toFixed(0)}% negative · {d.pct_positive.toFixed(0)}% positive</p>
          </div>
          <div className="text-right">
            <p className="text-4xl font-black" style={{ color: nssColor(d.nss) }}>{formatNSS(d.nss)}</p>
            <p className="text-xs text-gray-400 mt-0.5">Net Sentiment Score</p>
          </div>
        </div>
      </div>

      {/* AI Strategic Recommendation */}
      <ChartCard title="Strategic Recommendation">
        {rec.loading ? (
          <div className="space-y-2 py-2">
            <Skeleton className="h-4 w-3/4 rounded" />
            <Skeleton className="h-4 w-full rounded" />
            <Skeleton className="h-4 w-5/6 rounded" />
            <Skeleton className="h-4 w-full rounded" />
            <Skeleton className="h-4 w-2/3 rounded" />
            <p className="text-xs text-gray-400 mt-2 animate-pulse">Generating AI analysis… (cached after first load)</p>
          </div>
        ) : rec.error ? (
          <p className="text-xs text-red-500 py-2">Failed to load AI recommendation: {rec.error}</p>
        ) : rec.data ? (
          <div className="prose prose-sm max-w-none prose-headings:text-gray-800 prose-headings:font-bold prose-h2:text-base prose-h3:text-sm prose-p:text-gray-700 prose-li:text-gray-700 prose-strong:text-gray-900 pt-1">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{rec.data.content}</ReactMarkdown>
          </div>
        ) : null}
      </ChartCard>

      {/* KPI tiles */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Tile label="NSS" value={formatNSS(d.nss)} color={nssColor(d.nss)} />
        <Tile label="Positive" value={`${d.pct_positive.toFixed(1)}%`} color="#4CAF50" sub={`${Math.round(d.rows * d.pct_positive / 100).toLocaleString()} reviews`} />
        <Tile label="Negative" value={`${d.pct_negative.toFixed(1)}%`} color="#F44336" sub={`${Math.round(d.rows * d.pct_negative / 100).toLocaleString()} reviews`} />
        <Tile label="Trend" value={`${tm.icon} ${d.trend.delta >= 0 ? "+" : ""}${d.trend.delta.toFixed(1)}pp`} color={tm.color} sub={`Recent ${formatNSS(d.trend.recent_nss)} vs prior ${formatNSS(d.trend.prior_nss)}`} />
      </div>

      {/* NSS over time */}
      <ChartCard title="NSS over time — volume (bars) + sentiment trend (line)">
        <NSSTimeline data={d.yearly_nss} />
      </ChartCard>

      {/* Sentiment + Rating */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ChartCard title="Sentiment breakdown">
          <SentimentDonut pos={d.pct_positive} neu={d.pct_neutral} neg={d.pct_negative} />
        </ChartCard>
        <ChartCard title="Rating distribution">
          <RatingBar dist={d.rating_dist} />
        </ChartCard>
      </div>

      {/* Bigrams */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ChartCard title="What riders complain about">
          <BigramList items={d.top_complaints} color="#F44336" />
        </ChartCard>
        <ChartCard title="What riders praise">
          <BigramList items={d.top_praises} color="#4CAF50" />
        </ChartCard>
      </div>

      {/* Source NSS */}
      {d.source_dist.length > 0 && (
        <ChartCard title="NSS by source platform">
          <SourceBreakdownChart data={d.source_dist} />
        </ChartCard>
      )}

      {/* Sample reviews */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {d.sample_negative.length > 0 && (
          <ChartCard title={`Negative reviews (${d.sample_negative.length} samples)`}>
            <div className="space-y-2 pt-1">
              {d.sample_negative.map((r, i) => <ReviewCard key={i} r={r} />)}
            </div>
          </ChartCard>
        )}
        {d.sample_positive.length > 0 && (
          <ChartCard title={`Positive reviews (${d.sample_positive.length} samples)`}>
            <div className="space-y-2 pt-1">
              {d.sample_positive.map((r, i) => <ReviewCard key={i} r={r} />)}
            </div>
          </ChartCard>
        )}
      </div>
    </div>
  );
}
