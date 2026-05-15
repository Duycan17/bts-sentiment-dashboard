import { useFilters } from "../hooks/useFilters";
import { useApi } from "../hooks/useApi";
import { api } from "../api/endpoints";
import { filtersToQuery, formatNSS, formatNum, formatPct, nssColor } from "../lib/utils";
import { InsightCard } from "../components/InsightCard";
import { ChartCard } from "../components/ui/ChartCard";
import { SentimentDonut } from "../components/charts/Charts";
import { Skeleton } from "../components/ui/ChartCard";
import type { KPIsResponse, InsightsResponse, AspectNSSResponse } from "../api/types";

// ─── Hero NSS banner ─────────────────────────────────────────────────────────

function NSSHero({ kpis, loading }: { kpis: KPIsResponse | null; loading: boolean }) {
  if (loading || !kpis) return <Skeleton className="h-36 rounded-2xl" />;
  const arc = Math.min(Math.max((kpis.nss + 100) / 200, 0), 1);
  return (
    <div className="rounded-2xl bg-gradient-to-br from-indigo-600 to-indigo-800 text-white px-8 py-6 flex items-center justify-between shadow-lg">
      <div>
        <p className="text-indigo-200 text-xs font-semibold uppercase tracking-widest mb-1">Overall Net Sentiment Score</p>
        <p className="text-6xl font-black tracking-tight" style={{ color: kpis.nss >= 0 ? "#86efac" : "#fca5a5" }}>
          {formatNSS(kpis.nss)}
        </p>
        <p className="text-indigo-200 text-sm mt-2">{formatNum(kpis.rows)} reviews · {kpis.date_min} → {kpis.date_max}</p>
      </div>
      <div className="hidden md:flex flex-col items-center gap-3">
        {/* Mini gauge */}
        <svg width="140" height="80" viewBox="0 0 140 80">
          <path d="M10,70 A60,60 0 0,1 130,70" fill="none" stroke="rgba(255,255,255,0.15)" strokeWidth="12" strokeLinecap="round" />
          <path
            d="M10,70 A60,60 0 0,1 130,70"
            fill="none"
            stroke={kpis.nss >= 0 ? "#86efac" : "#fca5a5"}
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={`${arc * 188} 188`}
          />
          <text x="70" y="68" textAnchor="middle" fill="white" fontSize="13" fontWeight="700">{formatNSS(kpis.nss)}</text>
        </svg>
        <div className="flex gap-4 text-xs text-indigo-200">
          <span>😞 −100</span>
          <span>😊 +100</span>
        </div>
      </div>
      <div className="hidden lg:grid grid-cols-1 gap-3 text-right">
        {[
          { label: "Positive", pct: kpis.pct_positive, color: "#86efac" },
          { label: "Neutral", pct: kpis.pct_neutral, color: "#cbd5e1" },
          { label: "Negative", pct: kpis.pct_negative, color: "#fca5a5" },
        ].map(({ label, pct, color }) => (
          <div key={label} className="flex items-center gap-3">
            <span className="text-xs text-indigo-200 w-14 text-right">{label}</span>
            <div className="w-28 h-2 rounded-full bg-white/20 overflow-hidden">
              <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
            </div>
            <span className="text-xs font-semibold w-10" style={{ color }}>{pct.toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── KPI cards ───────────────────────────────────────────────────────────────

interface KPICardProps { label: string; value: string; sub?: string; color?: string; icon: string }

function KPICard({ label, value, sub, color, icon }: KPICardProps) {
  return (
    <div className="rounded-xl border border-gray-100 bg-white px-5 py-4 shadow-sm flex items-start gap-3 hover:shadow-md transition-shadow">
      <span className="text-2xl mt-0.5">{icon}</span>
      <div className="min-w-0">
        <p className="text-xs font-semibold uppercase tracking-wide text-gray-400 truncate">{label}</p>
        <p className="text-xl font-bold mt-0.5 leading-tight" style={color ? { color } : undefined}>{value}</p>
        {sub && <p className="text-xs text-gray-400 mt-0.5 truncate">{sub}</p>}
      </div>
    </div>
  );
}

function KPIRow({ kpis, loading }: { kpis: KPIsResponse | null; loading: boolean }) {
  if (loading || !kpis) {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-20 rounded-xl" />)}
      </div>
    );
  }
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
      <KPICard icon="📝" label="Total reviews" value={formatNum(kpis.rows)} sub={`${kpis.n_sources} sources`} />
      <KPICard icon="✅" label="Positive" value={formatPct(kpis.pct_positive)} color="#4CAF50" sub={`${formatNum(Math.round(kpis.rows * kpis.pct_positive / 100))} reviews`} />
      <KPICard icon="❌" label="Negative" value={formatPct(kpis.pct_negative)} color="#F44336" sub={`${formatNum(Math.round(kpis.rows * kpis.pct_negative / 100))} reviews`} />
      <KPICard icon="🎯" label="Avg confidence" value={`${(kpis.avg_confidence * 100).toFixed(1)}%`} sub={`${kpis.n_aspects} aspects tracked`} />
    </div>
  );
}

// ─── Aspect mini-bars ────────────────────────────────────────────────────────

function AspectMiniBar({ data, loading }: { data: AspectNSSResponse | null; loading: boolean }) {
  if (loading || !data) return <Skeleton className="h-64 rounded-xl" />;
  const sorted = [...data.aspects].sort((a, b) => a.nss - b.nss);
  const maxAbs = Math.max(...sorted.map((r) => Math.abs(r.nss)), 1);
  return (
    <div className="space-y-2">
      {sorted.map((row) => {
        const pct = Math.abs(row.nss) / maxAbs * 100;
        const color = nssColor(row.nss);
        return (
          <div key={row.aspect} className="flex items-center gap-2 group">
            <span className="text-xs text-gray-600 w-44 truncate shrink-0 group-hover:text-brand transition-colors">{row.aspect}</span>
            <div className="flex-1 h-5 rounded bg-gray-100 overflow-hidden relative">
              <div
                className="h-full rounded transition-all duration-500"
                style={{ width: `${pct}%`, backgroundColor: color, opacity: 0.85 }}
              />
              <span className="absolute inset-0 flex items-center px-2 text-xs font-mono font-semibold" style={{ color: pct > 40 ? "#fff" : color }}>
                {formatNSS(row.nss)}
              </span>
            </div>
            <span className="text-xs text-gray-400 w-14 text-right shrink-0">{formatNum(row.rows)} rev</span>
          </div>
        );
      })}
    </div>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function Overview() {
  const { filters } = useFilters();
  const qs = filtersToQuery(filters);

  const kpis = useApi<KPIsResponse>(() => api.kpis(qs), [qs]);
  const insights = useApi<InsightsResponse>(() => api.insights(qs), [qs]);
  const aspects = useApi<AspectNSSResponse>(() => api.aspects(qs), [qs]);

  const criticalCards = (insights.data?.cards ?? []).filter((c) => c.severity === "critical");
  const strengthCards = (insights.data?.cards ?? []).filter((c) => c.severity === "good");
  const otherCards = (insights.data?.cards ?? []).filter((c) => c.severity !== "critical" && c.severity !== "good");

  return (
    <div className="space-y-6 max-w-7xl">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 leading-tight">Bangkok BTS Skytrain</h1>
          <p className="text-sm text-gray-500 mt-0.5">Strategic Sentiment Dashboard · Aspect-based analysis · Predicted labels</p>
        </div>
        <span className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-50 text-indigo-700 text-xs font-semibold border border-indigo-100">
          🤖 Model predictions
        </span>
      </div>

      {/* NSS Hero */}
      <NSSHero kpis={kpis.data} loading={kpis.loading} />

      {/* KPI row */}
      <KPIRow kpis={kpis.data} loading={kpis.loading} />

      {/* Main content: aspects + donut */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        <ChartCard title="NSS by aspect" className="lg:col-span-3" loading={aspects.loading}>
          <AspectMiniBar data={aspects.data} loading={aspects.loading} />
        </ChartCard>
        <ChartCard title="Sentiment breakdown" className="lg:col-span-2" loading={kpis.loading}>
          {kpis.data && (
            <SentimentDonut
              pos={kpis.data.pct_positive}
              neu={kpis.data.pct_neutral}
              neg={kpis.data.pct_negative}
            />
          )}
        </ChartCard>
      </div>

      {/* Insights — critical first, then strengths, then others */}
      {!insights.loading && insights.data && (
        <div className="space-y-4">
          {criticalCards.length > 0 && (
            <div>
              <h2 className="text-sm font-semibold text-red-600 uppercase tracking-wide mb-2 flex items-center gap-1.5">
                <span>🔴</span> Critical pain points
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
                {criticalCards.map((c, i) => <InsightCard key={i} card={c} />)}
              </div>
            </div>
          )}
          {strengthCards.length > 0 && (
            <div>
              <h2 className="text-sm font-semibold text-green-600 uppercase tracking-wide mb-2 flex items-center gap-1.5">
                <span>🟢</span> Strengths to amplify
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
                {strengthCards.map((c, i) => <InsightCard key={i} card={c} />)}
              </div>
            </div>
          )}
          {otherCards.length > 0 && (
            <div>
              <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1.5">
                <span>🔵</span> Notes
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {otherCards.map((c, i) => <InsightCard key={i} card={c} />)}
              </div>
            </div>
          )}
        </div>
      )}
      {insights.loading && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-28 rounded-xl" />)}
        </div>
      )}
    </div>
  );
}

