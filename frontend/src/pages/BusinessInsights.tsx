import { useNavigate } from "react-router-dom";
import { useFilters } from "../hooks/useFilters";
import { useApi } from "../hooks/useApi";
import { api } from "../api/endpoints";
import { filtersToQuery, formatNSS, formatNum, nssColor } from "../lib/utils";
import { ChartCard } from "../components/ui/ChartCard";
import { Skeleton } from "../components/ui/ChartCard";
import type { BusinessResponse, AspectDeepDive, YoYRow } from "../api/types";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import { BRAND } from "../lib/colors";

const TREND_META = {
  improving: { icon: "↑", color: "#4CAF50" },
  declining: { icon: "↓", color: "#F44336" },
  stable:    { icon: "→", color: "#9E9E9E" },
};

const PRIORITY_META = {
  urgent:   { color: "#F44336", bg: "#FFF5F5", label: "Urgent",   icon: "🔴" },
  monitor:  { color: "#FF9800", bg: "#FFFBF0", label: "Monitor",  icon: "🟠" },
  leverage: { color: "#4CAF50", bg: "#F5FFF5", label: "Leverage", icon: "🟢" },
};


function YoYChart({ data }: { data: YoYRow[] }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 8, right: 20, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="year" tick={{ fontSize: 12 }} />
        <YAxis tickFormatter={(v) => `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(0)}%`} tick={{ fontSize: 11 }} />
        <Tooltip formatter={(v) => `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(1)}%`} labelFormatter={(l) => `Year: ${l}`} />
        <ReferenceLine y={0} stroke="#F44336" strokeDasharray="4 2" />
        <Line type="monotone" dataKey="nss" stroke={BRAND} strokeWidth={2.5} dot={{ r: 5, fill: BRAND }} name="NSS" />
      </LineChart>
    </ResponsiveContainer>
  );
}

function DeepDiveCard({ d, onClick }: { d: AspectDeepDive; onClick: () => void }) {
  const pm = PRIORITY_META[d.priority];
  const tm = TREND_META[d.trend.direction];
  return (
    <div
      className="rounded-xl border bg-white shadow-sm overflow-hidden hover:shadow-md transition-all cursor-pointer group"
      style={{ borderLeftWidth: 4, borderLeftColor: pm.color }}
      onClick={onClick}
    >
      <div className="px-4 pt-4 pb-3" style={{ backgroundColor: pm.bg }}>
        <div className="flex items-center gap-2 mb-0.5 flex-wrap">
          <span className="text-xs font-bold uppercase tracking-wide" style={{ color: pm.color }}>{pm.icon} {pm.label}</span>
          <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-white border font-semibold" style={{ color: nssColor(d.nss) }}>
            NSS {formatNSS(d.nss)}
          </span>
          <span className="text-xs font-semibold ml-auto" style={{ color: tm.color }}>
            {tm.icon} {d.trend.delta >= 0 ? "+" : ""}{d.trend.delta.toFixed(1)}pp
          </span>
        </div>
        <p className="font-bold text-gray-900 text-sm">{d.aspect}</p>
        <p className="text-xs text-gray-500 mt-0.5">{formatNum(d.rows)} reviews · {d.pct_negative.toFixed(0)}% neg · {d.pct_positive.toFixed(0)}% pos</p>
      </div>

      <div className="flex h-1.5">
        <div style={{ width: `${d.pct_positive}%`, backgroundColor: "#4CAF50" }} />
        <div style={{ width: `${100 - d.pct_positive - d.pct_negative}%`, backgroundColor: "#E5E7EB" }} />
        <div style={{ width: `${d.pct_negative}%`, backgroundColor: "#F44336" }} />
      </div>

      <div className="px-4 py-3 grid grid-cols-2 gap-3 border-t border-gray-50">
        <div>
          <p className="text-xs font-semibold text-red-500 mb-1">Top complaints</p>
          <ul className="space-y-0.5">
            {d.top_complaints.slice(0, 3).map((c) => (
              <li key={c} className="text-xs text-gray-600 flex items-start gap-1">
                <span className="text-red-400 shrink-0">•</span>{c}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <p className="text-xs font-semibold text-green-600 mb-1">Top praises</p>
          <ul className="space-y-0.5">
            {d.top_praises.slice(0, 3).map((c) => (
              <li key={c} className="text-xs text-gray-600 flex items-start gap-1">
                <span className="text-green-500 shrink-0">•</span>{c}
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="px-4 py-3 bg-gray-50 border-t border-gray-100 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-0.5">Recommended action</p>
          <p className="text-xs text-gray-800 leading-relaxed line-clamp-2">{d.action}</p>
        </div>
        <span className="text-xs text-brand font-semibold shrink-0 group-hover:underline mt-0.5 whitespace-nowrap">
          Full analysis →
        </span>
      </div>
    </div>
  );
}

function Section({ title, sub, items, onSelect }: {
  title: string; sub: string; items: AspectDeepDive[]; onSelect: (d: AspectDeepDive) => void;
}) {
  if (!items.length) return null;
  return (
    <div>
      <h2 className="text-sm font-bold uppercase tracking-wide mb-3 flex items-center gap-2">
        <span>{title}</span>
        <span className="text-xs font-normal text-gray-400 normal-case">{sub}</span>
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {items.map((d) => <DeepDiveCard key={d.aspect} d={d} onClick={() => onSelect(d)} />)}
      </div>
    </div>
  );
}

export default function BusinessInsights() {
  const { filters } = useFilters();
  const qs = filtersToQuery(filters);
  const biz = useApi<BusinessResponse>(() => api.business(qs), [qs]);
  const navigate = useNavigate();

  if (biz.loading) {
    return (
      <div className="space-y-4 max-w-7xl">
        <Skeleton className="h-52 rounded-xl" />
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-64 rounded-xl" />)}
        </div>
      </div>
    );
  }

  if (!biz.data) return null;
  const { summary, aspect_deep_dives, yoy_nss } = biz.data;

  function handleSelect(d: AspectDeepDive) {
    navigate(`/business/${encodeURIComponent(d.aspect)}`);
  }

  return (
    <div className="space-y-6 max-w-7xl">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Business Insights</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          Agency-grade analysis · Click any card for full detail · {summary.date_range}
        </p>
      </div>

      <ChartCard title="Year-over-year NSS trend">
        <YoYChart data={yoy_nss} />
      </ChartCard>

      <Section
        title="🔴 Urgent — act now"
        sub="High volume · Negative NSS"
        items={aspect_deep_dives.filter((d) => d.priority === "urgent")}
        onSelect={handleSelect}
      />
      <Section
        title="🟠 Monitor — watch closely"
        sub="Mixed signals or low volume"
        items={aspect_deep_dives.filter((d) => d.priority === "monitor")}
        onSelect={handleSelect}
      />
      <Section
        title="🟢 Leverage — amplify strengths"
        sub="High volume · Positive NSS"
        items={aspect_deep_dives.filter((d) => d.priority === "leverage")}
        onSelect={handleSelect}
      />
    </div>
  );
}
