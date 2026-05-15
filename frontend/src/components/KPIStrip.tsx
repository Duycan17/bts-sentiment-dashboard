import { formatNSS, formatNum, formatPct } from "../lib/utils";
import type { KPIsResponse } from "../api/types";
import { Skeleton } from "./ui/ChartCard";

interface KPICardProps {
  label: string;
  value: string;
  sub?: string;
  color?: string;
}

function KPICard({ label, value, sub, color }: KPICardProps) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm flex flex-col gap-1">
      <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">{label}</span>
      <span className="text-2xl font-bold" style={color ? { color } : undefined}>{value}</span>
      {sub && <span className="text-xs text-gray-400">{sub}</span>}
    </div>
  );
}

export function KPIStrip({ kpis, loading }: { kpis: KPIsResponse | null; loading: boolean }) {
  if (loading || !kpis) {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-24 rounded-xl" />)}
      </div>
    );
  }
  const nssColor = kpis.nss < 0 ? "#F44336" : kpis.nss < 30 ? "#FF9800" : "#4CAF50";
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
      <KPICard label="Reviews" value={formatNum(kpis.rows)} />
      <KPICard label="NSS" value={formatNSS(kpis.nss)} color={nssColor} />
      <KPICard label="Positive" value={formatPct(kpis.pct_positive)} color="#4CAF50" />
      <KPICard label="Negative" value={formatPct(kpis.pct_negative)} color="#F44336" />
      <KPICard label="Aspects" value={String(kpis.n_aspects)} />
      <KPICard label="Date span" value={kpis.date_min} sub={`→ ${kpis.date_max}`} />
    </div>
  );
}
