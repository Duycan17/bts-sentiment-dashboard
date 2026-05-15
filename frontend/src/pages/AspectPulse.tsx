import { useFilters } from "../hooks/useFilters";
import { useApi } from "../hooks/useApi";
import { api } from "../api/endpoints";
import { filtersToQuery, formatNSS } from "../lib/utils";
import { ChartCard } from "../components/ui/ChartCard";
import { NSSBarChart, VolumeNSSScatter } from "../components/charts/Charts";
import { InsightCard } from "../components/InsightCard";
import type { AspectNSSResponse, InsightsResponse } from "../api/types";
import { nssColor } from "../lib/utils";

export default function AspectPulse() {
  const { filters } = useFilters();
  const qs = filtersToQuery(filters);
  const aspects = useApi<AspectNSSResponse>(() => api.aspects(qs), [qs]);
  const insights = useApi<InsightsResponse>(() => api.insights(qs), [qs]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Aspect Pulse</h1>
        <p className="text-sm text-gray-500 mt-1">Where the brand wins or loses — ranked by NSS and review volume.</p>
      </div>

      <ChartCard title="Net Sentiment Score by aspect" loading={aspects.loading}>
        {aspects.data && <NSSBarChart data={aspects.data.aspects} />}
      </ChartCard>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartCard title="Volume vs. NSS — where to invest" loading={aspects.loading}>
          {aspects.data && <VolumeNSSScatter data={aspects.data.aspects} />}
        </ChartCard>

        <ChartCard title="Aspect breakdown" loading={aspects.loading}>
          {aspects.data && (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-gray-100">
                    {["Aspect", "Reviews", "NSS", "% Neg"].map((h) => (
                      <th key={h} className="text-left py-2 px-2 text-gray-500 font-semibold uppercase tracking-wide">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[...aspects.data.aspects].sort((a, b) => a.nss - b.nss).map((row) => (
                    <tr key={row.aspect} className="border-b border-gray-50 hover:bg-gray-50">
                      <td className="py-2 px-2 font-medium text-gray-800">{row.aspect}</td>
                      <td className="py-2 px-2 text-gray-600">{row.rows.toLocaleString()}</td>
                      <td className="py-2 px-2 font-mono font-semibold" style={{ color: nssColor(row.nss) }}>{formatNSS(row.nss)}</td>
                      <td className="py-2 px-2 text-gray-600">{row.pct_negative.toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </ChartCard>
      </div>

      <div>
        <h2 className="text-base font-semibold text-gray-700 mb-3">Aspect-driven recommendations</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {(insights.data?.cards ?? [])
            .filter((c) => c.severity === "critical" || c.severity === "good")
            .map((c, i) => <InsightCard key={i} card={c} />)}
        </div>
      </div>
    </div>
  );
}
