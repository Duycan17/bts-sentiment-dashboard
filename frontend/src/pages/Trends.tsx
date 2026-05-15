import { useFilters } from "../hooks/useFilters";
import { useApi } from "../hooks/useApi";
import { api } from "../api/endpoints";
import { filtersToQuery } from "../lib/utils";
import { ChartCard } from "../components/ui/ChartCard";
import { MonthlyTrendChart, SeasonalityHeatmap } from "../components/charts/Charts";
import type { TrendResponse } from "../api/types";

export default function Trends() {
  const { filters } = useFilters();
  const qs = filtersToQuery(filters);
  const trends = useApi<TrendResponse>(() => api.trends(qs), [qs]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Trends Over Time</h1>
        <p className="text-sm text-gray-500 mt-1">How rider sentiment moved across months and seasons (2012–2026).</p>
      </div>

      <ChartCard title="Monthly volume & NSS" loading={trends.loading}>
        {trends.data && <MonthlyTrendChart data={trends.data.monthly} />}
      </ChartCard>

      <ChartCard title="Volume seasonality (year × month)" loading={trends.loading}>
        {trends.data && <SeasonalityHeatmap data={trends.data.seasonality} />}
      </ChartCard>

      {/* NSS summary table */}
      <ChartCard title="Monthly NSS — last 24 months" loading={trends.loading}>
        {trends.data && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-gray-100">
                  {["Month", "Reviews", "NSS"].map((h) => (
                    <th key={h} className="text-left py-2 px-3 text-gray-500 font-semibold uppercase tracking-wide">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[...trends.data.monthly].reverse().slice(0, 24).map((row) => (
                  <tr key={row.year_month} className="border-b border-gray-50 hover:bg-gray-50">
                    <td className="py-1.5 px-3 font-medium text-gray-700">{row.year_month}</td>
                    <td className="py-1.5 px-3 text-gray-600">{row.rows.toLocaleString()}</td>
                    <td className="py-1.5 px-3 font-mono font-semibold" style={{ color: row.nss < 0 ? "#F44336" : row.nss < 30 ? "#FF9800" : "#4CAF50" }}>
                      {row.nss >= 0 ? "+" : ""}{row.nss.toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </ChartCard>
    </div>
  );
}
