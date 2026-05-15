import { useFilters } from "../hooks/useFilters";
import { useApi } from "../hooks/useApi";
import { api } from "../api/endpoints";
import { filtersToQuery } from "../lib/utils";
import { ChartCard } from "../components/ui/ChartCard";
import { ConfusionMatrix, PerClassMetricsBar, ConfidenceCalibration } from "../components/charts/Charts";
import type { PerformanceResponse } from "../api/types";

export default function ModelPerformance() {
  const { filters } = useFilters();
  const qs = filtersToQuery(filters);
  const perf = useApi<PerformanceResponse>(() => api.performance(qs), [qs]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Model Performance</h1>
        <p className="text-sm text-gray-500 mt-1">Where the sentiment model can be trusted — and where to be cautious.</p>
      </div>

      {/* Top metrics */}
      {perf.data && (
        <div className="grid grid-cols-3 gap-3">
          {[
            { label: "Accuracy", value: (perf.data.accuracy * 100).toFixed(1) + "%" },
            { label: "Macro F1", value: perf.data.macro_f1.toFixed(3) },
            { label: "Aspect agreement", value: (perf.data.aspect_agreement * 100).toFixed(1) + "%",
              note: perf.data.aspect_agreement < 0.5 ? "⚠ Below 50% — use ground truth labels" : undefined },
          ].map(({ label, value, note }) => (
            <div key={label} className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
              <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">{label}</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
              {note && <p className="text-xs text-orange-500 mt-1">{note}</p>}
            </div>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartCard title="Confusion matrix — counts" loading={perf.loading}>
          {perf.data && <ConfusionMatrix labels={perf.data.confusion_matrix.labels} matrix={perf.data.confusion_matrix.counts} />}
        </ChartCard>
        <ChartCard title="Confusion matrix — normalized" loading={perf.loading}>
          {perf.data && <ConfusionMatrix labels={perf.data.confusion_matrix.labels} matrix={perf.data.confusion_matrix.normalized} />}
        </ChartCard>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartCard title="Per-class precision / recall / F1" loading={perf.loading}>
          {perf.data && <PerClassMetricsBar data={perf.data.per_class} />}
        </ChartCard>
        <ChartCard title="Confidence calibration" loading={perf.loading}>
          {perf.data && <ConfidenceCalibration data={perf.data.calibration} />}
        </ChartCard>
      </div>

      <ChartCard title="Per-aspect weighted F1" loading={perf.loading}>
        {perf.data && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-gray-100">
                  {["Aspect", "Reviews", "Accuracy", "Macro F1", "Weighted F1"].map((h) => (
                    <th key={h} className="text-left py-2 px-3 text-gray-500 font-semibold uppercase tracking-wide">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {perf.data.per_aspect_f1.map((row) => (
                  <tr key={row.aspect} className="border-b border-gray-50 hover:bg-gray-50">
                    <td className="py-2 px-3 font-medium text-gray-800">{row.aspect}</td>
                    <td className="py-2 px-3 text-gray-600">{row.rows.toLocaleString()}</td>
                    <td className="py-2 px-3 font-mono">{(row.accuracy * 100).toFixed(1)}%</td>
                    <td className="py-2 px-3 font-mono">{row.macro_f1.toFixed(3)}</td>
                    <td className="py-2 px-3 font-mono font-semibold" style={{ color: row.weighted_f1 > 0.8 ? "#4CAF50" : row.weighted_f1 > 0.6 ? "#FF9800" : "#F44336" }}>
                      {row.weighted_f1.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </ChartCard>

      <ChartCard title="Aspect prediction agreement (ground truth rows vs. predicted cols)" loading={perf.loading}>
        {perf.data && (
          <ConfusionMatrix
            labels={perf.data.aspect_confusion.labels}
            matrix={perf.data.aspect_confusion.normalized}
          />
        )}
      </ChartCard>
    </div>
  );
}
