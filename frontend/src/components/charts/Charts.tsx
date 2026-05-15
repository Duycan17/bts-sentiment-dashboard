import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, ReferenceLine, ComposedChart, Legend,
  PieChart, Pie, ScatterChart, Scatter, ZAxis, Line,
} from "recharts";
import { nssColor, formatNSS } from "../../lib/utils";
import { SENTIMENT_COLORS, BRAND } from "../../lib/colors";
import type { AspectRow, MonthlyRow, NgramRow, ClassMetrics, CalibrationBin, SeasonalityCell } from "../../api/types";

// ─── NSS Diverging Bar ───────────────────────────────────────────────────────

export function NSSBarChart({ data }: { data: AspectRow[] }) {
  const sorted = [...data].sort((a, b) => a.nss - b.nss);
  return (
    <ResponsiveContainer width="100%" height={420}>
      <BarChart data={sorted} layout="vertical" margin={{ left: 180, right: 60, top: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" tickFormatter={(v) => `${v > 0 ? "+" : ""}${v.toFixed(0)}%`} />
        <YAxis type="category" dataKey="aspect" width={175} tick={{ fontSize: 12 }} />
        <Tooltip formatter={(v) => formatNSS(Number(v))} labelFormatter={(l) => `Aspect: ${l}`} />
        <ReferenceLine x={0} stroke="#333" strokeWidth={1.5} />
        <Bar dataKey="nss" name="NSS" radius={[0, 4, 4, 0]}>
          {sorted.map((row) => (
            <Cell key={row.aspect} fill={nssColor(row.nss)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ─── Sentiment Donut ─────────────────────────────────────────────────────────

export function SentimentDonut({ pos, neu, neg }: { pos: number; neu: number; neg: number }) {
  const data = [
    { name: "Positive", value: pos },
    { name: "Neutral", value: neu },
    { name: "Negative", value: neg },
  ];
  return (
    <ResponsiveContainer width="100%" height={280}>
      <PieChart>
        <Pie data={data} cx="50%" cy="50%" innerRadius={70} outerRadius={110} dataKey="value" label={(props) => `${props.name ?? ""} ${((props.percent ?? 0) * 100).toFixed(1)}%`} labelLine={false}>
          {data.map((d) => (
            <Cell key={d.name} fill={SENTIMENT_COLORS[d.name]} />
          ))}
        </Pie>
        <Tooltip formatter={(v) => Number(v).toLocaleString()} />
      </PieChart>
    </ResponsiveContainer>
  );
}

// ─── Monthly Trend (dual axis: volume bars + NSS line) ───────────────────────

export function MonthlyTrendChart({ data }: { data: MonthlyRow[] }) {
  const step = Math.max(1, Math.floor(data.length / 18));
  const ticks = data.filter((_, i) => i % step === 0).map((d) => d.year_month);
  return (
    <ResponsiveContainer width="100%" height={320}>
      <ComposedChart data={data} margin={{ left: 0, right: 40, top: 8, bottom: 40 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="year_month" ticks={ticks} angle={-45} textAnchor="end" tick={{ fontSize: 11 }} interval={0} />
        <YAxis yAxisId="vol" orientation="left" tickFormatter={(v) => v.toLocaleString()} />
        <YAxis yAxisId="nss" orientation="right" tickFormatter={(v) => `${v > 0 ? "+" : ""}${v.toFixed(0)}%`} />
        <Tooltip />
        <Legend verticalAlign="top" />
        <Bar yAxisId="vol" dataKey="rows" name="Volume" fill="#CBD5E1" radius={[2, 2, 0, 0]} />
        <Line yAxisId="nss" type="monotone" dataKey="nss" name="NSS (%)" stroke="#4CAF50" strokeWidth={2} dot={false} />
        <ReferenceLine yAxisId="nss" y={0} stroke="#F44336" strokeDasharray="4 2" />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// ─── Top N-grams Bar ─────────────────────────────────────────────────────────

export function TopNgramsChart({ data, color }: { data: NgramRow[]; color: string }) {
  const sorted = [...data].sort((a, b) => a.count - b.count);
  return (
    <ResponsiveContainer width="100%" height={420}>
      <BarChart data={sorted} layout="vertical" margin={{ left: 130, right: 40, top: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" />
        <YAxis type="category" dataKey="ngram" width={125} tick={{ fontSize: 12 }} />
        <Tooltip />
        <Bar dataKey="count" name="Count" fill={color} radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

// ─── Per-class Metrics Bar ───────────────────────────────────────────────────

export function PerClassMetricsBar({ data }: { data: ClassMetrics[] }) {
  const chartData = data.map((d) => ({ label: d.label, Precision: d.precision, Recall: d.recall, F1: d.f1 }));
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={chartData} margin={{ top: 8, right: 20, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="label" />
        <YAxis domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} />
        <Tooltip formatter={(v) => Number(v).toFixed(3)} />
        <Legend />
        <Bar dataKey="Precision" fill={BRAND} radius={[2, 2, 0, 0]} />
        <Bar dataKey="Recall" fill="#FF9800" radius={[2, 2, 0, 0]} />
        <Bar dataKey="F1" fill="#4CAF50" radius={[2, 2, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

// ─── Confidence Calibration ──────────────────────────────────────────────────

export function ConfidenceCalibration({ data }: { data: CalibrationBin[] }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={data} margin={{ top: 8, right: 20, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="avg_conf" tickFormatter={(v) => v.toFixed(2)} label={{ value: "Confidence", position: "insideBottom", offset: -4 }} />
        <YAxis domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} label={{ value: "Accuracy", angle: -90, position: "insideLeft" }} />
        <Tooltip formatter={(v) => Number(v).toFixed(3)} />
        <ReferenceLine stroke="#9E9E9E" strokeDasharray="4 2" segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} />
        <Line type="monotone" dataKey="accuracy" stroke={BRAND} strokeWidth={2} dot={{ r: 4 }} name="Observed accuracy" />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// ─── Seasonality Heatmap (CSS grid) ─────────────────────────────────────────

const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

export function SeasonalityHeatmap({ data }: { data: SeasonalityCell[] }) {
  const map = new Map<string, number>();
  let maxVal = 1;
  data.forEach((d) => {
    const key = `${d.year}-${d.month}`;
    map.set(key, d.rows);
    if (d.rows > maxVal) maxVal = d.rows;
  });
  const years = [...new Set(data.map((d) => d.year))].sort();
  return (
    <div className="overflow-x-auto">
      <table className="text-xs border-collapse">
        <thead>
          <tr>
            <th className="p-1 text-gray-500 font-normal w-12">Year</th>
            {MONTHS.map((m) => <th key={m} className="p-1 text-gray-500 font-normal w-10">{m}</th>)}
          </tr>
        </thead>
        <tbody>
          {years.map((yr) => (
            <tr key={yr}>
              <td className="p-1 font-semibold text-gray-600">{yr}</td>
              {Array.from({ length: 12 }, (_, i) => i + 1).map((mo) => {
                const v = map.get(`${yr}-${mo}`) ?? 0;
                const intensity = Math.round((v / maxVal) * 255);
                const bg = v === 0 ? "#F3F4F6" : `rgb(${255 - Math.round(intensity * 0.4)}, ${255 - Math.round(intensity * 0.2)}, ${255 - intensity})`;
                return (
                  <td key={mo} className="p-1 text-center rounded" style={{ backgroundColor: bg, color: intensity > 180 ? "#fff" : "#374151" }} title={`${yr}-${String(mo).padStart(2,"0")}: ${v}`}>
                    {v > 0 ? v : ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Volume × NSS Scatter ────────────────────────────────────────────────────

export function VolumeNSSScatter({ data }: { data: AspectRow[] }) {
  const medianVol = [...data].sort((a, b) => a.rows - b.rows)[Math.floor(data.length / 2)]?.rows ?? 0;
  return (
    <ResponsiveContainer width="100%" height={380}>
      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="rows" name="Volume" label={{ value: "Review volume", position: "insideBottom", offset: -10 }} />
        <YAxis dataKey="nss" name="NSS" tickFormatter={(v) => `${v > 0 ? "+" : ""}${v.toFixed(0)}%`} label={{ value: "NSS (%)", angle: -90, position: "insideLeft" }} />
        <ZAxis range={[80, 400]} />
        <Tooltip cursor={{ strokeDasharray: "3 3" }} content={({ payload }) => {
          if (!payload?.length) return null;
          const d = payload[0].payload as AspectRow;
          return (
            <div className="bg-white border rounded shadow p-2 text-xs">
              <p className="font-semibold">{d.aspect}</p>
              <p>Volume: {d.rows.toLocaleString()}</p>
              <p>NSS: {formatNSS(d.nss)}</p>
              <p>Negative: {d.pct_negative.toFixed(1)}%</p>
            </div>
          );
        }} />
        <ReferenceLine y={0} stroke="#F44336" strokeDasharray="4 2" />
        <ReferenceLine x={medianVol} stroke="#9E9E9E" strokeDasharray="4 2" />
        <Scatter data={data} fill={BRAND}>
          {data.map((d) => <Cell key={d.aspect} fill={nssColor(d.nss)} />)}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}

// ─── Confusion Matrix ────────────────────────────────────────────────────────

export function ConfusionMatrix({ labels, matrix, title }: { labels: string[]; matrix: number[][]; title?: string }) {
  const maxVal = Math.max(...matrix.flat(), 1);
  return (
    <div>
      {title && <p className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wide">{title}</p>}
      <div className="overflow-x-auto">
        <table className="text-xs border-collapse">
          <thead>
            <tr>
              <th className="p-1 text-gray-400 font-normal">True ↓ / Pred →</th>
              {labels.map((l) => <th key={l} className="p-2 font-semibold text-gray-700">{l}</th>)}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, ri) => (
              <tr key={labels[ri]}>
                <td className="p-2 font-semibold text-gray-700">{labels[ri]}</td>
                {row.map((v, ci) => {
                  const intensity = v / maxVal;
                  const bg = `rgba(63,81,181,${(intensity * 0.8).toFixed(2)})`;
                  return (
                    <td key={ci} className="p-2 text-center rounded font-mono" style={{ backgroundColor: bg, color: intensity > 0.5 ? "#fff" : "#374151", minWidth: 48 }}>
                      {typeof v === "number" && v < 1 ? v.toFixed(2) : v}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── SHAP Bar ────────────────────────────────────────────────────────────────

export function ShapBar({ contributions, label }: { contributions: { token: string; value: number }[]; label: string }) {
  const posColor = label === "Negative" ? "#F44336" : label === "Neutral" ? "#9E9E9E" : "#4CAF50";
  const data = [...contributions].sort((a, b) => a.value - b.value);
  return (
    <ResponsiveContainer width="100%" height={Math.max(300, data.length * 28)}>
      <BarChart data={data} layout="vertical" margin={{ left: 140, right: 60, top: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" tickFormatter={(v) => v.toFixed(2)} />
        <YAxis type="category" dataKey="token" width={135} tick={{ fontSize: 12 }} />
        <Tooltip formatter={(v) => Number(v).toFixed(4)} />
        <ReferenceLine x={0} stroke="#333" strokeWidth={1.5} />
        <Bar dataKey="value" name="SHAP" radius={[0, 4, 4, 0]}>
          {data.map((d, i) => <Cell key={i} fill={d.value >= 0 ? posColor : "#9E9E9E"} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
