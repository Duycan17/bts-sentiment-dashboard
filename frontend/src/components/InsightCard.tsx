import { SEVERITY_BG, SEVERITY_COLORS } from "../lib/colors";
import type { InsightCard as InsightCardType } from "../api/types";

const BADGE: Record<string, string> = {
  critical: "🔴 Critical",
  warning: "🟠 Warning",
  good: "🟢 Strength",
  info: "🔵 Note",
};

export function InsightCard({ card }: { card: InsightCardType }) {
  const color = SEVERITY_COLORS[card.severity];
  const bg = SEVERITY_BG[card.severity];
  return (
    <div
      className="rounded-xl border p-4 shadow-sm"
      style={{ borderLeftWidth: 5, borderLeftColor: color, backgroundColor: bg }}
    >
      <div className="flex items-center gap-2 mb-1">
        <span className="text-xs font-bold uppercase tracking-wide" style={{ color }}>
          {BADGE[card.severity]}
        </span>
        {card.metric && (
          <span className="ml-auto text-xs font-mono font-semibold px-2 py-0.5 rounded-full bg-white border" style={{ color }}>
            {card.metric}
          </span>
        )}
      </div>
      <p className="font-semibold text-gray-900 text-sm mb-1">{card.title}</p>
      <p className="text-xs text-gray-700 mb-1"><span className="font-medium">Finding: </span>{card.finding}</p>
      <p className="text-xs text-gray-700"><span className="font-medium">Recommendation: </span>{card.recommendation}</p>
    </div>
  );
}
