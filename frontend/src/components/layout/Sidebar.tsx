import { NavLink } from "react-router-dom";
import { useFilters } from "../../hooks/useFilters";
import { useApi } from "../../hooks/useApi";
import { api } from "../../api/endpoints";
import { cn } from "../../lib/utils";
import type { MetaResponse } from "../../api/types";

const NAV = [
  { to: "/", label: "Overview", icon: "📊" },
  { to: "/business", label: "Business Insights", icon: "💼" },
  { to: "/chat", label: "AI Chatbot", icon: "🤖" },
  { to: "/aspects", label: "Aspect Pulse", icon: "🎯" },
  { to: "/trends", label: "Trends", icon: "📈" },
  { to: "/voice", label: "Voice of Customer", icon: "💬" },
  { to: "/performance", label: "Model Performance", icon: "🤖" },
  { to: "/errors", label: "Error Analysis", icon: "🔬" },
];

export function Sidebar() {
  const { filters, dispatch } = useFilters();
  const { data: meta } = useApi<MetaResponse>(() => api.meta(), []);

  function multi(field: "aspects" | "sentiments", value: string) {
    const cur = filters[field] as string[];
    const next = cur.includes(value) ? cur.filter((v) => v !== value) : [...cur, value];
    dispatch({ type: "SET", payload: { [field]: next } });
  }

  return (
    <aside className="w-64 shrink-0 bg-white border-r border-gray-200 flex flex-col h-screen sticky top-0 overflow-y-auto">
      {/* Logo */}
      <div className="px-5 py-4 border-b border-gray-100">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🚆</span>
          <div>
            <p className="font-bold text-brand text-sm leading-tight">BTS Sentiment</p>
            <p className="text-xs text-gray-400">Strategic Dashboard</p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="px-3 py-3 border-b border-gray-100">
        {NAV.map(({ to, label, icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              cn("flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors mb-0.5",
                isActive ? "bg-brand text-white" : "text-gray-600 hover:bg-gray-100")
            }
          >
            <span>{icon}</span>
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Filters */}
      <div className="px-4 py-3 flex-1 space-y-4 text-xs">
        <div className="flex items-center justify-between">
          <span className="font-semibold text-gray-700 uppercase tracking-wide text-xs">Filters</span>
          <button onClick={() => dispatch({ type: "RESET" })} className="text-brand hover:underline text-xs">Reset</button>
        </div>

        {/* Date range */}
        <div>
          <label className="block text-gray-500 mb-1 font-medium">Date range</label>
          <input type="date" value={filters.dateStart} onChange={(e) => dispatch({ type: "SET", payload: { dateStart: e.target.value } })}
            className="w-full border border-gray-200 rounded px-2 py-1 mb-1 text-xs" />
          <input type="date" value={filters.dateEnd} onChange={(e) => dispatch({ type: "SET", payload: { dateEnd: e.target.value } })}
            className="w-full border border-gray-200 rounded px-2 py-1 text-xs" />
        </div>

        {/* Sentiment */}
        <div>
          <label className="block text-gray-500 mb-1 font-medium">Sentiment</label>
          <div className="flex gap-1 flex-wrap">
            {["Positive", "Neutral", "Negative"].map((s) => (
              <button key={s} onClick={() => multi("sentiments", s)}
                className={cn("px-2 py-0.5 rounded-full border text-xs font-medium transition-colors",
                  filters.sentiments.includes(s) ? "text-white border-transparent" : "bg-white text-gray-600 border-gray-200 hover:border-gray-400")}
                style={filters.sentiments.includes(s) ? { backgroundColor: s === "Positive" ? "#4CAF50" : s === "Negative" ? "#F44336" : "#9E9E9E" } : undefined}>
                {s}
              </button>
            ))}
          </div>
        </div>

        {/* Aspects */}
        {meta && (
          <div>
            <label className="block text-gray-500 mb-1 font-medium">Aspects</label>
            <div className="space-y-0.5 max-h-40 overflow-y-auto">
              {meta.aspects.map((a) => (
                <label key={a} className="flex items-center gap-1.5 cursor-pointer hover:text-brand">
                  <input type="checkbox" checked={filters.aspects.includes(a)} onChange={() => multi("aspects", a)} className="accent-brand" />
                  <span className="truncate">{a}</span>
                </label>
              ))}
            </div>
          </div>
        )}

      </div>

      <div className="px-4 py-3 border-t border-gray-100 text-xs text-gray-400">
        20,782 reviews · 2012–2026
      </div>
    </aside>
  );
}
