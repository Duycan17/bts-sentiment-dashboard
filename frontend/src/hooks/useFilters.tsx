import { createContext, useContext, useReducer, type ReactNode } from "react";
import { DEFAULT_FILTERS, type FilterState } from "../lib/utils";

type Action =
  | { type: "SET"; payload: Partial<FilterState> }
  | { type: "RESET" };

function reducer(state: FilterState, action: Action): FilterState {
  if (action.type === "RESET") return DEFAULT_FILTERS;
  return { ...state, ...action.payload };
}

interface FilterCtx {
  filters: FilterState;
  dispatch: React.Dispatch<Action>;
}

const Ctx = createContext<FilterCtx | null>(null);

export function FilterProvider({ children }: { children: ReactNode }) {
  const [filters, dispatch] = useReducer(reducer, DEFAULT_FILTERS);
  return <Ctx.Provider value={{ filters, dispatch }}>{children}</Ctx.Provider>;
}

export function useFilters() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useFilters must be inside FilterProvider");
  return ctx;
}
