import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "../api/endpoints";
import { cn } from "../lib/utils";
import type { ChatMessage, ReviewSource } from "../api/types";

const SENTIMENT_COLOR: Record<string, string> = {
  Positive: "#4CAF50",
  Negative: "#F44336",
  Neutral: "#9E9E9E",
};

const STARTERS = [
  "How to improve Accessibility?",
  "How is Staff & Service performing?",
  "What about Cleanliness?",
  "Reduce Crowding & Comfort complaints?",
];

function SourceChip({ s }: { s: ReviewSource }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-lg border border-gray-100 bg-gray-50 text-xs overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-gray-100 transition-colors text-left"
      >
        <span
          className="shrink-0 w-2 h-2 rounded-full"
          style={{ backgroundColor: SENTIMENT_COLOR[s.sentiment] ?? "#9E9E9E" }}
        />
        <span className="font-semibold text-gray-700 truncate">{s.aspect}</span>
        <span className="text-gray-400 shrink-0 ml-auto">{s.date}</span>
        <span className="text-gray-400 shrink-0">{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div className="px-2 pb-2 pt-1 text-gray-600 border-t border-gray-100 leading-relaxed">
          {s.text}
        </div>
      )}
    </div>
  );
}

function AssistantBubble({ msg }: { msg: ChatMessage }) {
  const [showSources, setShowSources] = useState(false);
  return (
    <div className="flex flex-col gap-1.5 max-w-[90%]">
      <div className="rounded-2xl rounded-tl-sm bg-white border border-gray-200 shadow-sm px-3 py-2 text-xs text-gray-800 leading-relaxed prose prose-xs max-w-none prose-p:my-1 prose-li:my-0 prose-headings:my-1">
        {msg.content
          ? <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
          : <span className="text-gray-300 animate-pulse">Thinking…</span>
        }
      </div>
      {msg.sources && msg.sources.length > 0 && (
        <div>
          <button
            onClick={() => setShowSources((v) => !v)}
            className="text-xs text-indigo-500 hover:underline font-medium"
          >
            {showSources ? "Hide" : "Show"} {msg.sources.length} sources
          </button>
          {showSources && (
            <div className="mt-1 space-y-1">
              {msg.sources.map((s, i) => <SourceChip key={i} s={s} />)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function UserBubble({ msg }: { msg: ChatMessage }) {
  return (
    <div className="self-end max-w-[80%] rounded-2xl rounded-tr-sm bg-indigo-600 text-white px-3 py-2 text-xs leading-relaxed">
      {msg.content}
    </div>
  );
}

export function ChatBalloon() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, open]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  async function send(text: string) {
    const question = text.trim();
    if (!question || streaming) return;

    const history = messages.map((m) => ({ role: m.role, content: m.content }));
    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "", sources: [] },
    ]);
    setInput("");
    setStreaming(true);

    try {
      const res = await api.chat(question, history);
      if (!res.body) throw new Error("No response body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        let nextIsSources = false;
        for (const line of lines) {
          if (line.startsWith("event: sources")) { nextIsSources = true; continue; }
          if (line.startsWith("data: ")) {
            const payload = line.slice(6);
            if (payload === "[DONE]") { nextIsSources = false; continue; }
            try {
              const parsed = JSON.parse(payload);
              if (nextIsSources && Array.isArray(parsed)) {
                setMessages((prev) => {
                  const next = [...prev];
                  next[next.length - 1] = { ...next[next.length - 1], sources: parsed };
                  return next;
                });
                nextIsSources = false;
              } else if (typeof parsed === "string") {
                setMessages((prev) => {
                  const next = [...prev];
                  next[next.length - 1] = {
                    ...next[next.length - 1],
                    content: next[next.length - 1].content + parsed,
                  };
                  return next;
                });
              }
            } catch { /* skip */ }
          }
        }
      }
    } catch (e) {
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = {
          ...next[next.length - 1],
          content: `Error: ${e instanceof Error ? e.message : "Unknown error"}`,
        };
        return next;
      });
    } finally {
      setStreaming(false);
      inputRef.current?.focus();
    }
  }

  return (
    <>
      {/* Floating balloon button */}
      <button
        onClick={() => setOpen((v) => !v)}
        className={cn(
          "fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full shadow-lg flex items-center justify-center text-2xl transition-all",
          open ? "bg-indigo-700 rotate-12" : "bg-indigo-600 hover:bg-indigo-700 hover:scale-110"
        )}
        aria-label="Open AI Chatbot"
      >
        {open ? "✕" : "🤖"}
      </button>

      {/* Chat panel */}
      {open && (
        <div className="fixed bottom-24 right-6 z-50 w-96 max-h-[70vh] flex flex-col rounded-2xl shadow-2xl border border-gray-200 bg-gray-50 overflow-hidden">
          {/* Header */}
          <div className="flex items-center gap-2 px-4 py-3 bg-indigo-600 text-white shrink-0">
            <span className="text-lg">🤖</span>
            <div className="flex-1 min-w-0">
              <p className="font-semibold text-sm leading-tight">BTS AI Analyst</p>
              <p className="text-indigo-200 text-xs">Ask about passenger sentiment</p>
            </div>
            <button
              onClick={() => setMessages([])}
              className="text-indigo-200 hover:text-white text-xs"
              title="Clear chat"
            >
              Clear
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-3 space-y-3 min-h-0">
            {messages.length === 0 && (
              <div className="space-y-2">
                <p className="text-xs text-gray-400 font-medium">Try asking:</p>
                <div className="flex flex-wrap gap-1.5">
                  {STARTERS.map((s) => (
                    <button
                      key={s}
                      onClick={() => send(s)}
                      className="px-2 py-1 rounded-lg border border-gray-200 bg-white text-xs text-gray-600 hover:border-indigo-400 hover:text-indigo-600 transition-colors"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((msg, i) =>
              msg.role === "user" ? (
                <div key={i} className="flex justify-end">
                  <UserBubble msg={msg} />
                </div>
              ) : (
                <div key={i} className="flex justify-start">
                  <AssistantBubble msg={msg} />
                </div>
              )
            )}
            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div className="shrink-0 flex gap-2 px-3 py-2 border-t border-gray-200 bg-white">
            <input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") send(input); }}
              placeholder="Ask anything…"
              disabled={streaming}
              className="flex-1 border border-gray-200 rounded-lg px-3 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-400"
            />
            <button
              onClick={() => send(input)}
              disabled={streaming || !input.trim()}
              className={cn(
                "px-3 py-2 rounded-lg text-xs font-semibold transition-colors",
                streaming || !input.trim()
                  ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                  : "bg-indigo-600 text-white hover:bg-indigo-700"
              )}
            >
              {streaming ? "…" : "Send"}
            </button>
          </div>
        </div>
      )}
    </>
  );
}
