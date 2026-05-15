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
  "What should I do to improve Accessibility?",
  "How is Staff & Service performing?",
  "What are passengers saying about Cleanliness?",
  "How can we reduce Crowding & Comfort complaints?",
  "What is the trend for Fare & Payment System?",
];

function SourceChip({ s }: { s: ReviewSource }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-lg border border-gray-100 bg-white text-xs overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 hover:bg-gray-50 transition-colors text-left"
      >
        <span
          className="shrink-0 w-2 h-2 rounded-full"
          style={{ backgroundColor: SENTIMENT_COLOR[s.sentiment] ?? "#9E9E9E" }}
        />
        <span className="font-semibold text-gray-700 truncate">{s.aspect}</span>
        <span className="text-gray-400 shrink-0">{s.sentiment}</span>
        <span className="text-gray-300 shrink-0 ml-auto">{s.date}</span>
        <span className="text-gray-400 shrink-0">{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div className="px-3 pb-3 pt-1 text-gray-600 border-t border-gray-50 leading-relaxed">
          {s.text}
        </div>
      )}
    </div>
  );
}

function AssistantBubble({ msg }: { msg: ChatMessage }) {
  const [showSources, setShowSources] = useState(false);
  return (
    <div className="flex flex-col gap-2 max-w-[85%]">
      <div className="rounded-2xl rounded-tl-sm bg-white border border-gray-200 shadow-sm px-4 py-3 text-sm text-gray-800 leading-relaxed prose prose-sm max-w-none prose-p:my-1 prose-li:my-0.5 prose-headings:my-2">
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
            <div className="mt-2 space-y-1.5">
              {msg.sources.map((s, i) => (
                <SourceChip key={i} s={s} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function UserBubble({ msg }: { msg: ChatMessage }) {
  return (
    <div className="self-end max-w-[75%] rounded-2xl rounded-tr-sm bg-indigo-600 text-white px-4 py-3 text-sm leading-relaxed">
      {msg.content}
    </div>
  );
}

export default function Chatbot() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

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
          if (line.startsWith("event: sources")) {
            nextIsSources = true;
            continue;
          }
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
            } catch {
              // non-JSON delta — skip
            }
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

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-3rem)] max-w-3xl">
      {/* Header */}
      <div className="mb-4 shrink-0">
        <h1 className="text-2xl font-bold text-gray-900">AI Chatbot</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          Ask anything about BTS passenger sentiment — powered by RAG over real reviews.
        </p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-1">
        {messages.length === 0 && (
          <div className="space-y-3">
            <p className="text-xs font-semibold uppercase tracking-wide text-gray-400">Suggested questions</p>
            <div className="flex flex-wrap gap-2">
              {STARTERS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="px-3 py-2 rounded-xl border border-gray-200 bg-white text-xs text-gray-600 hover:border-indigo-400 hover:text-indigo-600 transition-colors shadow-sm"
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
      <div className="shrink-0 mt-4 flex gap-2 items-end">
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={2}
          placeholder="Ask about any aspect… (Enter to send, Shift+Enter for newline)"
          className="flex-1 border border-gray-200 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400 resize-none shadow-sm"
          disabled={streaming}
        />
        <button
          onClick={() => send(input)}
          disabled={streaming || !input.trim()}
          className={cn(
            "px-5 py-3 rounded-xl text-sm font-semibold transition-colors shadow-sm",
            streaming || !input.trim()
              ? "bg-gray-100 text-gray-400 cursor-not-allowed"
              : "bg-indigo-600 text-white hover:bg-indigo-700"
          )}
        >
          {streaming ? "…" : "Send"}
        </button>
      </div>
    </div>
  );
}
