import { BrowserRouter, Routes, Route } from "react-router-dom";
import { FilterProvider } from "./hooks/useFilters";
import { Sidebar } from "./components/layout/Sidebar";
import { ChatBalloon } from "./components/ChatBalloon";
import Overview from "./pages/Overview";
import BusinessInsights from "./pages/BusinessInsights";
import AspectDetail from "./pages/AspectDetail";
import AspectPulse from "./pages/AspectPulse";
import Trends from "./pages/Trends";
import VoiceOfCustomer from "./pages/VoiceOfCustomer";
import ModelPerformance from "./pages/ModelPerformance";
import Chatbot from "./pages/Chatbot";
import ErrorAnalysis from "./pages/ErrorAnalysis";

export default function App() {
  return (
    <BrowserRouter>
      <FilterProvider>
        <div className="flex min-h-screen bg-gray-50">
          <Sidebar />
          <main className="flex-1 p-6 overflow-y-auto">
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/business" element={<BusinessInsights />} />
              <Route path="/business/:aspect" element={<AspectDetail />} />
              <Route path="/aspects" element={<AspectPulse />} />
              <Route path="/trends" element={<Trends />} />
              <Route path="/voice" element={<VoiceOfCustomer />} />
              <Route path="/performance" element={<ModelPerformance />} />
              <Route path="/errors" element={<ErrorAnalysis />} />
              <Route path="/chat" element={<Chatbot />} />
            </Routes>
          </main>
          <ChatBalloon />
        </div>
      </FilterProvider>
    </BrowserRouter>
  );
}
