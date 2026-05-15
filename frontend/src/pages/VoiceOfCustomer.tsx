import { useFilters } from "../hooks/useFilters";
import { useApi } from "../hooks/useApi";
import { api } from "../api/endpoints";
import { filtersToQuery } from "../lib/utils";
import { ChartCard } from "../components/ui/ChartCard";
import { TopNgramsChart } from "../components/charts/Charts";
import type { VoiceResponse } from "../api/types";

export default function VoiceOfCustomer() {
  const { filters } = useFilters();
  const qs = filtersToQuery(filters);
  const voice = useApi<VoiceResponse>(() => api.voice(qs), [qs]);

  const wcPos = api.wordcloudUrl("positive", qs);
  const wcNeg = api.wordcloudUrl("negative", qs);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Voice of Customer</h1>
        <p className="text-sm text-gray-500 mt-1">What riders are actually saying — by polarity.</p>
      </div>

      {/* Word clouds */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ChartCard title="Positive reviews — word cloud">
          <img key={`pos-${qs}`} src={wcPos} alt="Positive word cloud" className="w-full rounded-lg" onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }} />
        </ChartCard>
        <ChartCard title="Negative reviews — word cloud">
          <img key={`neg-${qs}`} src={wcNeg} alt="Negative word cloud" className="w-full rounded-lg" onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }} />
        </ChartCard>
      </div>

      {/* Top bigrams */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ChartCard title="Top bigrams — positive reviews" loading={voice.loading}>
          {voice.data && <TopNgramsChart data={voice.data.positive_ngrams} color="#4CAF50" />}
        </ChartCard>
        <ChartCard title="Top bigrams — negative reviews" loading={voice.loading}>
          {voice.data && <TopNgramsChart data={voice.data.negative_ngrams} color="#F44336" />}
        </ChartCard>
      </div>
    </div>
  );
}
