import { PredictionResult, RiskLevel } from "@/lib/types";

const riskConfig: Record<RiskLevel, { color: string; bg: string; border: string; label: string }> = {
  HIGH: { color: "text-red-400", bg: "bg-red-900/30", border: "border-red-500", label: "High Risk" },
  MED:  { color: "text-orange-400", bg: "bg-orange-900/30", border: "border-orange-500", label: "Medium Risk" },
  LOW:  { color: "text-green-400", bg: "bg-green-900/30", border: "border-green-500", label: "Low Risk" },
};

interface Props {
  result: PredictionResult;
}

export default function ResultCard({ result }: Props) {
  const cfg = riskConfig[result.risk_level];
  const pct = Math.round(result.probability * 100);

  return (
    <div className={`rounded-2xl border ${cfg.border} ${cfg.bg} p-8`}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <p className="text-gray-400 text-sm uppercase tracking-wider mb-1">Attrition Risk</p>
          <p className={`text-5xl font-bold ${cfg.color}`}>{cfg.label}</p>
        </div>
        <div className="text-right">
          <p className="text-gray-400 text-sm mb-1">Probability</p>
          <p className={`text-5xl font-bold ${cfg.color}`}>{pct}%</p>
        </div>
      </div>

      {/* Probability bar */}
      <div className="w-full bg-gray-800 rounded-full h-3 mb-4">
        <div
          className={`h-3 rounded-full transition-all duration-500 ${
            result.risk_level === "HIGH" ? "bg-red-500" :
            result.risk_level === "MED" ? "bg-orange-500" : "bg-green-500"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>

      <div className="flex justify-between text-xs text-gray-500">
        <span>0% (Stays)</span>
        <span className="text-gray-400">30% MED</span>
        <span className="text-gray-400">60% HIGH</span>
        <span>100% (Leaves)</span>
      </div>

      <div className="mt-6 pt-4 border-t border-gray-700 flex justify-between text-sm text-gray-400">
        <span>Verdict: <span className="text-white font-medium">{result.prediction ? "Likely to Leave" : "Likely to Stay"}</span></span>
        <span>Model: <span className="text-white font-mono">{result.model_version}</span></span>
      </div>
    </div>
  );
}
