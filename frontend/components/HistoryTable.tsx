"use client";

import { HistoryRow, RiskLevel } from "@/lib/types";

const riskBadge: Record<RiskLevel, string> = {
  HIGH: "bg-red-900/50 text-red-400 border border-red-700",
  MED:  "bg-orange-900/50 text-orange-400 border border-orange-700",
  LOW:  "bg-green-900/50 text-green-400 border border-green-700",
};

function getRisk(probability: number): RiskLevel {
  if (probability >= 0.60) return "HIGH";
  if (probability >= 0.30) return "MED";
  return "LOW";
}

interface Props {
  rows: HistoryRow[];
}

export default function HistoryTable({ rows }: Props) {
  if (rows.length === 0) {
    return (
      <div className="text-center py-16 text-gray-500">
        No predictions yet. <a href="/" className="text-blue-400 hover:underline">Make your first one →</a>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-2xl border border-gray-800">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-gray-900 text-gray-400 text-xs uppercase tracking-wider">
            <th className="px-4 py-3 text-left">ID</th>
            <th className="px-4 py-3 text-left">Date</th>
            <th className="px-4 py-3 text-left">Job Role</th>
            <th className="px-4 py-3 text-center">Age</th>
            <th className="px-4 py-3 text-center">Risk</th>
            <th className="px-4 py-3 text-center">Probability</th>
            <th className="px-4 py-3 text-center">Model</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-800">
          {rows.map((row) => {
            const risk = getRisk(row.probability);
            return (
              <tr key={row.id} className="bg-gray-950 hover:bg-gray-900 transition-colors">
                <td className="px-4 py-3 font-mono text-gray-500 text-xs">
                  {row.id.split("-")[0]}
                </td>
                <td className="px-4 py-3 text-gray-300">
                  {new Date(row.created_at).toLocaleDateString("en-US", {
                    month: "short", day: "numeric", year: "numeric",
                  })}
                </td>
                <td className="px-4 py-3 text-white">{row.job_role}</td>
                <td className="px-4 py-3 text-center text-gray-300">{row.age}</td>
                <td className="px-4 py-3 text-center">
                  <span className={`px-2 py-1 rounded-full text-xs font-semibold ${riskBadge[risk]}`}>
                    {risk}
                  </span>
                </td>
                <td className="px-4 py-3 text-center text-white font-medium">
                  {Math.round(row.probability * 100)}%
                </td>
                <td className="px-4 py-3 text-center font-mono text-gray-500 text-xs">
                  {row.model_version}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
