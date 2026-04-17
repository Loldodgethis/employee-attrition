"use client";

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell,
} from "recharts";
import { ShapValue } from "@/lib/types";

interface Props {
  shapValues: ShapValue[];
}

export default function ShapChart({ shapValues }: Props) {
  const sorted = [...shapValues].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6">
      <h2 className="text-lg font-semibold text-white mb-1">Top Contributing Factors</h2>
      <p className="text-gray-400 text-sm mb-6">
        Red bars increase attrition risk · Green bars decrease it
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart
          data={sorted}
          layout="vertical"
          margin={{ top: 0, right: 40, left: 120, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
          <XAxis
            type="number"
            tickFormatter={(v) => v.toFixed(2)}
            stroke="#6b7280"
            tick={{ fill: "#9ca3af", fontSize: 12 }}
          />
          <YAxis
            type="category"
            dataKey="feature"
            stroke="#6b7280"
            tick={{ fill: "#d1d5db", fontSize: 13 }}
            width={115}
          />
          <Tooltip
            formatter={(value) => [
              typeof value === "number" ? value.toFixed(4) : value,
              "SHAP value",
            ]}
            contentStyle={{ background: "#1f2937", border: "1px solid #374151", borderRadius: "8px" }}
            labelStyle={{ color: "#f9fafb" }}
          />
          <ReferenceLine x={0} stroke="#4b5563" />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {sorted.map((entry, index) => (
              <Cell key={index} fill={entry.value >= 0 ? "#ef4444" : "#22c55e"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
