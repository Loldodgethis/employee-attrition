"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import ResultCard from "@/components/ResultCard";
import ShapChart from "@/components/ShapChart";
import { PredictionResult } from "@/lib/types";

export default function ResultPage() {
  const router = useRouter();
  const [result, setResult] = useState<PredictionResult | null>(null);

  useEffect(() => {
    const raw = sessionStorage.getItem("predictionResult");
    if (!raw) {
      router.replace("/");
      return;
    }
    setResult(JSON.parse(raw));
  }, [router]);

  if (!result) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <p className="text-gray-400">Loading result...</p>
      </div>
    );
  }

  return (
    <main className="min-h-screen bg-gray-950 text-white">
      <div className="max-w-3xl mx-auto px-4 py-12 space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold">Prediction Result</h1>
          <Link
            href="/"
            className="text-blue-400 hover:text-blue-300 text-sm transition-colors"
          >
            ← New Prediction
          </Link>
        </div>

        <ResultCard result={result} />
        <ShapChart shapValues={result.shap_values} />

        <div className="text-center">
          <Link
            href="/history"
            className="text-gray-400 hover:text-gray-300 text-sm transition-colors"
          >
            View prediction history →
          </Link>
        </div>
      </div>
    </main>
  );
}
