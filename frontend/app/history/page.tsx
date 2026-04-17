"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import HistoryTable from "@/components/HistoryTable";
import { fetchHistory } from "@/lib/api";
import { HistoryRow } from "@/lib/types";

export default function HistoryPage() {
  const [rows, setRows] = useState<HistoryRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHistory()
      .then(setRows)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <main className="min-h-screen bg-gray-950 text-white">
      <div className="max-w-5xl mx-auto px-4 py-12">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold">Prediction History</h1>
            <p className="text-gray-400 mt-1">Last 50 predictions — stored in PostgreSQL</p>
          </div>
          <Link
            href="/"
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm transition-colors"
          >
            + New Prediction
          </Link>
        </div>

        {loading && (
          <div className="text-center py-16 text-gray-400">Loading history...</div>
        )}

        {error && (
          <div className="bg-red-900/30 border border-red-700 rounded-xl px-6 py-4 text-red-400">
            {error}
          </div>
        )}

        {!loading && !error && <HistoryTable rows={rows} />}
      </div>
    </main>
  );
}
