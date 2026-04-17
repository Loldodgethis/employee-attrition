"use client";

import { useEffect, useState } from "react";
import { checkHealth } from "@/lib/api";

export default function ApiStatusBanner() {
  const [slow, setSlow] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setSlow(true), 3000);

    checkHealth()
      .then(() => {
        clearTimeout(timer);
        setSlow(false);
        setReady(true);
      })
      .catch(() => {
        clearTimeout(timer);
        setSlow(false);
      });

    return () => clearTimeout(timer);
  }, []);

  if (ready || !slow) return null;

  return (
    <div className="bg-yellow-900/40 border-b border-yellow-700 px-4 py-2 text-center text-yellow-300 text-sm">
      API is waking up (free tier cold start) — this may take up to 30 seconds...
    </div>
  );
}
