import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import ApiStatusBanner from "@/components/ApiStatusBanner";
import Link from "next/link";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Employee Attrition Predictor",
  description: "Predict employee attrition with machine learning and SHAP explainability",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-gray-950`}>
        <ApiStatusBanner />
        <nav className="border-b border-gray-800 px-6 py-3 flex items-center gap-6 bg-gray-950">
          <span className="text-white font-semibold">Attrition Predictor</span>
          <Link href="/" className="text-gray-400 hover:text-white text-sm transition-colors">Predict</Link>
          <Link href="/history" className="text-gray-400 hover:text-white text-sm transition-colors">History</Link>
        </nav>
        {children}
      </body>
    </html>
  );
}
