"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { submitPrediction } from "@/lib/api";
import { PredictionInput } from "@/lib/types";

const JOB_ROLES = [
  "Healthcare Representative", "Human Resources", "Laboratory Technician",
  "Manager", "Manufacturing Director", "Research Director", "Research Scientist",
  "Sales Executive", "Sales Representative",
];
const DEPARTMENTS = ["Human Resources", "Research & Development", "Sales"];
const BUSINESS_TRAVEL = ["Non-Travel", "Travel_Frequently", "Travel_Rarely"];
const MARITAL_STATUS = ["Divorced", "Married", "Single"];
const SATISFACTION_LEVELS = [
  { value: 1, label: "1 — Low" },
  { value: 2, label: "2 — Medium" },
  { value: 3, label: "3 — High" },
  { value: 4, label: "4 — Very High" },
];

const defaultValues: PredictionInput = {
  BusinessTravel: "Travel_Rarely",
  Department: "Sales",
  JobRole: "Sales Executive",
  MaritalStatus: "Single",
  OverTime: "No",
  Age: 35,
  DistanceFromHome: 10,
  EnvironmentSatisfaction: 3,
  JobSatisfaction: 3,
  MonthlyIncome: 5000,
  NumCompaniesWorked: 2,
  TotalWorkingYears: 8,
  WorkLifeBalance: 3,
  YearsAtCompany: 4,
};

export default function PredictionForm() {
  const router = useRouter();
  const [form, setForm] = useState<PredictionInput>(defaultValues);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function setField<K extends keyof PredictionInput>(key: K, value: PredictionInput[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const result = await submitPrediction(form);
      sessionStorage.setItem("predictionResult", JSON.stringify(result));
      router.push("/result");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Row 1 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Age</label>
          <input
            type="number" min={18} max={65} required
            value={form.Age}
            onChange={(e) => setField("Age", parseInt(e.target.value))}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Monthly Income ($)</label>
          <input
            type="number" min={1000} max={20000} required
            value={form.MonthlyIncome}
            onChange={(e) => setField("MonthlyIncome", parseInt(e.target.value))}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Job Role</label>
          <select
            value={form.JobRole}
            onChange={(e) => setField("JobRole", e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {JOB_ROLES.map((r) => <option key={r}>{r}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Department</label>
          <select
            value={form.Department}
            onChange={(e) => setField("Department", e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {DEPARTMENTS.map((d) => <option key={d}>{d}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Years at Company</label>
          <input
            type="number" min={0} max={40} required
            value={form.YearsAtCompany}
            onChange={(e) => setField("YearsAtCompany", parseInt(e.target.value))}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Total Working Years</label>
          <input
            type="number" min={0} max={40} required
            value={form.TotalWorkingYears}
            onChange={(e) => setField("TotalWorkingYears", parseInt(e.target.value))}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Overtime</label>
          <select
            value={form.OverTime}
            onChange={(e) => setField("OverTime", e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option>Yes</option>
            <option>No</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Business Travel</label>
          <select
            value={form.BusinessTravel}
            onChange={(e) => setField("BusinessTravel", e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {BUSINESS_TRAVEL.map((t) => <option key={t}>{t}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Marital Status</label>
          <select
            value={form.MaritalStatus}
            onChange={(e) => setField("MaritalStatus", e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {MARITAL_STATUS.map((m) => <option key={m}>{m}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Distance from Home (km)</label>
          <input
            type="number" min={1} max={29} required
            value={form.DistanceFromHome}
            onChange={(e) => setField("DistanceFromHome", parseInt(e.target.value))}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Companies Worked At</label>
          <input
            type="number" min={0} max={9} required
            value={form.NumCompaniesWorked}
            onChange={(e) => setField("NumCompaniesWorked", parseInt(e.target.value))}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>

      {/* Satisfaction sliders row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {(
          [
            ["Job Satisfaction", "JobSatisfaction"],
            ["Environment Satisfaction", "EnvironmentSatisfaction"],
            ["Work-Life Balance", "WorkLifeBalance"],
          ] as [string, keyof PredictionInput][]
        ).map(([label, key]) => (
          <div key={key}>
            <label className="block text-sm font-medium text-gray-300 mb-1">{label}</label>
            <select
              value={form[key] as number}
              onChange={(e) => setField(key, parseInt(e.target.value))}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {SATISFACTION_LEVELS.map((s) => (
                <option key={s.value} value={s.value}>{s.label}</option>
              ))}
            </select>
          </div>
        ))}
      </div>

      {error && (
        <div className="bg-red-900/50 border border-red-500 rounded-lg px-4 py-3 text-red-300 text-sm">
          {error}
        </div>
      )}

      <button
        type="submit"
        disabled={loading}
        className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-lg transition-colors"
      >
        {loading ? "Analyzing..." : "Predict Attrition →"}
      </button>
    </form>
  );
}
