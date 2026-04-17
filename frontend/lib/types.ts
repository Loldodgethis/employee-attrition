export interface PredictionInput {
  BusinessTravel: string;
  Department: string;
  JobRole: string;
  MaritalStatus: string;
  OverTime: string;
  Age: number;
  DistanceFromHome: number;
  EnvironmentSatisfaction: number;
  JobSatisfaction: number;
  MonthlyIncome: number;
  NumCompaniesWorked: number;
  TotalWorkingYears: number;
  WorkLifeBalance: number;
  YearsAtCompany: number;
}

export type RiskLevel = "HIGH" | "MED" | "LOW";

export interface ShapValue {
  feature: string;
  value: number;
}

export interface PredictionResult {
  prediction: boolean;
  probability: number;
  risk_level: RiskLevel;
  shap_values: ShapValue[];
  model_version: string;
}

export interface HistoryRow {
  id: string;
  age: number;
  monthly_income: number;
  job_role: string;
  years_at_company: number;
  overtime: boolean;
  satisfaction_level: number;
  prediction: boolean;
  probability: number;
  shap_json: ShapValue[];
  model_version: string;
  created_at: string;
}
