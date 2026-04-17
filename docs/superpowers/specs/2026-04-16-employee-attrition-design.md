# Employee Attrition Prediction System — Design Spec

**Date:** 2026-04-16  
**Status:** Approved  
**Goal:** Resume/portfolio project demonstrating end-to-end ML, REST API, and frontend skills for recruiter review.

---

## Architecture

Monorepo with three independently deployable layers:

```
employee-retention/
  ml/          ← training notebook + saved model
  api/         ← Flask REST API
  frontend/    ← Next.js 14 app
  README.md
```

**Deployment:**
- Frontend → Vercel (free)
- API → Render (free, web service)
- Database → Supabase (free, PostgreSQL)

**Data flow:**
1. User fills prediction form in Next.js
2. Next.js POSTs to Flask `/predict`
3. Flask loads `model.pkl`, runs prediction + SHAP, logs to Supabase
4. Response `{ prediction, probability, shap_values, model_version }` returned
5. Next.js redirects to `/result` page with full SHAP chart

---

## ML Layer (`ml/`)

**Dataset:** IBM HR Analytics (Kaggle) — 1,470 rows, 35 features, binary target `Attrition`.

**Preprocessing:**
- Drop 4 constant/useless columns: `EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours`
- `OneHotEncoder` for categoricals (JobRole, Department, MaritalStatus, etc.)
- `StandardScaler` for numericals
- Final feature space: ~20 features after encoding

**Training:**
- Train Logistic Regression and Random Forest
- Select best model by ROC-AUC on 20% holdout
- Expected: ~87% accuracy, ~0.82 ROC-AUC (Random Forest)
- Save with `joblib`: `ml/model.pkl`
- Model version stored in `ml/VERSION` (e.g. `v1.0.0`)

**Explainability:**
- SHAP `TreeExplainer` loaded at Flask startup
- Top 5 SHAP feature contributions returned per prediction

---

## Backend API (`api/`)

**Stack:** Flask, scikit-learn, SHAP, joblib, psycopg2, python-dotenv

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Returns `{ status: "ok", model_version }` |
| POST | `/predict` | Accepts employee JSON, returns prediction + SHAP |
| GET | `/history` | Returns last 50 predictions from DB |

**POST `/predict` request body:**
```json
{
  "Age": 35,
  "MonthlyIncome": 5200,
  "JobRole": "Manager",
  "YearsAtCompany": 4,
  "OverTime": "Yes",
  "JobSatisfaction": 2,
  ...
}
```

**POST `/predict` response:**
```json
{
  "prediction": true,
  "probability": 0.78,
  "risk_level": "HIGH",   // HIGH: probability >= 0.60 | MED: 0.30–0.59 | LOW: < 0.30
  "shap_values": [
    { "feature": "OverTime", "value": 0.42 },
    { "feature": "JobSatisfaction", "value": 0.31 },
    { "feature": "MonthlyIncome", "value": -0.18 },
    { "feature": "Age", "value": -0.11 },
    { "feature": "YearsAtCompany", "value": 0.09 }
  ],
  "model_version": "v1.0.0"
}
```

**Error responses:**
```json
{ "error": "Missing required field: Age", "code": 400 }
```

**Startup:**
- Load `model.pkl` and SHAP explainer once at startup (not per request)
- Read `VERSION` file and attach to every prediction row

**CORS:** Allow `https://<project>.vercel.app` and `http://localhost:3000`

---

## Database (`Supabase PostgreSQL`)

```sql
CREATE TABLE predictions (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  age              INT,
  monthly_income   INT,
  job_role         TEXT,
  years_at_company INT,
  overtime         BOOLEAN,
  satisfaction_level INT,
  input_json       JSONB,        -- full raw input (all features)
  prediction       BOOLEAN,      -- true = leaves
  probability      FLOAT,
  shap_json        JSONB,        -- top 5 SHAP values
  model_version    TEXT,         -- e.g. "v1.0.0"
  created_at       TIMESTAMPTZ DEFAULT now()
);
```

`input_json` stores the complete raw input so schema changes don't require migrations. `shap_json` caches top-5 SHAP values so history page doesn't need to re-run SHAP.

---

## Frontend (`frontend/`)

**Stack:** Next.js 14, TypeScript, Tailwind CSS, Recharts (for SHAP bar chart)

**Pages:**

| Route | Component | Description |
|-------|-----------|-------------|
| `/` | `PredictionForm` | Employee input form, submits to Flask API |
| `/result` | `ResultCard` + `ShapChart` | Risk level, probability gauge, SHAP explanation |
| `/history` | `HistoryTable` | Last 50 predictions fetched from `/history` |

**Key components:**
- `PredictionForm.tsx` — controlled form with 14 input fields (dropdowns + number inputs), client-side validation, shows "Waking up API..." spinner on cold start
- `ResultCard.tsx` — displays risk level (HIGH/MED/LOW), probability percentage, color-coded badge
- `ShapChart.tsx` — horizontal bar chart (Recharts), red bars = increases risk, green = decreases risk
- `HistoryTable.tsx` — table with employee ID (auto-generated), date, risk level, probability, model version

**API calls:** All via a `lib/api.ts` client that reads `NEXT_PUBLIC_API_URL` from env.

**Cold start UX:** On page load, frontend calls `GET /health`. If it takes >3s, shows "API is waking up (free tier), please wait..." banner.

---

## Error Handling

- Flask: all exceptions caught, return `{ error, code }` JSON — never 500 HTML
- Supabase write failure: logged to stderr, prediction response still returned (non-blocking)
- Next.js: all fetch calls wrapped in try/catch, inline error messages (never blank screen)
- Cold start: loading spinner + "Waking up API..." message after 3s delay

---

## Folder Structure

```
employee-retention/
  ml/
    train.ipynb
    model.pkl
    VERSION
    requirements.txt
  api/
    app.py
    requirements.txt
    Procfile
    .env.example
  frontend/
    app/
      page.tsx
      result/
        page.tsx
      history/
        page.tsx
    components/
      PredictionForm.tsx
      ResultCard.tsx
      ShapChart.tsx
      HistoryTable.tsx
    lib/
      api.ts
    .env.example
  .gitignore
  README.md
```

---

## Deployment Checklist

- [ ] Supabase: create project, run SQL migration, copy `DATABASE_URL`
- [ ] Render: connect GitHub repo, set `DATABASE_URL` + `FLASK_ENV=production` env vars
- [ ] Vercel: connect GitHub repo, set `NEXT_PUBLIC_API_URL=https://<render-url>` env var
- [ ] Test: submit prediction end-to-end, verify row appears in Supabase dashboard
- [ ] README: add live URL, architecture diagram screenshot, model metrics

---

## Resume Talking Points

- "Trained a Random Forest on the IBM HR dataset, achieving 87% accuracy and 0.82 ROC-AUC"
- "Built a Flask REST API with SHAP explainability — every prediction returns the top 5 contributing factors"
- "Deployed on Render + Vercel + Supabase — fully live, zero cost"
- "Stamped every prediction with a model version field for traceability — a standard MLOps practice"
- "Built a prediction history log backed by PostgreSQL to demonstrate real data persistence"
