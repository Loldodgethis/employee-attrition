# Employee Attrition Prediction System

Predict whether an employee will leave using machine learning. Built end-to-end: data → model → API → frontend → database.

**Live:** [[https://employee-attrition.vercel.app](https://employee-attrition-eight.vercel.app/)]
## Architecture

```
Next.js (Vercel) → Flask API (Render) → model.pkl + Supabase PostgreSQL
```

- **ML:** Random Forest trained on IBM HR Analytics dataset — ~87% accuracy, ~0.82 ROC-AUC
- **Explainability:** SHAP values returned with every prediction (top 5 contributing factors)
- **MLOps:** Every prediction stamped with `model_version` for traceability
- **Persistence:** All predictions logged to PostgreSQL

## Local Development

### 1. Train the model

```bash
cd ml
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Download IBM HR dataset from Kaggle → ml/data/WA_Fn-UseC_-HR-Employee-Attrition.csv
python train.py
```

### 2. Start the API

```bash
cd api
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in DATABASE_URL from Supabase
python app.py
```

### 3. Start the frontend

```bash
cd frontend
npm install
cp .env.example .env.local  # NEXT_PUBLIC_API_URL=http://localhost:5000
npm run dev
```

Open http://localhost:3000

## Running Tests

```bash
cd api && source venv/bin/activate && pytest tests/ -v
```

## Deployment

| Service | Purpose | Free tier |
|---------|---------|-----------|
| Render | Flask API | ✅ Yes |
| Vercel | Next.js frontend | ✅ Yes |
| Supabase | PostgreSQL | ✅ Yes |

### Render (Flask API)
1. Push repo to GitHub (`https://github.com/loldodgethis/employee-attrition`)
2. Render → New Web Service → connect repo
3. Root directory: `api`, Build: `pip install -r requirements.txt`, Start: `gunicorn app:app`
4. Env vars: `DATABASE_URL` (from Supabase) + `FRONTEND_URL` (your Vercel URL)

### Vercel (Next.js)
1. Vercel → New Project → import repo
2. Root directory: `frontend`
3. Env var: `NEXT_PUBLIC_API_URL=https://<your-render-url>.onrender.com`

## Model

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest (200 trees) |
| Dataset | IBM HR Analytics (1,470 rows) |
| Accuracy | ~87% |
| ROC-AUC | ~0.82 |
| Features | 14 (5 categorical, 9 numerical) |
| Explainability | SHAP TreeExplainer |
