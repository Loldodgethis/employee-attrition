import PredictionForm from "@/components/PredictionForm";

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-950 text-white">
      <div className="max-w-3xl mx-auto px-4 py-12">
        <div className="mb-10 text-center">
          <h1 className="text-4xl font-bold text-white mb-3">
            Employee Attrition Predictor
          </h1>
          <p className="text-gray-400 text-lg">
            Enter employee details to predict the likelihood of attrition using machine learning.
          </p>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-8 shadow-xl">
          <PredictionForm />
        </div>
        <p className="text-center text-gray-600 text-sm mt-6">
          Powered by Random Forest · IBM HR Analytics Dataset · SHAP Explainability
        </p>
      </div>
    </main>
  );
}
