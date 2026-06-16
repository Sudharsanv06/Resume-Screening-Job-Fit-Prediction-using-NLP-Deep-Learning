import { useState, useEffect } from "react";

import { Header }       from "./components/Header";
import { DropZone }     from "./components/DropZone";
import { CategoryGrid } from "./components/CategoryGrid";
import { ResultCard }   from "./components/ResultCard";
import { Top3Chart }    from "./components/Top3Chart";
import { ResumeDNA }    from "./components/ResumeDNA";
import SkeletonLoader   from "./components/SkeletonLoader";
import ErrorBanner      from "./components/ErrorBanner";
import ExportButton     from "./components/ExportButton";

import { predictFromFile, predictFromText } from "./api/predict";
import type { PredictionResponse }          from "./types";

export default function App() {
  const [result,   setResult]   = useState<PredictionResponse | null>(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState<string | null>(null);
  const [darkMode, setDarkMode] = useState(false);

  // Listen for dna-role-click events fired from ResumeDNA alternative paths
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent<{ role: string }>).detail;
      console.debug("dna-role-click", detail?.role);
    };
    window.addEventListener("dna-role-click", handler);
    return () => window.removeEventListener("dna-role-click", handler);
  }, []);

  // ── API handlers ────────────────────────────────────────────────────────────

  const handleFile = async (file: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictFromFile(file);
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const handleText = async (text: string) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictFromText(text);
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
  };

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className={darkMode ? "dark" : ""}>
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900/20 to-slate-900 text-white">

        <Header
          darkMode={darkMode}
          setDarkMode={setDarkMode}
        />

        <main className="mx-auto max-w-4xl space-y-6 px-4 py-8">

          {/* Upload / paste area — always visible */}
          <DropZone
            onAnalyzeFile={handleFile}
            onAnalyzeText={handleText}
            loading={loading}
            onReset={handleReset}
            hasResult={!!result}
          />

          {/* Error banner */}
          {error && (
            <ErrorBanner
              message={error}
              onDismiss={() => setError(null)}
            />
          )}

          {/* Loading skeleton */}
          {loading && <SkeletonLoader />}

          {/* Results */}
          {result && !loading && (
            <div className="space-y-6">

              {/* Export + Reset row */}
              <div className="flex items-center justify-between">
                <button
                  onClick={handleReset}
                  className="text-sm text-white/40 hover:text-white/70 transition-colors"
                >
                  ← Analyse another resume
                </button>
                <ExportButton result={result} />
              </div>

              {/* Model version + stats bar */}
              <div className="flex flex-wrap gap-4 text-xs text-white/40">
                <span>Model {result.model_version}</span>
                <span>{result.word_count} words</span>
                <span>{result.processing_time_ms} ms</span>
              </div>

              <ResultCard result={result} onReset={handleReset} />
              <Top3Chart  result={result} darkMode={darkMode} />
              <ResumeDNA  dna={result.resume_dna} />
            </div>
          )}

          {/* Default state — show supported roles */}
          {!result && !loading && !error && (
            <CategoryGrid />
          )}

        </main>
      </div>
    </div>
  );
}
