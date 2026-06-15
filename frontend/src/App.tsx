import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { DropZone } from './components/DropZone';
import { ResultCard } from './components/ResultCard';
import { Top3Chart } from './components/Top3Chart';
import { CategoryGrid } from './components/CategoryGrid';
import { ResumeDNA } from './components/ResumeDNA';
import { predictFile, predictText } from './api/predict';
import type { PredictionResponse } from './types';
import { AlertCircle, X, Sparkles } from 'lucide-react';
import './App.css';

function App() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [darkMode, setDarkMode] = useState<boolean>(false);

  // Initialize theme from localStorage or system preference
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
      setDarkMode(true);
      document.documentElement.classList.add('dark');
    } else {
      setDarkMode(false);
      document.documentElement.classList.remove('dark');
    }
  }, []);

  const handleAnalyzeFile = async (file: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictFile(file);
      setResult(data);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred during file analysis.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeText = async (text: string) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictText(text);
      setResult(data);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred during text analysis.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 flex flex-col transition-colors duration-300">
      {/* Background ambient glows */}
      <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-blue-500/10 dark:bg-blue-600/5 rounded-full blur-[100px] pointer-events-none z-0"></div>
      <div className="absolute bottom-10 right-1/4 w-[400px] h-[400px] bg-indigo-500/10 dark:bg-indigo-600/5 rounded-full blur-[80px] pointer-events-none z-0"></div>

      <Header darkMode={darkMode} setDarkMode={setDarkMode} />

      <main className="flex-1 max-w-5xl w-full mx-auto px-4 py-8 md:py-12 z-10 flex flex-col gap-10">
        
        {/* Intro Hero Section */}
        <section className="text-center max-w-2xl mx-auto mb-6 md:mb-10">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-500/10 dark:bg-blue-400/10 text-blue-600 dark:text-blue-400 text-xs font-semibold mb-6 border border-blue-500/20 dark:border-blue-450/20">
            <Sparkles className="w-3.5 h-3.5" />
            Empowered by NLP & Deep Learning
          </div>
          <h1 
            style={{ fontSize: 'clamp(2rem, 5vw, 3.5rem)' }}
            className="font-extrabold tracking-tight text-slate-900 dark:text-white leading-tight"
          >
            Screen Resumes with <span className="bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400 bg-clip-text text-transparent" style={{ background: 'linear-gradient(135deg, #3B82F6, #06B6D4)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>AI Precision</span>
          </h1>
          <p className="mt-4 text-sm md:text-base text-slate-500 dark:text-slate-300 max-w-lg mx-auto leading-relaxed">
            Upload your PDF or Word resume, or paste plain text directly. Get instantaneous predictions on matching professional categories and probability models.
          </p>
        </section>

        {/* Error Banner */}
        {error && (
          <div className="w-full max-w-2xl mx-auto p-4 bg-rose-500/10 border border-rose-500/30 rounded-2xl flex items-start justify-between gap-3 text-rose-600 dark:text-rose-450 shadow-sm animate-fade-in">
            <div className="flex gap-2.5 items-start">
              <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-bold">Analysis Failed</p>
                <p className="text-xs text-rose-500 mt-0.5 leading-relaxed">{error}</p>
              </div>
            </div>
            <button
              onClick={() => setError(null)}
              className="p-1 rounded-lg hover:bg-rose-500/10 text-rose-400 hover:text-rose-600 dark:hover:text-rose-200 transition-colors focus:outline-none"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Dropzone container */}
        <section>
          <DropZone
            onAnalyzeFile={handleAnalyzeFile}
            onAnalyzeText={handleAnalyzeText}
            loading={loading}
            onReset={handleReset}
            hasResult={!!result}
          />
        </section>

        {/* Results layout container with smooth transitions */}
        {result && (
          <section className="w-full flex flex-col gap-8 animate-fade-in">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-stretch">
              <ResultCard result={result} onReset={handleReset} />
              <Top3Chart result={result} darkMode={darkMode} />
            </div>
            
            <ResumeDNA dna={result.dna} />
            
            <CategoryGrid predictedRole={result.predicted_role} />
          </section>
        )}

        {/* Fallback Categories display when there are no results yet */}
        {!result && !loading && (
          <section className="mt-2 animate-fade-in">
            <CategoryGrid />
          </section>
        )}
      </main>

      <footer className="py-6 border-t border-slate-200/50 dark:border-slate-900 bg-white/40 dark:bg-slate-950/40 text-center text-xs text-slate-400 dark:text-slate-600">
        <p>© 2026 ResumeAI. Powered by SentenceTransformers and FastAPI.</p>
      </footer>
    </div>
  );
}

export default App;
