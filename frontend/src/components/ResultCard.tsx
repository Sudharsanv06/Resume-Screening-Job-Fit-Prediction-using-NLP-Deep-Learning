import React from 'react';
import type { PredictionResponse } from '../types';
import { Award, Clock, FileText, RefreshCw } from 'lucide-react';

interface ResultCardProps {
  result: PredictionResponse;
  onReset: () => void;
}

export const ResultCard: React.FC<ResultCardProps> = ({ result, onReset }) => {
  const confidence = result.confidence; // 0.0 to 1.0
  const confidencePct = Math.round(confidence * 100);

  // Determine color scheme based on confidence percentage
  let badgeColorClass = '';
  let progressColor = '';

  if (confidencePct >= 80) {
    badgeColorClass = 'bg-gradient-to-r from-emerald-500 to-teal-600 text-white shadow-emerald-500/20';
    progressColor = 'url(#green-grad)';
  } else if (confidencePct >= 60) {
    badgeColorClass = 'bg-gradient-to-r from-amber-500 to-orange-500 text-white shadow-amber-500/20';
    progressColor = 'url(#amber-grad)';
  } else {
    badgeColorClass = 'bg-gradient-to-r from-rose-500 to-red-600 text-white shadow-rose-500/20';
    progressColor = 'url(#red-grad)';
  }

  // Circular progress calculations (Radius = 40, Circumference = 2 * PI * R ≈ 251.32)
  const radius = 40;
  const strokeWidth = 8;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (confidencePct / 100) * circumference;

  return (
    <div className="w-full glass-card bg-white/60 dark:bg-slate-900/60 p-6 md:p-8 shadow-xl border border-slate-200/40 dark:border-slate-800/40 animate-slide-in flex flex-col justify-between transition-all duration-300">
      <div>
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2 text-slate-500 dark:text-slate-300 text-xs font-bold uppercase tracking-wider">
            <Award className="w-4.5 h-4.5 text-blue-500" />
            Classification Result
          </div>
          <span className={`px-3 py-1 rounded-full text-xs font-semibold shadow-md ${badgeColorClass}`}>
            {confidencePct}% Match
          </span>
        </div>

        {/* Prediction Headline */}
        <div className="mb-6">
          <p className="text-xs text-slate-400 dark:text-slate-300 font-medium">Predicted Job Role</p>
          <h2 className="text-2xl md:text-3xl font-extrabold text-slate-900 dark:text-white mt-1 tracking-tight leading-tight">
            {result.predicted_role}
          </h2>
        </div>

        {/* Visual Progress Section */}
        <div className="flex items-center justify-center py-6">
          <div className="relative flex items-center justify-center">
            {/* Circular Progress Ring */}
            <svg className="w-32 h-32 transform -rotate-90" viewBox="0 0 100 100">
              <defs>
                <linearGradient id="green-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#10B981" />
                  <stop offset="100%" stopColor="#059669" />
                </linearGradient>
                <linearGradient id="amber-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#F59E0B" />
                  <stop offset="100%" stopColor="#D97706" />
                </linearGradient>
                <linearGradient id="red-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#F43F5E" />
                  <stop offset="100%" stopColor="#DC2626" />
                </linearGradient>
              </defs>
              {/* Background Ring */}
              <circle
                cx="50"
                cy="50"
                r={radius}
                className="stroke-slate-200 dark:stroke-slate-800"
                strokeWidth={strokeWidth}
                fill="transparent"
              />
              {/* Foreground Ring */}
              <circle
                cx="50"
                cy="50"
                r={radius}
                stroke={progressColor}
                strokeWidth={strokeWidth}
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="round"
                fill="transparent"
                className="transition-all duration-1000 ease-out"
              />
            </svg>
            <div className="absolute text-center">
              <span className="text-3xl font-black text-slate-900 dark:text-white tracking-tighter">
                {confidencePct}%
              </span>
              <p className="text-[10px] text-slate-400 dark:text-slate-300 font-bold uppercase tracking-wider mt-0.5">
                Confidence
              </p>
            </div>
          </div>
        </div>

        {/* Metadata Pills */}
        <div className="grid grid-cols-2 gap-3.5 mt-2 mb-6">
          <div className="flex items-center gap-2.5 p-3 rounded-xl bg-slate-100/60 dark:bg-slate-800/40 border border-slate-200/30 dark:border-slate-700/30">
            <FileText className="w-5 h-5 text-slate-400 dark:text-slate-300 flex-shrink-0" />
            <div className="min-w-0">
              <p className="text-[10px] text-slate-400 dark:text-slate-300 font-bold uppercase tracking-wider">
                Word Count
              </p>
              <p className="text-sm font-semibold text-slate-700 dark:text-slate-200 mt-0.5">
                {result.word_count} words
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5 p-3 rounded-xl bg-slate-100/60 dark:bg-slate-800/40 border border-slate-200/30 dark:border-slate-700/30">
            <Clock className="w-5 h-5 text-slate-400 dark:text-slate-300 flex-shrink-0" />
            <div className="min-w-0">
              <p className="text-[10px] text-slate-400 dark:text-slate-300 font-bold uppercase tracking-wider">
                Latency
              </p>
              <p className="text-sm font-semibold text-slate-700 dark:text-slate-200 mt-0.5">
                {result.processing_time_ms} ms
              </p>
            </div>
          </div>
        </div>
      </div>

      <button
        onClick={onReset}
        className="w-full py-3 px-6 rounded-xl bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 font-semibold text-sm flex items-center justify-center gap-2 border border-slate-200 dark:border-slate-700 hover:shadow-md transition-all duration-300 cursor-pointer"
      >
        <RefreshCw className="w-4 h-4" />
        Analyze Another
      </button>
    </div>
  );
};
