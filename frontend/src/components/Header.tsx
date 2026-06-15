import React, { useEffect, useState } from 'react';
import { Briefcase, Sun, Moon } from 'lucide-react';
import axios from 'axios';

interface HeaderProps {
  darkMode: boolean;
  setDarkMode: (val: boolean) => void;
}

export const Header: React.FC<HeaderProps> = ({ darkMode, setDarkMode }) => {
  const [healthStatus, setHealthStatus] = useState<'online' | 'offline' | 'checking'>('checking');

  // Toggle dark mode on documentElement
  const toggleDarkMode = () => {
    const newVal = !darkMode;
    setDarkMode(newVal);
    if (newVal) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  };

  // Poll health status every 30 seconds
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await axios.get('/api/health');
        if (response.data && response.data.status === 'ok') {
          setHealthStatus('online');
        } else {
          setHealthStatus('offline');
        }
      } catch (err) {
        setHealthStatus('offline');
      }
    };

    // Run immediately on mount
    checkHealth();

    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="w-full py-4 px-6 flex flex-col sm:flex-row items-center justify-between border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-[#1E293B] sticky top-0 z-50 transition-all duration-300">
      <div className="flex items-center gap-3.5 mb-4 sm:mb-0">
        <div className="p-2.5 bg-gradient-to-tr from-blue-600 to-indigo-500 rounded-xl text-white shadow-lg shadow-blue-500/20 dark:shadow-indigo-500/10 flex items-center justify-center transform hover:scale-105 transition-transform duration-200">
          <Briefcase className="w-6 h-6 animate-pulse" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold tracking-tight text-[#0F172A] dark:text-white">
              ResumeAI
            </h1>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-slate-100 dark:bg-slate-850 text-slate-650 dark:text-slate-300 border border-slate-200/50 dark:border-slate-700/50">
              v2.0
            </span>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5 font-medium">
            Intelligent Job Role Classifier
          </p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* Health status badge */}
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-semibold transition-colors duration-300 ${
          healthStatus === 'online'
            ? 'bg-emerald-50/50 dark:bg-emerald-950/20 border-emerald-250 dark:border-emerald-800/30 text-emerald-600 dark:text-emerald-400'
            : healthStatus === 'offline'
            ? 'bg-rose-50/50 dark:bg-rose-950/20 border-rose-250 dark:border-rose-800/30 text-rose-600 dark:text-rose-400'
            : 'bg-amber-50/50 dark:bg-amber-950/20 border-amber-250 dark:border-amber-800/30 text-amber-605 dark:text-amber-400'
        }`}>
          <span className="relative flex h-2.5 w-2.5">
            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
              healthStatus === 'online'
                ? 'bg-emerald-400'
                : healthStatus === 'offline'
                ? 'bg-rose-400'
                : 'bg-amber-450'
            }`}></span>
            <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${
              healthStatus === 'online'
                ? 'bg-emerald-500'
                : healthStatus === 'offline'
                ? 'bg-rose-500'
                : 'bg-amber-500'
            }`}></span>
          </span>
          <span>
            {healthStatus === 'online'
              ? 'API Online'
              : healthStatus === 'offline'
              ? 'API Offline'
              : 'Checking API...'}
          </span>
        </div>

        {/* Theme Toggle Button */}
        <button
          onClick={toggleDarkMode}
          className="p-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-[#1E293B] text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800/60 shadow-sm hover:shadow-md transition-all duration-200 flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-blue-500/20"
          aria-label="Toggle light/dark theme"
        >
          {darkMode ? (
            <Sun className="w-5 h-5 text-amber-500 transition-transform duration-300 hover:rotate-45" />
          ) : (
            <Moon className="w-5 h-5 text-indigo-600 transition-transform duration-300 hover:-rotate-12" />
          )}
        </button>
      </div>
    </header>
  );
};
