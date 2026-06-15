import React from 'react';
import { Tag } from 'lucide-react';

interface CategoryGridProps {
  predictedRole?: string;
}

const CATEGORIES = [
  'Backend Developer',
  'Business Analyst',
  'Cloud Architect',
  'Data Analyst',
  'Data Engineer',
  'Data Scientist',
  'DevOps Engineer',
  'Frontend Developer',
  'Java Developer',
  'Mobile Developer',
  'Python Developer',
  'QA Engineer',
  'Security Analyst',
  'Web Developer',
];

export const CategoryGrid: React.FC<CategoryGridProps> = ({ predictedRole }) => {
  return (
    <div className="w-full glass-card bg-white/60 dark:bg-slate-900/60 p-6 md:p-8 shadow-md border border-slate-200/40 dark:border-slate-800/40 transition-all duration-300">
      <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-xs font-bold uppercase tracking-wider mb-6">
        <Tag className="w-4.5 h-4.5 text-blue-500" />
        Supported Classifications
      </div>

      <div className="flex flex-wrap gap-2.5 justify-center md:justify-start">
        {CATEGORIES.map((category) => {
          const isPredicted = predictedRole?.toLowerCase() === category.toLowerCase();

          return (
            <div
              key={category}
              className={`px-4 py-2 rounded-full text-xs font-semibold tracking-wide border transition-all duration-300 transform ${
                isPredicted
                  ? 'bg-gradient-to-r from-blue-600 to-indigo-600 border-blue-500 text-white shadow-md shadow-blue-500/25 scale-105'
                  : 'bg-white/80 hover:bg-white dark:bg-slate-800/50 dark:hover:bg-slate-800 border-slate-200/60 dark:border-slate-800 text-slate-600 dark:text-slate-350 hover:border-blue-400 dark:hover:border-slate-700 hover:scale-[1.03] cursor-default'
              }`}
            >
              {category}
            </div>
          );
        })}
      </div>

      <div className="mt-5 text-center md:text-left">
        <p className="text-[11px] text-slate-400 dark:text-slate-500 font-medium">
          The machine learning classification engine predicts candidate suitability across these {CATEGORIES.length} roles.
        </p>
      </div>
    </div>
  );
};
