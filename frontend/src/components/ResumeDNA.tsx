import React from 'react';
import type { ResumeDNA as ResumeDNAType } from '../types';
import { 
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis, 
  Radar, 
  ResponsiveContainer, 
  Tooltip 
} from 'recharts';
import { 
  Dna, 
  CheckCircle2, 
  AlertTriangle, 
  TrendingUp, 
  Sparkles 
} from 'lucide-react';

interface ResumeDNAProps {
  dna: ResumeDNAType;
}

export const ResumeDNA: React.FC<ResumeDNAProps> = ({ dna }) => {
  const { cluster_scores, skill_gap, alternative_paths } = dna;
  const fitPct = skill_gap.fit_pct;

  // Radar Chart Data formatting
  const radarData = Object.entries(cluster_scores).map(([cluster, score]) => ({
    cluster,
    score: Math.round(score),
  }));

  // Style configurations based on fit score
  let fitColorClass = '';
  let fitBgClass = '';
  let fitTextLabel = '';

  if (fitPct >= 70) {
    fitColorClass = 'text-emerald-500 dark:text-emerald-400';
    fitBgClass = 'bg-emerald-500/10 border-emerald-500/20 dark:border-emerald-500/10';
    fitTextLabel = 'Strong Fit';
  } else if (fitPct >= 40) {
    fitColorClass = 'text-amber-500 dark:text-amber-400';
    fitBgClass = 'bg-amber-500/10 border-amber-500/20 dark:border-amber-500/10';
    fitTextLabel = 'Moderate Fit';
  } else {
    fitColorClass = 'text-rose-500 dark:text-rose-450';
    fitBgClass = 'bg-rose-500/10 border-rose-500/20 dark:border-rose-500/10';
    fitTextLabel = 'Skills Gap Observed';
  }

  // Handle alternative path selection event dispatcher
  const handleAlternativeClick = (role: string) => {
    const event = new CustomEvent('dna-role-click', {
      detail: { role },
    });
    window.dispatchEvent(event);
  };

  // Custom tooltips matching theme
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="px-3 py-2 bg-slate-900/95 dark:bg-slate-950/95 backdrop-blur-md text-white text-xs rounded-xl shadow-lg border border-slate-700/50 flex flex-col gap-0.5">
          <p className="font-bold tracking-wide">{payload[0].payload.cluster}</p>
          <p className="text-blue-400 font-semibold">Match Score: {payload[0].value}%</p>
        </div>
      );
    }
    return null;
  };

  const presentSkills = skill_gap.present.slice(0, 8);
  const missingSkills = skill_gap.missing.slice(0, 8);

  return (
    <div className="w-full glass-card bg-white/60 dark:bg-slate-900/60 p-6 md:p-8 shadow-xl border border-slate-200/40 dark:border-slate-800/40 rounded-3xl animate-slide-in transition-all duration-300">
      
      {/* Header */}
      <div className="flex items-center gap-2.5 text-slate-500 dark:text-slate-400 text-xs font-bold uppercase tracking-wider mb-8">
        <Dna className="w-5 h-5 text-blue-500 animate-pulse" />
        Resume DNA & Career Mapping
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-stretch">
        
        {/* Column 1: Career Radar Chart & Overall Score */}
        <div className="flex flex-col justify-between p-5 rounded-2xl bg-slate-50/40 dark:bg-slate-900/30 border border-slate-200/20 dark:border-slate-800/20">
          <div>
            <h3 className="text-base font-bold text-slate-800 dark:text-white mb-1 flex items-center gap-2">
              Career Area Radar
            </h3>
            <p className="text-xs text-slate-400 dark:text-slate-350 mb-4">
              Visual map illustrating fit across 6 core technical role clusters.
            </p>
            
            {/* Radar Chart */}
            <div style={{ width: '100%', height: 280 }} className="flex justify-center items-center">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                  <PolarGrid stroke="rgba(148, 163, 184, 0.2)" />
                  <PolarAngleAxis 
                    dataKey="cluster" 
                    tick={{ fill: 'currentColor', fontSize: 10, fontWeight: 600 }}
                    className="text-slate-400 dark:text-slate-300"
                  />
                  <PolarRadiusAxis 
                    angle={30} 
                    domain={[0, 100]} 
                    tick={false}
                    axisLine={false}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Radar
                    name="Match Score"
                    dataKey="score"
                    stroke="#3B82F6"
                    fill="#3B82F6"
                    fillOpacity={0.2}
                    strokeWidth={2}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Skill Fit Panel */}
          <div className={`mt-6 p-4 rounded-xl border flex items-center justify-between gap-4 transition-all duration-300 ${fitBgClass}`}>
            <div className="min-w-0">
              <p className="text-[10px] text-slate-400 dark:text-slate-300 font-bold uppercase tracking-wider">
                Keyword Fit score
              </p>
              <h4 className="text-sm font-bold text-slate-700 dark:text-slate-200 mt-1 truncate">
                Keyword fit for <span className="text-blue-500 font-extrabold">{skill_gap.role}</span>
              </h4>
            </div>
            <div className="text-right flex-shrink-0">
              <span className={`text-3xl font-black ${fitColorClass} tracking-tighter`}>
                {fitPct}%
              </span>
              <p className={`text-[10px] font-bold uppercase tracking-widest ${fitColorClass} mt-0.5`}>
                {fitTextLabel}
              </p>
            </div>
          </div>
        </div>

        {/* Column 2: Skill Gaps and Alternative Paths */}
        <div className="flex flex-col gap-6">
          
          {/* Skill Gap Analysis */}
          <div className="p-5 rounded-2xl bg-slate-50/40 dark:bg-slate-900/30 border border-slate-200/20 dark:border-slate-800/20 flex-1 flex flex-col justify-between">
            <div>
              <h3 className="text-base font-bold text-slate-800 dark:text-white mb-4">
                Skill Keywords Audit
              </h3>
              
              {/* Detected Skills */}
              <div className="mb-5">
                <div className="flex items-center gap-1.5 text-xs text-emerald-600 dark:text-emerald-400 font-bold uppercase tracking-wider mb-2.5">
                  <CheckCircle2 className="w-4 h-4" />
                  Skills Detected
                </div>
                <div className="flex flex-wrap gap-2">
                  {presentSkills.length > 0 ? (
                    presentSkills.map((skill, idx) => (
                      <span 
                        key={idx}
                        className="px-2.5 py-1 bg-emerald-500/10 text-emerald-700 dark:text-emerald-350 border border-emerald-500/20 dark:border-emerald-500/10 rounded-full text-xs font-semibold"
                      >
                        {skill}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs text-slate-400 dark:text-slate-350 italic">
                      No matching keywords detected in the resume text.
                    </span>
                  )}
                </div>
              </div>

              {/* Missing Skills */}
              <div>
                <div className="flex items-center gap-1.5 text-xs text-rose-500 dark:text-rose-450 font-bold uppercase tracking-wider mb-2.5">
                  <AlertTriangle className="w-4 h-4" />
                  Add to your resume
                </div>
                <div className="flex flex-wrap gap-2">
                  {missingSkills.length > 0 ? (
                    missingSkills.map((skill, idx) => (
                      <span 
                        key={idx}
                        className="px-2.5 py-1 bg-rose-500/10 text-rose-700 dark:text-rose-350 border border-rose-500/20 dark:border-rose-500/10 rounded-full text-xs font-semibold"
                      >
                        {skill}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs text-emerald-500 dark:text-emerald-400 italic">
                      Outstanding match! All core keywords detected.
                    </span>
                  )}
                </div>
              </div>
            </div>

            <div className="text-[10px] text-slate-400 dark:text-slate-500 font-medium mt-6 pt-4 border-t border-slate-200/40 dark:border-slate-800/40">
              Matches are computed from a dictionary of 15 core skills per job category.
            </div>
          </div>

          {/* Alternative Careers */}
          <div className="p-5 rounded-2xl bg-slate-50/40 dark:bg-slate-900/30 border border-slate-200/20 dark:border-slate-800/20">
            <h3 className="text-base font-bold text-slate-800 dark:text-white mb-1 flex items-center gap-2">
              <TrendingUp className="w-4.5 h-4.5 text-blue-500" />
              Alternative Career Paths
            </h3>
            <p className="text-xs text-slate-400 dark:text-slate-350 mb-4">
              Explore secondary role recommendations with calculated gaps.
            </p>

            <div className="flex flex-col gap-3">
              {alternative_paths.map((path, idx) => {
                const pathScorePct = Math.round(path.score * 100);
                
                return (
                  <div 
                    key={idx}
                    onClick={() => handleAlternativeClick(path.role)}
                    className="p-3.5 rounded-xl bg-white/40 dark:bg-slate-900/40 hover:bg-white/80 dark:hover:bg-slate-800/60 border border-slate-200/40 dark:border-slate-800/60 cursor-pointer transition-all duration-300 group flex items-center justify-between gap-4"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-xs font-bold text-slate-700 dark:text-slate-200 group-hover:text-blue-500 transition-colors">
                          {path.role}
                        </span>
                        <span className="text-[10px] font-bold text-slate-400 dark:text-slate-300">
                          {pathScorePct}% Match
                        </span>
                      </div>
                      
                      {/* Progress Bar */}
                      <div className="w-full h-1.5 bg-slate-200 dark:bg-slate-850 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-blue-500 rounded-full transition-all duration-500" 
                          style={{ width: `${pathScorePct}%` }}
                        />
                      </div>
                    </div>

                    {/* Gap Count Badge */}
                    <div className="text-right flex-shrink-0">
                      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-slate-100 dark:bg-slate-800 border border-slate-200/50 dark:border-slate-700/50 text-[10px] font-bold text-slate-500 dark:text-slate-400">
                        <Sparkles className="w-3 h-3 text-amber-500" />
                        {path.gap_count} gaps
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

        </div>

      </div>

    </div>
  );
};
