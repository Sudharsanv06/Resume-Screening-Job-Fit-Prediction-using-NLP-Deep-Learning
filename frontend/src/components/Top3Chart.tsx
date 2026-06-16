import React from 'react';
import type { PredictionResponse } from '../types';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { BarChart3 } from 'lucide-react';

interface Top3ChartProps {
  result: PredictionResponse;
  darkMode: boolean;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    payload: {
      name: string;
      probability: number;
      rawPercentage: string;
    };
  }>;
}

export const Top3Chart: React.FC<Top3ChartProps> = ({ result, darkMode }) => {
  // Map predictions to chart data format
  const chartData = [...result.top3]
    .sort((a, b) => b.score - a.score) // Ensure descending order
    .map((item) => ({
      name: item.label,
      probability: Math.round(item.score * 100),
      rawPercentage: `${(item.score * 100).toFixed(1)}%`,
    }));


  // Define colors based on light/dark mode
  const secondaryBarColor = darkMode ? '#475569' : '#CBD5E1'; // Slate-600 vs Slate-300
  const axisColor = darkMode ? '#94A3B8' : '#64748B'; // Slate-400 vs Slate-500

  // Custom tooltips matching theme
  const CustomTooltip = ({ active, payload }: CustomTooltipProps) => {
    if (active && payload && payload.length) {
      return (
        <div className="px-3.5 py-2.5 bg-slate-900/90 dark:bg-slate-950/95 backdrop-blur-md text-white text-xs rounded-xl shadow-lg border border-slate-700/50 flex flex-col gap-0.5">
          <p className="font-bold tracking-wide">{payload[0].payload.name}</p>
          <p className="text-blue-400 font-semibold">Confidence: {payload[0].payload.rawPercentage}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full glass-card bg-white/60 dark:bg-slate-900/60 p-6 md:p-8 shadow-xl border border-slate-200/40 dark:border-slate-800/40 animate-slide-in transition-all duration-300">
      <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-xs font-bold uppercase tracking-wider mb-6">
        <BarChart3 className="w-4.5 h-4.5 text-blue-500" />
        Top Predictions
      </div>

      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 10, bottom: 5 }}
          >
            {/* Grid Line simulation through border styling */}
            <XAxis
              type="number"
              domain={[0, 100]}
              stroke={axisColor}
              fontSize={11}
              fontWeight={500}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `${value}%`}
            />
            <YAxis
              dataKey="name"
              type="category"
              stroke={axisColor}
              fontSize={12}
              fontWeight={600}
              tickLine={false}
              axisLine={false}
              width={140}
            />
            <Tooltip
              content={<CustomTooltip />}
              cursor={{ fill: darkMode ? 'rgba(51, 65, 85, 0.15)' : 'rgba(226, 232, 240, 0.4)' }}
            />
            <Bar
              dataKey="probability"
              radius={[0, 6, 6, 0]}
              barSize={20}
              isAnimationActive={true}
              animationDuration={800}
            >
              {chartData.map((_entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={index === 0 ? '#3B82F6' : secondaryBarColor}
                  className="transition-all duration-300"
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 pt-4 border-t border-slate-200/40 dark:border-slate-800/40 flex items-center justify-between text-xs text-slate-400 dark:text-slate-500 font-medium">
        <span>Model Inference Probability Distribution</span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2.5 rounded bg-blue-500"></span> Primary Role
          <span className="inline-block w-2.5 h-2.5 rounded bg-slate-300 dark:bg-slate-600 ml-2"></span> Alternatives
        </span>
      </div>
    </div>
  );
};
