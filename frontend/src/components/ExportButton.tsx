import { Download } from "lucide-react";
import type { PredictionResponse } from "../types";

interface ExportButtonProps {
  result: PredictionResponse;
}

export default function ExportButton({ result }: ExportButtonProps) {

  const handleExport = () => {
    const exportData = {
      exported_at:       new Date().toISOString(),
      model_version:     result.model_version,
      prediction: {
        label:      result.label,
        confidence: `${(result.confidence * 100).toFixed(1)}%`,
        top3:       result.top3.map((item) => ({
          role:  item.label,
          score: `${(item.score * 100).toFixed(1)}%`,
        })),
      },
      stats: {
        word_count:         result.word_count,
        processing_time_ms: result.processing_time_ms,
      },
      resume_dna: {
        keyword_fit_score: `${result.resume_dna.skill_gap.fit_pct.toFixed(1)}%`,
        cluster_scores:    Object.fromEntries(
          Object.entries(result.resume_dna.cluster_scores).map(([k, v]) => [
            k,
            `${v.toFixed(1)}%`,
          ])
        ),
        detected_skills:  result.resume_dna.skill_gap.present,
        missing_skills:   result.resume_dna.skill_gap.missing,
        alternative_paths: result.resume_dna.alternative_paths.map((path) => ({
          role: path.role,
          score: `${(path.score * 100).toFixed(1)}%`,
          gap_count: path.gap_count,
        })),
      },
    };

    const blob     = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    });
    const url      = URL.createObjectURL(blob);
    const anchor   = document.createElement("a");
    const filename = `resume-prediction-${result.label.replace(/\s+/g, "-").toLowerCase()}-${Date.now()}.json`;

    anchor.href     = url;
    anchor.download = filename;
    anchor.click();

    URL.revokeObjectURL(url);
  };

  return (
    <button
      onClick={handleExport}
      aria-label="Export prediction results as JSON"
      className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/5
                 px-4 py-2 text-sm text-white/70 backdrop-blur-md
                 transition-all hover:border-white/20 hover:bg-white/10 hover:text-white"
    >
      <Download size={15} />
      Export Results
    </button>
  );
}
