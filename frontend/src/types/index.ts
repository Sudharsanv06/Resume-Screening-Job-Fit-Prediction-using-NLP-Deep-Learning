export interface Top3Item {
  role: string;
  score: number;
  score_pct: string;
}

export interface SkillGap {
  role: string;
  present: string[];
  missing: string[];
  fit_pct: number;
}

export interface AlternativePath {
  role: string;
  score: number;
  gap_count: number;
}

export interface ResumeDNA {
  cluster_scores: Record<string, number>;
  skill_gap: SkillGap;
  alternative_paths: AlternativePath[];
}

export interface PredictionResponse {
  predicted_role: string;
  confidence: number;
  confidence_pct: string;
  top3: Top3Item[];
  word_count: number;
  processing_time_ms: number;
  dna: ResumeDNA;
}
