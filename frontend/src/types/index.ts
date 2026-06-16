// ── Core prediction types ─────────────────────────────────────────────────────

export interface Top3Item {
  label: string;
  score: number;
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

export interface ClusterScores {
  "Data Science":      number;
  "Backend & Python":  number;
  "DevOps & Cloud":    number;
  "Frontend & Mobile": number;
  "Security":          number;
  "Business":          number;
}

export interface ResumeDNA {
  cluster_scores:    ClusterScores;
  skill_gap:         SkillGap;
  alternative_paths: AlternativePath[];
}

export interface PredictionResponse {
  label:               string;
  confidence:          number;
  top3:                Top3Item[];
  all_probs:           Record<string, number>;
  word_count:          number;
  processing_time_ms:  number;
  model_version:       string;
  resume_dna:          ResumeDNA;
}

// ── Batch types ───────────────────────────────────────────────────────────────

export interface BatchResultItem {
  index:      number;
  label:      string | null;
  confidence?: number;
  top3?:      Top3Item[];
  error?:     string;
}

export interface BatchResponse {
  results:            BatchResultItem[];
  total:              number;
  model_version:      string;
  processing_time_ms: number;
}

// ── Error type (RFC 7807) ─────────────────────────────────────────────────────

export interface ApiError {
  type:   string;
  title:  string;
  status: number;
  detail: string | Record<string, unknown>[];
}
