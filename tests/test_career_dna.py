"""Unit tests for src/career_dna.py — cluster scoring and skill gap analysis."""
import pytest
from src.career_dna import ROLE_CLUSTERS, get_cluster_scores, get_resume_dna

ALL_ROLES = [
    "Python Developer", "Backend Developer", "Java Developer",
    "Data Scientist",   "Data Analyst",      "Data Engineer",
    "DevOps Engineer",  "Cloud Architect",
    "Frontend Developer", "Mobile Developer", "Web Developer",
    "Security Analyst", "Business Analyst",  "QA Engineer",
]

# All roles share equal probability
UNIFORM_PROBS = {role: 1 / len(ALL_ROLES) for role in ALL_ROLES}

# High confidence on Python Developer
PYTHON_DEV_PROBS = {role: 0.01 for role in ALL_ROLES}
PYTHON_DEV_PROBS["Python Developer"] = 0.85

SAMPLE_TEXT = (
    "Python developer with Django, Flask, FastAPI, PostgreSQL, Redis, "
    "Docker, Kubernetes, AWS, CI/CD, REST APIs, and microservices experience."
)


class TestRoleClusters:

    def test_exactly_six_clusters(self):
        assert len(ROLE_CLUSTERS) == 6

    def test_expected_cluster_names_present(self):
        expected = {
            "Data Science", "Backend & Python", "DevOps & Cloud",
            "Frontend & Mobile", "Security", "Business",
        }
        assert set(ROLE_CLUSTERS.keys()) == expected

    def test_total_roles_across_all_clusters_is_14(self):
        flat = [role for roles in ROLE_CLUSTERS.values() for role in roles]
        assert len(flat) == 14

    def test_no_role_belongs_to_multiple_clusters(self):
        flat = [role for roles in ROLE_CLUSTERS.values() for role in roles]
        assert len(flat) == len(set(flat)), "A role appears in more than one cluster"

    def test_data_science_has_correct_roles(self):
        roles = ROLE_CLUSTERS["Data Science"]
        assert "Data Scientist" in roles
        assert "Data Analyst"   in roles
        assert "Data Engineer"  in roles

    def test_backend_cluster_has_correct_roles(self):
        roles = ROLE_CLUSTERS["Backend & Python"]
        assert "Python Developer"  in roles
        assert "Backend Developer" in roles
        assert "Java Developer"    in roles


class TestGetClusterScores:

    def test_returns_a_dict(self):
        assert isinstance(get_cluster_scores(UNIFORM_PROBS), dict)

    def test_returns_all_six_cluster_keys(self):
        result = get_cluster_scores(UNIFORM_PROBS)
        assert set(result.keys()) == set(ROLE_CLUSTERS.keys())

    def test_all_scores_are_floats(self):
        result = get_cluster_scores(UNIFORM_PROBS)
        for score in result.values():
            assert isinstance(score, float)

    def test_all_scores_are_percentages_0_to_100(self):
        result = get_cluster_scores(PYTHON_DEV_PROBS)
        for cluster, score in result.items():
            assert 0.0 <= score <= 100.0, f"{cluster}: {score} out of [0,100]"

    def test_uniform_probs_give_roughly_equal_scores(self):
        result = get_cluster_scores(UNIFORM_PROBS)
        scores = list(result.values())
        # No cluster should dominate — spread must be small
        assert max(scores) - min(scores) < 2.0

    def test_high_python_dev_lifts_backend_cluster(self):
        result = get_cluster_scores(PYTHON_DEV_PROBS)
        assert result["Backend & Python"] > result["Data Science"]
        assert result["Backend & Python"] > result["Security"]

    def test_cluster_average_math_is_correct(self):
        # Backend & Python = (Python Dev + Backend Dev + Java Dev) / 3 * 100
        probs = {role: 0.0 for role in ALL_ROLES}
        probs["Python Developer"]  = 0.60
        probs["Backend Developer"] = 0.30
        probs["Java Developer"]    = 0.00
        result   = get_cluster_scores(probs)
        expected = round((0.60 + 0.30 + 0.00) / 3 * 100, 2)
        assert result["Backend & Python"] == pytest.approx(expected, abs=0.01)

    def test_missing_role_in_probs_defaults_to_zero(self):
        # Pass only one role — should not raise KeyError
        result = get_cluster_scores({"Python Developer": 0.9})
        assert isinstance(result, dict)
        assert len(result) == 6


class TestGetResumeDna:

    def test_returns_a_dict(self):
        result = get_resume_dna(SAMPLE_TEXT, PYTHON_DEV_PROBS, "Python Developer")
        assert isinstance(result, dict)

    def test_contains_cluster_scores_key(self):
        result = get_resume_dna(SAMPLE_TEXT, PYTHON_DEV_PROBS, "Python Developer")
        assert "cluster_scores" in result

    def test_contains_skill_gap_key(self):
        result = get_resume_dna(SAMPLE_TEXT, PYTHON_DEV_PROBS, "Python Developer")
        assert "skill_gap" in result
        assert isinstance(result["skill_gap"], dict)

    def test_skill_gap_contains_present_and_missing_skills(self):
        result = get_resume_dna(SAMPLE_TEXT, PYTHON_DEV_PROBS, "Python Developer")
        assert "present" in result["skill_gap"]
        assert "missing" in result["skill_gap"]
        assert isinstance(result["skill_gap"]["present"], list)
        assert isinstance(result["skill_gap"]["missing"], list)

    def test_skill_gap_contains_fit_pct(self):
        result = get_resume_dna(SAMPLE_TEXT, PYTHON_DEV_PROBS, "Python Developer")
        assert "fit_pct" in result["skill_gap"]
        assert 0.0 <= result["skill_gap"]["fit_pct"] <= 100.0

    def test_detected_plus_missing_equals_15(self):
        # The overview states exactly 15 keywords per category
        result  = get_resume_dna(SAMPLE_TEXT, PYTHON_DEV_PROBS, "Python Developer")
        total   = len(result["skill_gap"]["present"]) + len(result["skill_gap"]["missing"])
        assert total == 15

    def test_fit_score_matches_detected_ratio(self):
        result   = get_resume_dna(SAMPLE_TEXT, PYTHON_DEV_PROBS, "Python Developer")
        detected = len(result["skill_gap"]["present"])
        total    = detected + len(result["skill_gap"]["missing"])
        expected = int(round((detected / total) * 100)) if total else 0
        assert result["skill_gap"]["fit_pct"] == expected

    def test_no_skill_appears_in_both_present_and_missing(self):
        result   = get_resume_dna(SAMPLE_TEXT, PYTHON_DEV_PROBS, "Python Developer")
        overlap  = set(result["skill_gap"]["present"]) & set(result["skill_gap"]["missing"])
        assert len(overlap) == 0, f"Skills in both lists: {overlap}"

    def test_empty_resume_does_not_crash(self):
        result = get_resume_dna("", PYTHON_DEV_PROBS, "Python Developer")
        assert isinstance(result, dict)

    def test_cluster_scores_in_dna_matches_standalone_function(self):
        dna            = get_resume_dna(SAMPLE_TEXT, PYTHON_DEV_PROBS, "Python Developer")
        standalone     = get_cluster_scores(PYTHON_DEV_PROBS)
        assert dna["cluster_scores"] == standalone
