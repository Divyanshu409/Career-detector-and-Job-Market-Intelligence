"""
recommender.py - Career recommendation engine using cosine similarity + skill gap
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import json

from modules.data_processing import SKILL_TAXONOMY, FLAT_SKILLS
from modules.model import CAREER_REQUIRED_SKILLS, analyse_skill_gap


# ── Role similarity engine ────────────────────────────────────────────────────

def build_role_skill_matrix() -> tuple[pd.DataFrame, MultiLabelBinarizer]:
    """Build a skills × roles matrix from our career taxonomy."""
    rows = []
    for role, skills in CAREER_REQUIRED_SKILLS.items():
        rows.append({'role': role, 'skills': skills})
    role_df = pd.DataFrame(rows)
    mlb = MultiLabelBinarizer()
    mat = mlb.fit_transform(role_df['skills'])
    result = pd.DataFrame(mat, columns=mlb.classes_, index=role_df['role'])
    return result, mlb


def similar_roles(career_goal: str, top_n: int = 4) -> list[dict]:
    """Find roles most similar to the target goal."""
    matrix, _ = build_role_skill_matrix()
    if career_goal not in matrix.index:
        return []
    target = matrix.loc[career_goal].values.reshape(1, -1)
    sims   = cosine_similarity(target, matrix.values)[0]
    sim_df = pd.Series(sims, index=matrix.index).sort_values(ascending=False)
    sim_df = sim_df[sim_df.index != career_goal]
    return [
        {'role': role, 'similarity': round(float(score), 3)}
        for role, score in sim_df.head(top_n).items()
    ]


# ── Learning path generator ───────────────────────────────────────────────────

LEARNING_RESOURCES = {
    "python":            {"type": "language",   "difficulty": 1, "weeks": 4},
    "sql":               {"type": "database",   "difficulty": 1, "weeks": 3},
    "machine learning":  {"type": "ml",         "difficulty": 3, "weeks": 10},
    "deep learning":     {"type": "ml",         "difficulty": 4, "weeks": 12},
    "tensorflow":        {"type": "framework",  "difficulty": 3, "weeks": 6},
    "pytorch":           {"type": "framework",  "difficulty": 3, "weeks": 6},
    "docker":            {"type": "devops",     "difficulty": 2, "weeks": 3},
    "kubernetes":        {"type": "devops",     "difficulty": 3, "weeks": 5},
    "aws":               {"type": "cloud",      "difficulty": 2, "weeks": 6},
    "react":             {"type": "framework",  "difficulty": 2, "weeks": 6},
    "javascript":        {"type": "language",   "difficulty": 2, "weeks": 5},
    "typescript":        {"type": "language",   "difficulty": 2, "weeks": 3},
    "spark":             {"type": "data_eng",   "difficulty": 3, "weeks": 6},
    "kafka":             {"type": "data_eng",   "difficulty": 3, "weeks": 4},
    "tableau":           {"type": "viz",        "difficulty": 1, "weeks": 2},
    "power bi":          {"type": "viz",        "difficulty": 1, "weeks": 2},
    "git":               {"type": "tool",       "difficulty": 1, "weeks": 1},
    "linux":             {"type": "os",         "difficulty": 2, "weeks": 3},
    "statistics":        {"type": "math",       "difficulty": 2, "weeks": 6},
    "nlp":               {"type": "ml",         "difficulty": 3, "weeks": 8},
    "computer vision":   {"type": "ml",         "difficulty": 4, "weeks": 10},
    "terraform":         {"type": "devops",     "difficulty": 3, "weeks": 4},
    "airflow":           {"type": "data_eng",   "difficulty": 2, "weeks": 3},
    "pandas":            {"type": "library",    "difficulty": 1, "weeks": 3},
    "scikit-learn":      {"type": "library",    "difficulty": 2, "weeks": 4},
    "r":                 {"type": "language",   "difficulty": 2, "weeks": 5},
    "java":              {"type": "language",   "difficulty": 2, "weeks": 8},
    "nodejs":            {"type": "runtime",    "difficulty": 2, "weeks": 4},
    "rest api":          {"type": "concept",    "difficulty": 2, "weeks": 2},
    "microservices":     {"type": "concept",    "difficulty": 3, "weeks": 4},
    "mongodb":           {"type": "database",   "difficulty": 2, "weeks": 2},
    "postgresql":        {"type": "database",   "difficulty": 2, "weeks": 3},
}

def generate_learning_path(career_goal: str, current_skills: list[str],
                            timeline_weeks: int = 24) -> dict:
    """
    Generate a phased, week-by-week learning roadmap.
    Returns phases with skills, duration, and rationale.
    """
    gap_result = analyse_skill_gap(career_goal, current_skills)
    if 'error' in gap_result:
        return gap_result

    missing = gap_result['missing_skills']

    # Sort by difficulty (easiest first) then by market demand
    enriched = []
    for sk in missing:
        info = LEARNING_RESOURCES.get(sk, {'difficulty': 2, 'weeks': 4, 'type': 'skill'})
        enriched.append({
            'skill':      sk,
            'difficulty': info['difficulty'],
            'weeks':      info['weeks'],
            'type':       info.get('type', 'skill')
        })
    enriched.sort(key=lambda x: (x['difficulty'], x['weeks']))

    # Build phases
    phases = []
    week   = 0
    for item in enriched:
        if week >= timeline_weeks:
            break
        phases.append({
            'phase':      len(phases) + 1,
            'skill':      item['skill'],
            'start_week': week + 1,
            'end_week':   week + item['weeks'],
            'difficulty': item['difficulty'],
            'type':       item['type'],
            'rationale':  _skill_rationale(item['skill'], career_goal)
        })
        week += item['weeks']

    return {
        'career_goal':     gap_result['career_goal'],
        'match_pct':       gap_result['match_pct'],
        'readiness_level': gap_result['readiness_level'],
        'matched_skills':  gap_result['matched_skills'],
        'phases':          phases,
        'total_weeks':     week,
        'similar_roles':   similar_roles(career_goal.lower())
    }


def _skill_rationale(skill: str, role: str) -> str:
    rationales = {
        "python":            "Python is the backbone of modern data science and automation.",
        "sql":               "SQL is essential for querying databases in virtually every data role.",
        "machine learning":  "Core competency for any ML/AI career path.",
        "deep learning":     "Powers modern AI from NLP to computer vision.",
        "tensorflow":        "Industry-standard deep learning framework at Google.",
        "pytorch":           "Dominant in research; growing fast in production.",
        "docker":            "Container technology used in nearly every production system.",
        "kubernetes":        "Required for orchestrating containers at scale.",
        "aws":               "Largest cloud provider; widely demanded in job postings.",
        "react":             "Leading frontend framework for web applications.",
        "javascript":        "The language of the web; unavoidable for frontend work.",
        "statistics":        "Foundational for understanding data and model behaviour.",
        "git":               "Version control is a non-negotiable baseline skill.",
        "linux":             "Most servers run Linux; essential for backend/DevOps.",
    }
    base = rationales.get(skill, f'Highly demanded for {role} roles in the current job market.')
    return base


# ── Collaborative filtering (user–skill matrix) ───────────────────────────────

def collaborative_skill_suggest(df: pd.DataFrame,
                                  user_skills: list[str],
                                  top_n: int = 8) -> list[dict]:
    """
    Find the most similar job postings to the user's skill profile,
    then recommend skills those postings have that the user doesn't.
    """
    mlb = MultiLabelBinarizer()
    mat = mlb.fit_transform(df['skills_list'])
    classes = mlb.classes_

    # Encode user
    user_enc = np.zeros((1, len(classes)))
    for sk in user_skills:
        if sk in classes:
            idx = list(classes).index(sk)
            user_enc[0, idx] = 1

    if user_enc.sum() == 0:
        return []

    sims  = cosine_similarity(user_enc, mat)[0]
    top_i = np.argsort(sims)[::-1][:50]

    # Collect skills from similar postings
    candidate_skills = {}
    for i in top_i:
        for sk in df.iloc[i]['skills_list']:
            if sk not in user_skills:
                candidate_skills[sk] = candidate_skills.get(sk, 0) + sims[i]

    ranked = sorted(candidate_skills.items(), key=lambda x: -x[1])
    return [{'skill': sk, 'score': round(float(s), 3)} for sk, s in ranked[:top_n]]
