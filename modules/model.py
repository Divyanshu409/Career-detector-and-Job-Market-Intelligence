import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               HistGradientBoostingRegressor, StackingRegressor)
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

from modules.data_processing import SKILL_TAXONOMY, FLAT_SKILLS, EXPERIENCE_ORDINAL


# ── Salary Prediction ────────────────────────────────────────────────────────

class SalaryPredictor:
    def __init__(self):
        # FIX: stronger models + stacking ensemble
        self.base_models = {
            'Ridge Regression':           Ridge(alpha=10.0),
            'Random Forest':              RandomForestRegressor(
                                              n_estimators=300, max_depth=12,
                                              min_samples_leaf=5, max_features=0.4,
                                              random_state=42, n_jobs=-1),
            'Hist Gradient Boosting':     HistGradientBoostingRegressor(
                                              max_iter=300, max_depth=6,
                                              learning_rate=0.05, min_samples_leaf=20,
                                              random_state=42),
        }
        # FIX: stacking ensemble uses the 3 base models with Ridge as meta-learner
        self.stacking_model = StackingRegressor(
            estimators=[
                ('rf',  RandomForestRegressor(n_estimators=200, max_depth=10,
                                              max_features=0.4, random_state=42, n_jobs=-1)),
                ('hgb', HistGradientBoostingRegressor(max_iter=200, max_depth=5,
                                                      learning_rate=0.05, random_state=42)),
                ('ridge', Ridge(alpha=10.0)),
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            n_jobs=-1
        )
        self.models = {**self.base_models, 'Stacking Ensemble': self.stacking_model}

        self.mlb      = MultiLabelBinarizer()
        self.le_role  = LabelEncoder()
        self.le_loc   = LabelEncoder()   # FIX: location encoder
        self.scaler   = StandardScaler()
        self.trained  = False
        self.feature_names = []
        self.use_log_target = True  # FIX: log-transform target

    def _build_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        if fit:
            skill_mat = self.mlb.fit_transform(df['skills_list'])
        else:
            skill_mat = self.mlb.transform(df['skills_list'])

        # Role encoding
        roles = df['role'].fillna('other')
        if fit:
            role_enc = self.le_role.fit_transform(roles)
        else:
            roles = roles.apply(lambda r: r if r in self.le_role.classes_ else 'other')
            role_enc = self.le_role.transform(roles)

        # FIX: ordinal experience (correct numeric order, not alphabetical)
        exp_ordinal = df['experience_ordinal'].fillna(2).values \
            if 'experience_ordinal' in df.columns \
            else df['experience_level'].map(EXPERIENCE_ORDINAL).fillna(2).values

        # FIX: continuous experience years (stronger signal than ordinal buckets)
        exp_years = df['experience_years'].fillna(3.0).values \
            if 'experience_years' in df.columns \
            else exp_ordinal * 2.0

        # FIX: location encoding
        if 'location_clean' in df.columns:
            locs = df['location_clean'].fillna('Unknown')
        elif 'location' in df.columns:
            locs = df['location'].fillna('Unknown')
        else:
            locs = pd.Series(['Unknown'] * len(df))

        if fit:
            # FIX: fit on classes + 'Unknown' sentinel so inference never fails on unseen locs
            self.le_loc.fit(pd.concat([locs, pd.Series(['Unknown'])], ignore_index=True))
            loc_enc = self.le_loc.transform(locs)
        else:
            locs = locs.apply(lambda l: l if l in self.le_loc.classes_ else 'Unknown')
            loc_enc = self.le_loc.transform(locs)

        skill_count = df['skill_count'].fillna(0).values

        # FIX: combined numeric block: role, exp_ordinal, exp_years, location, skill_count
        num_features = np.column_stack([
            role_enc, exp_ordinal, exp_years, loc_enc, skill_count
        ])
        X = np.hstack([num_features, skill_mat])

        if fit:
            self.feature_names = (
                ['role_enc', 'exp_ordinal', 'exp_years', 'loc_enc', 'skill_count'] +
                list(self.mlb.classes_)
            )

        return X

    def train(self, df: pd.DataFrame) -> dict:
        train_df = df.dropna(subset=['salary']).copy()
        if len(train_df) < 20:
            return {'error': f'Need ≥20 rows with salary data (found {len(train_df)})'}

        X = self._build_features(train_df, fit=True)
        y = train_df['salary'].values

        # FIX: log-transform salary target to reduce right-skew
        if self.use_log_target:
            y_fit = np.log1p(y)
        else:
            y_fit = y

        X = self.scaler.fit_transform(X)

        results = {}
        best_score = -np.inf
        self.best_model_name = None

        n_cv = min(3, max(3, len(y) // 500))

        for name, mdl in self.models.items():
            try:
                cv = cross_val_score(mdl, X, y_fit, cv=n_cv,
                                     scoring='r2', n_jobs=-1)
                results[name] = {'r2_mean': round(float(cv.mean()), 3),
                                 'r2_std':  round(float(cv.std()),  3)}
                mdl.fit(X, y_fit)
                if cv.mean() > best_score:
                    best_score = cv.mean()
                    self.best_model_name = name
            except Exception as e:
                results[name] = {'error': str(e)}

        self.trained = True
        self.train_df_ = train_df
        return results

    def predict(self, role: str, experience: str, skills: list,
                location: str = 'Unknown') -> dict:
        if not self.trained:
            return {'error': 'Model not trained yet'}

        from modules.data_processing import extract_experience_years, EXPERIENCE_ORDINAL
        exp_years   = extract_experience_years(experience)
        exp_ordinal = EXPERIENCE_ORDINAL.get(experience.lower(), 2)

        row = pd.DataFrame([{
            'role':             role,
            'experience_level': experience,
            'experience_ordinal': exp_ordinal,
            'experience_years': exp_years,
            'skills_list':      skills,
            'skill_count':      len(skills),
            'location_clean':   location,
        }])
        try:
            X = self._build_features(row, fit=False)
            X = self.scaler.transform(X)
        except Exception as e:
            return {'error': str(e)}

        preds = {}
        for name, mdl in self.models.items():
            try:
                raw = float(mdl.predict(X)[0])
                # FIX: inverse log-transform
                val = np.expm1(raw) if self.use_log_target else raw
                preds[name] = max(0, round(val, 0))
            except Exception:
                pass

        best = preds.get(self.best_model_name, list(preds.values())[0] if preds else 0)

        # FIX: use HistGradientBoosting feature importance instead of RF
        importance = {}
        try:
            rf = self.models['Random Forest']
            fi = rf.feature_importances_
            # FIX: guard against index out of range
            n = min(len(fi), len(self.feature_names))
            top_idx = np.argsort(fi[:n])[::-1][:10]
            importance = {self.feature_names[i]: round(float(fi[i]), 4) for i in top_idx}
        except Exception:
            pass

        return {
            'best_model': self.best_model_name,
            'predicted_salary': max(0, best),
            'all_predictions': preds,
            'feature_importance': importance
        }

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.trained:
            return pd.DataFrame()
        try:
            rf = self.models['Random Forest']
            fi = rf.feature_importances_
            n = min(len(fi), len(self.feature_names))
            df = pd.DataFrame({'feature': self.feature_names[:n], 'importance': fi[:n]})
            return df.sort_values('importance', ascending=False).head(20)
        except Exception:
            return pd.DataFrame()


# ── Job Role Clustering ──────────────────────────────────────────────────────

class JobClusterer:
    def __init__(self, n_clusters: int = 7):
        self.n_clusters = n_clusters
        self.mlb     = MultiLabelBinarizer()
        self.kmeans  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.pca     = PCA(n_components=2, random_state=42)
        self.trained = False

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.mlb.fit_transform(df['skills_list'])
        if X.shape[0] < self.n_clusters:
            self.n_clusters = max(2, X.shape[0] // 2)
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)

        labels = self.kmeans.fit_predict(X)
        coords = self.pca.fit_transform(X)

        result = df[['title_raw', 'role', 'company', 'salary']].copy()
        result['cluster'] = labels
        result['pca_x']   = coords[:, 0]
        result['pca_y']   = coords[:, 1]
        self.trained = True
        return result

    def cluster_summary(self, df: pd.DataFrame, clustered: pd.DataFrame) -> dict:
        summary = {}
        for cid in range(self.n_clusters):
            idx = clustered[clustered['cluster'] == cid].index
            sub = df.loc[idx]
            skills = [s for sl in sub['skills_list'] for s in sl]
            top = pd.Series(skills).value_counts().head(5).index.tolist()
            summary[cid] = {
                'top_skills': top,
                'job_count':  len(idx),
                'avg_salary': round(sub['salary'].mean(), 0) if sub['salary'].notna().any() else None
            }
        return summary


# ── Skill Gap Analyser ────────────────────────────────────────────────────────

CAREER_REQUIRED_SKILLS = {
    "data scientist": [
        "python","machine learning","deep learning","tensorflow","pytorch",
        "pandas","numpy","sql","statistics","scikit-learn","nlp",
        "data visualization","r","matplotlib","jupyter"
    ],
    "data analyst": [
        "sql","excel","python","tableau","power bi","statistics",
        "pandas","data visualization","r","matplotlib","seaborn"
    ],
    "machine learning engineer": [
        "python","machine learning","deep learning","tensorflow","pytorch",
        "docker","kubernetes","mlops","sql","scikit-learn","aws","git"
    ],
    "software engineer": [
        "python","java","javascript","sql","git","docker","linux",
        "algorithms","data structures","rest api","agile"
    ],
    "frontend developer": [
        "javascript","typescript","react","html","css","git",
        "responsive design","vue","angular","nodejs","figma"
    ],
    "backend developer": [
        "python","java","node","sql","postgresql","redis","docker",
        "kubernetes","rest api","microservices","aws","git"
    ],
    "full stack developer": [
        "javascript","typescript","react","nodejs","python","sql",
        "mongodb","docker","git","rest api","html","css"
    ],
    "data engineer": [
        "python","sql","spark","kafka","airflow","dbt","aws",
        "postgresql","mongodb","docker","etl","data pipeline"
    ],
    "devops engineer": [
        "linux","docker","kubernetes","terraform","ansible","aws",
        "jenkins","git","python","bash","prometheus","grafana"
    ],
    "ai engineer": [
        "python","machine learning","deep learning","tensorflow","pytorch",
        "nlp","computer vision","aws","docker","kubernetes","mlops","git"
    ],
    "cloud architect": [
        "aws","azure","gcp","terraform","kubernetes","docker",
        "microservices","networking","security","python"
    ],
    "web developer": [
        "html","css","javascript","react","git","sql",
        "nodejs","responsive design","typescript","figma"
    ]
}

def analyse_skill_gap(career_goal: str, current_skills: list) -> dict:
    goal = career_goal.lower()
    required = CAREER_REQUIRED_SKILLS.get(goal, [])
    if not required:
        for k, v in CAREER_REQUIRED_SKILLS.items():
            if any(w in k for w in goal.split()):
                required = v
                goal = k
                break
    if not required:
        return {'error': f'Career goal "{career_goal}" not found'}

    current_set  = set(s.lower() for s in current_skills)
    required_set = set(required)
    matched      = current_set & required_set
    missing      = required_set - current_set
    match_pct    = round(len(matched) / len(required_set) * 100, 1)

    skill_counts = {}
    for role_skills in CAREER_REQUIRED_SKILLS.values():
        for sk in role_skills:
            skill_counts[sk] = skill_counts.get(sk, 0) + 1

    missing_prioritised = sorted(missing, key=lambda s: -skill_counts.get(s, 0))

    return {
        'career_goal':      goal,
        'required_skills':  required,
        'matched_skills':   sorted(matched),
        'missing_skills':   missing_prioritised,
        'match_pct':        match_pct,
        'top_recommended':  missing_prioritised[:5],
        'readiness_level':  _readiness_label(match_pct)
    }


def _readiness_label(pct: float) -> str:
    if pct >= 80: return "Job-Ready"
    if pct >= 60: return "Almost Ready"
    if pct >= 40: return "Intermediate"
    if pct >= 20: return "Beginner"
    return "Just Starting"


def market_demand_for_skills(df: pd.DataFrame, skills: list) -> dict:
    total = len(df)
    result = {}
    for sk in skills:
        count = df['skills_list'].apply(lambda x: sk in x).sum()
        result[sk] = {'count': int(count), 'pct': round(count/total*100, 1)}
    return result
