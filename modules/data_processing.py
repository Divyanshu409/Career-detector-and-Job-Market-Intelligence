"""
data_processing.py - Data cleaning, feature engineering, and skill extraction
"""
import pandas as pd
import numpy as np
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── Comprehensive skill taxonomy ──────────────────────────────────────────────
SKILL_TAXONOMY = {
    "Programming Languages": [
        "python","java","javascript","typescript","c++","c#","ruby","go","golang",
        "rust","swift","kotlin","scala","r","matlab","php","perl","bash","shell",
        "powershell","dart","elixir","haskell","julia","lua","groovy","vba"
    ],
    "Web Frameworks": [
        "react","reactjs","angular","angularjs","vue","vuejs","django","flask",
        "fastapi","spring","springboot","express","expressjs","nextjs","nuxt",
        "rails","laravel","asp.net","node","nodejs","svelte","gatsby","remix"
    ],
    "Data Science & ML": [
        "machine learning","deep learning","nlp","natural language processing",
        "computer vision","tensorflow","pytorch","keras","scikit-learn","sklearn",
        "pandas","numpy","scipy","xgboost","lightgbm","catboost","hugging face",
        "transformers","bert","gpt","llm","reinforcement learning","neural network",
        "random forest","gradient boosting","svm","support vector",
        # FIX: added missing but common skills from job descriptions
        "rag","langchain","fine-tuning","vllm","mlops","generative ai",
        "stable diffusion","llama","mistral","openai","embeddings","vector database"
    ],
    "Data Engineering": [
        "sql","mysql","postgresql","mongodb","redis","elasticsearch","kafka",
        "spark","hadoop","airflow","dbt","etl","data pipeline","databricks",
        "snowflake","bigquery","redshift","datalake","hive","presto","flink",
        "cassandra","dynamodb","neo4j","influxdb","clickhouse","pinecone","weaviate"
    ],
    "Cloud & DevOps": [
        "aws","azure","gcp","google cloud","docker","kubernetes","k8s","terraform",
        "ansible","jenkins","github actions","ci/cd","devops","mlops","linux",
        "ubuntu","helm","prometheus","grafana","cloudformation","pulumi","argocd",
        "istio","microservices","serverless","lambda","cloud functions"
    ],
    "Data Visualization": [
        "tableau","power bi","matplotlib","seaborn","plotly","d3","d3.js",
        "bokeh","looker","qlik","superset","metabase","grafana","kibana","dash"
    ],
    "Soft Skills & Domain": [
        "communication","leadership","teamwork","problem solving","agile","scrum",
        "project management","stakeholder","presentation","analytical","research",
        "statistics","probability","linear algebra","calculus","optimization"
    ],
    "Mobile & Other": [
        "android","ios","flutter","react native","xamarin","unity","unreal",
        "blockchain","solidity","web3","cybersecurity","penetration testing",
        "networking","embedded","iot","robotics","ar","vr",
        # FIX: added more specialised roles
        "qa","testing","selenium","pytest","junit","cypress","playwright",
        "security","cryptography","zero trust","soc","siem"
    ]
}

FLAT_SKILLS = {skill: category
               for category, skills in SKILL_TAXONOMY.items()
               for skill in skills}

# FIX: Massively expanded ROLE_NORMALIZER to cover unmapped titles (was 31% 'other')
ROLE_NORMALIZER = {
    "data scientist": [
        "data science","data sciences","data scientist","senior data scientist"
    ],
    "data analyst": [
        "data analyst","data analytics","analyst","business analyst","bi analyst"
    ],
    "machine learning engineer": [
        "ml engineer","machine learning engineer","ml dev","mlops engineer",
        "ml ops","mlops"
    ],
    "software engineer": [
        "software engineer","software developer","swe","sde","qa engineer",
        "quality assurance","test engineer","quality engineer"
    ],
    "frontend developer": [
        "frontend","front-end","front end","ui developer","ui engineer"
    ],
    "backend developer": [
        "backend","back-end","back end","api developer","api engineer"
    ],
    "full stack developer": [
        "full stack","fullstack","full-stack"
    ],
    "data engineer": [
        "data engineer","data engineering","etl developer","etl engineer",
        "analytics engineer"
    ],
    "devops engineer": [
        "devops","site reliability","sre","platform engineer","infrastructure engineer",
        "cloud engineer","cloud infrastructure"
    ],
    "product manager": [
        "product manager","pm","product owner","po","program manager"
    ],
    "ai engineer": [
        "ai engineer","artificial intelligence","ai developer","ai ml",
        "ai research","llm engineer","generative ai","gen ai engineer",
        "computer vision engineer","nlp engineer","ai research scientist",
        "research scientist","applied scientist","applied ml"
    ],
    "cloud architect": [
        "cloud architect","solutions architect","cloud engineer","enterprise architect"
    ],
    "cybersecurity analyst": [
        "cybersecurity","security analyst","infosec","pentest","security engineer",
        "penetration tester","appsec","devsecops","blockchain developer","blockchain"
    ],
    "mobile developer": [
        "mobile developer","android developer","ios developer","mobile engineer",
        "embedded systems","embedded engineer","iot engineer","firmware"
    ],
    "web developer": [
        "web developer","web dev","website developer"
    ]
}

EXPERIENCE_MAP = {
    "entry": 0, "junior": 1, "associate": 1,
    "mid": 2, "mid-level": 2, "intermediate": 2,
    "senior": 3, "sr.": 3, "lead": 3,
    "principal": 4, "staff": 4,
    "manager": 5, "director": 6, "vp": 7, "head": 6
}

# FIX: Ordinal encoding map so the model sees correct numeric order
EXPERIENCE_ORDINAL = {
    "entry": 0, "associate": 1, "junior": 2, "mid": 3, "mid-level": 3,
    "intermediate": 3, "senior": 4, "lead": 4, "sr.": 4,
    "staff": 5, "principal": 5, "manager": 6, "head": 6,
    "director": 7, "vp": 8, "unknown": 2  # default to mid
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^\w\s\+\#\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_skills(text: str) -> list:
    cleaned = _clean_text(text)
    found = set()
    for skill in FLAT_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, cleaned):
            found.add(skill)
    return sorted(found)


def normalize_role(title: str) -> str:
    if not isinstance(title, str):
        return "other"
    t = _clean_text(title)
    for canonical, variants in ROLE_NORMALIZER.items():
        for v in variants:
            if v in t:
                return canonical
    return "other"


def extract_experience_level(text: str) -> str:
    if not isinstance(text, str):
        return "unknown"
    t = text.lower()
    for level in EXPERIENCE_MAP:
        if level in t:
            return level
    patterns = [
        (r'(\d+)\+?\s*years?', lambda m: int(m.group(1))),
        (r'(\d+)\s*-\s*(\d+)\s*years?', lambda m: (int(m.group(1))+int(m.group(2)))//2),
    ]
    for pat, fn in patterns:
        m = re.search(pat, t)
        if m:
            yrs = fn(m)
            if yrs <= 1:   return "entry"
            elif yrs <= 3: return "junior"
            elif yrs <= 5: return "mid"
            elif yrs <= 8: return "senior"
            else:           return "principal"
    return "unknown"


# FIX: extract numeric years directly (stronger signal than bucketed level)
def extract_experience_years(text: str) -> float:
    """Return numeric years of experience; falls back to level-based estimate."""
    if not isinstance(text, str):
        return 3.0
    t = text.lower()
    patterns = [
        (r'(\d+)\+\s*years?', lambda m: int(m.group(1)) + 1),
        (r'(\d+)\s*-\s*(\d+)\s*years?', lambda m: (int(m.group(1)) + int(m.group(2))) / 2),
        (r'(\d+)\s*years?', lambda m: int(m.group(1))),
    ]
    for pat, fn in patterns:
        m = re.search(pat, t)
        if m:
            return float(fn(m))
    level_defaults = {
        "entry": 0.5, "junior": 2.0, "associate": 2.0, "mid": 4.0,
        "mid-level": 4.0, "intermediate": 4.0, "senior": 7.0, "sr.": 7.0,
        "lead": 8.0, "principal": 10.0, "staff": 10.0,
        "manager": 8.0, "director": 12.0, "vp": 15.0, "head": 12.0
    }
    for level, yrs in level_defaults.items():
        if level in t:
            return yrs
    return 3.0


def parse_salary(value) -> float:
    if pd.isna(value):
        return None
    s = str(value).lower().replace(',', '').replace('$', '').strip()
    multiplier = 1
    if 'k' in s:
        s = s.replace('k', '')
        multiplier = 1000
    if 'per hour' in s or '/hr' in s or 'hourly' in s:
        multiplier *= 2080
    s = re.sub(r'[^0-9\.\-]', ' ', s).strip()
    parts = re.split(r'[\s\-–]+', s)
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except ValueError:
            pass
    if not nums:
        return None
    return round(np.mean(nums) * multiplier, 2)


# ── Main preprocessing pipeline ──────────────────────────────────────────────

def load_and_preprocess(filepath: str) -> dict:
    try:
        df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1', low_memory=False)

    df.columns = [c.strip().lower().replace(' ', '_').replace('-', '_')
                  for c in df.columns]

    col_map = _detect_columns(df)

    wdf = pd.DataFrame()

    wdf['title_raw'] = df[col_map['title']].astype(str) if col_map['title'] else 'unknown'
    wdf['company']   = df[col_map['company']].astype(str).str.strip().str.title() \
                       if col_map['company'] else 'Unknown'
    wdf['location']  = df[col_map['location']].astype(str).str.strip().str.title() \
                       if col_map['location'] else 'Unknown'
    wdf['description'] = df[col_map['description']].astype(str) \
                       if col_map['description'] else ''

    if col_map['salary']:
        wdf['salary_raw'] = df[col_map['salary']]
        # FIX: if salary column is already numeric (not a string), skip parse_salary
        if pd.api.types.is_numeric_dtype(df[col_map['salary']]):
            wdf['salary'] = df[col_map['salary']].astype(float)
        else:
            wdf['salary'] = wdf['salary_raw'].apply(parse_salary)
    else:
        wdf['salary'] = np.nan

    if col_map['date']:
        wdf['date_posted'] = pd.to_datetime(df[col_map['date']], errors='coerce')
    else:
        wdf['date_posted'] = pd.NaT

    exp_col = col_map.get('experience')
    if exp_col:
        wdf['experience_level'] = df[exp_col].apply(extract_experience_level)
        # FIX: also store continuous years as a separate feature
        wdf['experience_years'] = df[exp_col].apply(extract_experience_years)
    else:
        wdf['experience_level'] = wdf['title_raw'].apply(extract_experience_level)
        mask = wdf['experience_level'] == 'unknown'
        wdf.loc[mask, 'experience_level'] = \
            wdf.loc[mask, 'description'].apply(extract_experience_level)
        wdf['experience_years'] = wdf['experience_level'].map(
            {"entry": 0.5, "junior": 2.0, "mid": 4.0, "senior": 7.0,
             "principal": 10.0, "unknown": 3.0}
        ).fillna(3.0)

    # FIX: ordinal encode experience (not LabelEncoder which is alphabetical)
    wdf['experience_ordinal'] = wdf['experience_level'].map(EXPERIENCE_ORDINAL).fillna(2)

    wdf['role'] = wdf['title_raw'].apply(normalize_role)

    combined_text = wdf['title_raw'].str.lower() + ' ' + wdf['description'].str.lower()
    wdf['skills_list'] = combined_text.apply(extract_skills)
    wdf['skills_str']  = wdf['skills_list'].apply(lambda x: ', '.join(x))
    wdf['skill_count'] = wdf['skills_list'].apply(len)

    # FIX: use wider IQR-based outlier removal (1.5x IQR) instead of hard 5-95 percentile
    # which was silently dropping 10% of valid rows
    sal = wdf['salary'].dropna()
    if len(sal) > 10:
        q1, q3 = sal.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = max(0, q1 - 3.0 * iqr)   # 3x IQR = generous, only true outliers
        upper = q3 + 3.0 * iqr
        wdf.loc[(wdf['salary'] < lower) | (wdf['salary'] > upper), 'salary'] = np.nan

    wdf['year_month'] = wdf['date_posted'].dt.to_period('M').astype(str)

    # FIX: add location as a feature column
    wdf['location_clean'] = wdf['location'].str.strip().str.title()

    stats = {
        'total_rows': len(wdf),
        'columns_detected': col_map,
        'missing_salary_pct': round(wdf['salary'].isna().mean() * 100, 1),
        'roles_found': wdf['role'].nunique(),
        'date_range': (
            str(wdf['date_posted'].min())[:10],
            str(wdf['date_posted'].max())[:10]
        ) if col_map['date'] else None
    }

    return {'df': wdf, 'stats': stats, 'col_map': col_map}


def _detect_columns(df: pd.DataFrame) -> dict:
    cols = df.columns.tolist()

    def match(keywords):
        for c in cols:
            for k in keywords:
                if k in c:
                    return c
        return None

    return {
        'title':       match(['title','job_title','position','role','job_name']),
        'company':     match(['company','employer','organization','firm','corp']),
        'location':    match(['location','city','state','place','country','region']),
        'description': match(['description','details','summary','requirements','body','text','posting']),
        'salary':      match(['salary','pay','compensation','wage','ctc','annual','income']),
        'date':        match(['date','posted','published','created','timestamp','time']),
        'experience':  match(['experience','exp','level','seniority','years']),
    }


# ── Skill-frequency helpers ───────────────────────────────────────────────────

def skill_frequency_df(df: pd.DataFrame) -> pd.DataFrame:
    all_skills = [s for skills in df['skills_list'] for s in skills]
    freq = pd.Series(all_skills).value_counts().reset_index()
    freq.columns = ['skill', 'count']
    freq['category'] = freq['skill'].map(FLAT_SKILLS).fillna('Other')
    freq['pct'] = (freq['count'] / len(df) * 100).round(1)
    return freq


def trending_skills_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df[['year_month', 'skills_list']].iterrows():
        for sk in row['skills_list']:
            rows.append({'year_month': row['year_month'], 'skill': sk})
    if not rows:
        return pd.DataFrame(columns=['year_month','skill','count'])
    trend = pd.DataFrame(rows)
    trend = trend.groupby(['year_month','skill']).size().reset_index(name='count')
    trend = trend.sort_values('year_month')
    return trend


def salary_by_role(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby('role')['salary']
              .agg(['mean','median','count'])
              .reset_index()
              .rename(columns={'mean':'avg_salary','median':'median_salary','count':'job_count'})
              .dropna()
              .sort_values('avg_salary', ascending=False))


def tfidf_top_skills(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    docs = df['description'].fillna('').tolist()
    if len(docs) < 5:
        return pd.DataFrame(columns=['term','tfidf_score'])
    tv = TfidfVectorizer(max_features=200, stop_words='english',
                         ngram_range=(1, 2), min_df=2)
    try:
        mat = tv.fit_transform(docs)
    except Exception:
        return pd.DataFrame(columns=['term','tfidf_score'])
    scores = np.asarray(mat.mean(axis=0)).ravel()
    terms  = tv.get_feature_names_out()
    tfidf_df = pd.DataFrame({'term': terms, 'tfidf_score': scores})
    tfidf_df = tfidf_df.sort_values('tfidf_score', ascending=False).head(top_n)
    return tfidf_df
