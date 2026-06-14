
import io, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Palette ───────────────────────────────────────────────────────────────────
PRIMARY   = '#6C63FF'
SECONDARY = '#00D9A3'
ACCENT    = '#FF6584'
WARM      = '#FFB347'
NEUTRAL   = '#1C1C2E'
LIGHT     = '#F0EFFF'

PALETTE   = [PRIMARY, SECONDARY, ACCENT, WARM, '#4ECDC4', '#FF8B94',
             '#A8E6CF', '#FFEAA7', '#DDA0DD', '#98D8C8']

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _base_fig(w=10, h=5, dark=True):
    bg = NEUTRAL if dark else '#FAFAFA'
    fig = plt.figure(figsize=(w, h), facecolor=bg)
    return fig, bg

def _style_ax(ax, bg):
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_edgecolor('#3A3A5C')
    ax.tick_params(colors='#CCCCDD', labelsize=9)
    ax.xaxis.label.set_color('#CCCCDD')
    ax.yaxis.label.set_color('#CCCCDD')
    ax.title.set_color('#EBEBFF')


def top_skills_bar(freq_df: pd.DataFrame, top_n: int = 15) -> str:
    data = freq_df.head(top_n)
    fig, bg = _base_fig(10, 6)
    ax = fig.add_subplot(111)
    _style_ax(ax, bg)

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(data))]
    bars = ax.barh(data['skill'][::-1], data['count'][::-1],
                   color=colors[::-1], height=0.7, edgecolor='none')

    for bar, val in zip(bars, data['count'][::-1]):
        ax.text(bar.get_width() + max(data['count']) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{int(val):,}', va='center', color='#CCCCDD', fontsize=8)

    ax.set_xlabel('Frequency in Job Postings', color='#CCCCDD')
    ax.set_title('Top In-Demand Skills', color='#EBEBFF', fontsize=13, pad=12, fontweight='bold')
    ax.grid(axis='x', color='#2A2A4A', linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _fig_to_b64(fig)


def skill_category_donut(freq_df: pd.DataFrame) -> str:
    cat_counts = freq_df.groupby('category')['count'].sum().sort_values(ascending=False)
    fig, bg = _base_fig(8, 6)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg)

    wedges, texts, autotexts = ax.pie(
        cat_counts.values,
        labels=None,
        autopct=lambda p: f'{p:.1f}%' if p > 4 else '',
        colors=PALETTE[:len(cat_counts)],
        startangle=140,
        pctdistance=0.82,
        wedgeprops=dict(width=0.55, edgecolor=bg, linewidth=2)
    )
    for at in autotexts:
        at.set_color('#EBEBFF')
        at.set_fontsize(8)

    ax.legend(wedges, cat_counts.index, loc='lower right',
              facecolor='#1C1C2E', edgecolor='#3A3A5C', labelcolor='#CCCCDD',
              fontsize=8, framealpha=0.8)
    ax.set_title('Skills by Category', color='#EBEBFF', fontsize=12, pad=12, fontweight='bold')
    ax.text(0, 0, 'Skills\nBreakdown', ha='center', va='center',
            color='#EBEBFF', fontsize=10, fontweight='bold')
    return _fig_to_b64(fig)


def salary_by_role_chart(df: pd.DataFrame, top_roles: int = 10) -> str:
    sal_df = df.dropna(subset=['salary'])
    if sal_df.empty:
        return _placeholder('No salary data available')

    role_counts = sal_df['role'].value_counts()
    top_r = role_counts[role_counts >= 3].head(top_roles).index.tolist()
    plot_df = sal_df[sal_df['role'].isin(top_r)]
    if plot_df.empty:
        return _placeholder('Insufficient salary data per role')

    order = (plot_df.groupby('role')['salary'].median()
             .sort_values(ascending=True).index)

    fig, bg = _base_fig(11, 6)
    ax = fig.add_subplot(111)
    _style_ax(ax, bg)

    bp = ax.boxplot(
        [plot_df[plot_df['role'] == r]['salary'].values for r in order],
        vert=False, patch_artist=True,
        boxprops=dict(facecolor=PRIMARY+'55', edgecolor=PRIMARY),
        medianprops=dict(color=SECONDARY, linewidth=2),
        whiskerprops=dict(color='#6666AA'),
        capprops=dict(color='#6666AA'),
        flierprops=dict(marker='o', color=ACCENT, markersize=3, alpha=0.5)
    )

    ax.set_yticks(range(1, len(order)+1))
    ax.set_yticklabels([r.title() for r in order], fontsize=9)
    ax.set_xlabel('Annual Salary (USD)', color='#CCCCDD')
    ax.set_title('Salary Distribution by Job Role', color='#EBEBFF', fontsize=13, pad=12, fontweight='bold')
    ax.grid(axis='x', color='#2A2A4A', linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _fig_to_b64(fig)


def skill_salary_heatmap(df: pd.DataFrame, top_skills: int = 20) -> str:
    sal_df = df.dropna(subset=['salary'])
    if len(sal_df) < 10:
        return _placeholder('Need ≥10 rows with salary for heatmap')

    all_skills = [s for sl in sal_df['skills_list'] for s in sl]
    top = pd.Series(all_skills).value_counts().head(top_skills).index.tolist()

    rows = []
    for sk in top:
        has    = sal_df[sal_df['skills_list'].apply(lambda x: sk in x)]['salary']
        has_not= sal_df[sal_df['skills_list'].apply(lambda x: sk not in x)]['salary']
        if len(has) >= 3 and len(has_not) >= 3:
            rows.append({'skill': sk,
                         'with_skill':    has.mean(),
                         'without_skill': has_not.mean(),
                         'premium':       has.mean() - has_not.mean()})

    if not rows:
        return _placeholder('Insufficient data for heatmap')

    heat_df = pd.DataFrame(rows).sort_values('premium', ascending=False)
    mat     = heat_df[['with_skill', 'without_skill']].values

    fig, bg = _base_fig(9, max(5, len(heat_df) * 0.45))
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg)

    im = ax.imshow(mat, cmap='RdYlGn', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['With Skill', 'Without Skill'], color='#CCCCDD', fontsize=10)
    ax.set_yticks(range(len(heat_df)))
    ax.set_yticklabels(heat_df['skill'], color='#CCCCDD', fontsize=8)

    for i in range(len(heat_df)):
        for j in range(2):
            val = mat[i, j]
            ax.text(j, i, f'${val:,.0f}', ha='center', va='center',
                    color='black', fontsize=7, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Avg Salary')
    ax.set_title('Skill–Salary Correlation', color='#EBEBFF', fontsize=12, pad=12, fontweight='bold')
    fig.tight_layout()
    return _fig_to_b64(fig)


def trending_skills_line(trend_df: pd.DataFrame, top_skills: int = 6) -> str:
    if trend_df.empty or trend_df['year_month'].nunique() < 2:
        return _placeholder('Not enough time-series data for trend chart')

    top = (trend_df.groupby('skill')['count'].sum()
           .sort_values(ascending=False).head(top_skills).index.tolist())
    plot = trend_df[trend_df['skill'].isin(top)]

    fig, bg = _base_fig(11, 5)
    ax = fig.add_subplot(111)
    _style_ax(ax, bg)

    months = sorted(trend_df['year_month'].unique())
    for i, sk in enumerate(top):
        sub = plot[plot['skill'] == sk].set_index('year_month').reindex(months, fill_value=0)
        ax.plot(range(len(months)), sub['count'].values,
                color=PALETTE[i], linewidth=2, marker='o', markersize=4, label=sk)

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Month', color='#CCCCDD')
    ax.set_ylabel('Frequency', color='#CCCCDD')
    ax.set_title('Trending Skills Over Time', color='#EBEBFF', fontsize=13, pad=12, fontweight='bold')
    ax.legend(facecolor='#1C1C2E', edgecolor='#3A3A5C', labelcolor='#CCCCDD', fontsize=8)
    ax.grid(color='#2A2A4A', linestyle='--', alpha=0.5)
    fig.tight_layout()
    return _fig_to_b64(fig)


def company_hiring_chart(df: pd.DataFrame, top_n: int = 15) -> str:
    counts = (df['company'].value_counts()
              .drop('Unknown', errors='ignore')
              .head(top_n))
    if counts.empty:
        return _placeholder('No company data available')

    fig, bg = _base_fig(10, 6)
    ax = fig.add_subplot(111)
    _style_ax(ax, bg)

    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color=SECONDARY, height=0.65, edgecolor='none')
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + counts.max()*0.01, bar.get_y() + bar.get_height()/2,
                str(val), va='center', color='#CCCCDD', fontsize=8)

    ax.set_xlabel('Number of Job Postings', color='#CCCCDD')
    ax.set_title('Top Hiring Companies', color='#EBEBFF', fontsize=13, pad=12, fontweight='bold')
    ax.grid(axis='x', color='#2A2A4A', linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _fig_to_b64(fig)


def cluster_scatter(clustered_df: pd.DataFrame) -> str:
    fig, bg = _base_fig(9, 6)
    ax = fig.add_subplot(111)
    _style_ax(ax, bg)

    n_clust = clustered_df['cluster'].nunique()
    colors  = PALETTE[:n_clust]

    for cid in range(n_clust):
        sub = clustered_df[clustered_df['cluster'] == cid]
        ax.scatter(sub['pca_x'], sub['pca_y'],
                   c=colors[cid], s=30, alpha=0.7, label=f'Cluster {cid}')

    ax.set_xlabel('PCA Component 1', color='#CCCCDD')
    ax.set_ylabel('PCA Component 2', color='#CCCCDD')
    ax.set_title('Job Role Clusters (PCA projection)', color='#EBEBFF',
                 fontsize=12, pad=12, fontweight='bold')
    ax.legend(facecolor='#1C1C2E', edgecolor='#3A3A5C', labelcolor='#CCCCDD',
              fontsize=8, ncol=2)
    ax.grid(color='#2A2A4A', linestyle='--', alpha=0.4)
    fig.tight_layout()
    return _fig_to_b64(fig)


def skill_gap_radar(gap_result: dict) -> str:
    required = gap_result['required_skills'][:12]
    matched  = set(gap_result['matched_skills'])

    angles = np.linspace(0, 2*np.pi, len(required), endpoint=False).tolist()
    angles += angles[:1]

    user_vals   = [1 if r in matched else 0 for r in required] + [1 if required[0] in matched else 0]
    target_vals = [1] * (len(required) + 1)

    fig, bg = _base_fig(7, 7)
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(bg)
    ax.spines['polar'].set_color('#3A3A5C')
    ax.set_thetagrids(np.degrees(angles[:-1]), labels=required,
                      color='#CCCCDD', fontsize=7)
    ax.set_ylim(0, 1.2)
    ax.yaxis.set_visible(False)
    ax.grid(color='#3A3A5C', linestyle='--', alpha=0.5)

    ax.plot(angles, target_vals, color=SECONDARY, linewidth=1.5, linestyle='--', alpha=0.6)
    ax.fill(angles, target_vals, color=SECONDARY, alpha=0.08)

    ax.plot(angles, user_vals, color=PRIMARY, linewidth=2)
    ax.fill(angles, user_vals, color=PRIMARY, alpha=0.35)

    ax.set_title(f'Skill Gap — {gap_result["career_goal"].title()}',
                 color='#EBEBFF', fontsize=11, pad=20, fontweight='bold')

    patches = [mpatches.Patch(color=PRIMARY, label='Your Skills'),
               mpatches.Patch(color=SECONDARY, label='Required Skills')]
    ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.3, 1.1),
              facecolor='#1C1C2E', edgecolor='#3A3A5C', labelcolor='#CCCCDD', fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def learning_roadmap_gantt(phases: list) -> str:
    if not phases:
        return _placeholder('No learning phases to display')

    colors_by_diff = {1: SECONDARY, 2: WARM, 3: PRIMARY, 4: ACCENT}
    diff_labels    = {1: 'Beginner', 2: 'Intermediate', 3: 'Advanced', 4: 'Expert'}

    fig, bg = _base_fig(12, max(5, len(phases) * 0.55))
    ax = fig.add_subplot(111)
    _style_ax(ax, bg)

    for i, phase in enumerate(phases):
        d   = phase['difficulty']
        col = colors_by_diff.get(d, PRIMARY)
        dur = phase['end_week'] - phase['start_week'] + 1
        ax.barh(i, dur, left=phase['start_week']-1, color=col,
                height=0.55, edgecolor=bg, linewidth=1)
        ax.text(phase['start_week'] - 0.5 + dur/2, i,
                phase['skill'], va='center', ha='center',
                color='white', fontsize=7.5, fontweight='bold')

    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels([f"Phase {p['phase']}" for p in phases], fontsize=8)
    ax.set_xlabel('Weeks', color='#CCCCDD')
    ax.set_title('Personalised Learning Roadmap', color='#EBEBFF', fontsize=12, pad=12, fontweight='bold')
    ax.grid(axis='x', color='#2A2A4A', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    patches = [mpatches.Patch(color=v, label=diff_labels[k]) for k, v in colors_by_diff.items()]
    ax.legend(handles=patches, loc='lower right', facecolor='#1C1C2E',
              edgecolor='#3A3A5C', labelcolor='#CCCCDD', fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def feature_importance_chart(fi_df: pd.DataFrame) -> str:
    if fi_df.empty:
        return _placeholder('Train a model first to see feature importance')
    top = fi_df.head(15)

    fig, bg = _base_fig(9, 5)
    ax = fig.add_subplot(111)
    _style_ax(ax, bg)

    cols = [SECONDARY if 'skill' in c else WARM for c in top['feature']]
    ax.barh(top['feature'][::-1], top['importance'][::-1],
            color=cols[::-1], height=0.65, edgecolor='none')
    ax.set_xlabel('Importance Score', color='#CCCCDD')
    ax.set_title('Model Feature Importance', color='#EBEBFF', fontsize=12, pad=12, fontweight='bold')
    ax.grid(axis='x', color='#2A2A4A', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _fig_to_b64(fig)


def salary_prediction_chart(predictions: dict) -> str:
    names  = list(predictions.keys())
    values = list(predictions.values())

    fig, bg = _base_fig(7, 4)
    ax = fig.add_subplot(111)
    _style_ax(ax, bg)

    bars = ax.bar(names, values, color=PALETTE[:len(names)], width=0.5, edgecolor='none')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'${val:,.0f}', ha='center', va='bottom', color='#EBEBFF', fontsize=9)

    ax.set_ylabel('Predicted Annual Salary (USD)', color='#CCCCDD')
    ax.set_title('Salary Predictions by Model', color='#EBEBFF', fontsize=12, pad=12, fontweight='bold')
    ax.grid(axis='y', color='#2A2A4A', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _fig_to_b64(fig)


def experience_distribution(df: pd.DataFrame) -> str:
    exp_order = ['entry','junior','mid','senior','principal','manager','director']
    counts = df['experience_level'].value_counts()
    counts = counts[counts.index.isin(exp_order)].reindex(
        [e for e in exp_order if e in counts.index])
    if counts.empty:
        return _placeholder('No experience level data')

    fig, bg = _base_fig(8, 4)
    ax = fig.add_subplot(111)
    _style_ax(ax, bg)

    bars = ax.bar(counts.index, counts.values,
                  color=PALETTE[:len(counts)], width=0.6, edgecolor='none')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + counts.max()*0.01,
                str(val), ha='center', va='bottom', color='#CCCCDD', fontsize=8)

    ax.set_xlabel('Experience Level', color='#CCCCDD')
    ax.set_ylabel('Job Count', color='#CCCCDD')
    ax.set_title('Jobs by Experience Level', color='#EBEBFF', fontsize=12, pad=12, fontweight='bold')
    ax.grid(axis='y', color='#2A2A4A', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _placeholder(msg: str) -> str:
    fig, bg = _base_fig(6, 3)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg)
    ax.text(0.5, 0.5, msg, ha='center', va='center', transform=ax.transAxes,
            color='#8888AA', fontsize=11)
    ax.axis('off')
    return _fig_to_b64(fig)
