import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import pi
from sklearn.metrics import silhouette_score
from scipy.stats import zscore


os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# 1. Load and prepare dataset
df = pd.read_csv("Linhac24-25_Sportlogiq.csv")
df = df.sort_values(by=['gameid', 'compiledgametime'])

# ========== 2. Event flags with renamed variables ==========
df['num_passes'] = (df['eventname'] == 'pass').astype(int)
df['num_zone_entries'] = (df['eventname'] == 'controlledentry').astype(int)
df['num_assists'] = (df['eventname'] == 'assist').astype(int)
df['num_blocks'] = (df['eventname'] == 'block').astype(int)
df['num_entries_against'] = (df['eventname'] == 'controlledentryagainst').astype(int)
df['num_dumpouts'] = (df['eventname'] == 'dumpout').astype(int)
df['num_shots'] = df['eventname'].isin(['shot', 'goal']).astype(int)

# ========== 3. Success flags ==========
df['num_successful_passes'] = ((df['eventname'] == 'pass') & (df['outcome'] == 'successful')).astype(int)
df['num_successful_zone_entries'] = ((df['eventname'] == 'controlledentry') & (df['outcome'] == 'successful')).astype(int)
df['num_successful_entries_against'] = ((df['eventname'] == 'controlledentryagainst') & (df['outcome'] == 'successful')).astype(int)

# ========== 4. Carry duration & distance ==========
df['is_carry'] = (df['eventname'] == 'carry').astype(int)
carry_df = df[df['is_carry'] == 1].copy()
carry_df['next_x'] = df['xadjcoord'].shift(-1)
carry_df['next_y'] = df['yadjcoord'].shift(-1)
carry_df['carry_distance'] = np.sqrt((carry_df['next_x'] - carry_df['xadjcoord'])**2 + (carry_df['next_y'] - carry_df['yadjcoord'])**2)
carry_df['carry_duration'] = df['compiledgametime'].shift(-1) - carry_df['compiledgametime']
carry_avg = carry_df.groupby(['teamid', 'gameid'])[['carry_distance', 'carry_duration']].mean().reset_index()
carry_avg.rename(columns={'carry_distance': 'avg_carry_distance', 'carry_duration': 'avg_carry_duration'}, inplace=True)

# ========== 5. Aggregated metrics ==========
agg_cols = [
    'num_passes', 'num_successful_passes', 'num_zone_entries', 'num_successful_zone_entries',
    'num_assists', 'num_blocks', 'num_entries_against', 'num_dumpouts', 'num_shots'
]
agg_dict = {col: 'sum' for col in agg_cols}
agg_dict['xg_allattempts'] = 'mean'
agg_df = df.groupby(['teamid', 'gameid']).agg(agg_dict).reset_index()
agg_df = agg_df.merge(carry_avg, on=['teamid', 'gameid'], how='left')

# ========== 6. Final score + result ==========
last_score = df.sort_values(['gameid', 'compiledgametime']).groupby(['teamid', 'gameid'])['scoredifferential'].last().reset_index()
agg_df = agg_df.drop(columns='scoredifferential', errors='ignore')
agg_df = agg_df.merge(last_score, on=['teamid', 'gameid'], how='left')
agg_df['result'] = agg_df['scoredifferential'].apply(lambda x: 1 if x > 0 else (0 if x < 0 else 0.5))

# ========== 7. Derived success rates ==========
agg_df['pass_success_rate'] = agg_df['num_successful_passes'] / agg_df['num_passes'].replace(0, np.nan)
agg_df['entry_success_rate'] = agg_df['num_successful_zone_entries'] / agg_df['num_zone_entries'].replace(0, np.nan)

entry_against_stats = df.groupby(['teamid', 'gameid'])[['num_successful_entries_against', 'num_entries_against']].sum().reset_index()
entry_against_stats['entry_against_success_rate'] = entry_against_stats['num_successful_entries_against'] / entry_against_stats['num_entries_against'].replace(0, np.nan)
agg_df = agg_df.merge(entry_against_stats[['teamid', 'gameid', 'entry_against_success_rate']], on=['teamid', 'gameid'], how='left')

# ========== 8. Feature list for clustering ==========
features_v2 = [
    'num_passes', 'num_zone_entries', 'num_assists', 'num_blocks',
    'num_entries_against', 'num_dumpouts', 'num_shots', 'xg_allattempts',
    'pass_success_rate', 'entry_success_rate', 'entry_against_success_rate',
    'avg_carry_duration', 'avg_carry_distance'
]
X = StandardScaler().fit_transform(agg_df[features_v2])



# 8. Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
agg_df['cluster'] = kmeans.fit_predict(X)
style_names = {0: 'Puck Control Play', 1: 'Defensive Counterattack', 2: 'High-Pressure Offense'}
agg_df['style'] = agg_df['cluster'].map(style_names)

# 9. Radar chart with clipped normalization
style_features = agg_df.groupby('style')[features_v2].mean()
style_features_norm = (style_features - style_features.min()) / (style_features.max() - style_features.min())
style_features_norm = style_features_norm.clip(lower=0.05)

labels = style_features.columns.tolist()
angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
angles += angles[:1]

plt.figure(figsize=(8, 6))
colors = sns.color_palette("Set2", n_colors=3)

for idx, (style, row) in enumerate(style_features_norm.iterrows()):
    values = row.tolist()
    values += values[:1]
    plt.polar(angles, values, label=style, color=colors[idx])
    plt.fill(angles, values, alpha=0.1, color=colors[idx])

plt.xticks(angles[:-1], labels, fontsize=8)
plt.title("Team Style Feature Radar Chart (Clipped Normalized)", size=14)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()



# 11. Pairwise style win rate matrix
df_matches = agg_df[['gameid', 'teamid', 'style', 'result']].copy()
df_matches = df_matches.merge(df_matches, on='gameid')
df_matches = df_matches[df_matches['teamid_x'] < df_matches['teamid_y']].copy()

records = []
for _, row in df_matches.iterrows():
    s1, r1 = row['style_x'], row['result_x']
    s2, r2 = row['style_y'], row['result_y']
    if r1 == 1 and r2 == 0:
        records.append((s1, s2, 'win'))
        records.append((s2, s1, 'loss'))
    elif r2 == 1 and r1 == 0:
        records.append((s1, s2, 'loss'))
        records.append((s2, s1, 'win'))
    elif r1 == 0.5 and r2 == 0.5:
        records.append((s1, s2, 'draw'))
        records.append((s2, s1, 'draw'))

results_df = pd.DataFrame(records, columns=['style_team', 'style_opp', 'outcome'])
winrate_numeric = results_df.assign(win=results_df['outcome'].map({'win': 1, 'draw': 0.5, 'loss': 0}))
winrate_table = winrate_numeric.groupby(['style_team', 'style_opp'])['win'].mean().unstack().round(2)

# normalize winrate table
winrate_norm = winrate_table.sub(winrate_table.min(axis=1), axis=0)
winrate_norm = winrate_norm.div(winrate_table.max(axis=1) - winrate_table.min(axis=1), axis=0).fillna(0)

style_avg_winrate = winrate_table.mean(axis=1).round(2)

plt.figure(figsize=(8, 6))
sns.heatmap(winrate_norm, annot=winrate_table, cmap="YlOrBr", fmt=".2f", cbar_kws={"label": "Normalized Value"})
plt.title("Normalized Style Win Rate Matrix\n(Annotations = Actual Win Rate)")
plt.xlabel("Opponent Style")
plt.ylabel("Team Style")
plt.tight_layout()
plt.show()




