# 在博弈论方面做出的尝试的报告

## 背景
1. 将我们的match up的矩阵，也就是figure4.png抽象为pay-off matrix A

2. hybrid stategy vector

 对于一个队伍而言， 他的style组成可以用vector x表示: x = [freq1， freq2. freq3], freq表示采取一个风格进行比赛的频率。

 3. 预测胜率

现在有队伍x, y 我们预测胜率的公式为: predict = x.T @ y
即队伍x面对队伍y时的期望胜率。


## 现有的工作

### 1. 计算pay-off matrix

其实就是figure4.png的内容

```
def compute_payoff_matrix(agg_df, filename="figure4.png", plot_only=False):
    """
    Compute and visualize a win-rate matrix for matchups between different
    team styles.

    Args:
        agg_df (pd.DataFrame): Aggregated dataframe with game results and style labels.
    """
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

    if not plot_only:
        return winrate_table

    plt.figure(figsize=(8, 6))
    sns.heatmap(winrate_table, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Win Rate by Team Style vs Opponent Style")
    plt.xlabel("Opponent Style")
    plt.ylabel("Team Style")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

```

转换成矩阵形式

```
payoff = compute_payoff_matrix(agg_df)
style_names = {0: 'Puck Control Play', 1: 'Defensive Counterattack', 2: 'High-Pressure Offense'}

style_order = [style_names[i] for i in range(3)]
payoff = payoff.loc[style_order, style_order]

# 转换为矩阵（numpy array）
payoff_matrix = payoff.to_numpy()
payoff_matrix
```

结果：
array([[0.5 , 0.18, 0.14],
       [0.82, 0.5 , 0.55],
       [0.86, 0.45, 0.5 ]])


### 最佳风格求解

使用game theory求出在对手y期望队伍x的胜率最小的情况下，胜率最大的style vector

···import numpy as np
from scipy.optimize import linprog

# 定义 payoff 矩阵 A（我方的胜率）
A = payoff_matrix

# 为了使用 linprog，需要将 maximin 问题转换成线性规划的对偶形式：
# min v
# s.t. Ay <= v, sum(y) = 1, y >= 0

# 转换为标准线性规划形式：
# min c^T y
# s.t. A_ub y <= b_ub, A_eq y = b_eq, y >= 0

# 变量 y 有 3 个，对应对手的策略混合
num_strategies = A.shape[1]

# c 是目标函数系数：min v = sum(A[i] * y) → 对应 v 是最大最小值，但这里是对偶问题
c = np.ones(num_strategies)

# 原问题是 max min x^T A y → 对偶是 min max y^T A^T x
# 我们转换为标准形式：min z  s.t. A^T x <= z, sum(x) = 1, x >= 0
# 所以在对偶中，我们是最小化最大行的期望值

# 构造线性规划问题：
# 对偶形式是：
# min z
# s.t. A^T x <= z
# 即：我们定义 x = [x1, x2, x3, v]
# 目标函数为：min -v → 实际为 max v

# 在 primal 中构造如下：
# max v
# s.t. A[i] @ y >= v, ∑y = 1, y >= 0
# 转换为对偶形式，变为 min z, s.t. A^T x <= z

# 使用 linprog 求解玩家1的最优混合策略
# 注意要对 A 做转置
c = [0, 0, 0, -1]  # 我们最大化 v，等价于最小化 -v
A_ub = np.hstack([-A.T, np.ones((3, 1))])  # A.T @ x - v <= 0 → -A.T @ x + v >= 0
b_ub = np.zeros(3)
A_eq = [[1, 1, 1, 0]]  # sum(x) = 1
b_eq = [1]

# bounds for x1, x2, x3, v
bounds = [(0, 1)] * 3 + [(None, None)]

# 求解线性规划
res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

# 结果
res.success, res.fun, res.x
···

结果：
(True, -0.5, array([ 0. ,  1. , -0. ,  0.5]))
