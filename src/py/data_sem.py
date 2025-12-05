"""生成用于结构方程模型(SEM)分析的合成心理学数据集。

功能：
- 生成多个潜变量（例如五大人格 + 压力/社会支持/自尊/幸福感）
- 每个潜变量由多个观测题项（Likert 1-7）测量
- 包含一些协变量：`age`, `gender`, `ses`
- 结果变量（抑郁、焦虑、幸福感）由潜变量按指定路径生成
- 引入测量误差、潜变量相关性、部分非正态性
- 引入 MCAR 和 MAR 缺失值机制
- 将结果保存为 `data/processed/sem_synthetic.csv`

用法：
	python src/py/data_sem.py --n 1000 --seed 123
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
from typing import Iterable


def _to_likert(x: np.ndarray, minv: int = 1, maxv: int = 7) -> np.ndarray:
	"""把连续分数映射到 Likert 1..7（尽量保留顺序和一些偏态）。"""
	# 使用分位数映射以保留分布形状
	ranks = pd.qcut(x, q=maxv, labels=False, duplicates='drop')
	# qcut 有时会返回少于 q 个箱（当重复值多时），用 rank fallback
	if (ranks.isna()).any():
		ranks = x.rank(method='average')
		ranks = pd.cut(ranks, bins=maxv, labels=False)
	return (ranks.astype(int) + minv).to_numpy()


def generate_sem_data(n: int = 1000, seed: int = 123, save_path: str = 'data/processed/sem_synthetic.csv') -> pd.DataFrame:
	rng = np.random.default_rng(seed)

	# --- 协变量 ---
	age = rng.normal(35, 12, size=n).round(1)
	age = np.clip(age, 18, 90)
	gender = rng.binomial(1, 0.53, size=n)  # 0 = female, 1 = male
	ses_cont = rng.normal(0, 1, size=n)  # 连续 SES 指标
	ses_cat = pd.cut(ses_cont, bins=[-np.inf, -0.7, 0.7, np.inf], labels=[1, 2, 3]).astype(int)

	# --- 潜变量定义（9 个潜变量）---
	latent_names = [
		'Extraversion', 'Neuroticism', 'Conscientiousness', 'Openness', 'Agreeableness',
		'SocialSupport', 'Stress', 'SelfEsteem', 'WellBeing'
	]
	m = len(latent_names)

	# 构建潜变量协方差矩阵（设定一些相关性）
	base_cov = np.eye(m)
	# 让人格维度互相关不高
	for i in range(5):
		for j in range(5):
			if i != j:
				base_cov[i, j] = 0.25
	# Stress 与 Neuroticism 相关，高负相关 SelfEsteem
	li = latent_names.index
	base_cov[li('Neuroticism'), li('Stress')] = 0.6
	base_cov[li('Stress'), li('Neuroticism')] = 0.6
	base_cov[li('SelfEsteem'), li('Neuroticism')] = -0.5
	base_cov[li('WellBeing'), li('SelfEsteem')] = 0.6
	base_cov[li('SocialSupport'), li('WellBeing')] = 0.45

	# 确保协方差矩阵对称并正定（轻微调整以避免数值问题）
	base_cov = (base_cov + base_cov.T) / 2.0
	eigs = np.linalg.eigvalsh(base_cov)
	if eigs.min() <= 0:
		base_cov += np.eye(m) * (abs(eigs.min()) + 1e-4)

	# 潜变量均值受协变量影响（小效应）
	mean_effects = np.zeros((m,))
	# 例如：年龄对自尊和幸福感有小负效应
	mean_effects[li('SelfEsteem')] = -0.01
	mean_effects[li('WellBeing')] = -0.005

	# 为每个被试生成潜变量值
	# 先生成多元正态，然后对部分潜变量做非线性变换以产生轻微非正态性
	z = rng.multivariate_normal(mean=np.zeros(m), cov=base_cov, size=n)
	latents = z + (np.outer(age - age.mean(), mean_effects) * 0.02) + np.outer(ses_cont, 0.05 * np.ones(m))
	# 对一两个潜变量引入非线性（轻微偏态）
	latents[:, li('Stress')] = np.exp(latents[:, li('Stress')] / 2.5)
	latents[:, li('Neuroticism')] = np.tanh(latents[:, li('Neuroticism')]) * 1.2

	# 标准化潜变量（便于设置载荷）
	latents = (latents - latents.mean(axis=0)) / latents.std(axis=0)

	rows: dict[str, Iterable] = {
		'id': np.arange(1, n + 1),
		'age': age,
		'gender': gender,
		'ses_cat': ses_cat,
	}

	# --- 为每个潜变量生成观测题项（4 个题项/潜变量）---
	rng_float = rng
	for k, lname in enumerate(latent_names):
		for item in range(1, 5):
			col = f'{lname[:3].lower()}_it{item}'
			# 随机载荷在 0.6-0.9 之间
			loading = rng_float.uniform(0.6, 0.9)
			noise_sd = np.sqrt(1 - loading ** 2) * 0.9
			cont_scores = loading * latents[:, k] + rng_float.normal(0, noise_sd, size=n)
			# 将部分题目反向编码（增加现实感）
			if (item % 4) == 0 and (k % 2 == 1):
				cont_scores = -cont_scores
			likert = _to_likert(pd.Series(cont_scores), 1, 7)
			rows[col] = likert

	# --- 额外观测量表（例如认知测验，反应时间）---
	rows['cog_score'] = (latents[:, li('Openness')] * 5 + rng.normal(0, 1, n)).round(2)
	rows['rt_mean'] = np.clip(np.exp(3 - 0.3 * latents[:, li('Conscientiousness')] + rng.normal(0, 0.3, n)), 0.2, 20).round(3)

	# --- 结果变量（按结构路径从潜变量生成）---
	# 抑郁（Dep）由 Neuroticism、Stress、SocialSupport 决定
	dep = (
		0.6 * latents[:, li('Neuroticism')]
		+ 0.5 * latents[:, li('Stress')]
		- 0.4 * latents[:, li('SocialSupport')]
		+ rng.normal(0, 0.8, n)
	)
	# 焦虑
	anx = 0.5 * latents[:, li('Neuroticism')] + 0.35 * latents[:, li('Stress')] + rng.normal(0, 0.7, n)
	# 幸福感（WellBeing）由自尊、社会支持、神经质负向影响
	wb = 0.7 * latents[:, li('SelfEsteem')] + 0.35 * latents[:, li('SocialSupport')] - 0.3 * latents[:, li('Neuroticism')] + rng.normal(0, 0.6, n)

	rows['depression_cont'] = np.round(dep, 3)
	rows['anxiety_cont'] = np.round(anx, 3)
	rows['wellbeing_cont'] = np.round(wb, 3)

	df = pd.DataFrame(rows)

	# --- 引入缺失值 ---
	# MCAR: 每个观测题目随机缺失 2%-8%
	item_cols = [c for c in df.columns if c.endswith(tuple(f'it{i}' for i in range(1, 5)))]
	for c in item_cols:
		p = float(rng_float.uniform(0.02, 0.08))
		mask = rng_float.choice([True, False], size=n, p=[p, 1 - p])
		df.loc[mask, c] = np.nan

	# MAR: 年龄较大者在自尊题目上缺失更高
	se_items = [c for c in df.columns if c.startswith('sel') and c.endswith(tuple(f'it{i}' for i in range(1, 5)))]
	if not se_items:
		# 如果没有 sel_ 前缀（安全回退），使用 any selfesteem items by name
		se_items = [c for c in df.columns if 'self' in c.lower() or 'self' in c]
	for c in se_items:
		p_base = 0.03
		# 年龄 > 65 增加缺失概率
		mask_old = df['age'] > 65
		miss_mask = rng_float.random(size=n) < (p_base + 0.15 * mask_old.astype(float))
		df.loc[miss_mask, c] = np.nan

	# 让结果变量有少量缺失（例如未完成问卷）
	for c in ['depression_cont', 'anxiety_cont']:
		p = 0.03
		df.loc[rng_float.random(size=n) < p, c] = np.nan

	# --- 保存文件 ---
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	df.to_csv(save_path, index=False)

	return df


def _parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description='Generate synthetic SEM data')
	p.add_argument('--n', type=int, default=1000)
	p.add_argument('--seed', type=int, default=123)
	p.add_argument('--out', type=str, default='data/processed/sem_synthetic.csv')
	return p.parse_args()


if __name__ == '__main__':
	args = _parse_args()
	df = generate_sem_data(n=args.n, seed=args.seed, save_path=args.out)
	print(f'Generated {len(df)} rows and saved to: {args.out}')

