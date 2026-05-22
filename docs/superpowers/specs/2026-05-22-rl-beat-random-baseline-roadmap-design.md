# AlphaGen 开发路线:让 RL 跑赢随机基线

- 日期:2026-05-22
- 状态:设计已确认,待写实现计划
- 来源:仓库根目录 `TODO.md` 的 4 项待办

## 背景

2026-05-21 的 `lr=0` 实验确认:在 20k 步预算下,RL 相对随机搜索没有可复现增量——所有 run
的最佳 checkpoint(按 `valid_rank_icir` 选)都落在 2048–6144 步,即策略尚未有效更新时。
`lr=3e-6` 的 run 与 `lr=0` 逐字节相同,实际上是一次意外的随机基线;
`notebooks/runs/aggregate/` 里 test rank_ICIR ≈ 0.22 的汇总其实就是那个随机基线。

本路线围绕 `TODO.md` 的 4 项展开,目标是让 RL alpha 生成器**可复现地跑赢随机基线**。

## 目标与非目标

**目标**
- 训练能跑满更大预算,并在 valid 指标停滞时自动早停。
- 一条命令完成超参数网格 × 多 seed 的批量实验,并产出跨配置对比表。
- 去重阈值与复杂度惩罚成为可配置项,可被 sweep 扫描。

**成功判定(方向性提升)**
同预算下,跨 seed 的 test rank_ICIR 均值稳定为正增量(高于随机基线均值)即算成功;
重点在方向而非幅度。随机基线参考:test rank_ICIR ≈ 0.22。

**非目标**
- 不重写 PPO / 环境 / pool 的核心算法。
- 不引入 LR 进一步细扫为独立目标(预算与 reward 设计才是主杠杆)。
- 不做与本路线无关的重构。

## 依赖分析与阶段划分

`TODO.md` 的 4 项中,#1(早停 + 大预算)与 #2(sweep runner)是**实验基础设施**;
#3(去重阈值)与 #4(复杂度惩罚)代码改动很小,但其效果**只有靠 #1+#2 的预算与批量对比
才能验证**,否则会像 lr=0 那样被 seed 噪声淹没。因此采用「基础设施优先」的三阶段顺序:

| 阶段 | 内容 | TODO 项 | 交付物 |
|------|------|---------|--------|
| 1 | 早停机制 + 增大训练步数 | #1 | `CustomCallback` 内的早停状态机 |
| 2 | Sweep runner | #2 | `scripts/sweep.py` + 评估器重构 + 对比表 |
| 3 | 去重阈值 + 复杂度惩罚,作为可配置项一起扫 | #3, #4 | 两个 config 开关 + sweep 结果 |

**跨阶段约定**:所有新参数默认值保留当前行为(`complexity_penalty=0`、
`ic_mut_threshold=0.99`、`early_stop_patience` 可设 0 关闭),且全部写入
`run_config.json`,供 sweep 与评估器读取。向后兼容,不惊扰现有流程。

---

## 阶段 1:早停机制 + 增大训练步数(TODO #1)

**改动位置**:`scripts/rl.py` 的 `CustomCallback`,纯内部状态机,不新增文件。

### 早停状态机

监控指标:`valid` split 的 `rank_icir`,从已有的 `_compute_split_metrics()` 取(无新计算成本)。

每个 `_on_rollout_end`:
1. 计算 `valid_rank_icir`,与全程 `best` 比较。
2. 提升幅度 `> min_delta` → 刷新 `best`、`no_improve_rollouts = 0`;否则 `no_improve_rollouts += 1`。
3. 若 `num_timesteps >= warmup_steps` 且 `no_improve_rollouts >= patience` → 置 `self._should_stop = True`。

`_on_step` 返回 `not self._should_stop`。PPO 在回调返回 `False` 时停止采集,并仍会走
`_on_training_end`(强制存最后一个 checkpoint)。

### 关键语义

- **`best` 全程记录**,warmup 期的指标也参与更新 `best`。
- **`no_improve` 计数器从头累加**,warmup 期间也累加。
- **warmup 只抑制「触发停止」这一个动作**,不抑制记录与计数。
- 推论:若 `best` 落在 warmup 期(如 lr=0 实验里 rollout 2 的运气峰值),warmup 一结束且
  `no_improve >= patience` 时会立即停止。这是有意为之——等价于「策略刷不过自己的随机初始
  池就停」,正贴合「检验 RL 是否真有增量」的研究目的。
- 早停判定只影响**是否停止训练**;checkpoint 的最终择优仍由离线评估器
  `evaluate_local_runs.py` 跨全部 checkpoint 选,不受早停影响。

### 新增 CLI 参数

`main` / `local` / `local_smoke` / `colab` 全部透传:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `early_stop_patience` | `3` | 连续无提升的 rollout 数;设 `0` 关闭早停 |
| `early_stop_warmup_steps` | `20_000` | warmup 上限;期间不触发停止 |
| `early_stop_min_delta` | `1e-3` | 视为「提升」的最小幅度 |

监控的 split/metric 固定为 `valid` + `rank_icir`(与 `checkpoint_selection_metric` 一致),
不做成参数,避免过度设计。

`early_stop_warmup_steps = 20_000` 在 Colab `ppo_n_steps=2048` 下约为 10 个 rollout,
覆盖 lr=0 实验中 rollout 1–3 的运气峰值区。

### 预算

`steps` 作为上限设大——取 `5e5`(落在 `TODO.md` 建议的 `1e5`–`1e6` 区间内,
当前 Colab 默认约 `2.5e5`)。它是早停的天花板而非目标值;早停优先于跑满。
具体写法由实现计划决定(调 `DEFAULT_STEPS` 表,或靠 `--steps` 覆盖)。

### 可观测性

- `status.json` 增加 `early_stop` 字段:`best` 值、`best_step`、`no_improve_rollouts`、
  `stopped`(bool)、`stop_reason`。
- `monitor.log` 在触发早停时写一行。

### 验收

- valid 仍在升时能持续训练;停滞后自动停。
- 最佳 checkpoint 不再总落在前 3 个 rollout。

---

## 阶段 2:Sweep runner(TODO #2)

采用**方案 A:进程内 Python sweep**——复用 `run_single_experiment` 与现有评估器,
单进程顺序执行,断点续跑机制匹配 Colab(12h 会话、易断线)的运行环境。

### 组件 1:`scripts/sweep.py`(新文件)

- Sweep 配置 = 一个 dict:
  - `grid`:参数名 → 候选值列表;
  - `seeds`:seed 列表;
  - `base`:profile 名 + 固定 overrides。
- 笛卡尔积展开 `grid × seeds` → 一串运行配置。总 run 数 =
  `(∏ 每个 grid 参数候选值个数) × seed 个数`;单值 grid 参数贡献因子 1,不增加 run 数。
- 输出布局:`out/sweeps/<sweep_name>/<config_slug>/`。`config_slug` 由所有被扫参数
  (含 seed)拼成可读字符串 + 短 hash。
- **断点续跑**:每个配置启动前检查 `<config_slug>/status.json` 是否已是
  `event == "training_end"`,是则跳过。Colab 断线后重跑同一命令即可接着跑。
- 单进程顺序执行。

**`grid` 与 `base` 的语义区分**:固定某参数时,放进 `base` 与放进单值 `grid` 列表效果
等价,但单值 `grid` 项会作为一列出现在对比表里(整列同值)。约定:真正要对比的维度放
`grid`,纯固定值放 `base`,使对比表更干净。

### 组件 2:`run_single_experiment` 新增两个参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `output_dir` | `None` | 给定时直接用作 `save_path`,不再走 `./out/results/<name_prefix>` |
| `save_model_checkpoints` | `True` | 透传给 `CustomCallback` |

### 组件 3:`CustomCallback` 新增 `save_model_checkpoints` 开关

为 `False` 时 `save_checkpoint` 跳过 `self.model.save(path)`(即 `*_steps.zip`),
保留 pool json、TensorBoard event、`status.json`、`monitor.log`、`run_config.json`。
这是 sweep 省空间的核心。

### 组件 4:`evaluate_local_runs.py` 重构

现状把 US runtime(`DELTA_TIMES`、features)硬编码在 `_patch_us_runtime`。
抽出可导入函数 `evaluate_runs(runs_dir, qlib_data, region, features)`,
`region` / `features` 变为参数。sweep 跑完后直接调用它,对 `out/sweeps/<name>/` 下每个
run 产出 `checkpoint_metrics.csv` / `best_segment_metrics.csv`。
`main()` CLI 入口保留,内部改为调用 `evaluate_runs`。

### 组件 5:跨配置对比表

`sweep.py` 读每个 run 的 `run_config.json`(含全部被扫参数)+ `best_segment_metrics.csv`,
透视成 `out/sweeps/<sweep_name>/sweep_comparison.csv`:一行一个配置,列出各被扫参数 +
跨 seed 的 test/valid rank_ICIR 均值与 std,按 test rank_ICIR 均值排序。

### 数据流

```
sweep config (grid × seeds)
  → 展开为运行配置列表
  → for each config (跳过已完成的):
        run_single_experiment(output_dir=..., save_model_checkpoints=False, ...)
  → evaluate_runs(out/sweeps/<name>, ...) 产出 per-run CSV
  → 透视 run_config.json + best_segment_metrics.csv → sweep_comparison.csv
```

### 验收

- 一条命令跑完整个网格。
- sweep run 目录无 `.zip`、体积显著下降。
- `sweep_comparison.csv` 可直接读出最优配置。
- Colab 断线重跑可续。

---

## 阶段 3:去重阈值 + 复杂度惩罚(TODO #3 + #4)

两项落地为可配置开关后,用阶段 2 的 sweep 一起扫。

### #3 — `ic_mut_threshold` 可配置

- `LinearAlphaPool.__init__` 新增 `ic_mut_threshold: float = 0.99`,存 `self._ic_mut_threshold`。
- `try_new_expr` 用 `self._ic_mut_threshold` 替换 `linear_alpha_pool.py:62` 的硬编码 `0.99`。
- `MseAlphaPool` 与 `MeanStdAlphaPool` 构造函数透传该参数。
- `run_single_experiment.build_pool` 传入;新增 CLI 参数 `ic_mut_threshold: float = 0.99`。
- `force_load_exprs` 里的 `_calc_ics(..., ic_mut_threshold=None)` 保持不变(force load 不去重)。
- 默认 `0.99` 保留现行为;sweep 时扫 `{0.7, 0.75, 0.8, 0.99}`。

### #4 — 复杂度惩罚

- **度量:token 数**。`AlphaEnvCore.step` 中 `self._tokens` 现成可用,无需遍历树;
  按 `MAX_EXPR_LENGTH` 归一化,惩罚落在 `[0, coef]`。(树深度为备选方案,但需额外遍历树,
  v1 取 token 数更简洁。)
- 公式:`reward = pool_reward - complexity_penalty * (len(tokens) / MAX_EXPR_LENGTH)`。
- 施加点:`AlphaEnvCore.step` 中由 `_evaluate()` 得到 reward 的两个分支(SEP 结束、
  达 `MAX_EXPR_LENGTH` 且 `is_valid()`);非法表达式的 `-1` 分支不加惩罚(已是最差)。
- 新增参数 `complexity_penalty: float = 0.0`(默认 0 = 关闭,保留现行为),
  沿 `run_single_experiment → AlphaEnv → AlphaEnvCore` 透传。
- 量纲:pool objective 量级约 0–0.5,归一化惩罚 ∈ [0, coef];sweep 扫
  `coef ∈ {0, 0.01, 0.02, 0.05}`。

### 扫法

#3、#4 落地后,用阶段 2 的 sweep 把 `ic_mut_threshold` 与 `complexity_penalty` 一起扫,
`sweep_comparison.csv` 直接对比。

### 验收

- 池内因子两两相关性下降。
- 池内平均表达式复杂度下降。
- train/test IC 的过拟合差距(当前约 9x)收窄。
- 样本外(valid/test)rank_ICIR 在同等预算下不劣于现状。

---

## 测试策略

- **阶段 1**:用 `local_smoke` profile(步数极小)验证早停状态机——构造一次注定停滞的
  run,确认 warmup 内不停、warmup 后按 `patience` 停;确认 `patience=0` 时完全不早停。
- **阶段 2**:用 `local_smoke` 跑一个 2×2 的小 grid,确认:run 数符合公式、断点续跑能
  跳过已完成项、`save_model_checkpoints=False` 时无 `.zip`、`sweep_comparison.csv` 行列正确。
- **阶段 3**:单元层面确认 `ic_mut_threshold` 与 `complexity_penalty` 被正确读取并影响
  pool / reward;端到端用小 grid 确认两个开关进入 `run_config.json` 与对比表。
- 所有阶段:确认新参数默认值下,行为与改动前一致(向后兼容回归)。

## 受影响文件清单

| 文件 | 改动 |
|------|------|
| `scripts/rl.py` | `CustomCallback` 早停状态机;`run_single_experiment` 与 4 个 CLI 入口新增参数;`save_model_checkpoints` / `output_dir` 透传 |
| `scripts/sweep.py` | 新文件:网格展开、断点续跑、对比表 |
| `scripts/evaluate_local_runs.py` | 抽出可导入的 `evaluate_runs(...)`,region/features 参数化 |
| `alphagen/models/linear_alpha_pool.py` | `LinearAlphaPool` / `MseAlphaPool` / `MeanStdAlphaPool` 新增 `ic_mut_threshold` 参数 |
| `alphagen/rl/env/core.py` | `AlphaEnvCore` 新增 `complexity_penalty`,在 `step` 施加惩罚 |
| `alphagen/rl/env/wrapper.py` | `AlphaEnv` 透传 `complexity_penalty` |
