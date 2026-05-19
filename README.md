# AlphaGen

<p align="center">
    <img src="images/logo.jpg" width=275 />
</p>

Automatic formulaic alpha generation with reinforcement learning.

This repository contains the code for our paper *Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning* accepted by [KDD 2023](https://kdd.org/kdd2023/), Applied Data Science (ADS) track, publically available on [ACM DL](https://dl.acm.org/doi/10.1145/3580305.3599831). Some extensions upon this work are also included in this repo.

## Repository Structure

- `/alphagen` contains the basic data structures and the essential modules for starting an alpha mining pipeline;
- `/alphagen_qlib` contains the qlib-specific APIs for data preparation;
- `/alphagen_generic` contains data structures and utils designed for our baselines, which basically follow [gplearn](https://github.com/trevorstephens/gplearn) APIs, but with modifications for quant pipeline;
- `/alphagen_llm` contains LLM client abstractions and a set of prompts useful for LLM-based alpha generation, and also provides some LLM-based automatic iterative alpha-generation routines.
- `/gplearn` and `/dso` contains modified versions of our baselines;
- `/scripts` contains several scripts for running the experiments.

## Quick Start

Create and activate a local virtual environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pytest
```

Run the local regression tests with:

```bash
python -m pytest -q
```

Notes on dependencies:

- `requirements.txt` targets the main AlphaGen runtime on a modern Python environment.
- The vendored `dso/` baseline still depends on an older TensorFlow-based stack declared separately in `dso/setup.py`.

## Result Reproduction

Note that you can either use our builtin alpha calculation pipeline (see Choice 1), or implement an adapter to your own pipeline (see Choice 2).

### Choice 1: Stock data preparation

Builtin pipeline requires Qlib library and local-storaged stock data.

- READ THIS! We need some of the metadata (but not the actual stock price/volume data) given by Qlib, so follow the data preparing process in [Qlib](https://github.com/microsoft/qlib#data-preparation) first.
- The actual stock data we use are retrieved from [baostock](http://baostock.com/baostock/index.php/%E9%A6%96%E9%A1%B5), due to concerns on the timeliness and truthfulness of the data source used by Qlib.
- The data can be downloaded by running the script `data_collection/fetch_baostock_data.py`. The newly downloaded data is saved into `~/.qlib/qlib_data/cn_data_baostock_fwdadj` by default. This path can be customized to fit your specific needs, but make sure to use the correct path when loading the data (In `alphagen_qlib/stock_data.py`, function `StockData._init_qlib`, the path should be passed to qlib with `qlib.init(provider_uri=path)`).

### Choice 2: Adapt to external pipelines

Maybe you have better implements of alpha calculation, you can implement an adapter of `alphagen.data.calculator.AlphaCalculator`. The interface is defined as follows:

```python
class AlphaCalculator(metaclass=ABCMeta):
    @abstractmethod
    def calc_single_IC_ret(self, expr: Expression) -> float:
        'Calculate IC between a single alpha and a predefined target.'

    @abstractmethod
    def calc_single_rIC_ret(self, expr: Expression) -> float:
        'Calculate Rank IC between a single alpha and a predefined target.'

    @abstractmethod
    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        'Calculate both IC and Rank IC between a single alpha and a predefined target.'

    @abstractmethod
    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        'Calculate IC between two alphas.'

    @abstractmethod
    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        'First combine the alphas linearly,'
        'then Calculate IC between the linear combination and a predefined target.'

    @abstractmethod
    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        'First combine the alphas linearly,'
        'then Calculate Rank IC between the linear combination and a predefined target.'

    @abstractmethod
    def calc_pool_all_ret(self, exprs: List[Expression], weights: List[float]) -> Tuple[float, float]:
        'First combine the alphas linearly,'
        'then Calculate both IC and Rank IC between the linear combination and a predefined target.'
```

Reminder: the values evaluated from different alphas may have drastically different scales, we recommend that you should normalize them before combination.

### Before running

The current RL entrypoint is [scripts/rl.py](scripts/rl.py), not `train_maskable_ppo.py`.
The scripts under `scripts/` are the most up-to-date entrypoints in this repository.

These parameters may help you build an `AlphaCalculator`:

- instruments (Set of instruments)
- start_time & end_time (Data range for each dataset)
- target (Target stock trend, e.g., 20d return rate)

These parameters will define a RL run:

- batch_size (PPO batch size)
- features_extractor_kwargs (Arguments for LSTM shared net)
- device (PyTorch device)
- save_path (Path for checkpoints)
- tensorboard_log (Path for TensorBoard)

Practical runtime notes from the current code:

- The Qlib provider URI defaults to `~/.qlib/qlib_data/cn_data`.
- Several code paths still default to `torch.device("cuda:0")`, so CPU-only runs may require passing a different device or editing the script.
- `scripts/rl.py` and `scripts/llm_only.py` use `fire.Fire(...)`, so they should be launched directly as scripts from the repository root.

### Run the experiments

From the repository root, run the experiment entrypoints directly:

```bash
python scripts/rl.py --help
python scripts/llm_only.py --help
python scripts/llm_test_validity.py --help
```

- `scripts/rl.py`: Main experiments of AlphaGen/HARLA
- `scripts/llm_only.py`: Alpha generator based solely on iterative interactions with an LLM.
- `scripts/llm_test_validity.py`: Tests on how the system prompt affects the valid alpha rate of an LLM.

### After running

- Model checkpoints and alpha pools are located in `save_path`;
    - The model is compatiable with [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
    - Alpha pools are formatted in human-readable JSON.
- Tensorboard logs are located in `tensorboard_log`.
    - Per-split scalars are namespaced by `segment_names`: pass `segment_names=("train","valid","test")` to `run_single_experiment` and you get `valid/{ic,rank_ic,icir,rank_icir}` and `test/{ic,rank_ic,icir,rank_icir}` keys (vs. the legacy `test/ic_{i}` numeric scheme).
    - Set `tb_log_every_n_steps` (>= 32) to record those scalars within a rollout rather than only at rollout end. Default `0` keeps the original per-rollout cadence.
- `monitor.log` carries a one-liner per heartbeat/rollout/training event. Rollout and training-end lines include per-split `rank_icir`, e.g. `rollout_end: ... best_ic=+0.087 valid_rank_icir=+0.30 test_rank_icir=+0.18`.
- `status.json` is overwritten on every event. Latest `rollout_end` / `training_end` entries carry a `splits` dict with the full `{ic, rank_ic, icir, rank_icir}` per non-train segment.
- For an offline pass over an existing tree of `*_pool.json` checkpoints (e.g. results synced down from Colab), run `python scripts/evaluate_local_runs.py --runs_dir <DIR> --qlib_data <PATH>`. It writes `checkpoint_metrics.csv`, `checkpoint_selection.csv`, `best_segment_metrics.csv` per run and aggregates into `<DIR>/aggregate/`.

## Baselines

### GP-based methods

[gplearn](https://github.com/trevorstephens/gplearn) implements Genetic Programming, a commonly used method for symbolic regression. We maintained a modified version of gplearn to make it compatiable with our task. The corresponding experiment scipt is [gp.py](gp.py)

### Deep Symbolic Regression

[DSO](https://github.com/brendenpetersen/deep-symbolic-optimization) is a mature deep learning framework for symbolic optimization tasks. We maintained a minimal version of DSO to make it compatiable with our task. The corresponding experiment scipt is [dso.py](dso.py)

Note that this baseline is tied to the legacy dependency set in `dso/setup.py`, not the main project `requirements.txt`.

## Trading (Experimental)

We implemented some trading strategies based on Qlib. See [backtest.py](backtest.py) and [trade_decision.py](trade_decision.py) for demos.

## Citing our work

```bibtex
@inproceedings{alphagen,
    author = {Yu, Shuo and Xue, Hongyan and Ao, Xiang and Pan, Feiyang and He, Jia and Tu, Dandan and He, Qing},
    title = {Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning},
    year = {2023},
    doi = {10.1145/3580305.3599831},
    booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
}
```

## Contributing

Feel free to submit Issues or Pull requests.

## Contributors

This work is maintained by the MLDM research group, [IIP, ICT, CAS](http://iip.ict.ac.cn/).

Maintainers include:

- [Hongyan Xue](https://github.com/xuehongyanL)
- [Shuo Yu](https://github.com/Chlorie)

Thanks to the following contributors:

- [@yigaza](https://github.com/yigaza)

Thanks to the following in-depth research on our project:

- *因子选股系列之九十五: DFQ强化学习因子组合挖掘系统*
