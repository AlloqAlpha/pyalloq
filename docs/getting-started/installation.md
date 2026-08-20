# Installation

PyAlloq requires **Python 3.11+**.

---

## Using `uv` (Recommended)

[`uv`](https://docs.astral.sh/uv/) is the recommended package manager for PyAlloq.

```bash
uv add pyalloq
```

To also install optional machine learning or deep learning extras:

```bash
# Deep learning (PyTorch-based models)
uv add "pyalloq[dl]"

# Reinforcement learning (Stable-Baselines3 + Gymnasium)
uv add "pyalloq[rl]"

# All ML extras
uv add "pyalloq[all-ml]"
```

---

## Using `pip`

```bash
pip install pyalloq
```

With optional extras:

```bash
pip install "pyalloq[dl]"
pip install "pyalloq[rl]"
pip install "pyalloq[all-ml]"
```

---

## Development Installation (from source)

Clone the repository and install all workspace packages together:

```bash
git clone https://github.com/AlloqAlpha/pyalloq.git
cd pyalloq

# Install all workspace packages + dev dependencies
uv sync --all-extras
```

This installs `pyalloq`, `pyalloq-core`, `pyalloq-backtest`, `pyalloq-data-connector`, and `pyalloq-features` as editable workspace packages.

---

## Optional Dependencies

| Extra | Key packages | Use case |
|-------|-------------|----------|
| `[dl]` | `torch>=2.0` | Deep learning return/covariance estimators |
| `[rl]` | `stable-baselines3`, `gymnasium` | Reinforcement learning allocators |
| `[all-ml]` | All of the above | Full ML stack |

---

## Verify Installation

```python
import pyalloq
import pyalloq_core
import pyalloq_backtest
import pyalloq_data_connector

print("PyAlloq installed successfully!")
```
