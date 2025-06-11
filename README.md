# Data Pruning for System Identification

The research results for the 9th Edition of the Workshop on Nonlinear System Identification Benchmarks.

## 1. Install Python package and project manager `uv`

The detailed installation instructions can be found [uv installation](https://docs.astral.sh/uv/getting-started/installation/).

### For macOS and Linux

Use `curl` to download the script and execute it with `sh`:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If your system doesn't have `curl`, you can use `wget`:

```shell
wget -qO- https://astral.sh/uv/install.sh | sh
```

### For Windows

Use `irm` to download the script and execute it with `iex`:

```shell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 2. Run command

Generate box plot results.

```shell
uv run plot_box.py
```

To set the dataset, run

```shell
uv run plot_box.py --dataset emps
```

To see help, run

```shell
uv run plot_box.py --help
```

Generate errorbar plot results.

```shell
uv run plot_errorbar.py
```

Generate PCA plot results.

```shell
uv run plot_pca.py
```

To set random state, run

```shell
uv run plot_pca.py --random 123
```

Generate all results for `dsed-eq` dataset

```shell
chmod +x run_dsed_eq.sh
uv run ./run_dsed_eq.sh
```

Generate all results for `dsed-tr` dataset

```shell
chmod +x run_dsed_tr.sh
uv run ./run_dsed_tr.sh
```

Print NARX terms and training R2 score

```shell
uv run print_terms.py --dataset emps --delay 6
```
