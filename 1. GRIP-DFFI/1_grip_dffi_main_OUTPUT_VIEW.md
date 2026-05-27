```python
import os
import gc
import re
import glob
import json
import time
import math
import random
import warnings
import zlib
from io import BytesIO
from html import escape as html_escape
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    from IPython.display import display, Image as IPImage, HTML
    IN_NOTEBOOK = True
except Exception:
    HTML = None
    IN_NOTEBOOK = False

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    cohen_kappa_score,
)
from sklearn.preprocessing import label_binarize

# ==================================================================================================
# REPRODUCIBILITY / DEVICE
# ==================================================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEM = DEVICE == "cuda"
NUM_WORKERS = 2 if DEVICE == "cuda" else 0
print(f"DEVICE: {DEVICE} | PIN_MEMORY: {PIN_MEM}")

# ==================================================================================================
# STYLE / PLOTTING
# ==================================================================================================
BG_CLR = "#f8fafc"
PANEL_CLR = "#ffffff"
GRID_CLR = "#d9e2ec"
TEXT_CLR = "#102a43"
SUBTEXT_CLR = "#486581"
ACCENT_1 = "#2563eb"
ACCENT_2 = "#06b6d4"
ACCENT_3 = "#10b981"
ACCENT_4 = "#f97316"
ACCENT_5 = "#ef4444"
ACCENT_6 = "#8b5cf6"
PALETTE = [
    "#2563eb", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
    "#0ea5e9", "#22c55e", "#f97316", "#db2777", "#84cc16", "#7c3aed"
]

PROCESS_NAME = "GRIP-DFFI"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.facecolor": BG_CLR,
    "axes.facecolor": PANEL_CLR,
    "savefig.facecolor": BG_CLR,
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": TEXT_CLR,
    "xtick.color": TEXT_CLR,
    "ytick.color": TEXT_CLR,
    "text.color": TEXT_CLR,
    "axes.titlecolor": TEXT_CLR,
    "grid.color": GRID_CLR,
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "axes.grid": True,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "legend.framealpha": 0.96,
    "legend.facecolor": "white",
    "legend.edgecolor": "#d1d5db",
})

SPLIT_POLICY = "merge-compatible-labeled-sources-then-fresh-split"
PRESERVE_OFFICIAL_BENCHMARK_SPLITS = False


def save_or_show(fig, path=None):
    fig.tight_layout()
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    if IN_NOTEBOOK:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
        buf.seek(0)
        display(IPImage(data=buf.read()))
    plt.close(fig)


def prettify_ax(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(PANEL_CLR)
    for sp in ax.spines.values():
        sp.set_edgecolor("#cbd5e1")
    ax.grid(True, color=GRID_CLR, linestyle="--", linewidth=0.8, alpha=0.85)
    ax.tick_params(colors=TEXT_CLR)
    if title is not None:
        ax.set_title(title, pad=10)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


_TABLE_CSS_EMITTED = False


def _emit_table_css_once():
    global _TABLE_CSS_EMITTED
    if not IN_NOTEBOOK or _TABLE_CSS_EMITTED:
        return
    css = """
    <style>
      .grip-table-block {
        margin: 12px 0 18px 0;
        border: 1px solid #dbe4ee;
        border-radius: 12px;
        overflow-x: auto;
        background: #ffffff;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.05);
      }
      .grip-table-title {
        padding: 10px 14px;
        font-weight: 700;
        color: #0f172a;
        background: #f8fafc;
        border-bottom: 1px solid #e2e8f0;
        font-size: 15px;
      }
      .grip-table-block table.dataframe {
        border-collapse: separate !important;
        border-spacing: 0;
        width: max-content;
        min-width: 100%;
        margin: 0;
        font-size: 13px;
      }
      .grip-table-block table.dataframe thead th {
        position: sticky;
        top: 0;
        background: #eff6ff;
        color: #0f172a;
        text-align: left;
        padding: 9px 10px;
        border-bottom: 1px solid #cbd5e1;
        white-space: nowrap;
      }
      .grip-table-block table.dataframe tbody td {
        padding: 8px 10px;
        border-bottom: 1px solid #eef2f7;
        vertical-align: top;
        white-space: normal;
        word-break: break-word;
      }
      .grip-table-block table.dataframe tbody tr:nth-child(even) {
        background: #fcfdff;
      }
      .grip-table-block table.dataframe tbody tr:hover {
        background: #f8fbff;
      }
      .grip-note {
        color: #475569;
        font-size: 12px;
        margin: 6px 0 12px 2px;
      }
    </style>
    """
    display(HTML(css))
    _TABLE_CSS_EMITTED = True


def wrap_sequence_for_display(seq, items_per_line=8):
    seq = list(seq)
    if not seq:
        return '[]'
    lines = []
    for i in range(0, len(seq), items_per_line):
        chunk = ', '.join(str(x) for x in seq[i:i + items_per_line])
        lines.append(chunk)
    return '[' + ',\n '.join(lines) + ']'


def _prepare_table_df(df, float_decimals=4):
    out = df.copy()
    for col in out.columns:
        vals = []
        for v in out[col].tolist():
            if isinstance(v, (list, tuple, np.ndarray)):
                vals.append(wrap_sequence_for_display(v))
            elif isinstance(v, (float, np.floating)):
                vals.append('' if pd.isna(v) else f"{float(v):.{float_decimals}f}")
            elif isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)):
                vals.append(f"{int(v):,}")
            elif isinstance(v, (bool, np.bool_)):
                vals.append('True' if bool(v) else 'False')
            elif pd.isna(v):
                vals.append('')
            else:
                vals.append(str(v))
        out[col] = vals
    return out


def show_table(title, df, index=False, rename_map=None, columns=None, float_decimals=4, note=None):
    if df is None:
        return
    view = df.copy()
    if columns is not None:
        existing = [c for c in columns if c in view.columns]
        view = view[existing]
    if rename_map is not None:
        view = view.rename(columns=rename_map)
    view = _prepare_table_df(view, float_decimals=float_decimals)

    if IN_NOTEBOOK:
        _emit_table_css_once()
        html_df = view.copy()
        html_df.columns = [html_escape(str(c)) for c in html_df.columns]
        for col in html_df.columns:
            html_df[col] = html_df[col].map(lambda s: html_escape(str(s)).replace('\n', '<br>'))
        table_html = html_df.to_html(index=index, escape=False, border=0)
        note_html = f'<div class="grip-note">{html_escape(str(note))}</div>' if note else ''
        display(HTML(f'<div class="grip-table-block"><div class="grip-table-title">{html_escape(str(title))}</div>{table_html}</div>{note_html}'))
    else:
        print('\n' + '=' * 100)
        print(title)
        print('-' * 100)
        try:
            print(view.to_markdown(index=index))
        except Exception:
            print(view.to_string(index=index))
        if note:
            print(note)
        print('=' * 100)


def show_key_value_table(title, items, note=None):
    df = pd.DataFrame({
        'Field': list(items.keys()),
        'Value': [wrap_sequence_for_display(v) if isinstance(v, (list, tuple, np.ndarray)) else v for v in items.values()],
    })
    show_table(title, df, index=False, float_decimals=4, note=note)


# ==================================================================================================
# DATASET CONFIG
# ==================================================================================================
DATASET_SPECS = [
    {
        "name": "I23Sub",
        "source_type": "kaggle",
        "slug": "wittigenz/hydras",
        "domain_id": 0,
        "task_preference": "multiclass",
        "prefer_multiclass": ["attack_cat", "attack_type", "attack", "category", "type", "class_type"],
        "prefer_binary": ["label", "Label", "binary_label", "target", "is_attack", "Class"],
        "normal_aliases": ["normal", "benign", "benign_traffic", "background", "legitimate", "0"],
        "file_rank": None,
        "unsw_nb15_raw": False,
    },
    {
        "name": "K99Sub",
        "source_type": "kaggle",
        "slug": "sampadab17/network-intrusion-detection",
        "domain_id": 1,
        "task_preference": "multiclass",
        "prefer_multiclass": ["attack_cat", "attack_type", "attack", "category", "type", "class", "Class"],
        "prefer_binary": ["label", "Label", "binary_label", "target", "is_attack"],
        "normal_aliases": ["normal", "benign", "0"],
        "file_rank": None,
        "unsw_nb15_raw": False,
    },
    {
        "name": "NTD1",
        "source_type": "kaggle",
        "slug": "rebsonramalho/network-threat-detection-dataset",
        "domain_id": 2,
        "task_preference": "multiclass",
        "prefer_multiclass": ["attack_cat", "attack_type", "attack", "category", "type", "class", "Class"],
        "prefer_binary": ["label", "Label", "binary_label", "target", "is_attack"],
        "normal_aliases": ["normal", "benign", "benign_traffic", "legitimate", "0"],
        "file_rank": 0,
        "unsw_nb15_raw": False,
    },
    {
        "name": "NTD2",
        "source_type": "kaggle",
        "slug": "rebsonramalho/network-threat-detection-dataset",
        "domain_id": 3,
        "task_preference": "multiclass",
        "prefer_multiclass": ["attack_cat", "attack_type", "attack", "category", "type", "class", "Class"],
        "prefer_binary": ["label", "Label", "binary_label", "target", "is_attack"],
        "normal_aliases": ["normal", "benign", "benign_traffic", "legitimate", "0"],
        "file_rank": 1,
        "unsw_nb15_raw": False,
    },
    {
        "name": "WII21",
        "source_type": "kaggle",
        "slug": "annaamalaiu/wustl-iiot-2021-dataset",
        "domain_id": 4,
        "task_preference": "multiclass",
        "prefer_multiclass": ["attack_cat", "attack_type", "attack", "category", "type", "class", "Class", "label"],
        "prefer_binary": ["label", "Label", "binary_label", "target", "is_attack"],
        "normal_aliases": ["normal", "benign", "benign_traffic", "background", "legitimate", "0"],
        "file_rank": None,
        "unsw_nb15_raw": False,
    },
]

TARGET_LIKE_NAMES = {
    "label", "target", "class", "y", "outcome", "attack_cat", "attack", "category",
    "type", "label_type", "class_type", "attack_type", "binary_label", "is_attack"
}

UNSW_COLS = [
    "srcip","sport","dstip","dsport","proto","state","dur","sbytes","dbytes",
    "sttl","dttl","sloss","dloss","service","Sload","Dload","Spkts","Dpkts",
    "swin","dwin","stcpb","dtcpb","smeansz","dmeansz","trans_depth","res_bdy_len",
    "Sjit","Djit","Stime","Ltime","Sintpkt","Dintpkt","tcprtt","synack","ackdat",
    "is_sm_ips_ports","ct_state_ttl","ct_flw_http_mthd","is_ftp_login","ct_ftp_cmd",
    "ct_srv_src","ct_srv_dst","ct_dst_ltm","ct_src_ltm","ct_src_dport_ltm",
    "ct_dst_sport_ltm","ct_dst_src_ltm","attack_cat","label"
]

# ==================================================================================================
# DOWNLOAD / MATERIALIZE EXTERNAL SOURCES
# ==================================================================================================
import kagglehub
from datasets import load_dataset

HF_MATERIALIZED_ROOT = os.path.abspath("./hf_materialized_datasets")


def materialize_hf_dataset(hf_id, local_name):
    os.makedirs(HF_MATERIALIZED_ROOT, exist_ok=True)
    out_dir = os.path.join(HF_MATERIALIZED_ROOT, local_name)
    os.makedirs(out_dir, exist_ok=True)

    existing = []
    for ext in ["csv", "tsv", "parquet"]:
        existing += glob.glob(os.path.join(out_dir, f"**/*.{ext}"), recursive=True)
    if existing:
        print(f"[HF cached] {hf_id} -> {out_dir}")
        return out_dir

    ds_obj = load_dataset(hf_id)
    if hasattr(ds_obj, "keys"):
        split_names = list(ds_obj.keys())
        for split in split_names:
            df = ds_obj[split].to_pandas()
            out_path = os.path.join(out_dir, f"{split}.parquet")
            df.to_parquet(out_path, index=False)
            print(f"[HF materialized] {hf_id}::{split} -> {out_path} | shape={df.shape}")
    else:
        df = ds_obj.to_pandas()
        out_path = os.path.join(out_dir, "train.parquet")
        df.to_parquet(out_path, index=False)
        print(f"[HF materialized] {hf_id} -> {out_path} | shape={df.shape}")
    return out_dir


_download_cache = {}
for spec in DATASET_SPECS:
    source_type = spec.get("source_type", "kaggle")
    if source_type == "huggingface":
        hf_id = spec["hf_id"]
        cache_key = ("hf", hf_id)
        if cache_key not in _download_cache:
            _download_cache[cache_key] = materialize_hf_dataset(hf_id, spec["name"])
        spec["path"] = _download_cache[cache_key]
        print(f"[{spec['name']}] HF dataset path -> {spec['path']}")
    else:
        slug = spec["slug"]
        cache_key = ("kaggle", slug)
        if cache_key not in _download_cache:
            _download_cache[cache_key] = kagglehub.dataset_download(slug)
            print(f"[{slug}] -> {_download_cache[cache_key]}")
        spec["path"] = _download_cache[cache_key]

# ==================================================================================================
# UTILITIES
# ==================================================================================================
IP_RE = re.compile(r"^\s*\d{1,3}(\.\d{1,3}){3}\s*$")
TRAIN_RE = re.compile(r"(?:^|[_\-\s])(train|training)(?:[_\-\s\.]|$)", re.I)
VAL_RE = re.compile(r"(?:^|[_\-\s])(val|valid|validation|dev)(?:[_\-\s\.]|$)", re.I)
TEST_RE = re.compile(r"(?:^|[_\-\s])(test|testing)(?:[_\-\s\.]|$)", re.I)


def safe_norm(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    rng = arr.max() - arr.min()
    return (arr - arr.min()) / (rng + 1e-6)


def sanitize_numeric_1d(x, clip=1e6):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
    x = np.clip(x, -clip, clip)
    return x.astype(np.float32)


def list_table_files(root):
    files = []
    for ext in ["csv", "tsv", "parquet"]:
        files += glob.glob(os.path.join(root, f"**/*.{ext}"), recursive=True)
    return sorted(files)


def load_one(path, force_cols=None):
    if path.endswith(".csv"):
        if force_cols is not None:
            df = pd.read_csv(path, header=None, low_memory=False)
            if df.shape[1] == len(force_cols):
                first = df.iloc[0].astype(str).tolist()
                if any(v in force_cols for v in first):
                    df = df.iloc[1:].reset_index(drop=True)
                df.columns = force_cols
                return df
        return pd.read_csv(path, low_memory=False)
    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t", low_memory=False)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(path)


def combine_tables(files, force_cols=None, max_rows=None):
    dfs = []
    for fp in files:
        try:
            d = load_one(fp, force_cols=force_cols)
            if len(d) > 0:
                dfs.append(d)
                print(f"    loaded {os.path.basename(fp)}: {d.shape}")
        except Exception as e:
            print(f"    [WARN] failed {fp}: {e}")
    if not dfs:
        return None
    cols = sorted(set().union(*[set(d.columns) for d in dfs]))
    aligned = []
    for d in dfs:
        dd = d.copy()
        for c in cols:
            if c not in dd.columns:
                dd[c] = np.nan
        aligned.append(dd[cols])
    out = pd.concat(aligned, axis=0, ignore_index=True)
    if max_rows is not None and len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=SEED).reset_index(drop=True)
    return out


def categorize_split_files(files):
    groups = {"train": [], "val": [], "test": [], "other": []}
    for fp in files:
        bn = os.path.basename(fp)
        if TRAIN_RE.search(bn):
            groups["train"].append(fp)
        elif VAL_RE.search(bn):
            groups["val"].append(fp)
        elif TEST_RE.search(bn):
            groups["test"].append(fp)
        else:
            groups["other"].append(fp)
    return groups


def choose_files_for_spec(spec, all_files):
    if spec.get("unsw_nb15_raw", False):
        nb_names = [f"UNSW-NB15_{i}.csv" for i in range(1, 5)]
        chosen = [f for f in all_files if os.path.basename(f) in nb_names]
        if not chosen:
            chosen = [f for f in all_files if re.search(r"UNSW.NB15_[1-4]\.csv$", f, re.I)]
        if not chosen:
            chosen = sorted(all_files, key=os.path.getsize, reverse=True)[:4]
        return sorted(chosen)
    if spec.get("file_rank") is not None:
        by_size = sorted(all_files, key=os.path.getsize, reverse=True)
        rank = int(spec["file_rank"])
        if rank < 0 or rank >= len(by_size):
            raise ValueError(f"{spec['name']}: file_rank={rank} out of range for {len(by_size)} files")
        return [by_size[rank]]
    return all_files


def clean_target_series(s, spec=None):
    x = s.astype("string").fillna("missing").str.strip()
    x = x.replace({"": "missing", "nan": "missing", "None": "missing", "<NA>": "missing"})
    if spec is not None and spec["name"] == "ds3":
        x = x.replace({" ": "Normal", "": "Normal"})
    return x


def detect_target_column(df, spec):
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}

    def existing(cands):
        return [lower_map[str(c).lower()] for c in cands if str(c).lower() in lower_map]

    multiclass_cands = existing(spec.get("prefer_multiclass", []))
    binary_cands = existing(spec.get("prefer_binary", []))

    def viable(c, want_multi):
        vals = clean_target_series(df[c], spec)
        nunq = vals.nunique(dropna=True)
        if want_multi:
            return nunq > 2
        return nunq == 2

    if spec.get("task_preference", "multiclass") == "multiclass":
        for c in multiclass_cands:
            if viable(c, want_multi=True):
                return c, "multiclass"
        for c in binary_cands:
            if viable(c, want_multi=False):
                return c, "binary"
    else:
        for c in binary_cands:
            if viable(c, want_multi=False):
                return c, "binary"
        for c in multiclass_cands:
            if viable(c, want_multi=True):
                return c, "multiclass"

    for c in cols:
        if str(c).lower() in TARGET_LIKE_NAMES:
            vals = clean_target_series(df[c], spec)
            nunq = vals.nunique(dropna=True)
            if nunq > 2:
                return c, "multiclass"
            if nunq == 2:
                return c, "binary"

    raise RuntimeError(f"{spec['name']}: could not resolve a real target column")


def drop_target_like_columns(X, keep_target, drop_cols=None):
    keep_lower = str(keep_target).lower()
    drop_cols = drop_cols or []
    safe_drop = [c for c in drop_cols if str(c).lower() != keep_lower and c in X.columns]
    if safe_drop:
        print(f"  [INFO] dropped verified target-sibling columns: {safe_drop}")
    return X.drop(columns=safe_drop, errors="ignore")


def candidate_target_like_columns(df, target_col):
    keep_lower = str(target_col).lower()
    out = []
    for c in df.columns:
        cl = str(c).lower()
        if cl == keep_lower:
            continue
        if cl in TARGET_LIKE_NAMES or keep_lower in cl or cl in keep_lower:
            out.append(c)
    return out


def find_target_sibling_columns(Xtr, y_train_raw, target_col, spec, deterministic_threshold=0.995):
    """
    Drop only columns that behave like alternate labels / deterministic remaps of the chosen target.
    This is intentionally conservative: name similarity alone is NOT enough.
    """
    y = clean_target_series(pd.Series(y_train_raw), spec).astype(str)
    drop_cols = []
    sibling_report = []
    for c in candidate_target_like_columns(Xtr, target_col):
        s = clean_target_series(pd.Series(Xtr[c]), spec).astype(str)
        tmp = pd.DataFrame({"feat": s, "target": y}).replace({"missing": np.nan}).dropna()
        if len(tmp) == 0:
            continue
        nunq_feat = int(tmp["feat"].nunique())
        nunq_tgt = int(tmp["target"].nunique())
        if nunq_feat <= 1 or nunq_tgt <= 1:
            continue

        per_feat_target_nunique = tmp.groupby("feat")["target"].nunique()
        deterministic_feat_to_target = int(per_feat_target_nunique.max()) == 1
        mean_majority_purity = float(
            tmp.groupby("feat")["target"].apply(lambda z: z.value_counts(normalize=True).iloc[0]).mean()
        )

        if deterministic_feat_to_target and mean_majority_purity >= deterministic_threshold:
            drop_cols.append(c)
            sibling_report.append((c, nunq_feat, nunq_tgt, round(mean_majority_purity, 4)))

    if sibling_report:
        print(f"  [INFO] verified target-sibling columns (dropped): {sibling_report}")
    else:
        print("  [INFO] no deterministic target-sibling feature columns detected")
    return drop_cols



def detect_normal_index(class_names, spec=None):
    aliases = {"normal", "benign", "benign_traffic", "background", "legitimate"}
    if spec is not None:
        aliases |= set([str(x).lower() for x in spec.get("normal_aliases", [])])

    for i, name in enumerate(class_names):
        s = str(name).strip().lower()
        if s in aliases:
            return i
        if re.fullmatch(r"normal[_\-\s]*traffic", s):
            return i
        if re.fullmatch(r"benign[_\-\s]*traffic", s):
            return i
        if s == "0" and len(class_names) == 2:
            return i
    if len(class_names) == 2:
        names = [str(x) for x in class_names]
        if set(names) == {"0", "1"}:
            return names.index("0")
    return None


def fit_label_mapping(y_train_raw):
    uniq = pd.Index(pd.unique(y_train_raw.astype(str)))
    mapping = {v: i for i, v in enumerate(uniq.tolist())}
    return mapping, uniq.tolist()


def apply_label_mapping(y_raw, mapping):
    y = y_raw.astype(str).map(mapping)
    keep = y.notna()
    return y[keep].astype(int).to_numpy(), keep.to_numpy()


def safe_stratify_labels(y_raw):
    vc = pd.Series(y_raw).value_counts()
    return y_raw if len(vc) > 1 and vc.min() >= 2 else None


def split_three_way(df, y_raw, seed=SEED):
    vc = pd.Series(y_raw).value_counts()
    valid = vc[vc >= 3].index
    dropped = vc.index.difference(valid)
    if len(dropped) > 0:
        print(f"  [WARN] dropping rare classes before split: {dropped.tolist()}")
        mask = pd.Series(y_raw).isin(valid).to_numpy()
        df = df.loc[mask].reset_index(drop=True)
        y_raw = pd.Series(y_raw).loc[mask].reset_index(drop=True)

    X_tr, X_tmp, y_tr_raw, y_tmp_raw = train_test_split(
        df, y_raw, test_size=0.30, random_state=seed, stratify=safe_stratify_labels(y_raw)
    )
    X_va, X_te, y_va_raw, y_te_raw = train_test_split(
        X_tmp, y_tmp_raw, test_size=0.50, random_state=seed, stratify=safe_stratify_labels(y_tmp_raw)
    )
    return (
        X_tr.reset_index(drop=True), pd.Series(y_tr_raw).reset_index(drop=True),
        X_va.reset_index(drop=True), pd.Series(y_va_raw).reset_index(drop=True),
        X_te.reset_index(drop=True), pd.Series(y_te_raw).reset_index(drop=True),
    )


def drop_id_cols(X):
    n = len(X)
    explicit = {"id", "uid", "uuid", "flow_id", "record_id", "srcip", "dstip"}
    drop = []
    for c in X.columns:
        cl = str(c).lower()
        if cl in explicit:
            drop.append(c)
            continue
        if n >= 5000:
            try:
                if X[c].nunique(dropna=True) == n:
                    drop.append(c)
            except Exception:
                pass
    kept = [c for c in X.columns if c not in set(drop)]
    if len(kept) == 0:
        return X.copy()
    if drop:
        print(f"  [INFO] dropped ID/high-uniqueness columns: {drop[:12]}{'...' if len(drop) > 12 else ''}")
    return X[kept].copy()


def infer_types(Xtr, thresh=0.90):
    num_cols, cat_cols = [], []
    for c in Xtr.columns:
        s = Xtr[c]
        if s.dtype == "O" or str(s.dtype).startswith(("category", "string")):
            sample = s.dropna().astype(str).head(250).tolist()
            if any(IP_RE.match(v) for v in sample):
                cat_cols.append(c)
                continue
            if pd.to_numeric(s, errors="coerce").notna().mean() >= thresh:
                num_cols.append(c)
            else:
                cat_cols.append(c)
        else:
            num_cols.append(c)
    cat_cols = [c for c in cat_cols if c not in num_cols]
    return num_cols, cat_cols


def fit_num_stats(Xtr, num_cols):
    means, stds = {}, {}
    for c in num_cols:
        s = pd.to_numeric(Xtr[c], errors="coerce")
        m = float(s.mean()) if s.notna().any() else 0.0
        sd = float(s.std()) if s.notna().any() else 1.0
        if not np.isfinite(sd) or sd < 1e-6:
            sd = 1.0
        means[c] = m
        stds[c] = sd
    return means, stds


def apply_num_cat_preproc(X, num_cols, cat_cols, means, stds):
    X = X.copy()
    ni = ci = 0
    for c in num_cols:
        s = pd.to_numeric(X[c], errors="coerce")
        ni += int(s.isna().sum())
        s = s.fillna(means.get(c, 0.0))
        s = (s - means.get(c, 0.0)) / (stds.get(c, 1.0) + 1e-6)
        X[c] = s.astype(np.float32)
    for c in cat_cols:
        s = X[c].astype("string")
        ci += int(s.isna().sum())
        X[c] = s.fillna("missing")
    return X, ni, ci


def target_signature(y_series, spec):
    y = clean_target_series(pd.Series(y_series), spec)
    vals = set(y.dropna().astype(str).unique().tolist())
    vals = {v for v in vals if v != "missing"}
    return vals


def target_compatible(train_y, other_y, spec):
    tr = target_signature(train_y, spec)
    ot = target_signature(other_y, spec)
    if len(tr) < 2 or len(ot) < 1:
        return False
    inter = tr & ot
    if tr == ot:
        return True
    if tr.issubset(ot) or ot.issubset(tr):
        return True
    overlap_ratio = len(inter) / max(min(len(tr), len(ot)), 1)
    if overlap_ratio >= 0.80:
        return True
    if tr.issubset({"0", "1"}) and ot.issubset({"0", "1"}):
        return True
    return False


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(to_jsonable(k)): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


# ==================================================================================================
# GRIP-DFFI CONFIGURATION
# ==================================================================================================
FEATURE_INTEL_ROUNDS = 3
FEATURE_INTEL_STEPS = 100
FEATURE_INTEL_LR = 1e-2
FEATURE_INTEL_MI_WEIGHT = 0.55
FEATURE_INTEL_FRN_WEIGHT = 0.45
UNIVERSALITY_THRESHOLD = 0.60
MIN_SHARED_OCCURRENCE = 2
SHARED_HASH_BUCKETS = 4096
RL_POLICY_EPISODES = 90
RL_POLICY_LR = 0.18
DIFFUSION_STEPS = 20
DIFFUSION_LOSS_WEIGHT = 0.05
ENABLE_DIFFUSION_SHARED_BACKBONE = True
CLIENT_DIRICHLET_ALPHA = 0.70


# ==================================================================================================
# GRAPH-REFINED FEATURE INTELLIGENCE
# ==================================================================================================
def feature_key(name, kind):
    return f"{kind}::{name}"


def feature_name_from_key(key):
    return key.split("::", 1)[1]


def feature_kind_from_key(key):
    return key.split("::", 1)[0]


def deterministic_hash_value(text, mod=SHARED_HASH_BUCKETS):
    return zlib.crc32(str(text).encode("utf-8")) % mod


def hash_categorical_frame(X, cols, mod=SHARED_HASH_BUCKETS):
    if not cols:
        return np.zeros((len(X), 0), np.int64)
    out = np.zeros((len(X), len(cols)), dtype=np.int64)
    for j, c in enumerate(cols):
        vals = X[c].astype(str).fillna("missing").tolist()
        out[:, j] = np.array([deterministic_hash_value(f"{c}={v}", mod=mod) for v in vals], dtype=np.int64)
    return out


def cross_plan(Xtr, ytr, cat_cols, max_base=6, seed=SEED):
    rng = np.random.RandomState(seed)
    base = cat_cols[:min(max_base, len(cat_cols))]
    if len(base) < 2:
        return {"use_triples": False, "base_cols": base}
    idx = rng.choice(len(Xtr), min(7000, len(Xtr)), replace=False)
    Xt, yt = Xtr.iloc[idx].copy(), ytr[idx]

    def mk(X, triples):
        Xc = X.copy()
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                Xc[f"xp_{i}_{j}"] = Xc[base[i]].astype(str) + "||" + Xc[base[j]].astype(str)
        if triples and len(base) >= 3:
            for i in range(len(base)):
                for j in range(i + 1, len(base)):
                    for k in range(j + 1, len(base)):
                        Xc[f"xt_{i}_{j}_{k}"] = (
                            Xc[base[i]].astype(str) + "||" +
                            Xc[base[j]].astype(str) + "||" +
                            Xc[base[k]].astype(str)
                        )
        return Xc

    def mi_sc(Xc):
        cols = [c for c in Xc.columns if c.startswith(("xp_", "xt_"))]
        if not cols:
            return 0.0
        M = np.stack([
            pd.factorize(Xc[c].astype(str).fillna("missing"))[0].astype(np.float32) for c in cols
        ], axis=1)
        mi = mutual_info_classif(M, yt, discrete_features=True, random_state=SEED)
        return float(np.nan_to_num(mi).mean())

    sp = mi_sc(mk(Xt, False))
    st = mi_sc(mk(Xt, True))
    return {"use_triples": st > sp * 1.03, "base_cols": base, "pair_mi": float(sp), "triple_mi": float(st)}



def apply_crosses(X, plan):
    X = X.copy()
    base = plan["base_cols"]
    new = []
    if len(base) >= 2:
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                cn = f"cross_{base[i]}__{base[j]}"
                X[cn] = X[base[i]].astype(str) + "||" + X[base[j]].astype(str)
                new.append(cn)
    if plan["use_triples"] and len(base) >= 3:
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                for k in range(j + 1, len(base)):
                    cn = f"cross_{base[i]}__{base[j]}__{base[k]}"
                    X[cn] = (
                        X[base[i]].astype(str) + "||" +
                        X[base[j]].astype(str) + "||" +
                        X[base[k]].astype(str)
                    )
                    new.append(cn)
    return X, new


class GraphRefinedRelevanceNet(nn.Module):
    def __init__(self, in_dim=2, hid=32):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.out = nn.Linear(hid, 1)

    def forward(self, X, A):
        h = F.relu(self.fc1(A @ X))
        h = F.relu(self.fc2(A @ h))
        return torch.sigmoid(self.out(h)).squeeze(1)



def build_feature_graph(Xtr, ytr, feat_names, num_cols, cat_cols, max_rows=8000, top_e=8):
    if not feat_names:
        return np.zeros((0, 2), np.float32), np.zeros((0, 0), np.float32)
    if len(Xtr) > max_rows:
        idx = np.random.choice(len(Xtr), max_rows, replace=False)
        Xt, yt = Xtr.iloc[idx], ytr[idx]
    else:
        Xt, yt = Xtr, ytr

    M, disc, names = [], [], []
    for c in num_cols:
        if c in Xt.columns:
            M.append(sanitize_numeric_1d(pd.to_numeric(Xt[c], errors="coerce").to_numpy()))
            disc.append(False)
            names.append(c)
    for c in cat_cols:
        if c in Xt.columns:
            cd, _ = pd.factorize(Xt[c].astype(str).fillna("missing"))
            M.append(cd.astype(np.float32))
            disc.append(True)
            names.append(c)

    Fdim = len(feat_names)
    if not M:
        return np.zeros((Fdim, 2), np.float32), np.eye(Fdim, dtype=np.float32)

    Mm = np.stack(M, axis=1)
    mi = mutual_info_classif(Mm, yt, discrete_features=disc, random_state=SEED)
    mi = np.nan_to_num(mi)
    mi_map = {names[i]: float(mi[i]) for i in range(len(names))}

    mi_v, var_v = [], []
    for f in feat_names:
        mi_v.append(mi_map.get(f, 0.0))
        if f in num_cols and f in Xt.columns:
            vals = sanitize_numeric_1d(pd.to_numeric(Xt[f], errors="coerce").to_numpy())
            var_v.append(float(np.nanstd(vals)))
        else:
            var_v.append(float(Xt[f].astype(str).nunique() / max(len(Xt), 1)) if f in Xt.columns else 0.0)

    Xn = np.stack([np.array(mi_v, np.float32), np.array(var_v, np.float32)], axis=1)
    Xn = (Xn - Xn.mean(0, keepdims=True)) / (Xn.std(0, keepdims=True) + 1e-6)

    A = np.eye(Fdim, dtype=np.float32)
    ni = [i for i, f in enumerate(feat_names) if f in num_cols and f in Xt.columns]
    if len(ni) >= 2:
        Xnm = np.stack([
            sanitize_numeric_1d(pd.to_numeric(Xt[feat_names[i]], errors="coerce").to_numpy()) for i in ni
        ], axis=1)
        C = np.nan_to_num(np.corrcoef(Xnm, rowvar=False))
        for ii, i in enumerate(ni):
            for jj in np.argsort(np.abs(C[ii]))[::-1][1:top_e + 1]:
                j = ni[jj]
                A[i, j] = max(A[i, j], float(abs(C[ii, jj])))
                A[j, i] = max(A[j, i], float(abs(C[ii, jj])))
    ci_idx = [i for i, f in enumerate(feat_names) if f in cat_cols and f in Xt.columns]
    for i in ci_idx:
        sv = set(Xt[feat_names[i]].astype(str).value_counts().head(20).index)
        for j in ci_idx:
            if j <= i:
                continue
            sw = set(Xt[feat_names[j]].astype(str).value_counts().head(20).index)
            sim = len(sv & sw) / (len(sv | sw) + 1e-6)
            if sim > 0.15:
                A[i, j] = A[j, i] = max(A[i, j], float(sim))
    D = A.sum(1)
    Dinv = 1.0 / np.sqrt(D + 1e-6)
    A = (A * Dinv[:, None]) * Dinv[None, :]
    return Xn.astype(np.float32), A.astype(np.float32)



def train_graph_refined_relevance_net(Xnode, A, steps=FEATURE_INTEL_STEPS, lr=FEATURE_INTEL_LR, init_state=None):
    if Xnode.shape[0] == 0:
        dummy = GraphRefinedRelevanceNet(in_dim=2, hid=32)
        if init_state is not None:
            try:
                dummy.load_state_dict(init_state, strict=False)
            except Exception:
                pass
        return dummy.state_dict(), np.array([], dtype=np.float32)
    Xt = torch.tensor(Xnode, dtype=torch.float32)
    At = torch.tensor(A, dtype=torch.float32)
    net = GraphRefinedRelevanceNet(in_dim=Xt.shape[1], hid=32)
    if init_state is not None:
        try:
            net.load_state_dict(init_state, strict=False)
        except Exception:
            pass
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mi_col = Xt[:, 0]
    mi_norm = (mi_col - mi_col.min()) / (mi_col.max() - mi_col.min() + 1e-6)
    for _ in range(steps):
        opt.zero_grad()
        pred = net(Xt, At)
        loss = F.mse_loss(pred, mi_norm.detach())
        loss.backward()
        opt.step()
    with torch.no_grad():
        scores = net(Xt, At).cpu().numpy()
    return net.state_dict(), scores



def score_graph_refined_relevance_net(Xnode, A, state):
    if Xnode.shape[0] == 0:
        return np.array([], dtype=np.float32)
    Xt = torch.tensor(Xnode, dtype=torch.float32)
    At = torch.tensor(A, dtype=torch.float32)
    net = GraphRefinedRelevanceNet(in_dim=Xt.shape[1], hid=32)
    if state is not None:
        net.load_state_dict(state, strict=False)
    with torch.no_grad():
        return net(Xt, At).cpu().numpy().astype(np.float32)



def compute_mi(Xtr, ytr, num_cols, cat_cols, max_rows=9000):
    if len(Xtr) > max_rows:
        idx = np.random.choice(len(Xtr), max_rows, replace=False)
        Xt, yt = Xtr.iloc[idx], ytr[idx]
    else:
        Xt, yt = Xtr, ytr
    M, disc, names = [], [], []
    for c in num_cols:
        if c in Xt.columns:
            M.append(sanitize_numeric_1d(pd.to_numeric(Xt[c], errors="coerce").to_numpy()))
            disc.append(False)
            names.append(c)
    for c in cat_cols:
        if c in Xt.columns:
            cd, _ = pd.factorize(Xt[c].astype(str).fillna("missing"))
            M.append(cd.astype(np.float32))
            disc.append(True)
            names.append(c)
    if not M:
        return [], np.array([], np.float32)
    mi = mutual_info_classif(np.stack(M, axis=1), yt, discrete_features=disc, random_state=SEED)
    return names, np.nan_to_num(mi).astype(np.float32)



def rl_inspired_logistic_subset_policy(feat_names, scores, eps=RL_POLICY_EPISODES, lr=RL_POLICY_LR, min_pct=0.30, max_pct=0.90):
    n = len(feat_names)
    if n == 0:
        return feat_names
    if n <= 3:
        return feat_names
    sc = (scores - scores.mean()) / (scores.std() + 1e-6)
    logits = np.zeros(n, np.float32)
    min_k = max(1, int(min_pct * n))
    max_k = max(min_k + 1, int(max_pct * n))
    for _ in range(eps):
        p = 1.0 / (1.0 + np.exp(-logits))
        sel = np.random.rand(n) < p
        if sel.sum() == 0:
            sel[np.argmax(p)] = True
        k_sel = int(sel.sum())
        if k_sel < min_k:
            size_pen = (min_k - k_sel) / max(min_k, 1)
        elif k_sel > max_k:
            size_pen = (k_sel - max_k) / max(n - max_k, 1)
        else:
            size_pen = 0.0
        quality = float(sc[sel].mean())
        diversity = float(np.std(sc[sel])) * 0.10
        reward = quality + diversity - 0.40 * size_pen
        logits += lr * (sel.astype(np.float32) - p) * reward
    p = 1.0 / (1.0 + np.exp(-logits))
    sel_mask = p >= 0.50
    if sel_mask.sum() < min_k:
        sel_mask = np.zeros(n, bool)
        sel_mask[np.argsort(p)[::-1][:min_k]] = True
    return [feat_names[i] for i in np.where(sel_mask)[0]]



def feature_occurrence_counter(stage_records):
    counter = Counter()
    for st in stage_records:
        counter.update(st.get("selected_feature_keys", []))
    return counter



def compute_feature_universality(stage_records, min_occ=MIN_SHARED_OCCURRENCE):
    score_bank = defaultdict(list)
    for st in stage_records:
        for k, v in st.get("global_relevance_map", {}).items():
            score_bank[k].append(float(v))
    raw_var = {}
    occ = {k: len(v) for k, v in score_bank.items()}
    for k, vals in score_bank.items():
        if len(vals) >= 2:
            raw_var[k] = float(np.var(vals))
        else:
            raw_var[k] = 1.0
    vmax = max(raw_var.values()) if raw_var else 1.0
    uni = {}
    for k, v in raw_var.items():
        base = 1.0 - (v / (vmax + 1e-6))
        if occ.get(k, 0) < min_occ:
            base *= 0.35
        uni[k] = float(np.clip(base, 0.0, 1.0))
    return uni, occ


# ==================================================================================================
# CLIENT COUNT / CLIENT PARTITION
# ==================================================================================================
def entropy_adaptive_client_ratio(ytr):
    y_shift = ytr.astype(int) - ytr.astype(int).min()
    n_cls = len(np.unique(y_shift))
    counts = np.bincount(y_shift, minlength=n_cls).astype(float)
    probs = counts / (counts.sum() + 1e-9)
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    return float(np.clip(entropy / (np.log(n_cls + 1e-9) + 1e-9), 0.0, 1.0))



def choose_num_clients(n, ytr):
    ratio = entropy_adaptive_client_ratio(ytr)
    if n <= 30000:
        lo, hi = 1, 4
    elif n <= 300000:
        lo, hi = 8, 12
    else:
        lo, hi = 13, 100
    return int(np.clip(round(lo + ratio * (hi - lo)), lo, hi))



def make_client_indices(ytr, n_clients, alpha=CLIENT_DIRICHLET_ALPHA, seed=SEED):
    rng = np.random.RandomState(seed)
    ys = np.asarray(ytr)
    classes = np.unique(ys)
    client_bins = [[] for _ in range(n_clients)]
    for cls in classes:
        idx = np.where(ys == cls)[0]
        rng.shuffle(idx)
        if len(idx) < n_clients:
            chunks = np.array_split(idx, n_clients)
        else:
            props = rng.dirichlet(np.ones(n_clients) * alpha)
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            chunks = np.split(idx, cuts)
        for cid, ch in enumerate(chunks):
            client_bins[cid].extend(ch.tolist())

    empties = [i for i, arr in enumerate(client_bins) if len(arr) == 0]
    for e in empties:
        donors = sorted(range(n_clients), key=lambda k: len(client_bins[k]), reverse=True)
        for d in donors:
            if len(client_bins[d]) > 1:
                moved = client_bins[d].pop()
                client_bins[e].append(moved)
                break
    out = []
    for arr in client_bins:
        arr = np.array(arr, dtype=np.int64)
        rng.shuffle(arr)
        out.append(arr)
    used = np.concatenate(out) if out else np.array([], dtype=np.int64)
    if len(np.unique(used)) != len(used):
        raise RuntimeError("client partition repair produced duplicates")
    return out


# ==================================================================================================
# ROUTED VOCAB / DATASETS
# ==================================================================================================
class Vocab:
    def __init__(self):
        self.maps = {}
        self.sizes = {}

    def fit(self, X, cat_cols):
        for c in cat_cols:
            uniq = pd.unique(X[c].astype(str).fillna("missing").to_numpy())
            self.maps[c] = {v: i for i, v in enumerate(uniq)}
            self.sizes[c] = len(uniq) + 1

    def transform(self, X, cat_cols):
        if not cat_cols:
            return np.zeros((len(X), 0), np.int64)
        out = np.zeros((len(X), len(cat_cols)), np.int64)
        for j, c in enumerate(cat_cols):
            m = self.maps[c]
            oov = self.sizes[c] - 1
            out[:, j] = [m.get(v, oov) for v in X[c].astype(str).fillna("missing")]
        return out


class RouteTabDataset(Dataset):
    def __init__(self, Xsn, Xsc, Xpn, Xpc, y):
        self.Xsn = torch.tensor(Xsn, dtype=torch.float32)
        self.Xsc = torch.tensor(Xsc, dtype=torch.long)
        self.Xpn = torch.tensor(Xpn, dtype=torch.float32)
        self.Xpc = torch.tensor(Xpc, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.Xsn[idx], self.Xsc[idx], self.Xpn[idx], self.Xpc[idx], self.y[idx]

    def subset(self, idx):
        idx = np.asarray(idx)
        return RouteTabDataset(
            self.Xsn[idx].cpu().numpy(),
            self.Xsc[idx].cpu().numpy(),
            self.Xpn[idx].cpu().numpy(),
            self.Xpc[idx].cpu().numpy(),
            self.y[idx].cpu().numpy(),
        )



def mk_loader(ds, shuffle, batch_size=256):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        drop_last=False,
    )


# ==================================================================================================
# DIFFUSION-ENHANCED PERSONALIZED FEDERATED MODEL
# ==================================================================================================
class SharedRouteTokenizer(nn.Module):
    def __init__(self, n_shared_feature_ids, hash_buckets=SHARED_HASH_BUCKETS, d_model=192, dropout=0.10):
        super().__init__()
        self.n_shared_feature_ids = max(int(n_shared_feature_ids), 1)
        self.hash_buckets = int(hash_buckets)
        self.num_w = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.num_w, std=0.02)
        self.num_b = nn.Parameter(torch.zeros(d_model))
        self.feature_emb = nn.Embedding(self.n_shared_feature_ids, d_model)
        self.hash_emb = nn.Embedding(self.hash_buckets, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.drop = nn.Dropout(dropout)

    def forward(self, xsn, xsc, num_ids, cat_ids):
        B = xsn.size(0) if xsn.ndim == 2 else xsc.size(0)
        toks = [self.cls.expand(B, -1, -1)]
        if num_ids.numel() > 0:
            fid = self.feature_emb(num_ids).unsqueeze(0)
            num_tok = xsn.unsqueeze(-1) * self.num_w.view(1, 1, -1) + self.num_b.view(1, 1, -1) + fid
            toks.append(num_tok)
        if cat_ids.numel() > 0:
            fid = self.feature_emb(cat_ids).unsqueeze(0)
            cat_tok = self.hash_emb(xsc) + fid
            toks.append(cat_tok)
        return self.drop(torch.cat(toks, dim=1))


class RoutePrivateTokenizer(nn.Module):
    def __init__(self, n_num, cards, d_model=192, dropout=0.10):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cards)
        if n_num > 0:
            self.num_w = nn.Parameter(torch.empty(n_num, d_model))
            nn.init.kaiming_uniform_(self.num_w, a=math.sqrt(5))
            self.num_b = nn.Parameter(torch.zeros(n_num, d_model))
        else:
            self.num_w = None
            self.num_b = None
        self.cat_embs = nn.ModuleList([nn.Embedding(c, d_model) for c in cards]) if cards else nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, xpn, xpc):
        B = xpn.size(0) if xpn.ndim == 2 else xpc.size(0)
        toks = []
        if self.n_num > 0:
            toks.append(xpn.unsqueeze(-1) * self.num_w + self.num_b)
        if self.n_cat > 0:
            toks.append(torch.cat([emb(xpc[:, j]).unsqueeze(1) for j, emb in enumerate(self.cat_embs)], dim=1))
        if toks:
            return self.drop(torch.cat(toks, dim=1))
        return torch.zeros(B, 0, self.d_model, device=xpn.device)


class SharedBlock(nn.Module):
    def __init__(self, d=192, heads=8, ff=384, drop=0.10):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.drop1 = nn.Dropout(drop)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ff),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(ff, d),
        )
        self.drop2 = nn.Dropout(drop)
        self.gate = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())

    def forward(self, x):
        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + self.drop1(h)
        h = self.ffn(self.ln2(x))
        return x + self.drop2(h * self.gate(x))


class SharedBackbone(nn.Module):
    def __init__(self, d=192, blocks=3, heads=8, ff=384, drop=0.10):
        super().__init__()
        self.blocks = nn.ModuleList([SharedBlock(d, heads, ff, drop) for _ in range(blocks)])
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens):
        x = tokens
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x[:, 0])


class PrivateHead(nn.Module):
    def __init__(self, d=192, n_classes=2, drop=0.10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d, n_classes),
        )

    def forward(self, z):
        return self.mlp(z)


class DiffusionRegularizer(nn.Module):
    def __init__(self, d_model=192, n_steps=DIFFUSION_STEPS, hidden=256):
        super().__init__()
        self.d_model = d_model
        self.n_steps = n_steps
        beta = torch.linspace(1e-4, 2e-2, n_steps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("alpha_bar", alpha_bar)
        self.t_emb = nn.Embedding(n_steps, d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def loss(self, z):
        if z.numel() == 0:
            return z.sum() * 0.0
        B = z.size(0)
        t = torch.randint(0, self.n_steps, (B,), device=z.device)
        abar = self.alpha_bar[t].unsqueeze(1)
        noise = torch.randn_like(z)
        zt = torch.sqrt(abar) * z + torch.sqrt(1.0 - abar) * noise
        tvec = self.t_emb(t)
        pred = self.net(torch.cat([zt, tvec], dim=1))
        return F.mse_loss(pred, noise)


class GRIPDFFIModel(nn.Module):
    def __init__(self, meta, n_shared_feature_ids, d_model=192, n_blocks=3, n_heads=8, ff=384, drop=0.10):
        super().__init__()
        self.shared_tokenizer = SharedRouteTokenizer(n_shared_feature_ids, hash_buckets=SHARED_HASH_BUCKETS, d_model=d_model, dropout=drop)
        self.private_tokenizer = RoutePrivateTokenizer(len(meta["private_num_cols"]), meta["cards"], d_model=d_model, dropout=drop)
        self.backbone = SharedBackbone(d=d_model, blocks=n_blocks, heads=n_heads, ff=ff, drop=drop)
        self.head = PrivateHead(d=d_model, n_classes=meta["n_classes"], drop=drop)
        self.diffusion = DiffusionRegularizer(d_model=d_model, n_steps=DIFFUSION_STEPS) if ENABLE_DIFFUSION_SHARED_BACKBONE else None
        self.register_buffer("shared_num_ids", torch.tensor(meta["shared_num_global_ids"], dtype=torch.long))
        self.register_buffer("shared_cat_ids", torch.tensor(meta["shared_cat_global_ids"], dtype=torch.long))

    def forward(self, xsn, xsc, xpn, xpc):
        stoks = self.shared_tokenizer(xsn, xsc, self.shared_num_ids, self.shared_cat_ids)
        ptoks = self.private_tokenizer(xpn, xpc)
        toks = stoks if ptoks.size(1) == 0 else torch.cat([stoks, ptoks], dim=1)
        z = self.backbone(toks)
        logits = self.head(z)
        aux = {"latent": z}
        return logits, aux



def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}



def load_state(module, state):
    module.load_state_dict(state, strict=True)



def average_state_dicts(states, weights):
    out = {}
    for k in states[0]:
        out[k] = sum(weights[i] * states[i][k] for i in range(len(states)))
    return out



def _prefix_state(prefix, module_state):
    return {f"{prefix}.{k}": v for k, v in module_state.items()}



def _unprefix_state(prefix, full_state):
    plen = len(prefix) + 1
    return {k[plen:]: v for k, v in full_state.items() if k.startswith(prefix + ".")}



def get_shared_state(model):
    st = {}
    st.update(_prefix_state("shared_tokenizer", cpu_state(model.shared_tokenizer)))
    st.update(_prefix_state("backbone", cpu_state(model.backbone)))
    if model.diffusion is not None:
        st.update(_prefix_state("diffusion", cpu_state(model.diffusion)))
    return st



def load_shared_state(model, state):
    load_state(model.shared_tokenizer, _unprefix_state("shared_tokenizer", state))
    load_state(model.backbone, _unprefix_state("backbone", state))
    if model.diffusion is not None and any(k.startswith("diffusion.") for k in state):
        load_state(model.diffusion, _unprefix_state("diffusion", state))



def get_private_state(model):
    st = {}
    st.update(_prefix_state("private_tokenizer", cpu_state(model.private_tokenizer)))
    st.update(_prefix_state("head", cpu_state(model.head)))
    return st



def load_private_state(model, pstate):
    load_state(model.private_tokenizer, _unprefix_state("private_tokenizer", pstate))
    load_state(model.head, _unprefix_state("head", pstate))

# ==================================================================================================
# METRICS
# ==================================================================================================
def safe_auc(y_true, prob):
    K = prob.shape[1]
    present = set(np.unique(y_true))
    try:
        if K == 2:
            return float(roc_auc_score(y_true, prob[:, 1]))
        Y = label_binarize(y_true, classes=np.arange(K))
        pk = [k for k in range(K) if k in present]
        if len(pk) < 2:
            raise ValueError
        return float(roc_auc_score(Y[:, pk], prob[:, pk], average="macro", multi_class="ovr"))
    except Exception:
        scores = []
        for k in range(K):
            if k not in present:
                continue
            try:
                scores.append(roc_auc_score((y_true == k).astype(int), prob[:, k]))
            except Exception:
                pass
        return float(np.mean(scores)) if scores else 0.5


def ppv_npv_stats(y_true, y_pred, n_classes):
    """
    Zero-safe PPV/NPV metrics.
    - For binary tasks: ppv_positive / npv_negative use the standard confusion-matrix formulas.
    - For multiclass tasks: ppv_positive / npv_negative are populated with weighted one-vs-rest surrogates
      so report tables do not contain NaN for those columns.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    support = cm.sum(axis=1).astype(float)
    total = cm.sum().astype(float)
    ppv_list, npv_list = [], []
    ppv_valid, npv_valid = [], []
    for k in range(n_classes):
        tp = float(cm[k, k])
        fp = float(cm[:, k].sum() - tp)
        fn = float(cm[k, :].sum() - tp)
        tn = float(total - tp - fp - fn)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        ppv_list.append(ppv)
        npv_list.append(npv)
        ppv_valid.append((ppv, support[k]))
        npv_valid.append((npv, support[k]))
    ppv_macro = float(np.mean(ppv_list)) if len(ppv_list) else 0.0
    npv_macro = float(np.mean(npv_list)) if len(npv_list) else 0.0
    ppv_weighted = float(sum(v * w for v, w in ppv_valid) / max(sum(w for _, w in ppv_valid), 1.0)) if ppv_valid else 0.0
    npv_weighted = float(sum(v * w for v, w in npv_valid) / max(sum(w for _, w in npv_valid), 1.0)) if npv_valid else 0.0
    out = {
        "ppv_macro_ovr": ppv_macro,
        "npv_macro_ovr": npv_macro,
        "ppv_weighted_ovr": ppv_weighted,
        "npv_weighted_ovr": npv_weighted,
    }
    if n_classes == 2:
        tp = float(cm[1, 1]); fp = float(cm[0, 1]); fn = float(cm[1, 0]); tn = float(cm[0, 0])
        out["ppv_positive"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        out["npv_negative"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    else:
        out["ppv_positive"] = ppv_weighted
        out["npv_negative"] = npv_weighted
    return out


def compute_metrics(y_true, prob):
    pred = prob.argmax(1)
    acc = accuracy_score(y_true, pred)
    pr_m, rc_m, f1_m, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
    pr_w, rc_w, f1_w, _ = precision_recall_fscore_support(y_true, pred, average="weighted", zero_division=0)
    try:
        ll = log_loss(y_true, prob, labels=np.arange(prob.shape[1]))
    except Exception:
        ll = np.nan
    try:
        mcc = matthews_corrcoef(y_true, pred)
    except Exception:
        mcc = np.nan
    try:
        kappa = cohen_kappa_score(y_true, pred)
    except Exception:
        kappa = np.nan

    auc_macro = safe_auc(y_true, prob)
    K = prob.shape[1]
    try:
        if K == 2:
            auc_micro = float(roc_auc_score(y_true, prob[:, 1]))
            auc_weighted = float(roc_auc_score(y_true, prob[:, 1]))
            pr_auc = float(average_precision_score(y_true, prob[:, 1]))
        else:
            Y = label_binarize(y_true, classes=np.arange(K))
            auc_micro = float(roc_auc_score(Y, prob, average="micro", multi_class="ovr"))
            auc_weighted = float(roc_auc_score(Y, prob, average="weighted", multi_class="ovr"))
            pr_auc = float(average_precision_score(Y, prob, average="macro"))
    except Exception:
        auc_micro = np.nan
        auc_weighted = np.nan
        pr_auc = np.nan

    out = {
        "acc": float(acc),
        "precision_macro": float(pr_m),
        "recall_macro": float(rc_m),
        "f1_macro": float(f1_m),
        "precision_weighted": float(pr_w),
        "recall_weighted": float(rc_w),
        "f1_weighted": float(f1_w),
        "logloss": float(ll) if ll == ll else np.nan,
        "mcc": float(mcc) if mcc == mcc else np.nan,
        "kappa": float(kappa) if kappa == kappa else np.nan,
        "auc_roc_macro_ovr": float(auc_macro),
        "auc_roc_micro_ovr": float(auc_micro) if auc_micro == auc_micro else np.nan,
        "auc_roc_weighted_ovr": float(auc_weighted) if auc_weighted == auc_weighted else np.nan,
        "pr_auc_macro": float(pr_auc) if pr_auc == pr_auc else np.nan,
    }
    extra = ppv_npv_stats(y_true, pred, K)
    for k, v in extra.items():
        out[k] = float(v) if v == v else np.nan
    return out


def binary_view_from_multiclass(y_true, prob, normal_index):
    if normal_index is None:
        return None, None
    y_bin = (y_true != normal_index).astype(int)
    p_attack = 1.0 - prob[:, normal_index]
    p_bin = np.stack([1.0 - p_attack, p_attack], axis=1)
    return y_bin.astype(int), p_bin.astype(np.float32)


def collect_roc_data(y_true, prob, ds_name, label_names=None):
    K = prob.shape[1]
    if label_names is None:
        label_names = [str(i) for i in range(K)]
    data = {"ds_name": ds_name, "n_classes": K, "label_names": [str(x) for x in label_names], "curves": {}}
    for k in range(K):
        binary = (y_true == k).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            continue
        try:
            fpr, tpr, _ = roc_curve(binary, prob[:, k])
            auc_v = roc_auc_score(binary, prob[:, k])
        except Exception:
            fpr, tpr, auc_v = np.array([0, 1]), np.array([0, 1]), 0.5
        data["curves"][str(k)] = {
            "label": str(label_names[k]),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(auc_v),
        }
    return data


def collect_pr_data(y_true, prob, ds_name, positive_label="Attack"):
    if prob.shape[1] != 2:
        return None
    yb = y_true.astype(int)
    p1 = prob[:, 1]
    prec, rec, _ = precision_recall_curve(yb, p1)
    ap = average_precision_score(yb, p1)
    return {
        "ds_name": ds_name,
        "positive_label": positive_label,
        "precision": prec.tolist(),
        "recall": rec.tolist(),
        "ap": float(ap),
    }


def calibration_curve_binary(y_true_bin, p_attack, n_bins=10):
    y_true_bin = np.asarray(y_true_bin).astype(int)
    p_attack = np.asarray(p_attack).astype(float)
    p_attack = np.clip(p_attack, 1e-7, 1 - 1e-7)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p_attack, bins[1:-1], right=False)
    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": b, "bin_left": float(bins[b]), "bin_right": float(bins[b+1]), "count": 0, "mean_conf": np.nan, "emp_acc": np.nan, "gap": np.nan})
            continue
        conf = float(p_attack[mask].mean())
        acc = float(y_true_bin[mask].mean())
        rows.append({"bin": b, "bin_left": float(bins[b]), "bin_right": float(bins[b+1]), "count": n, "mean_conf": conf, "emp_acc": acc, "gap": abs(conf - acc)})
    return pd.DataFrame(rows)


def expected_calibration_error(y_true_bin, p_attack, n_bins=10):
    tab = calibration_curve_binary(y_true_bin, p_attack, n_bins=n_bins)
    total = max(tab["count"].sum(), 1)
    ece = float(np.nansum((tab["count"] / total) * tab["gap"].fillna(0.0)))
    return ece, tab


def brier_score_binary(y_true_bin, p_attack):
    y_true_bin = np.asarray(y_true_bin).astype(float)
    p_attack = np.asarray(p_attack).astype(float)
    return float(np.mean((p_attack - y_true_bin) ** 2))


def build_error_table(preds_dict, meta_list, split_name):
    rows = []
    meta_map = {m["name"]: m for m in meta_list}
    for ds_name, payload in preds_dict.items():
        meta = meta_map[ds_name]
        y = payload["y"]
        p = payload["p"]
        pred = p.argmax(1)
        conf = p.max(1)
        part = np.sort(p, axis=1)
        margin = part[:, -1] - part[:, -2] if p.shape[1] >= 2 else np.ones(len(y))
        entropy = -np.sum(np.clip(p, 1e-12, 1.0) * np.log(np.clip(p, 1e-12, 1.0)), axis=1)
        for i in range(len(y)):
            rows.append({
                "split": split_name,
                "dataset": ds_name,
                "true_idx": int(y[i]),
                "pred_idx": int(pred[i]),
                "true_label": str(meta["class_names"][int(y[i])]),
                "pred_label": str(meta["class_names"][int(pred[i])]),
                "correct": int(pred[i] == y[i]),
                "confidence": float(conf[i]),
                "margin": float(margin[i]),
                "entropy": float(entropy[i]),
                "dataset_class_key": f"{ds_name}::{meta['class_names'][int(y[i])]}",
            })
    return pd.DataFrame(rows)


# ==================================================================================================
# TRAIN / PREDICT
# ==================================================================================================
def train_client(model, loader, epochs=1, lr=1e-3, diffusion_weight=DIFFUSION_LOSS_WEIGHT):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        for xsn, xsc, xpn, xpc, y in loader:
            xsn = xsn.to(DEVICE, non_blocking=True)
            xsc = xsc.to(DEVICE, non_blocking=True)
            xpn = xpn.to(DEVICE, non_blocking=True)
            xpc = xpc.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits, aux = model(xsn, xsc, xpn, xpc)
            loss = F.cross_entropy(logits, y)
            if getattr(model, "diffusion", None) is not None and diffusion_weight > 0:
                loss = loss + diffusion_weight * model.diffusion.loss(aux["latent"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()



def predict(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xsn, xsc, xpn, xpc, y in loader:
            xsn = xsn.to(DEVICE, non_blocking=True)
            xsc = xsc.to(DEVICE, non_blocking=True)
            xpn = xpn.to(DEVICE, non_blocking=True)
            xpc = xpc.to(DEVICE, non_blocking=True)
            logits, _ = model(xsn, xsc, xpn, xpc)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            ys.append(y.numpy())
            ps.append(prob)
    return np.concatenate(ys), np.concatenate(ps)


# ==================================================================================================
# PRELIMINARY DATASET PREPARATION FOR FEDERATED FEATURE INTELLIGENCE
# ==================================================================================================
def prepare_dataset_for_feature_intelligence(spec, mi_pre=160):
    files_all = list_table_files(spec["path"])
    if not files_all:
        raise FileNotFoundError(f"{spec['name']}: no files found under {spec['path']}")

    selected_files = choose_files_for_spec(spec, files_all)
    groups = categorize_split_files(selected_files)
    has_named_splits = len(groups["train"]) > 0 or len(groups["test"]) > 0 or len(groups["val"]) > 0
    force_cols = UNSW_COLS if spec.get("unsw_nb15_raw", False) else None

    print("\n" + "=" * 100)
    print(f"[{spec['name'].upper()}] DATASET BUILD")
    print("-" * 100)
    print(f"  path              : {spec['path']}")
    print(f"  files_found       : {len(files_all)}")
    print(f"  files_selected    : {[os.path.basename(f) for f in selected_files]}")
    print(f"  split_hint        : {'named train/val/test files detected' if has_named_splits else 'no named split files'}")
    print(f"  split_policy      : {SPLIT_POLICY}")
    if not PRESERVE_OFFICIAL_BENCHMARK_SPLITS:
        print("  benchmark_note    : provided split files are treated as labeled sources, then re-split fresh")

    df_train = combine_tables(groups["train"], force_cols=force_cols) if groups["train"] else None
    df_val = combine_tables(groups["val"], force_cols=force_cols) if groups["val"] else None
    df_test = combine_tables(groups["test"], force_cols=force_cols) if groups["test"] else None
    df_other = combine_tables(groups["other"], force_cols=force_cols) if groups["other"] else None

    merged_source = None
    target_col = None
    task_kind = None
    source_parts_used = []

    if df_train is not None:
        target_col, task_kind = detect_target_column(df_train, spec)
        print(f"  target_col        : {target_col}")
        print(f"  task_kind         : {task_kind}")
        merge_parts = []
        if target_col in df_train.columns:
            merge_parts.append(("train", df_train))
        if df_test is not None and target_col in df_test.columns:
            if target_compatible(df_train[target_col], df_test[target_col], spec):
                print("  [INFO] test target is compatible with train target -> merge train+test")
                merge_parts.append(("test", df_test))
            else:
                print("  [INFO] test target incompatible -> ignore test for fresh split")
        if df_val is not None and target_col in df_val.columns:
            if target_compatible(df_train[target_col], df_val[target_col], spec):
                print("  [INFO] val target is compatible with train target -> merge val too")
                merge_parts.append(("val", df_val))
            else:
                print("  [INFO] val target incompatible -> ignore val for fresh split")
        if merge_parts:
            source_parts_used = [n for n, _ in merge_parts]
            merged_source = pd.concat([d for _, d in merge_parts], axis=0, ignore_index=True)
            print(f"  split_mode        : fresh-70/15/15 from merged labeled pieces {source_parts_used}")
        elif df_other is not None:
            merged_source = df_other
            source_parts_used = ["other"]
            target_col, task_kind = detect_target_column(merged_source, spec)
            print(f"  target_col        : {target_col}")
            print(f"  task_kind         : {task_kind}")
            print("  split_mode        : fresh-70/15/15 from unlabeled-split fallback table set")
        else:
            raise RuntimeError(f"{spec['name']}: no usable labeled table found")
    else:
        merged_source = combine_tables(selected_files, force_cols=force_cols)
        if merged_source is None:
            raise RuntimeError(f"{spec['name']}: failed to load selected files")
        source_parts_used = ["combined_tables"]
        target_col, task_kind = detect_target_column(merged_source, spec)
        print(f"  target_col        : {target_col}")
        print(f"  task_kind         : {task_kind}")
        print("  split_mode        : fresh-70/15/15 from combined tables")

    def prep_df_target(df):
        df = df.dropna(subset=[target_col]).reset_index(drop=True)
        y_raw = clean_target_series(df[target_col], spec).reset_index(drop=True)
        return df.reset_index(drop=True), y_raw

    df_full, y_full_raw = prep_df_target(merged_source)
    df_train, y_train_raw, df_val, y_val_raw, df_test, y_test_raw = split_three_way(df_full, y_full_raw, seed=SEED)

    Xtr = df_train.drop(columns=[target_col], errors="ignore").copy()
    Xva = df_val.drop(columns=[target_col], errors="ignore").copy()
    Xte = df_test.drop(columns=[target_col], errors="ignore").copy()

    sibling_drop_cols = find_target_sibling_columns(Xtr, y_train_raw, target_col, spec)
    Xtr = drop_target_like_columns(Xtr, keep_target=target_col, drop_cols=sibling_drop_cols)
    Xva = drop_target_like_columns(Xva, keep_target=target_col, drop_cols=sibling_drop_cols)
    Xte = drop_target_like_columns(Xte, keep_target=target_col, drop_cols=sibling_drop_cols)

    Xtr = drop_id_cols(Xtr)
    keep_cols = Xtr.columns.tolist()
    for X in [Xva, Xte]:
        for c in keep_cols:
            if c not in X.columns:
                X[c] = np.nan
    Xva = Xva[keep_cols].copy()
    Xte = Xte[keep_cols].copy()

    label_map, class_names = fit_label_mapping(y_train_raw)
    ytr, keep_tr = apply_label_mapping(y_train_raw, label_map)
    Xtr = Xtr.loc[keep_tr].reset_index(drop=True)

    yva, keep_va = apply_label_mapping(y_val_raw, label_map)
    dropped_val_unseen = int((~keep_va).sum())
    Xva = Xva.loc[keep_va].reset_index(drop=True)

    yte, keep_te = apply_label_mapping(y_test_raw, label_map)
    dropped_test_unseen = int((~keep_te).sum())
    Xte = Xte.loc[keep_te].reset_index(drop=True)

    class_names = np.array(class_names)
    normal_index = detect_normal_index(class_names, spec)
    n_classes = len(class_names)

    print(f"  train/val/test    : {len(Xtr)} / {len(Xva)} / {len(Xte)}")
    print(f"  n_classes         : {n_classes}")
    print(f"  class_names       : {class_names.tolist()}")
    print(f"  normal_class_idx  : {normal_index}")

    num_cols, cat_cols = infer_types(Xtr)
    means, stds = fit_num_stats(Xtr, num_cols)
    Xtr, ni_tr, ci_tr = apply_num_cat_preproc(Xtr, num_cols, cat_cols, means, stds)
    Xva, ni_va, ci_va = apply_num_cat_preproc(Xva, num_cols, cat_cols, means, stds)
    Xte, ni_te, ci_te = apply_num_cat_preproc(Xte, num_cols, cat_cols, means, stds)
    print(f"  initial_features  : {len(keep_cols)} | numeric={len(num_cols)} | categorical={len(cat_cols)}")
    print(f"  impute_counts     : train(n={ni_tr}, c={ci_tr}) | val(n={ni_va}, c={ci_va}) | test(n={ni_te}, c={ci_te})")

    plan = cross_plan(Xtr, ytr, cat_cols)
    Xtr, new_crosses = apply_crosses(Xtr, plan)
    Xva, _ = apply_crosses(Xva, plan)
    Xte, _ = apply_crosses(Xte, plan)
    print(f"  cross_features    : use_triples={plan['use_triples']} | base={plan['base_cols']} | new={len(new_crosses)}")

    num2, cat2 = infer_types(Xtr)
    feat_names, mi_scores = compute_mi(Xtr, ytr, num2, cat2)
    mi_pre_dyn = max(mi_pre, int(0.80 * len(feat_names))) if feat_names else 0
    if len(feat_names) > mi_pre_dyn > 0:
        ord_ = np.argsort(mi_scores)[::-1][:mi_pre_dyn]
        feat_names = [feat_names[i] for i in ord_]
        mi_scores = mi_scores[ord_]
    print(f"  mi_candidates     : {len(feat_names)}")

    fn_sel = [f for f in feat_names if f in Xtr.columns]
    if len(fn_sel) == 0:
        fn_sel = list(Xtr.columns)
        mi_scores = np.ones(len(fn_sel), np.float32)
        print(f"  [WARN] MI overlap empty; using all {len(fn_sel)} columns")

    nc_sel = [c for c in num2 if c in fn_sel]
    cc_sel = [c for c in cat2 if c in fn_sel]
    Xnode, A = build_feature_graph(Xtr[fn_sel], ytr, fn_sel, nc_sel, cc_sel)

    feat_keys = [feature_key(f, "num" if f in nc_sel else "cat") for f in fn_sel]
    feat_type_map = {k: {"name": feature_name_from_key(k), "kind": feature_kind_from_key(k)} for k in feat_keys}
    feat_key_to_mi = {feat_keys[i]: float(mi_scores[i]) for i in range(len(feat_keys))}

    print(f"  graph_nodes       : {len(fn_sel)}")
    print(f"  semantic_protocol : target-resolution + compatibility-gated merge + leakage-safe re-split")
    print("=" * 100)

    stage = {
        "spec": spec,
        "target_col": target_col,
        "task_kind": task_kind,
        "source_parts_used": source_parts_used,
        "sibling_drop_cols": sibling_drop_cols,
        "class_names": class_names.tolist(),
        "normal_index": None if normal_index is None else int(normal_index),
        "n_classes": int(n_classes),
        "label_map": label_map,
        "Xtr": Xtr,
        "Xva": Xva,
        "Xte": Xte,
        "ytr": ytr,
        "yva": yva,
        "yte": yte,
        "feat_names": fn_sel,
        "feat_keys": feat_keys,
        "feat_type_map": feat_type_map,
        "local_mi_scores": np.asarray([feat_key_to_mi[k] for k in feat_keys], dtype=np.float32),
        "Xnode": Xnode,
        "A": A,
        "num_cols_post": num2,
        "cat_cols_post": cat2,
        "dropped_unseen_labels": {"val": int(dropped_val_unseen), "test": int(dropped_test_unseen)},
        "class_dist_train": pd.Series(ytr).value_counts().sort_index().to_dict(),
        "class_dist_val": pd.Series(yva).value_counts().sort_index().to_dict(),
        "class_dist_test": pd.Series(yte).value_counts().sort_index().to_dict(),
        "n_train": int(len(ytr)),
        "cross_plan": plan,
        "new_crosses": new_crosses,
    }
    return stage


# ==================================================================================================
# FEDERATED FEATURE INTELLIGENCE
# ==================================================================================================
def run_federated_feature_intelligence(stage_records):
    print("\n" + "#" * 100)
    print(f"{PROCESS_NAME} FEDERATED FEATURE INTELLIGENCE")
    print("#" * 100)
    global_state = None
    weights = [st["n_train"] for st in stage_records]
    weights = [w / float(sum(weights)) for w in weights]
    for rnd in range(1, FEATURE_INTEL_ROUNDS + 1):
        print("\n" + "=" * 100)
        print(f"FEATURE INTELLIGENCE ROUND {rnd}/{FEATURE_INTEL_ROUNDS}")
        print("=" * 100)
        local_states = []
        for st in stage_records:
            lstate, lscores = train_graph_refined_relevance_net(st["Xnode"], st["A"], init_state=global_state)
            local_states.append(lstate)
            if len(lscores):
                print(f"  [{st['spec']['name']}] mean_local_relevance={float(np.mean(lscores)):.4f} | nodes={len(lscores)}")
            else:
                print(f"  [{st['spec']['name']}] mean_local_relevance=NA | nodes=0")
        global_state = average_state_dicts(local_states, weights)

    for st in stage_records:
        gscores = score_graph_refined_relevance_net(st["Xnode"], st["A"], global_state)
        mi_scores = st["local_mi_scores"]
        min_len = min(len(st["feat_keys"]), len(mi_scores), len(gscores)) if len(gscores) > 0 else len(st["feat_keys"])
        feat_keys = st["feat_keys"][:min_len]
        feat_names = [feature_name_from_key(k) for k in feat_keys]
        mi_scores_ = np.asarray(mi_scores[:min_len], np.float32) if len(mi_scores) > 0 else np.ones(min_len, np.float32)
        gscores_ = np.asarray(gscores[:min_len], np.float32) if len(gscores) > 0 else np.ones(min_len, np.float32)
        if min_len == 0:
            selected = list(st["Xtr"].columns)
            selected_keys = [feature_key(c, "num" if c in st["num_cols_post"] else "cat") for c in selected]
            combined = np.ones(len(selected), np.float32)
        else:
            combined = FEATURE_INTEL_MI_WEIGHT * safe_norm(mi_scores_) + FEATURE_INTEL_FRN_WEIGHT * safe_norm(gscores_)
            selected = rl_inspired_logistic_subset_policy(feat_names, combined)
            if len(feat_names) <= 20:
                min_keep = min(len(feat_names), max(8, int(math.ceil(0.70 * len(feat_names)))))
                if len(selected) < min_keep:
                    rank_idx = np.argsort(combined)[::-1]
                    ranked_feats = [feat_names[i] for i in rank_idx]
                    merged = list(dict.fromkeys(list(selected) + ranked_feats))
                    selected = merged[:min_keep]
            selected_keys = [k for k in feat_keys if feature_name_from_key(k) in set(selected)]
        st["global_relevance_scores"] = gscores_
        st["combined_scores"] = combined if len(feat_keys) else np.array([], np.float32)
        st["selected_features"] = selected
        st["selected_feature_keys"] = selected_keys
        st["global_relevance_map"] = {feat_keys[i]: float(gscores_[i]) for i in range(len(feat_keys))}
        print(f"  [{st['spec']['name']}] selected_features={len(selected)} / {max(len(feat_names), 1)}")
        print(f"  [{st['spec']['name']}] selected_top10={selected[:10]}{'...' if len(selected) > 10 else ''}")

    universality, occurrence = compute_feature_universality(stage_records, min_occ=MIN_SHARED_OCCURRENCE)
    shared_key_set = {k for k, u in universality.items() if u >= UNIVERSALITY_THRESHOLD and occurrence.get(k, 0) >= MIN_SHARED_OCCURRENCE}
    shared_keys_global = sorted(shared_key_set)
    shared_key_to_id = {k: i for i, k in enumerate(shared_keys_global)}

    print("\n" + "-" * 100)
    print(f"GLOBAL SHARED FEATURE KEYS : {len(shared_keys_global)}")
    print(f"UNIVERSALITY THRESHOLD     : {UNIVERSALITY_THRESHOLD:.2f}")
    print("-" * 100)
    return global_state, universality, occurrence, shared_keys_global, shared_key_to_id


# ==================================================================================================
# FINAL DATASET MATERIALIZATION AFTER FEDERATED FEATURE INTELLIGENCE
# ==================================================================================================
def finalize_stage_record(stage, universality, occurrence, shared_key_to_id):
    Xtr = stage["Xtr"].copy()
    Xva = stage["Xva"].copy()
    Xte = stage["Xte"].copy()
    ytr = stage["ytr"]
    yva = stage["yva"]
    yte = stage["yte"]

    selected_keys = stage["selected_feature_keys"]
    if not selected_keys:
        selected_keys = [feature_key(c, "num" if c in stage["num_cols_post"] else "cat") for c in Xtr.columns]
    selected_cols = [feature_name_from_key(k) for k in selected_keys]

    shared_keys = [k for k in selected_keys if universality.get(k, 0.0) >= UNIVERSALITY_THRESHOLD and occurrence.get(k, 0) >= MIN_SHARED_OCCURRENCE]
    private_keys = [k for k in selected_keys if k not in set(shared_keys)]

    shared_num_cols = [feature_name_from_key(k) for k in shared_keys if feature_kind_from_key(k) == "num"]
    shared_cat_cols = [feature_name_from_key(k) for k in shared_keys if feature_kind_from_key(k) == "cat"]
    private_num_cols = [feature_name_from_key(k) for k in private_keys if feature_kind_from_key(k) == "num"]
    private_cat_cols = [feature_name_from_key(k) for k in private_keys if feature_kind_from_key(k) == "cat"]

    Xtr = Xtr[selected_cols].copy()
    Xva = Xva[selected_cols].copy()
    Xte = Xte[selected_cols].copy()

    private_all_cols = private_num_cols + private_cat_cols
    if private_all_cols:
        means2, stds2 = fit_num_stats(Xtr[private_all_cols], private_num_cols)
        Xtr_private, _, _ = apply_num_cat_preproc(Xtr[private_all_cols], private_num_cols, private_cat_cols, means2, stds2)
        Xva_private, _, _ = apply_num_cat_preproc(Xva[private_all_cols], private_num_cols, private_cat_cols, means2, stds2)
        Xte_private, _, _ = apply_num_cat_preproc(Xte[private_all_cols], private_num_cols, private_cat_cols, means2, stds2)
    else:
        Xtr_private = pd.DataFrame(index=Xtr.index)
        Xva_private = pd.DataFrame(index=Xva.index)
        Xte_private = pd.DataFrame(index=Xte.index)

    shared_all_cols = shared_num_cols + shared_cat_cols
    if shared_all_cols:
        means_s, stds_s = fit_num_stats(Xtr[shared_all_cols], shared_num_cols)
        Xtr_shared, _, _ = apply_num_cat_preproc(Xtr[shared_all_cols], shared_num_cols, shared_cat_cols, means_s, stds_s)
        Xva_shared, _, _ = apply_num_cat_preproc(Xva[shared_all_cols], shared_num_cols, shared_cat_cols, means_s, stds_s)
        Xte_shared, _, _ = apply_num_cat_preproc(Xte[shared_all_cols], shared_num_cols, shared_cat_cols, means_s, stds_s)
    else:
        Xtr_shared = pd.DataFrame(index=Xtr.index)
        Xva_shared = pd.DataFrame(index=Xva.index)
        Xte_shared = pd.DataFrame(index=Xte.index)

    vocab = Vocab()
    vocab.fit(Xtr_private, private_cat_cols)
    cards = [vocab.sizes[c] for c in private_cat_cols]
    cat_vocab_sizes = {str(c): int(vocab.sizes[c]) for c in private_cat_cols}
    cat_oov_index = {str(c): int(vocab.sizes[c] - 1) for c in private_cat_cols}
    cat_vocab_preview = {}
    for c in private_cat_cols:
        items = list(vocab.maps.get(c, {}).items())[:8]
        cat_vocab_preview[str(c)] = [{"token": str(k), "index": int(v)} for k, v in items]

    Xtr_sn = Xtr_shared[shared_num_cols].to_numpy(np.float32) if shared_num_cols else np.zeros((len(Xtr_shared), 0), np.float32)
    Xva_sn = Xva_shared[shared_num_cols].to_numpy(np.float32) if shared_num_cols else np.zeros((len(Xva_shared), 0), np.float32)
    Xte_sn = Xte_shared[shared_num_cols].to_numpy(np.float32) if shared_num_cols else np.zeros((len(Xte_shared), 0), np.float32)

    Xtr_sc = hash_categorical_frame(Xtr_shared, shared_cat_cols, mod=SHARED_HASH_BUCKETS)
    Xva_sc = hash_categorical_frame(Xva_shared, shared_cat_cols, mod=SHARED_HASH_BUCKETS)
    Xte_sc = hash_categorical_frame(Xte_shared, shared_cat_cols, mod=SHARED_HASH_BUCKETS)

    Xtr_pn = Xtr_private[private_num_cols].to_numpy(np.float32) if private_num_cols else np.zeros((len(Xtr_private), 0), np.float32)
    Xva_pn = Xva_private[private_num_cols].to_numpy(np.float32) if private_num_cols else np.zeros((len(Xva_private), 0), np.float32)
    Xte_pn = Xte_private[private_num_cols].to_numpy(np.float32) if private_num_cols else np.zeros((len(Xte_private), 0), np.float32)

    Xtr_pc = vocab.transform(Xtr_private, private_cat_cols)
    Xva_pc = vocab.transform(Xva_private, private_cat_cols)
    Xte_pc = vocab.transform(Xte_private, private_cat_cols)

    tr_ds = RouteTabDataset(Xtr_sn, Xtr_sc, Xtr_pn, Xtr_pc, ytr)
    va_ds = RouteTabDataset(Xva_sn, Xva_sc, Xva_pn, Xva_pc, yva)
    te_ds = RouteTabDataset(Xte_sn, Xte_sc, Xte_pn, Xte_pc, yte)

    n_clients = choose_num_clients(len(tr_ds), ytr)
    client_idx = make_client_indices(ytr, n_clients, alpha=CLIENT_DIRICHLET_ALPHA, seed=SEED + stage["spec"]["domain_id"])
    client_sizes = [len(x) for x in client_idx]

    sequence_length = int(1 + len(shared_num_cols) + len(shared_cat_cols) + len(private_num_cols) + len(private_cat_cols))
    total_vocab_size = int(sum(cards) + (SHARED_HASH_BUCKETS if shared_cat_cols else 0))
    task_family = "binary" if int(stage["n_classes"]) == 2 else "multiclass"
    binary_view_available = bool(stage["normal_index"] is not None)
    selected_universality = {k: float(universality.get(k, 0.0)) for k in selected_keys}

    meta = {
        "name": stage["spec"]["name"],
        "domain_id": stage["spec"]["domain_id"],
        "dataset_path": stage["spec"]["path"],
        "target_col": stage["target_col"],
        "task_kind": stage["task_kind"],
        "class_names": stage["class_names"],
        "normal_index": stage["normal_index"],
        "n_classes": int(stage["n_classes"]),
        "task_family": task_family,
        "is_binary_dataset": bool(int(stage["n_classes"]) == 2),
        "binary_view_available": binary_view_available,
        "selected_features": [feature_name_from_key(k) for k in selected_keys],
        "selected_feature_keys": selected_keys,
        "num_cols": shared_num_cols + private_num_cols,
        "cat_cols": shared_cat_cols + private_cat_cols,
        "shared_num_cols": shared_num_cols,
        "shared_cat_cols": shared_cat_cols,
        "private_num_cols": private_num_cols,
        "private_cat_cols": private_cat_cols,
        "shared_num_global_ids": [shared_key_to_id[feature_key(c, 'num')] for c in shared_num_cols],
        "shared_cat_global_ids": [shared_key_to_id[feature_key(c, 'cat')] for c in shared_cat_cols],
        "cards": cards,
        "cat_vocab_sizes": cat_vocab_sizes,
        "cat_oov_index": cat_oov_index,
        "cat_vocab_preview": cat_vocab_preview,
        "total_vocab_size": total_vocab_size,
        "shared_hash_buckets": int(SHARED_HASH_BUCKETS),
        "sequence_length": sequence_length,
        "encoding_shapes": {
            "train_shared_num": [int(Xtr_sn.shape[0]), int(Xtr_sn.shape[1])],
            "val_shared_num": [int(Xva_sn.shape[0]), int(Xva_sn.shape[1])],
            "test_shared_num": [int(Xte_sn.shape[0]), int(Xte_sn.shape[1])],
            "train_shared_cat": [int(Xtr_sc.shape[0]), int(Xtr_sc.shape[1])],
            "val_shared_cat": [int(Xva_sc.shape[0]), int(Xva_sc.shape[1])],
            "test_shared_cat": [int(Xte_sc.shape[0]), int(Xte_sc.shape[1])],
            "train_private_num": [int(Xtr_pn.shape[0]), int(Xtr_pn.shape[1])],
            "val_private_num": [int(Xva_pn.shape[0]), int(Xva_pn.shape[1])],
            "test_private_num": [int(Xte_pn.shape[0]), int(Xte_pn.shape[1])],
            "train_private_cat": [int(Xtr_pc.shape[0]), int(Xtr_pc.shape[1])],
            "val_private_cat": [int(Xva_pc.shape[0]), int(Xva_pc.shape[1])],
            "test_private_cat": [int(Xte_pc.shape[0]), int(Xte_pc.shape[1])],
        },
        "n_clients": int(n_clients),
        "client_sizes": client_sizes,
        "split_mode": "fresh-70/15/15-after-compatibility-merge",
        "split_policy": SPLIT_POLICY,
        "preserve_official_benchmark_splits": PRESERVE_OFFICIAL_BENCHMARK_SPLITS,
        "source_parts_used": stage["source_parts_used"],
        "benchmark_note": "named split files are treated as labeled sources and re-split fresh; benchmark comparability is not preserved",
        "split_shapes": {"train": int(len(tr_ds)), "val": int(len(va_ds)), "test": int(len(te_ds))},
        "dropped_unseen_labels": stage["dropped_unseen_labels"],
        "class_dist_train": stage["class_dist_train"],
        "class_dist_val": stage["class_dist_val"],
        "class_dist_test": stage["class_dist_test"],
        "dropped_target_siblings": stage["sibling_drop_cols"],
        "cross_features_generated": int(len(stage["new_crosses"])),
        "cross_plan": stage["cross_plan"],
        "feature_intel_selected_count": int(len(selected_keys)),
        "feature_intel_shared_count": int(len(shared_num_cols) + len(shared_cat_cols)),
        "feature_intel_private_count": int(len(private_num_cols) + len(private_cat_cols)),
        "selected_universality": selected_universality,
        "global_relevance_mean": float(np.mean(stage.get("global_relevance_scores", np.array([0.0])))) if len(stage.get("global_relevance_scores", [])) else 0.0,
    }

    show_key_value_table(
        f"[{meta['name'].upper()}] FINAL ROUTED DATASET",
        {
            "selected_features": len(meta['selected_features']),
            "shared_route": f"num={len(shared_num_cols)} | cat={len(shared_cat_cols)}",
            "private_route": f"num={len(private_num_cols)} | cat={len(private_cat_cols)}",
            "sequence_length": sequence_length,
            "n_clients": n_clients,
            "client_sizes": client_sizes,
            "global_relevance_mean": meta['global_relevance_mean'],
        },
        note="Client sizes are wrapped across lines to keep notebook output readable.",
    )
    return meta, tr_ds, va_ds, te_ds, client_idx


# ==================================================================================================
# BUILD DATASETS
# ==================================================================================================
stage_records = []
for spec in DATASET_SPECS:
    print("\n" + "#" * 100)
    print(f"STARTING {spec['name'].upper()} PREPARATION")
    print("#" * 100)
    try:
        stage_records.append(prepare_dataset_for_feature_intelligence(spec, mi_pre=120))
    except Exception as e:
        print(f"[WARN] {spec['name']} skipped: {e}")

if len(stage_records) == 0:
    raise RuntimeError("No dataset preparation completed successfully.")

federated_relevance_state, feature_universality, feature_occurrence, global_shared_feature_keys, global_shared_key_to_id = run_federated_feature_intelligence(stage_records)

all_meta = []
all_tr = []
all_va = []
all_te = []
all_cidx = []
for st in stage_records:
    try:
        meta, tr_ds, va_ds, te_ds, cidx = finalize_stage_record(st, feature_universality, feature_occurrence, global_shared_key_to_id)
        all_meta.append(meta)
        all_tr.append(tr_ds)
        all_va.append(va_ds)
        all_te.append(te_ds)
        all_cidx.append(cidx)
    except Exception as e:
        print(f"[WARN] {st['spec']['name']} finalization skipped: {e}")

N_DS = len(all_meta)
if N_DS == 0:
    raise RuntimeError("No dataset pipeline built successfully after federated feature intelligence.")

GLOBAL_SHARED_FEATURE_COUNT = max(1, len(global_shared_feature_keys))

task_summary_rows = []
for meta in all_meta:
    task_summary_rows.append({
        "dataset": meta["name"],
        "task_kind": meta.get("task_kind", "unknown"),
        "task_family": meta.get("task_family", "binary" if int(meta.get("n_classes", 0)) == 2 else "multiclass"),
        "n_classes": int(meta.get("n_classes", 0)),
        "normal_index": meta.get("normal_index", None),
        "binary_view_available": bool(meta.get("binary_view_available", False)),
        "selected_features": int(len(meta.get("selected_features", []))),
        "shared_features": int(meta.get("feature_intel_shared_count", 0)),
        "private_features": int(meta.get("feature_intel_private_count", 0)),
        "numeric_features": int(len(meta.get("num_cols", []))),
        "categorical_features": int(len(meta.get("cat_cols", []))),
        "sequence_length": int(meta.get("sequence_length", 1 + len(meta.get("num_cols", [])) + len(meta.get("cat_cols", [])))),
        "total_vocab_size": int(meta.get("total_vocab_size", sum(meta.get("cards", [])))),
    })

task_summary_df = pd.DataFrame(task_summary_rows).sort_values("dataset").reset_index(drop=True)
show_table(
    f"{PROCESS_NAME} DATASET TASK / ENCODING SUMMARY",
    task_summary_df,
    index=False,
    columns=[
        "dataset", "task_kind", "task_family", "n_clients", "normal_index", "binary_view_available",
        "selected_features", "shared_features", "private_features", "numeric_features",
        "categorical_features", "sequence_length", "total_vocab_size",
    ],
    rename_map={
        "task_kind": "task",
        "task_family": "family",
        "n_clients": "clients",
        "normal_index": "normal_idx",
        "binary_view_available": "bin_view",
        "selected_features": "selected_feats",
        "shared_features": "shared_feats",
        "private_features": "private_feats",
        "numeric_features": "num_feats",
        "categorical_features": "cat_feats",
        "sequence_length": "seq_len",
        "total_vocab_size": "vocab_size",
    },
    note="Column labels are shortened only for display. Full names are preserved in artifacts/dataset_task_summary.csv.",
)
if (task_summary_df["task_family"] == "binary").all():
    overall_task_nature = "all_binary"
elif (task_summary_df["task_family"] == "multiclass").all():
    overall_task_nature = "all_multiclass"
else:
    overall_task_nature = "mixed_binary_multiclass"
print(f"\n{PROCESS_NAME} OVERALL TASK NATURE: {overall_task_nature.upper()}")
print("#" * 100)


# ==================================================================================================
# FEDERATED TRAINING
# ==================================================================================================
D_MODEL = 192
N_BLOCKS = 3
N_HEADS = 8
FF_DIM = 384
DROP = 0.10
ROUNDS = 15
LOCAL_EPOCHS = 1
LR = 1e-3
BATCH_SIZE = 256


def new_model(meta):
    return GRIPDFFIModel(
        meta=meta,
        n_shared_feature_ids=GLOBAL_SHARED_FEATURE_COUNT,
        d_model=D_MODEL,
        n_blocks=N_BLOCKS,
        n_heads=N_HEADS,
        ff=FF_DIM,
        drop=DROP,
    ).to(DEVICE)


seed_models = [new_model(meta) for meta in all_meta]
shared_global_state = get_shared_state(seed_models[0])
private_dataset_states = [get_private_state(m) for m in seed_models]

best_bundle = {
    "round": -1,
    "global_val_acc": -1.0,
    "shared_state": shared_global_state,
    "private_states": private_dataset_states,
}

va_loaders = [mk_loader(v, shuffle=False, batch_size=BATCH_SIZE) for v in all_va]
te_loaders = [mk_loader(t, shuffle=False, batch_size=BATCH_SIZE) for t in all_te]
round_history = []

for rnd in range(1, ROUNDS + 1):
    print("\n" + "=" * 100)
    print(f"FEDERATED ROUND {rnd}/{ROUNDS}")
    print("=" * 100)
    t_round = time.time()

    shared_candidates = []
    shared_sizes = []
    private_candidates = defaultdict(list)
    private_sizes = defaultdict(list)

    for di, (meta, tr_ds, client_idx) in enumerate(zip(all_meta, all_tr, all_cidx)):
        print(f"\n[{meta['name']}] {len(client_idx)} clients")
        for cid, idx in enumerate(client_idx):
            local_ds = tr_ds.subset(idx)
            local_loader = mk_loader(local_ds, shuffle=True, batch_size=BATCH_SIZE)
            model = new_model(meta)
            load_shared_state(model, shared_global_state)
            load_private_state(model, private_dataset_states[di])

            t0 = time.time()
            train_client(model, local_loader, epochs=LOCAL_EPOCHS, lr=LR)
            t_train = time.time() - t0

            yt_loc, pt_loc = predict(model, mk_loader(local_ds, shuffle=False, batch_size=BATCH_SIZE))
            mt_loc = compute_metrics(yt_loc, pt_loc)
            yv_loc, pv_loc = predict(model, va_loaders[di])
            mv_loc = compute_metrics(yv_loc, pv_loc)

            print(
                f"  client={cid:02d} | n={len(local_ds):5d} | "
                f"train_acc={mt_loc['acc']:.4f} | val_acc={mv_loc['acc']:.4f} | "
                f"val_f1={mv_loc['f1_macro']:.4f} | val_auc={mv_loc['auc_roc_macro_ovr']:.4f} | "
                f"time={t_train:.1f}s"
            )

            shared_candidates.append(get_shared_state(model))
            shared_sizes.append(len(local_ds))
            private_candidates[di].append(get_private_state(model))
            private_sizes[di].append(len(local_ds))

            del model, local_ds, local_loader
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    shared_weights = [n / float(sum(shared_sizes)) for n in shared_sizes]
    shared_global_state = average_state_dicts(shared_candidates, shared_weights)

    new_private_states = []
    for di in range(N_DS):
        plist = private_candidates[di]
        psz = private_sizes[di]
        w = [n / float(sum(psz)) for n in psz]
        new_private_states.append(average_state_dicts(plist, w))
    private_dataset_states = new_private_states

    val_ns = []
    val_ms = []
    dataset_val_metrics = {}
    for di, meta in enumerate(all_meta):
        model = new_model(meta)
        load_shared_state(model, shared_global_state)
        load_private_state(model, private_dataset_states[di])
        yv, pv = predict(model, va_loaders[di])
        mv = compute_metrics(yv, pv)
        dataset_val_metrics[meta["name"]] = mv
        val_ns.append(len(yv))
        val_ms.append(mv)
        print(
            f"  [VAL] {meta['name']} | acc={mv['acc']:.4f} | f1={mv['f1_macro']:.4f} | "
            f"auc={mv['auc_roc_macro_ovr']:.4f} | logloss={mv['logloss']:.4f}"
        )
        del model

    total_val = sum(val_ns)
    global_val_acc = sum(m['acc'] * n for m, n in zip(val_ms, val_ns)) / total_val
    global_val_f1 = sum(m['f1_macro'] * n for m, n in zip(val_ms, val_ns)) / total_val
    valid_ll = [m['logloss'] for m in val_ms if m['logloss'] == m['logloss']]
    global_val_logloss = (
        sum(m['logloss'] * n for m, n in zip(val_ms, val_ns) if m['logloss'] == m['logloss']) / total_val
        if valid_ll else np.nan
    )

    print("-" * 100)
    print(
        f"GLOBAL VAL | acc={global_val_acc:.4f} | f1={global_val_f1:.4f} | "
        f"logloss={global_val_logloss:.4f} | round_time={time.time() - t_round:.1f}s"
    )

    if global_val_acc > best_bundle["global_val_acc"]:
        best_bundle = {
            "round": rnd,
            "global_val_acc": float(global_val_acc),
            "shared_state": {k: v.clone() for k, v in shared_global_state.items()},
            "private_states": [{k: v.clone() for k, v in p.items()} for p in private_dataset_states],
        }

    round_history.append({
        "round": rnd,
        "global_val_acc": float(global_val_acc),
        "global_val_f1": float(global_val_f1),
        "global_val_logloss": float(global_val_logloss) if global_val_logloss == global_val_logloss else None,
        "best_round_so_far": int(best_bundle["round"]),
        "best_val_acc_so_far": float(best_bundle["global_val_acc"]),
        "per_dataset": {
            k: {kk: (float(vv) if vv == vv else None) for kk, vv in v.items()}
            for k, v in dataset_val_metrics.items()
        },
    })

print("\n" + "=" * 100)
print(f"BEST ROUND: {best_bundle['round']} | BEST GLOBAL VAL ACC: {best_bundle['global_val_acc']:.4f}")
print("=" * 100)

# ==================================================================================================
# FINAL EVAL USING BEST ROUND
# ==================================================================================================
best_shared_state = best_bundle["shared_state"]
best_private_states = best_bundle["private_states"]

val_preds = {}
test_preds = {}
val_metrics_each = []
test_metrics_each = []
rows = []

for di, meta in enumerate(all_meta):
    model = new_model(meta)
    load_shared_state(model, best_shared_state)
    load_private_state(model, best_private_states[di])

    yv, pv = predict(model, va_loaders[di])
    yt, pt = predict(model, te_loaders[di])
    mv = compute_metrics(yv, pv)
    mt = compute_metrics(yt, pt)

    val_preds[meta["name"]] = {"y": yv, "p": pv}
    test_preds[meta["name"]] = {"y": yt, "p": pt}
    val_metrics_each.append(mv)
    test_metrics_each.append(mt)

    rows.append({"split": "VAL", "dataset": meta["name"], **mv})
    rows.append({"split": "TEST", "dataset": meta["name"], **mt})
    del model


def weighted_metric(metric_list, ns, key):
    vals_weights = []
    for m, n in zip(metric_list, ns):
        v = m.get(key, np.nan)
        if v == v:
            vals_weights.append((float(v), float(n)))
    if not vals_weights:
        return np.nan
    return float(sum(v * n for v, n in vals_weights) / max(sum(n for _, n in vals_weights), 1.0))


val_ns = [len(val_preds[m["name"]]["y"]) for m in all_meta]
test_ns = [len(test_preds[m["name"]]["y"]) for m in all_meta]
all_metric_keys = sorted(set().union(*[m.keys() for m in val_metrics_each], *[m.keys() for m in test_metrics_each]))

rows.append({
    "split": "VAL",
    "dataset": "global_weighted",
    **{k: weighted_metric(val_metrics_each, val_ns, k) for k in all_metric_keys},
})
rows.append({
    "split": "TEST",
    "dataset": "global_weighted",
    **{k: weighted_metric(test_metrics_each, test_ns, k) for k in all_metric_keys},
})

report = pd.DataFrame(rows)
core_report_cols = [
    "split", "dataset", "acc", "precision_macro", "recall_macro", "f1_macro",
    "logloss", "mcc", "kappa", "auc_roc_macro_ovr", "pr_auc_macro",
]
show_table(
    "FINAL REPORT — CORE METRICS",
    report,
    index=False,
    columns=[c for c in core_report_cols if c in report.columns],
    rename_map={
        "precision_macro": "prec_macro",
        "recall_macro": "rec_macro",
        "auc_roc_macro_ovr": "auc_roc_macro",
        "pr_auc_macro": "pr_auc",
    },
    note="Full per-split metric export with every recorded column is saved to artifacts/final_report.csv.",
)
extra_report_cols = [c for c in report.columns if c not in core_report_cols]
extra_report_cols = [c for c in extra_report_cols if c not in ["split", "dataset"]]
if extra_report_cols:
    show_table(
        "FINAL REPORT — EXTENDED METRICS",
        report,
        index=False,
        columns=["split", "dataset"] + extra_report_cols,
        rename_map={
            "precision_weighted": "prec_weighted",
            "recall_weighted": "rec_weighted",
            "f1_weighted": "f1_weighted",
            "auc_roc_micro_ovr": "auc_roc_micro",
            "auc_roc_weighted_ovr": "auc_roc_weighted",
            "ppv_macro_ovr": "ppv_macro",
            "npv_macro_ovr": "npv_macro",
            "ppv_weighted_ovr": "ppv_weighted",
            "npv_weighted_ovr": "npv_weighted",
        },
    )

# ==================================================================================================
# SAVE CHECKPOINTS / TABLES
# ==================================================================================================
os.makedirs("artifacts/checkpoints", exist_ok=True)
os.makedirs("artifacts/plots", exist_ok=True)

torch.save(best_shared_state, "artifacts/checkpoints/shared_backbone_best.pth")
for di, meta in enumerate(all_meta):
    torch.save(best_private_states[di], f"artifacts/checkpoints/private_{meta['name']}_best.pth")
report.to_csv("artifacts/final_report.csv", index=False)
task_summary_df.to_csv("artifacts/dataset_task_summary.csv", index=False)
with open("artifacts/round_history.json", "w") as f:
    json.dump(to_jsonable(round_history), f, indent=2)
with open("artifacts/dataset_meta.json", "w") as f:
    json.dump(to_jsonable(all_meta), f, indent=2)

torch.save({
    "config": {
        "seed": SEED,
        "rounds": ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "lr": LR,
        "n_datasets": N_DS,
        "d_model": D_MODEL,
        "n_blocks": N_BLOCKS,
        "n_heads": N_HEADS,
    },
    "best_round": best_bundle["round"],
    "best_global_val_acc": best_bundle["global_val_acc"],
    "shared_state": best_shared_state,
    "private_states": best_private_states,
    "meta": all_meta,
    "round_history": round_history,
    "report_rows": report.to_dict(orient="records"),
}, "artifacts/full_process_all_in_one.pth")

# ==================================================================================================
# ROC / PR STORES + CALIBRATION + ERROR ANALYSIS TABLES
# ==================================================================================================
roc_store = {}
pr_store = {}
calibration_store = {}
combined_binary_val = []
combined_binary_test = []

for meta in all_meta:
    name = meta["name"]
    class_names = meta["class_names"]
    normal_idx = meta["normal_index"]

    yv, pv = val_preds[name]["y"], val_preds[name]["p"]
    yt, pt = test_preds[name]["y"], test_preds[name]["p"]

    roc_store[f"{name}_val_multiclass"] = collect_roc_data(yv, pv, f"{name}_val_multiclass", class_names)
    roc_store[f"{name}_test_multiclass"] = collect_roc_data(yt, pt, f"{name}_test_multiclass", class_names)

    yvb, pvb = binary_view_from_multiclass(yv, pv, normal_idx)
    ytb, ptb = binary_view_from_multiclass(yt, pt, normal_idx)

    if yvb is not None:
        roc_store[f"{name}_val_binary"] = collect_roc_data(yvb, pvb, f"{name}_val_binary", ["Normal", "Attack"])
        pr_store[f"{name}_val_binary"] = collect_pr_data(yvb, pvb, f"{name}_val_binary", positive_label="Attack")
        ece_v, tab_v = expected_calibration_error(yvb, pvb[:, 1], n_bins=10)
        calibration_store[f"{name}_val_binary"] = {
            "ece": float(ece_v),
            "brier": float(brier_score_binary(yvb, pvb[:, 1])),
            "attack_rate": float(np.mean(yvb)),
            "table": tab_v.to_dict(orient="records"),
        }
        combined_binary_val.append((name, yvb, pvb))
    if ytb is not None:
        roc_store[f"{name}_test_binary"] = collect_roc_data(ytb, ptb, f"{name}_test_binary", ["Normal", "Attack"])
        pr_store[f"{name}_test_binary"] = collect_pr_data(ytb, ptb, f"{name}_test_binary", positive_label="Attack")
        ece_t, tab_t = expected_calibration_error(ytb, ptb[:, 1], n_bins=10)
        calibration_store[f"{name}_test_binary"] = {
            "ece": float(ece_t),
            "brier": float(brier_score_binary(ytb, ptb[:, 1])),
            "attack_rate": float(np.mean(ytb)),
            "table": tab_t.to_dict(orient="records"),
        }
        combined_binary_test.append((name, ytb, ptb))

if combined_binary_val:
    y_all = np.concatenate([y for _, y, _ in combined_binary_val])
    p_all = np.concatenate([p for _, _, p in combined_binary_val], axis=0)
    roc_store["global_val_binary"] = collect_roc_data(y_all, p_all, "global_val_binary", ["Normal", "Attack"])
    pr_store["global_val_binary"] = collect_pr_data(y_all, p_all, "global_val_binary", positive_label="Attack")
    ece_gv, tab_gv = expected_calibration_error(y_all, p_all[:, 1], n_bins=10)
    calibration_store["global_val_binary"] = {
        "ece": float(ece_gv),
        "brier": float(brier_score_binary(y_all, p_all[:, 1])),
        "attack_rate": float(np.mean(y_all)),
        "table": tab_gv.to_dict(orient="records"),
    }
if combined_binary_test:
    y_all = np.concatenate([y for _, y, _ in combined_binary_test])
    p_all = np.concatenate([p for _, _, p in combined_binary_test], axis=0)
    roc_store["global_test_binary"] = collect_roc_data(y_all, p_all, "global_test_binary", ["Normal", "Attack"])
    pr_store["global_test_binary"] = collect_pr_data(y_all, p_all, "global_test_binary", positive_label="Attack")
    ece_gt, tab_gt = expected_calibration_error(y_all, p_all[:, 1], n_bins=10)
    calibration_store["global_test_binary"] = {
        "ece": float(ece_gt),
        "brier": float(brier_score_binary(y_all, p_all[:, 1])),
        "attack_rate": float(np.mean(y_all)),
        "table": tab_gt.to_dict(orient="records"),
    }

with open("artifacts/roc_data.json", "w") as f:
    json.dump(to_jsonable(roc_store), f, indent=2)
with open("artifacts/pr_data.json", "w") as f:
    json.dump(to_jsonable(pr_store), f, indent=2)
with open("artifacts/calibration_data.json", "w") as f:
    json.dump(to_jsonable(calibration_store), f, indent=2)

error_val_df = build_error_table(val_preds, all_meta, "VAL")
error_test_df = build_error_table(test_preds, all_meta, "TEST")
error_val_df.to_csv("artifacts/error_analysis_val_samples.csv", index=False)
error_test_df.to_csv("artifacts/error_analysis_test_samples.csv", index=False)

error_summary_rows = []
for ds_name, grp in error_test_df.groupby("dataset"):
    n = len(grp)
    correct = int(grp["correct"].sum())
    wrong = n - correct
    error_summary_rows.append({
        "dataset": ds_name,
        "n_samples": n,
        "accuracy": float(correct / max(n, 1)),
        "error_rate": float(wrong / max(n, 1)),
        "mean_conf_correct": float(grp.loc[grp["correct"] == 1, "confidence"].mean()) if (grp["correct"] == 1).any() else np.nan,
        "mean_conf_wrong": float(grp.loc[grp["correct"] == 0, "confidence"].mean()) if (grp["correct"] == 0).any() else np.nan,
        "mean_margin_correct": float(grp.loc[grp["correct"] == 1, "margin"].mean()) if (grp["correct"] == 1).any() else np.nan,
        "mean_margin_wrong": float(grp.loc[grp["correct"] == 0, "margin"].mean()) if (grp["correct"] == 0).any() else np.nan,
    })
error_summary_df = pd.DataFrame(error_summary_rows)
error_summary_df.to_csv("artifacts/error_analysis_summary_test.csv", index=False)

class_error_rows = []
for ds_name, grp in error_test_df.groupby("dataset"):
    for true_label, g in grp.groupby("true_label"):
        n = len(g)
        class_error_rows.append({
            "dataset": ds_name,
            "true_label": true_label,
            "dataset_class_key": f"{ds_name}::{true_label}",
            "support": n,
            "error_rate": float(1.0 - g["correct"].mean()),
            "mean_wrong_conf": float(g.loc[g["correct"] == 0, "confidence"].mean()) if (g["correct"] == 0).any() else np.nan,
        })
class_error_df = pd.DataFrame(class_error_rows)
class_error_df.to_csv("artifacts/error_analysis_by_class_test.csv", index=False)

conf_pair_counter = Counter(
    [(r["dataset"], r["true_label"], r["pred_label"]) for _, r in error_test_df[error_test_df["correct"] == 0].iterrows()]
)
top_conf_rows = []
for (ds, tl, pl), cnt in conf_pair_counter.most_common(25):
    top_conf_rows.append({
        "dataset": ds,
        "true_label": tl,
        "pred_label": pl,
        "count": int(cnt),
        "pair": f"{ds}: {tl} -> {pl}",
    })
top_conf_df = pd.DataFrame(top_conf_rows)
top_conf_df.to_csv("artifacts/error_analysis_top_confusions_test.csv", index=False)

# ==================================================================================================
# PLOTS
# ==================================================================================================
def plot_roc_multiclass(roc_data, title, out_path):
    curves = roc_data["curves"]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1.5, label="Chance")
    for i, (_, v) in enumerate(curves.items()):
        c = PALETTE[i % len(PALETTE)]
        ax.plot(v["fpr"], v["tpr"], color=c, lw=2.0, label=f"{v['label']} (AUC={v['auc']:.3f})", alpha=0.95)
    prettify_ax(ax, title=title, xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(loc="lower right", fontsize=8.7)
    save_or_show(fig, out_path)


def plot_roc_binary(roc_data, title, out_path):
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1.4, label="Chance")
    curve = roc_data["curves"].get("1", None)
    if curve is None:
        for _, v in roc_data["curves"].items():
            curve = v
            break
    if curve is not None:
        ax.plot(curve["fpr"], curve["tpr"], color=ACCENT_1, lw=2.5, label=f"Attack (AUC={curve['auc']:.3f})")
    prettify_ax(ax, title=title, xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(loc="lower right")
    save_or_show(fig, out_path)


def plot_pr_binary(pr_data, title, out_path):
    if pr_data is None:
        return
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.plot(pr_data["recall"], pr_data["precision"], color=ACCENT_3, lw=2.5, label=f"AP={pr_data['ap']:.3f}")
    prettify_ax(ax, title=title, xlabel="Recall", ylabel="Precision")
    ax.legend(loc="lower left")
    save_or_show(fig, out_path)


def plot_confusion(y_true, y_pred, labels, title, out_path):
    n = len(labels)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    cmn = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fs = max(6, min(10, 170 // max(n, 1)))
    fig, axes = plt.subplots(1, 2, figsize=(max(8, n * 1.0 + 3), max(5.5, n * 0.8 + 1.8)))
    cmaps = [
        LinearSegmentedColormap.from_list("cm_blue", ["#eef6ff", "#93c5fd", "#1d4ed8"]),
        LinearSegmentedColormap.from_list("cm_green", ["#f0fdf4", "#86efac", "#15803d"]),
    ]
    for ax, data, fmt, ttl, cmap in [
        (axes[0], cm, "d", "Counts", cmaps[0]),
        (axes[1], cmn, ".2f", "Row-normalized", cmaps[1]),
    ]:
        im = ax.imshow(data, cmap=cmap, aspect="equal", interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fs)
        ax.set_yticklabels(labels, fontsize=fs)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        prettify_ax(ax, title=f"{title}\n{ttl}")
        thresh = data.max() / 2.0 if data.size else 0.0
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i, format(data[i, j], fmt),
                    ha="center", va="center",
                    fontsize=max(5, fs - 1),
                    color="#0f172a" if data[i, j] >= thresh else "#334155"
                )
        ax.grid(False)
    save_or_show(fig, out_path)


def plot_round_history(history, out_path):
    rounds = [h["round"] for h in history]
    acc = [h["global_val_acc"] for h in history]
    f1 = [h["global_val_f1"] for h in history]
    best = [h["best_val_acc_so_far"] for h in history]

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.plot(rounds, acc, marker="o", lw=2.2, color=ACCENT_1, label="Global Val Accuracy")
    ax.plot(rounds, f1, marker="o", lw=2.2, color=ACCENT_3, label="Global Val F1")
    ax.plot(rounds, best, lw=2.4, color=ACCENT_6, linestyle="-.", label="Best Accuracy So Far")
    prettify_ax(ax, title=f"{PROCESS_NAME} - Federated Training History", xlabel="Round", ylabel="Score")
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right")
    save_or_show(fig, out_path)


def plot_dataset_metric_bars(report_df, split, out_path):
    df = report_df[(report_df["split"] == split) & (report_df["dataset"] != "global_weighted")].copy()
    names = df["dataset"].tolist()
    acc = df["acc"].tolist()
    f1 = df["f1_macro"].tolist()
    auc = df["auc_roc_macro_ovr"].tolist()

    x = np.arange(len(names))
    w = 0.24
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    ax.bar(x - w, acc, width=w, color=ACCENT_1, label="Accuracy", alpha=0.92)
    ax.bar(x, f1, width=w, color=ACCENT_3, label="F1 Macro", alpha=0.92)
    ax.bar(x + w, auc, width=w, color=ACCENT_4, label="ROC AUC Macro", alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    prettify_ax(ax, title=f"{split} Metrics by Dataset", xlabel="Dataset", ylabel="Score")
    ax.legend(loc="lower right")
    save_or_show(fig, out_path)


def plot_global_weighted_summary(report_df, out_path):
    df = report_df[report_df["dataset"] == "global_weighted"].copy()
    metrics = ["acc", "f1_macro", "ppv_macro_ovr", "npv_macro_ovr", "auc_roc_macro_ovr", "pr_auc_macro", "mcc", "kappa"]
    val_row = df[df["split"] == "VAL"].iloc[0]
    test_row = df[df["split"] == "TEST"].iloc[0]

    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    ax.bar(x - w / 2, [val_row[m] for m in metrics], width=w, color=ACCENT_2, label="VAL")
    ax.bar(x + w / 2, [test_row[m] for m in metrics], width=w, color=ACCENT_5, label="TEST")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    prettify_ax(ax, title=f"{PROCESS_NAME} - Global Weighted Summary", xlabel="Metric", ylabel="Score")
    ax.legend(loc="lower right")
    save_or_show(fig, out_path)


def plot_class_distribution(meta_list, out_path):
    fig, axes = plt.subplots(len(meta_list), 3, figsize=(13.5, 3.8 * len(meta_list)))
    if len(meta_list) == 1:
        axes = np.array([axes])
    split_keys = ["class_dist_train", "class_dist_val", "class_dist_test"]
    split_names = ["Train", "Val", "Test"]
    for r, meta in enumerate(meta_list):
        labels = [str(x) for x in meta["class_names"]]
        for c, (key, split_name) in enumerate(zip(split_keys, split_names)):
            ax = axes[r, c]
            dist = meta[key]
            vals = [dist.get(i, 0) for i in range(len(labels))]
            ax.bar(np.arange(len(labels)), vals, color=PALETTE[:len(labels)], alpha=0.9)
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            prettify_ax(ax, title=f"{meta['name']} - {split_name}", xlabel="Class", ylabel="Count")
    save_or_show(fig, out_path)


def plot_client_distribution(meta_list, out_path):
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for i, meta in enumerate(meta_list):
        xs = np.arange(len(meta["client_sizes"]))
        ys = meta["client_sizes"]
        ax.plot(xs, ys, marker="o", lw=2.0, label=meta["name"], color=PALETTE[i % len(PALETTE)])
    prettify_ax(ax, title=f"{PROCESS_NAME} - Client Size Distribution by Dataset", xlabel="Client ID", ylabel="Samples")
    ax.legend(loc="upper right")
    save_or_show(fig, out_path)


def plot_binary_roc_overlay(roc_store, keys, title, out_path):
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1.4, label="Chance")
    for i, key in enumerate(keys):
        if key not in roc_store:
            continue
        curve = roc_store[key]["curves"].get("1", None)
        if curve is None:
            for _, v in roc_store[key]["curves"].items():
                curve = v
                break
        if curve is None:
            continue
        ax.plot(curve["fpr"], curve["tpr"], lw=2.2, color=PALETTE[i % len(PALETTE)],
                label=f"{key.replace('_binary', '')} (AUC={curve['auc']:.3f})")
    prettify_ax(ax, title=title, xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(loc="lower right", fontsize=8.5)
    save_or_show(fig, out_path)

def plot_error_rate_summary(error_summary_df, out_path):
    df = error_summary_df.sort_values("error_rate", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(df))
    bars = ax.bar(x, df["error_rate"], color=[PALETTE[i % len(PALETTE)] for i in range(len(df))], alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"])
    ax.set_ylim(0, min(1.0, max(0.25, df["error_rate"].max() * 1.25 if len(df) else 0.25)))
    prettify_ax(ax, title="Test Error Rate by Dataset", xlabel="Dataset", ylabel="Error Rate")
    for rect, n in zip(bars, df["n_samples"]):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01, f"n={n}", ha="center", va="bottom", fontsize=9, color=SUBTEXT_CLR)
    save_or_show(fig, out_path)


def plot_confidence_correct_vs_wrong(error_df, out_path):
    stats = []
    for ds, grp in error_df.groupby("dataset"):
        mc = float(grp.loc[grp["correct"] == 1, "confidence"].mean()) if (grp["correct"] == 1).any() else np.nan
        mw = float(grp.loc[grp["correct"] == 0, "confidence"].mean()) if (grp["correct"] == 0).any() else np.nan
        stats.append((ds, mc, mw))
    stats_df = pd.DataFrame(stats, columns=["dataset", "mean_conf_correct", "mean_conf_wrong"])
    x = np.arange(len(stats_df))
    w = 0.34
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.bar(x - w / 2, stats_df["mean_conf_correct"], width=w, color=ACCENT_3, label="Correct")
    ax.bar(x + w / 2, stats_df["mean_conf_wrong"], width=w, color=ACCENT_5, label="Wrong")
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df["dataset"])
    ax.set_ylim(0, 1.05)
    prettify_ax(ax, title="Prediction Confidence: Correct vs Wrong (Test)", xlabel="Dataset", ylabel="Mean Max-Probability")
    ax.legend(loc="lower right")
    save_or_show(fig, out_path)


def plot_pooled_confidence_hist(error_df, out_path):
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    correct = error_df.loc[error_df["correct"] == 1, "confidence"].to_numpy()
    wrong = error_df.loc[error_df["correct"] == 0, "confidence"].to_numpy()
    bins = np.linspace(0, 1, 24)
    if len(correct):
        ax.hist(correct, bins=bins, alpha=0.60, color=ACCENT_3, label=f"Correct (n={len(correct)})", density=True)
    if len(wrong):
        ax.hist(wrong, bins=bins, alpha=0.60, color=ACCENT_5, label=f"Wrong (n={len(wrong)})", density=True)
    prettify_ax(ax, title=f"{PROCESS_NAME} - Pooled Confidence Distribution Across All Datasets (Test)", xlabel="Max Predicted Probability", ylabel="Density")
    ax.legend(loc="upper center")
    save_or_show(fig, out_path)


def plot_top_confusions(top_conf_df, out_path):
    if top_conf_df.empty:
        return
    df = top_conf_df.head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12.0, max(6.0, 0.38 * len(df) + 2)))
    ax.barh(np.arange(len(df)), df["count"], color=ACCENT_4, alpha=0.92)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["pair"], fontsize=9)
    prettify_ax(ax, title=f"{PROCESS_NAME} - Top Misclassification Pairs Across All Datasets (Test)", xlabel="Count", ylabel="True -> Pred")
    save_or_show(fig, out_path)


def plot_class_error_heatmap(class_error_df, out_path):
    if class_error_df.empty:
        return
    df = class_error_df.copy().sort_values(["dataset", "error_rate"], ascending=[True, False])
    labels = df["dataset_class_key"].tolist()
    arr = np.c_[df["error_rate"].to_numpy(), np.nan_to_num(df["mean_wrong_conf"].to_numpy(), nan=0.0)]
    fig_h = max(7.0, 0.28 * len(labels) + 2.4)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    cmap = LinearSegmentedColormap.from_list("errhm", ["#eff6ff", "#93c5fd", "#1d4ed8"])
    im = ax.imshow(arr, aspect="auto", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Error Rate", "Mean Wrong Conf."])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    prettify_ax(ax, title=f"{PROCESS_NAME} - Per-Class Error Difficulty Across All Datasets (Test)", xlabel="Metric", ylabel="Dataset::Class")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=7, color="#0f172a")
    ax.grid(False)
    save_or_show(fig, out_path)


def plot_reliability_diagram(calib_dict, title, out_path):
    tab = pd.DataFrame(calib_dict["table"])
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.4), gridspec_kw={"height_ratios": [3, 1.2]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1.5, label="Perfect calibration")
    mask = tab["count"] > 0
    ax.plot(tab.loc[mask, "mean_conf"], tab.loc[mask, "emp_acc"], marker="o", lw=2.3, color=ACCENT_1, label=f"ECE={calib_dict['ece']:.3f} | Brier={calib_dict['brier']:.3f}")
    prettify_ax(ax, title=title, xlabel="Predicted Attack Probability", ylabel="Observed Attack Frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax2 = axes[1]
    centers = 0.5 * (tab["bin_left"] + tab["bin_right"])
    ax2.bar(centers, tab["count"], width=0.09, color=ACCENT_2, alpha=0.88)
    prettify_ax(ax2, xlabel="Predicted Attack Probability", ylabel="Count")
    ax2.set_xlim(0, 1)
    save_or_show(fig, out_path)


def plot_dataset_calibration_bars(calibration_store, meta_list, split_key_suffix, out_path):
    rows = []
    for meta in meta_list:
        key = f"{meta['name']}_{split_key_suffix}"
        if key in calibration_store:
            rows.append({"dataset": meta["name"], "ece": calibration_store[key]["ece"], "brier": calibration_store[key]["brier"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    ax.bar(x - w / 2, df["ece"], width=w, color=ACCENT_6, label="ECE")
    ax.bar(x + w / 2, df["brier"], width=w, color=ACCENT_4, label="Brier Score")
    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"])
    prettify_ax(ax, title=f"Calibration Quality by Dataset ({split_key_suffix.replace('_', ' ').title()})", xlabel="Dataset", ylabel="Lower is Better")
    ax.legend(loc="upper right")
    save_or_show(fig, out_path)


def plot_dataset_reliability_overlay(calibration_store, meta_list, split_key_suffix, out_path):
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1.5, label="Perfect calibration")
    found = False
    for i, meta in enumerate(meta_list):
        key = f"{meta['name']}_{split_key_suffix}"
        if key not in calibration_store:
            continue
        found = True
        tab = pd.DataFrame(calibration_store[key]["table"])
        mask = tab["count"] > 0
        ax.plot(tab.loc[mask, "mean_conf"], tab.loc[mask, "emp_acc"], marker="o", lw=2.0, color=PALETTE[i % len(PALETTE)], label=f"{meta['name']} (ECE={calibration_store[key]['ece']:.3f})")
    if not found:
        plt.close(fig)
        return
    prettify_ax(ax, title=f"Reliability Curves Across All Datasets ({split_key_suffix.replace('_', ' ').title()})", xlabel="Predicted Attack Probability", ylabel="Observed Attack Frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8.4)
    save_or_show(fig, out_path)


def plot_multimetric_dataset_profile(report_df, split, out_path):
    df = report_df[(report_df["split"] == split) & (report_df["dataset"] != "global_weighted")].copy()
    if df.empty:
        return
    df = df.sort_values("dataset").reset_index(drop=True)
    metrics = [
        ("acc", "Accuracy"),
        ("f1_macro", "F1 Macro"),
        ("auc_roc_macro_ovr", "ROC AUC"),
        ("ppv_positive", "PPV"),
        ("npv_negative", "NPV"),
        ("mcc", "MCC"),
        ("kappa", "Kappa"),
    ]
    fig, ax = plt.subplots(figsize=(11.8, 6.4))
    xs = np.arange(len(df))
    for i, (col, label) in enumerate(metrics):
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ax.plot(xs, vals, marker="o", lw=2.2, color=PALETTE[i % len(PALETTE)], label=label, alpha=0.95)
    ax.set_xticks(xs)
    ax.set_xticklabels(df["dataset"].tolist(), rotation=20, ha="right")
    ax.set_ylim(min(-0.05, float(df[[m[0] for m in metrics]].min(numeric_only=True).min()) - 0.03), 1.05)
    prettify_ax(ax, title=f"Multi-Metric Dataset Profile ({split})", xlabel="Dataset", ylabel="Score")
    ax.legend(loc="upper left", ncol=2, fontsize=8.8)
    save_or_show(fig, out_path)


def plot_metric_heatmap(report_df, split, out_path):
    df = report_df[(report_df["split"] == split) & (report_df["dataset"] != "global_weighted")].copy()
    if df.empty:
        return
    metric_cols = ["acc", "f1_macro", "auc_roc_macro_ovr", "ppv_positive", "npv_negative", "mcc", "kappa", "pr_auc_macro"]
    present = [c for c in metric_cols if c in df.columns]
    mat = df[present].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11.0, max(5.6, 0.6 * len(df) + 2.5)))
    cmap = LinearSegmentedColormap.from_list("metrichm", ["#eff6ff", "#93c5fd", "#1d4ed8"])
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=max(-1.0, np.nanmin(mat)), vmax=max(1.0, np.nanmax(mat)))
    plt.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels([c.replace("_", " ") for c in present], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["dataset"].tolist())
    prettify_ax(ax, title=f"Dataset-vs-Metric Heatmap ({split})", xlabel="Metric", ylabel="Dataset")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#0f172a")
    ax.grid(False)
    save_or_show(fig, out_path)


def plot_error_profile(error_summary_df, out_path):
    if error_summary_df.empty:
        return
    df = error_summary_df.sort_values("dataset").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11.8, 6.4))
    xs = np.arange(len(df))
    metric_map = [
        ("error_rate", "Error Rate"),
        ("mean_conf_wrong", "Mean Wrong Conf."),
        ("mean_conf_correct", "Mean Correct Conf."),
        ("mean_margin_wrong", "Mean Wrong Margin"),
        ("mean_margin_correct", "Mean Correct Margin"),
    ]
    for i, (col, label) in enumerate(metric_map):
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ax.plot(xs, vals, marker="o", lw=2.2, color=PALETTE[i % len(PALETTE)], label=label, alpha=0.95)
    ax.set_xticks(xs)
    ax.set_xticklabels(df["dataset"].tolist(), rotation=20, ha="right")
    ax.set_ylim(-0.05, 1.05)
    prettify_ax(ax, title="Global Error Analysis Profile Across Datasets (Test)", xlabel="Dataset", ylabel="Value")
    ax.legend(loc="best", ncol=2, fontsize=8.8)
    save_or_show(fig, out_path)


def plot_error_scatter(error_summary_df, out_path):
    if error_summary_df.empty:
        return
    df = error_summary_df.sort_values("dataset").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.4, 6.6))
    sizes = 120 + 700 * pd.to_numeric(df["error_rate"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    xs = pd.to_numeric(df["mean_conf_wrong"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ys = pd.to_numeric(df["error_rate"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cols = pd.to_numeric(df["mean_margin_wrong"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    sc = ax.scatter(xs, ys, s=sizes, c=cols, cmap=LinearSegmentedColormap.from_list("errsc", ["#fef3c7", "#fb7185", "#7c3aed"]), alpha=0.82, edgecolors="#334155")
    for _, r in df.iterrows():
        ax.text(float(r["mean_conf_wrong"]) + 0.005 if pd.notna(r["mean_conf_wrong"]) else 0.01, float(r["error_rate"]) + 0.005, str(r["dataset"]), fontsize=8.8, color=TEXT_CLR)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="Mean Wrong Margin")
    prettify_ax(ax, title="Error Difficulty Map Across Datasets (Test)", xlabel="Mean Wrong Confidence", ylabel="Error Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_or_show(fig, out_path)


def plot_confidence_bin_error_curve(error_test_df, out_path):
    if error_test_df.empty:
        return
    df = error_test_df.copy()
    df["conf_bin"] = pd.cut(df["confidence"], bins=np.linspace(0, 1, 11), include_lowest=True)
    tab = df.groupby("conf_bin", observed=False).agg(
        n=("correct", "size"),
        mean_conf=("confidence", "mean"),
        error_rate=("correct", lambda z: float(1.0 - np.mean(z))),
        mean_margin=("margin", "mean"),
    ).reset_index()
    fig, ax1 = plt.subplots(figsize=(10.6, 6.2))
    ax1.plot(tab["mean_conf"], tab["error_rate"], marker="o", lw=2.4, color=ACCENT_5, label="Error Rate")
    ax1.plot(tab["mean_conf"], pd.to_numeric(tab["mean_margin"], errors="coerce").fillna(0.0), marker="o", lw=2.4, color=ACCENT_2, label="Mean Margin")
    ax2 = ax1.twinx()
    ax2.bar(tab["mean_conf"], tab["n"], width=0.07, color=ACCENT_3, alpha=0.22, label="Count")
    prettify_ax(ax1, title="Global Confidence-Bin Error Profile (Test)", xlabel="Mean Confidence per Bin", ylabel="Rate / Margin")
    ax2.set_ylabel("Count", color=SUBTEXT_CLR)
    ax2.tick_params(colors=SUBTEXT_CLR)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_xlim(0, 1)
    save_or_show(fig, out_path)


def plot_calibration_summary_profile(calibration_store, meta_list, split_key_suffix, out_path):
    rows = []
    for meta in meta_list:
        key = f"{meta['name']}_{split_key_suffix}"
        if key in calibration_store:
            rows.append({
                "dataset": meta["name"],
                "ece": calibration_store[key]["ece"],
                "brier": calibration_store[key]["brier"],
                "attack_rate": calibration_store[key].get("attack_rate", np.nan),
            })
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    xs = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(11.8, 6.4))
    ax.plot(xs, df["ece"], marker="o", lw=2.2, color=ACCENT_6, label="ECE")
    ax.plot(xs, df["brier"], marker="o", lw=2.2, color=ACCENT_4, label="Brier")
    if "attack_rate" in df.columns:
        ax.plot(xs, pd.to_numeric(df["attack_rate"], errors="coerce").fillna(0.0), marker="o", lw=2.2, color=ACCENT_2, label="Attack Rate")
    ax.set_xticks(xs)
    ax.set_xticklabels(df["dataset"].tolist(), rotation=20, ha="right")
    prettify_ax(ax, title=f"{PROCESS_NAME} - Calibration Profile Across Datasets ({split_key_suffix.replace('_', ' ').title()})", xlabel="Dataset", ylabel="Lower is Better / Reference Rate")
    ax.legend(loc="upper left")
    save_or_show(fig, out_path)


def plot_global_metric_ribbon(report_df, out_path):
    val_df = report_df[(report_df["split"] == "VAL") & (report_df["dataset"] != "global_weighted")].copy().sort_values("dataset")
    test_df = report_df[(report_df["split"] == "TEST") & (report_df["dataset"] != "global_weighted")].copy().sort_values("dataset")
    if val_df.empty or test_df.empty:
        return
    common = [d for d in val_df["dataset"] if d in set(test_df["dataset"])]
    val_df = val_df[val_df["dataset"].isin(common)].reset_index(drop=True)
    test_df = test_df[test_df["dataset"].isin(common)].reset_index(drop=True)
    xs = np.arange(len(common))
    fig, ax = plt.subplots(figsize=(12.4, 6.8))
    series = [
        ("acc", "Accuracy", ACCENT_1),
        ("f1_macro", "F1 Macro", ACCENT_3),
        ("auc_roc_macro_ovr", "ROC AUC", ACCENT_6),
    ]
    for col, label, color in series:
        v = pd.to_numeric(val_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        t = pd.to_numeric(test_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mid = 0.5 * (v + t)
        ax.plot(xs, mid, lw=2.5, marker="o", color=color, label=label)
        ax.fill_between(xs, np.minimum(v, t), np.maximum(v, t), color=color, alpha=0.12)
    ax.set_xticks(xs)
    ax.set_xticklabels(common, rotation=20, ha="right")
    ax.set_ylim(-0.05, 1.05)
    prettify_ax(ax, title="Global Multi-Metric Ribbon Across Datasets (VAL-TEST Band)", xlabel="Dataset", ylabel="Score")
    ax.legend(loc="upper left")
    save_or_show(fig, out_path)


def plot_vocab_encoding_overview(meta_list, out_path):
    if not meta_list:
        return
    rows = []
    for meta in meta_list:
        rows.append({
            "dataset": meta["name"],
            "selected_features": int(len(meta.get("selected_features", []))),
            "numeric_features": int(len(meta.get("num_cols", []))),
            "categorical_features": int(len(meta.get("cat_cols", []))),
            "sequence_length": int(meta.get("sequence_length", 1 + len(meta.get("num_cols", [])) + len(meta.get("cat_cols", [])))),
            "total_vocab_size": int(meta.get("total_vocab_size", sum(meta.get("cards", [])))),
        })
    df = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    xs = np.arange(len(df))
    fig, ax1 = plt.subplots(figsize=(12.8, 6.8))
    ax1.bar(xs, df["numeric_features"], color=ACCENT_1, alpha=0.86, label="Numeric Features")
    ax1.bar(xs, df["categorical_features"], bottom=df["numeric_features"], color=ACCENT_3, alpha=0.86, label="Categorical Features")
    ax1.plot(xs, df["selected_features"], color=ACCENT_6, lw=2.4, marker="o", label="Selected Features")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(df["dataset"].tolist(), rotation=20, ha="right")
    prettify_ax(ax1, title=f"{PROCESS_NAME} - Vocab / Encoding Overview Across Datasets", xlabel="Dataset", ylabel="Feature / Token Counts")
    ax2 = ax1.twinx()
    ax2.plot(xs, df["sequence_length"], color=ACCENT_4, lw=2.2, marker="D", linestyle="--", label="Sequence Length")
    ax2.plot(xs, df["total_vocab_size"], color=ACCENT_5, lw=2.2, marker="s", linestyle=":", label="Total Vocab Size")
    ax2.set_ylabel("Sequence / Vocab Scale", color=SUBTEXT_CLR)
    ax2.tick_params(colors=SUBTEXT_CLR)
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper left", ncol=2, fontsize=8.8)
    save_or_show(fig, out_path)


def plot_tasktype_overview(meta_list, out_path):
    if not meta_list:
        return
    rows = []
    for meta in meta_list:
        rows.append({
            "dataset": meta["name"],
            "n_classes": int(meta.get("n_classes", 0)),
            "task_family": str(meta.get("task_family", "binary" if int(meta.get("n_classes", 0)) == 2 else "multiclass")),
            "binary_view_available": bool(meta.get("binary_view_available", False)),
        })
    df = pd.DataFrame(rows).sort_values(["n_classes", "dataset"], ascending=[False, True]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11.8, max(5.4, 0.75 * len(df) + 2.2)))
    y = np.arange(len(df))
    colors = [ACCENT_1 if x == "binary" else ACCENT_6 for x in df["task_family"]]
    ax.barh(y, df["n_classes"], color=colors, alpha=0.90)
    ax.set_yticks(y)
    ax.set_yticklabels(df["dataset"].tolist())
    prettify_ax(ax, title=f"{PROCESS_NAME} - Resolved Task Type / Class Count by Dataset", xlabel="Number of Classes", ylabel="Dataset")
    for i, row in df.iterrows():
        flag = "binary-view" if row["binary_view_available"] else "no-binary-view"
        ax.text(float(row["n_classes"]) + 0.04, i, f"{row['task_family']} | {flag}", va="center", fontsize=9, color=TEXT_CLR)
    save_or_show(fig, out_path)


def plot_dataset_metric_scatter(report_df, split, out_path):
    df = report_df[(report_df["split"] == split) & (report_df["dataset"] != "global_weighted")].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9.6, 7.0))
    xs = pd.to_numeric(df["acc"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ys = pd.to_numeric(df["f1_macro"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    aucs = pd.to_numeric(df["auc_roc_macro_ovr"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    sizes = 220 + 900 * np.clip(aucs, 0, 1)
    sc = ax.scatter(xs, ys, s=sizes, c=aucs, cmap=LinearSegmentedColormap.from_list("score_scatter", ["#dbeafe", "#60a5fa", "#1d4ed8"]), edgecolors="#334155", alpha=0.84)
    for _, row in df.iterrows():
        ax.text(float(row["acc"]) + 0.004, float(row["f1_macro"]) + 0.004, str(row["dataset"]), fontsize=8.8, color=TEXT_CLR)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="ROC AUC")
    prettify_ax(ax, title=f"{PROCESS_NAME} - Dataset Score Landscape ({split})", xlabel="Accuracy", ylabel="F1 Macro")
    ax.set_xlim(max(0.0, xs.min() - 0.03), min(1.02, xs.max() + 0.05))
    ax.set_ylim(max(0.0, ys.min() - 0.03), min(1.02, ys.max() + 0.05))
    save_or_show(fig, out_path)


def plot_calibration_gap_heatmap(calibration_store, meta_list, split_key_suffix, out_path):
    labels = []
    gaps = []
    counts = []
    for meta in meta_list:
        key = f"{meta['name']}_{split_key_suffix}"
        if key not in calibration_store:
            continue
        tab = pd.DataFrame(calibration_store[key]["table"])
        labels.append(meta["name"])
        gaps.append(pd.to_numeric(tab["gap"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
        counts.append(pd.to_numeric(tab["count"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    gkey = f"global_{split_key_suffix}"
    if gkey in calibration_store:
        tab = pd.DataFrame(calibration_store[gkey]["table"])
        labels.append("GLOBAL")
        gaps.append(pd.to_numeric(tab["gap"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
        counts.append(pd.to_numeric(tab["count"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    if not labels:
        return
    gap_mat = np.vstack(gaps)
    cnt_mat = np.vstack(counts)
    fig, axes = plt.subplots(2, 1, figsize=(12.8, max(6.6, 0.7 * len(labels) + 4.0)), gridspec_kw={"height_ratios": [3.0, 2.1]})
    ax = axes[0]
    cmap_gap = LinearSegmentedColormap.from_list("calgap", ["#ecfeff", "#67e8f9", "#2563eb", "#7c3aed"])
    im = ax.imshow(gap_mat, aspect="auto", cmap=cmap_gap)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="|Confidence - Empirical Frequency|")
    ax.set_xticks(np.arange(gap_mat.shape[1]))
    ax.set_xticklabels([f"B{i}" for i in range(gap_mat.shape[1])])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    prettify_ax(ax, title=f"{PROCESS_NAME} - Calibration Gap Heatmap ({split_key_suffix.replace('_', ' ').title()})", xlabel="Calibration Bin", ylabel="Dataset")
    ax.grid(False)
    for i in range(gap_mat.shape[0]):
        for j in range(gap_mat.shape[1]):
            ax.text(j, i, f"{gap_mat[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="#0f172a")
    ax2 = axes[1]
    cmap_cnt = LinearSegmentedColormap.from_list("calcnt", ["#f8fafc", "#fde68a", "#fb7185"])
    im2 = ax2.imshow(np.log1p(cnt_mat), aspect="auto", cmap=cmap_cnt)
    plt.colorbar(im2, ax=ax2, fraction=0.03, pad=0.02, label="log(1 + Count)")
    ax2.set_xticks(np.arange(cnt_mat.shape[1]))
    ax2.set_xticklabels([f"B{i}" for i in range(cnt_mat.shape[1])])
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_yticklabels(labels)
    prettify_ax(ax2, xlabel="Calibration Bin", ylabel="Dataset")
    ax2.grid(False)
    save_or_show(fig, out_path)


def plot_calibration_tradeoff_bubble(calibration_store, meta_list, split_key_suffix, out_path):
    rows = []
    for meta in meta_list:
        key = f"{meta['name']}_{split_key_suffix}"
        if key in calibration_store:
            tab = pd.DataFrame(calibration_store[key]["table"])
            rows.append({
                "dataset": meta["name"],
                "ece": float(calibration_store[key]["ece"]),
                "brier": float(calibration_store[key]["brier"]),
                "attack_rate": float(calibration_store[key].get("attack_rate", np.nan)),
                "count": float(pd.to_numeric(tab["count"], errors="coerce").fillna(0.0).sum()),
            })
    gkey = f"global_{split_key_suffix}"
    if gkey in calibration_store:
        tab = pd.DataFrame(calibration_store[gkey]["table"])
        rows.append({
            "dataset": "GLOBAL",
            "ece": float(calibration_store[gkey]["ece"]),
            "brier": float(calibration_store[gkey]["brier"]),
            "attack_rate": float(calibration_store[gkey].get("attack_rate", np.nan)),
            "count": float(pd.to_numeric(tab["count"], errors="coerce").fillna(0.0).sum()),
        })
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["ece", "brier", "dataset"]).reset_index(drop=True)
    sizes = 220 + 900 * (df["count"].to_numpy(dtype=float) / max(float(df["count"].max()), 1.0))
    colors = pd.to_numeric(df["attack_rate"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.8, 7.2))
    sc = ax.scatter(df["ece"], df["brier"], s=sizes, c=colors, cmap=LinearSegmentedColormap.from_list("caltrade", ["#dbeafe", "#22d3ee", "#f97316"]), alpha=0.84, edgecolors="#334155")
    for _, row in df.iterrows():
        ax.text(float(row["ece"]) + 0.002, float(row["brier"]) + 0.002, str(row["dataset"]), fontsize=8.8, color=TEXT_CLR)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="Attack Rate")
    prettify_ax(ax, title=f"{PROCESS_NAME} - Calibration Trade-off Map ({split_key_suffix.replace('_', ' ').title()})", xlabel="ECE (Lower is Better)", ylabel="Brier Score (Lower is Better)")
    save_or_show(fig, out_path)


def plot_reliability_band_overlay(calibration_store, meta_list, split_key_suffix, out_path):
    series = []
    centers = None
    for meta in meta_list:
        key = f"{meta['name']}_{split_key_suffix}"
        if key not in calibration_store:
            continue
        tab = pd.DataFrame(calibration_store[key]["table"])
        centers = 0.5 * (pd.to_numeric(tab["bin_left"], errors="coerce").fillna(0.0).to_numpy(dtype=float) + pd.to_numeric(tab["bin_right"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
        emp = pd.to_numeric(tab["emp_acc"], errors="coerce").to_numpy(dtype=float)
        emp[pd.to_numeric(tab["count"], errors="coerce").fillna(0.0).to_numpy(dtype=float) <= 0] = np.nan
        series.append(emp)
    if centers is None or not series:
        return
    mat = np.vstack(series)
    med = np.nanmedian(mat, axis=0)
    lo = np.nanmin(mat, axis=0)
    hi = np.nanmax(mat, axis=0)
    fig, ax = plt.subplots(figsize=(9.4, 6.8))
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1.5, label="Perfect Calibration")
    ax.fill_between(centers, lo, hi, color=ACCENT_2, alpha=0.14, label="Dataset Min-Max Band")
    ax.plot(centers, med, color=ACCENT_6, lw=2.8, marker="o", label="Dataset Median Curve")
    gkey = f"global_{split_key_suffix}"
    if gkey in calibration_store:
        tabg = pd.DataFrame(calibration_store[gkey]["table"])
        mask = pd.to_numeric(tabg["count"], errors="coerce").fillna(0.0) > 0
        ax.plot(tabg.loc[mask, "mean_conf"], tabg.loc[mask, "emp_acc"], color=ACCENT_5, lw=2.6, marker="D", label=f"GLOBAL (ECE={calibration_store[gkey]['ece']:.3f})")
    prettify_ax(ax, title=f"{PROCESS_NAME} - Reliability Band Overlay ({split_key_suffix.replace('_', ' ').title()})", xlabel="Predicted Attack Probability", ylabel="Observed Attack Frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8.6)
    save_or_show(fig, out_path)


def plot_error_dashboard(error_summary_df, error_df, top_conf_df, out_path):
    if error_summary_df.empty or error_df.empty:
        return
    df = error_summary_df.sort_values("dataset").reset_index(drop=True)
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 11.2))
    ax = axes[0, 0]
    xs = np.arange(len(df))
    bars = ax.bar(xs, df["error_rate"], color=[PALETTE[i % len(PALETTE)] for i in range(len(df))], alpha=0.88, label="Error Rate")
    ax.set_xticks(xs)
    ax.set_xticklabels(df["dataset"].tolist(), rotation=20, ha="right")
    prettify_ax(ax, title=f"{PROCESS_NAME} - Error Rate by Dataset", xlabel="Dataset", ylabel="Error Rate")
    ax2 = ax.twinx()
    ax2.plot(xs, pd.to_numeric(df["mean_conf_wrong"], errors="coerce").fillna(0.0), color=ACCENT_5, marker="o", lw=2.2, label="Mean Wrong Confidence")
    ax2.set_ylabel("Mean Wrong Confidence", color=SUBTEXT_CLR)
    ax2.tick_params(colors=SUBTEXT_CLR)
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=8.5)

    ax = axes[0, 1]
    sizes = 180 + 900 * pd.to_numeric(df["error_rate"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cols = pd.to_numeric(df["mean_margin_wrong"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    sc = ax.scatter(pd.to_numeric(df["mean_conf_wrong"], errors="coerce").fillna(0.0), pd.to_numeric(df["mean_margin_wrong"], errors="coerce").fillna(0.0), s=sizes, c=cols, cmap=LinearSegmentedColormap.from_list("errdash", ["#fef3c7", "#60a5fa", "#7c3aed"]), edgecolors="#334155", alpha=0.84)
    for _, row in df.iterrows():
        ax.text(float(row["mean_conf_wrong"]) + 0.004 if pd.notna(row["mean_conf_wrong"]) else 0.01, float(row["mean_margin_wrong"]) + 0.004 if pd.notna(row["mean_margin_wrong"]) else 0.01, str(row["dataset"]), fontsize=8.4, color=TEXT_CLR)
    plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="Mean Wrong Margin")
    prettify_ax(ax, title=f"{PROCESS_NAME} - Error Difficulty Map", xlabel="Mean Wrong Confidence", ylabel="Mean Wrong Margin")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1, 0]
    corr = error_df.loc[error_df["correct"] == 1, "confidence"].to_numpy()
    wrg = error_df.loc[error_df["correct"] == 0, "confidence"].to_numpy()
    bins = np.linspace(0, 1, 26)
    if len(corr):
        ax.hist(corr, bins=bins, density=True, alpha=0.58, color=ACCENT_3, label=f"Correct (n={len(corr)})")
    if len(wrg):
        ax.hist(wrg, bins=bins, density=True, alpha=0.58, color=ACCENT_5, label=f"Wrong (n={len(wrg)})")
    prettify_ax(ax, title=f"{PROCESS_NAME} - Pooled Confidence Distribution", xlabel="Max Predicted Probability", ylabel="Density")
    ax.legend(loc="upper center")

    ax = axes[1, 1]
    topn = top_conf_df.head(10).iloc[::-1] if not top_conf_df.empty else pd.DataFrame(columns=["pair", "count"])
    if not topn.empty:
        ax.barh(np.arange(len(topn)), topn["count"], color=ACCENT_4, alpha=0.90)
        ax.set_yticks(np.arange(len(topn)))
        ax.set_yticklabels(topn["pair"].tolist(), fontsize=8.4)
    prettify_ax(ax, title=f"{PROCESS_NAME} - Top Error Pairs", xlabel="Count", ylabel="True -> Pred")
    save_or_show(fig, out_path)


def plot_error_entropy_violin(error_df, out_path):
    if error_df.empty:
        return
    datasets = sorted(error_df["dataset"].unique().tolist())
    fig, ax = plt.subplots(figsize=(13.6, max(6.0, 0.62 * len(datasets) + 4.0)))
    pos = []
    data = []
    colors = []
    labels = []
    for i, ds in enumerate(datasets):
        g = error_df[error_df["dataset"] == ds]
        corr = g.loc[g["correct"] == 1, "entropy"].dropna().to_numpy()
        wrg = g.loc[g["correct"] == 0, "entropy"].dropna().to_numpy()
        pos.extend([2 * i, 2 * i + 0.8])
        data.extend([corr if len(corr) else np.array([0.0]), wrg if len(wrg) else np.array([0.0])])
        colors.extend([ACCENT_3, ACCENT_5])
        labels.extend([f"{ds}\nCorrect", f"{ds}\nWrong"])
    vp = ax.violinplot(data, positions=pos, widths=0.65, showmeans=True, showmedians=False, showextrema=False)
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#334155")
        body.set_alpha(0.62)
    vp["cmeans"].set_color("#0f172a")
    vp["cmeans"].set_linewidth(1.2)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    prettify_ax(ax, title=f"{PROCESS_NAME} - Entropy Distribution by Dataset and Correctness", xlabel="Dataset / Outcome", ylabel="Predictive Entropy")
    save_or_show(fig, out_path)


plot_round_history(round_history, "artifacts/plots/round_history.png")
plot_dataset_metric_bars(report, "VAL", "artifacts/plots/val_metrics_bar.png")
plot_dataset_metric_bars(report, "TEST", "artifacts/plots/test_metrics_bar.png")
plot_global_weighted_summary(report, "artifacts/plots/global_weighted_summary.png")
plot_class_distribution(all_meta, "artifacts/plots/class_distribution_all_datasets.png")
plot_client_distribution(all_meta, "artifacts/plots/client_distribution.png")

for meta in all_meta:
    name = meta["name"]
    plot_roc_multiclass(
        roc_store[f"{name}_test_multiclass"],
        f"ROC-AUC - {name} Test (Multiclass One-vs-Rest)",
        f"artifacts/plots/{name}_test_multiclass_roc.png",
    )
    plot_roc_multiclass(
        roc_store[f"{name}_val_multiclass"],
        f"ROC-AUC - {name} Val (Multiclass One-vs-Rest)",
        f"artifacts/plots/{name}_val_multiclass_roc.png",
    )
    if f"{name}_test_binary" in roc_store:
        plot_roc_binary(
            roc_store[f"{name}_test_binary"],
            f"Attack-vs-Normal ROC - {name} Test",
            f"artifacts/plots/{name}_test_binary_roc.png",
        )
        plot_pr_binary(
            pr_store.get(f"{name}_test_binary"),
            f"Attack-vs-Normal PR Curve - {name} Test",
            f"artifacts/plots/{name}_test_binary_pr.png",
        )
    if f"{name}_val_binary" in roc_store:
        plot_roc_binary(
            roc_store[f"{name}_val_binary"],
            f"Attack-vs-Normal ROC - {name} Val",
            f"artifacts/plots/{name}_val_binary_roc.png",
        )
        plot_pr_binary(
            pr_store.get(f"{name}_val_binary"),
            f"Attack-vs-Normal PR Curve - {name} Val",
            f"artifacts/plots/{name}_val_binary_pr.png",
        )

    plot_confusion(
        test_preds[name]["y"],
        test_preds[name]["p"].argmax(1),
        [str(x) for x in meta["class_names"]],
        f"{name} Test Confusion Matrix",
        f"artifacts/plots/{name}_test_confusion.png",
    )
    plot_confusion(
        val_preds[name]["y"],
        val_preds[name]["p"].argmax(1),
        [str(x) for x in meta["class_names"]],
        f"{name} Val Confusion Matrix",
        f"artifacts/plots/{name}_val_confusion.png",
    )

overlay_test_keys = [f"{m['name']}_test_binary" for m in all_meta if f"{m['name']}_test_binary" in roc_store]
overlay_val_keys = [f"{m['name']}_val_binary" for m in all_meta if f"{m['name']}_val_binary" in roc_store]
if overlay_test_keys:
    plot_binary_roc_overlay(
        roc_store,
        overlay_test_keys,
        f"{PROCESS_NAME} - All Datasets Overlay Binary Test ROC",
        "artifacts/plots/all_datasets_test_binary_roc_overlay.png",
    )
if overlay_val_keys:
    plot_binary_roc_overlay(
        roc_store,
        overlay_val_keys,
        f"{PROCESS_NAME} - All Datasets Overlay Binary Val ROC",
        "artifacts/plots/all_datasets_val_binary_roc_overlay.png",
    )

if "global_test_binary" in roc_store:
    plot_roc_binary(
        roc_store["global_test_binary"],
        f"{PROCESS_NAME} - Global Binary ROC (All Datasets Test)",
        "artifacts/plots/global_test_binary_roc.png",
    )
    plot_pr_binary(
        pr_store.get("global_test_binary"),
        f"{PROCESS_NAME} - Global Binary PR Curve (All Datasets Test)",
        "artifacts/plots/global_test_binary_pr.png",
    )
    y_all = np.concatenate([y for _, y, _ in combined_binary_test])
    p_all = np.concatenate([p for _, _, p in combined_binary_test], axis=0)
    plot_confusion(
        y_all,
        p_all.argmax(1),
        ["Normal", "Attack"],
        f"{PROCESS_NAME} - Global Test Binary Confusion Matrix",
        "artifacts/plots/global_test_binary_confusion.png",
    )
    plot_reliability_diagram(
        calibration_store["global_test_binary"],
        f"{PROCESS_NAME} - Global Calibration (All Datasets Test)",
        "artifacts/plots/global_test_binary_calibration.png",
    )

if "global_val_binary" in roc_store:
    plot_roc_binary(
        roc_store["global_val_binary"],
        f"{PROCESS_NAME} - Global Binary ROC (All Datasets Val)",
        "artifacts/plots/global_val_binary_roc.png",
    )
    plot_pr_binary(
        pr_store.get("global_val_binary"),
        f"{PROCESS_NAME} - Global Binary PR Curve (All Datasets Val)",
        "artifacts/plots/global_val_binary_pr.png",
    )
    y_all = np.concatenate([y for _, y, _ in combined_binary_val])
    p_all = np.concatenate([p for _, _, p in combined_binary_val], axis=0)
    plot_confusion(
        y_all,
        p_all.argmax(1),
        ["Normal", "Attack"],
        f"{PROCESS_NAME} - Global Val Binary Confusion Matrix",
        "artifacts/plots/global_val_binary_confusion.png",
    )
    plot_reliability_diagram(
        calibration_store["global_val_binary"],
        f"{PROCESS_NAME} - Global Calibration (All Datasets Val)",
        "artifacts/plots/global_val_binary_calibration.png",
    )

plot_multimetric_dataset_profile(report, "VAL", "artifacts/plots/global_multimetric_profile_val.png")
plot_multimetric_dataset_profile(report, "TEST", "artifacts/plots/global_multimetric_profile_test.png")
plot_metric_heatmap(report, "VAL", "artifacts/plots/global_metric_heatmap_val.png")
plot_metric_heatmap(report, "TEST", "artifacts/plots/global_metric_heatmap_test.png")
plot_global_metric_ribbon(report, "artifacts/plots/global_metric_ribbon_val_test.png")
plot_dataset_metric_scatter(report, "VAL", "artifacts/plots/grip_fl_dataset_score_landscape_val.png")
plot_dataset_metric_scatter(report, "TEST", "artifacts/plots/grip_fl_dataset_score_landscape_test.png")
plot_vocab_encoding_overview(all_meta, "artifacts/plots/grip_fl_vocab_encoding_overview.png")
plot_tasktype_overview(all_meta, "artifacts/plots/grip_fl_task_type_overview.png")
plot_error_profile(error_summary_df, "artifacts/plots/global_error_profile_test.png")
plot_error_scatter(error_summary_df, "artifacts/plots/global_error_scatter_test.png")
plot_confidence_bin_error_curve(error_test_df, "artifacts/plots/global_confidence_bin_error_profile_test.png")
plot_error_dashboard(error_summary_df, error_test_df, top_conf_df, "artifacts/plots/grip_fl_error_dashboard_test.png")
plot_error_entropy_violin(error_test_df, "artifacts/plots/grip_fl_error_entropy_violin_test.png")
plot_calibration_summary_profile(calibration_store, all_meta, "val_binary", "artifacts/plots/global_calibration_profile_val.png")
plot_calibration_summary_profile(calibration_store, all_meta, "test_binary", "artifacts/plots/global_calibration_profile_test.png")
plot_dataset_calibration_bars(calibration_store, all_meta, "val_binary", "artifacts/plots/global_calibration_bars_val.png")
plot_dataset_calibration_bars(calibration_store, all_meta, "test_binary", "artifacts/plots/global_calibration_bars_test.png")
plot_dataset_reliability_overlay(calibration_store, all_meta, "val_binary", "artifacts/plots/global_reliability_overlay_val.png")
plot_dataset_reliability_overlay(calibration_store, all_meta, "test_binary", "artifacts/plots/global_reliability_overlay_test.png")
plot_calibration_gap_heatmap(calibration_store, all_meta, "val_binary", "artifacts/plots/grip_fl_calibration_gap_heatmap_val.png")
plot_calibration_gap_heatmap(calibration_store, all_meta, "test_binary", "artifacts/plots/grip_fl_calibration_gap_heatmap_test.png")
plot_calibration_tradeoff_bubble(calibration_store, all_meta, "val_binary", "artifacts/plots/grip_fl_calibration_tradeoff_val.png")
plot_calibration_tradeoff_bubble(calibration_store, all_meta, "test_binary", "artifacts/plots/grip_fl_calibration_tradeoff_test.png")
plot_reliability_band_overlay(calibration_store, all_meta, "val_binary", "artifacts/plots/grip_fl_reliability_band_val.png")
plot_reliability_band_overlay(calibration_store, all_meta, "test_binary", "artifacts/plots/grip_fl_reliability_band_test.png")

print("\nSaved artifacts:")
print("  - artifacts/final_report.csv")
print("  - artifacts/round_history.json")
print("  - artifacts/dataset_meta.json")
print("  - artifacts/dataset_task_summary.csv")
print("  - artifacts/roc_data.json")
print("  - artifacts/pr_data.json")
print("  - artifacts/calibration_data.json")
print("  - artifacts/error_analysis_val_samples.csv")
print("  - artifacts/error_analysis_test_samples.csv")
print("  - artifacts/error_analysis_summary_test.csv")
print("  - artifacts/error_analysis_by_class_test.csv")
print("  - artifacts/error_analysis_top_confusions_test.csv")
print("  - artifacts/checkpoints/shared_backbone_best.pth")
print("  - artifacts/checkpoints/private_<dataset>_best.pth")
print("  - artifacts/plots/*.png")
print("  - added global multi-metric / error / calibration profile plots")
print("\nDone.")
```

    DEVICE: cuda | PIN_MEMORY: True
    [wittigenz/hydras] -> /kaggle/input/datasets/wittigenz/hydras
    [sampadab17/network-intrusion-detection] -> /kaggle/input/datasets/sampadab17/network-intrusion-detection
    [rebsonramalho/network-threat-detection-dataset] -> /kaggle/input/datasets/rebsonramalho/network-threat-detection-dataset
    [annaamalaiu/wustl-iiot-2021-dataset] -> /kaggle/input/datasets/annaamalaiu/wustl-iiot-2021-dataset
    
    ####################################################################################################
    STARTING I23SUB PREPARATION
    ####################################################################################################
    
    ====================================================================================================
    [I23SUB] DATASET BUILD
    ----------------------------------------------------------------------------------------------------
      path              : /kaggle/input/datasets/wittigenz/hydras
      files_found       : 1
      files_selected    : ['data.csv']
      split_hint        : no named split files
      split_policy      : merge-compatible-labeled-sources-then-fresh-split
      benchmark_note    : provided split files are treated as labeled sources, then re-split fresh
        loaded data.csv: (23145, 18)
        loaded data.csv: (23145, 18)
      target_col        : label
      task_kind         : binary
      split_mode        : fresh-70/15/15 from combined tables
      [INFO] no deterministic target-sibling feature columns detected
      [INFO] dropped ID/high-uniqueness columns: ['Unnamed: 0']
      train/val/test    : 16201 / 3472 / 3472
      n_classes         : 2
      class_names       : ['Malicious', 'Benign']
      normal_class_idx  : 1
      initial_features  : 16 | numeric=10 | categorical=6
      impute_counts     : train(n=37476, c=14875) | val(n=7923, c=3206) | test(n=8073, c=3217)
      cross_features    : use_triples=True | base=['conn_state', 'history', 'id.orig_h', 'id.resp_h', 'proto', 'service'] | new=35
      mi_candidates     : 51
      graph_nodes       : 51
      semantic_protocol : target-resolution + compatibility-gated merge + leakage-safe re-split
    ====================================================================================================
    
    ####################################################################################################
    STARTING K99SUB PREPARATION
    ####################################################################################################
    
    ====================================================================================================
    [K99SUB] DATASET BUILD
    ----------------------------------------------------------------------------------------------------
      path              : /kaggle/input/datasets/sampadab17/network-intrusion-detection
      files_found       : 2
      files_selected    : ['Test_data.csv', 'Train_data.csv']
      split_hint        : named train/val/test files detected
      split_policy      : merge-compatible-labeled-sources-then-fresh-split
      benchmark_note    : provided split files are treated as labeled sources, then re-split fresh
        loaded Train_data.csv: (25192, 42)
        loaded Test_data.csv: (22544, 41)
      target_col        : class
      task_kind         : binary
      split_mode        : fresh-70/15/15 from merged labeled pieces ['train']
      [INFO] no deterministic target-sibling feature columns detected
      train/val/test    : 17634 / 3779 / 3779
      n_classes         : 2
      class_names       : ['anomaly', 'normal']
      normal_class_idx  : 1
      initial_features  : 41 | numeric=38 | categorical=3
      impute_counts     : train(n=0, c=0) | val(n=0, c=0) | test(n=0, c=0)
      cross_features    : use_triples=True | base=['flag', 'protocol_type', 'service'] | new=4
      mi_candidates     : 45
      graph_nodes       : 45
      semantic_protocol : target-resolution + compatibility-gated merge + leakage-safe re-split
    ====================================================================================================
    
    ####################################################################################################
    STARTING NTD1 PREPARATION
    ####################################################################################################
    
    ====================================================================================================
    [NTD1] DATASET BUILD
    ----------------------------------------------------------------------------------------------------
      path              : /kaggle/input/datasets/rebsonramalho/network-threat-detection-dataset
      files_found       : 2
      files_selected    : ['Dataset_completo.csv']
      split_hint        : no named split files
      split_policy      : merge-compatible-labeled-sources-then-fresh-split
      benchmark_note    : provided split files are treated as labeled sources, then re-split fresh
        loaded Dataset_completo.csv: (368017, 29)
        loaded Dataset_completo.csv: (368017, 29)
      target_col        : Label
      task_kind         : binary
      split_mode        : fresh-70/15/15 from combined tables
      [INFO] no deterministic target-sibling feature columns detected
      [INFO] dropped ID/high-uniqueness columns: ['dstip', 'srcip']
      train/val/test    : 257611 / 55203 / 55203
      n_classes         : 2
      class_names       : ['1', '0']
      normal_class_idx  : 1
      initial_features  : 26 | numeric=23 | categorical=3
      impute_counts     : train(n=132038, c=0) | val(n=28534, c=0) | test(n=28315, c=0)
      cross_features    : use_triples=True | base=['ackdat', 'proto', 'synack'] | new=4
      mi_candidates     : 30
      graph_nodes       : 30
      semantic_protocol : target-resolution + compatibility-gated merge + leakage-safe re-split
    ====================================================================================================
    
    ####################################################################################################
    STARTING NTD2 PREPARATION
    ####################################################################################################
    
    ====================================================================================================
    [NTD2] DATASET BUILD
    ----------------------------------------------------------------------------------------------------
      path              : /kaggle/input/datasets/rebsonramalho/network-threat-detection-dataset
      files_found       : 2
      files_selected    : ['Dataset_resumido.csv']
      split_hint        : no named split files
      split_policy      : merge-compatible-labeled-sources-then-fresh-split
      benchmark_note    : provided split files are treated as labeled sources, then re-split fresh
        loaded Dataset_resumido.csv: (55826, 25)
        loaded Dataset_resumido.csv: (55826, 25)
      target_col        : Label
      task_kind         : binary
      split_mode        : fresh-70/15/15 from combined tables
      [INFO] no deterministic target-sibling feature columns detected
      [INFO] dropped ID/high-uniqueness columns: ['dstip', 'srcip']
      train/val/test    : 39078 / 8374 / 8374
      n_classes         : 2
      class_names       : ['1', '0']
      normal_class_idx  : 1
      initial_features  : 22 | numeric=12 | categorical=10
      impute_counts     : train(n=145432, c=187971) | val(n=31850, c=40688) | test(n=30886, c=40104)
      cross_features    : use_triples=True | base=['Ltime', 'Stime', 'ackdat', 'ct_state_ttl', 'proto', 'res_bdy_len'] | new=35
      mi_candidates     : 57
      graph_nodes       : 57
      semantic_protocol : target-resolution + compatibility-gated merge + leakage-safe re-split
    ====================================================================================================
    
    ####################################################################################################
    STARTING WII21 PREPARATION
    ####################################################################################################
    
    ====================================================================================================
    [WII21] DATASET BUILD
    ----------------------------------------------------------------------------------------------------
      path              : /kaggle/input/datasets/annaamalaiu/wustl-iiot-2021-dataset
      files_found       : 1
      files_selected    : ['wustl_iiot_2021.csv']
      split_hint        : no named split files
      split_policy      : merge-compatible-labeled-sources-then-fresh-split
      benchmark_note    : provided split files are treated as labeled sources, then re-split fresh
        loaded wustl_iiot_2021.csv: (1194464, 49)
        loaded wustl_iiot_2021.csv: (1194464, 49)
      target_col        : Target
      task_kind         : binary
      split_mode        : fresh-70/15/15 from combined tables
      [INFO] no deterministic target-sibling feature columns detected
      train/val/test    : 836124 / 179170 / 179170
      n_classes         : 2
      class_names       : ['1', '0']
      normal_class_idx  : 1
      initial_features  : 48 | numeric=43 | categorical=5
      impute_counts     : train(n=0, c=0) | val(n=0, c=0) | test(n=0, c=0)
      cross_features    : use_triples=False | base=['DstAddr', 'LastTime', 'SrcAddr', 'StartTime', 'Traffic'] | new=10
      mi_candidates     : 58
      graph_nodes       : 58
      semantic_protocol : target-resolution + compatibility-gated merge + leakage-safe re-split
    ====================================================================================================
    
    ####################################################################################################
    GRIP-DFFI FEDERATED FEATURE INTELLIGENCE
    ####################################################################################################
    
    ====================================================================================================
    FEATURE INTELLIGENCE ROUND 1/3
    ====================================================================================================
      [I23Sub] mean_local_relevance=0.8125 | nodes=51
      [K99Sub] mean_local_relevance=0.3176 | nodes=45
      [NTD1] mean_local_relevance=0.3360 | nodes=30
      [NTD2] mean_local_relevance=0.7014 | nodes=57
      [WII21] mean_local_relevance=0.7563 | nodes=58
    
    ====================================================================================================
    FEATURE INTELLIGENCE ROUND 2/3
    ====================================================================================================
      [I23Sub] mean_local_relevance=0.8156 | nodes=51
      [K99Sub] mean_local_relevance=0.3198 | nodes=45
      [NTD1] mean_local_relevance=0.3358 | nodes=30
      [NTD2] mean_local_relevance=0.7029 | nodes=57
      [WII21] mean_local_relevance=0.7574 | nodes=58
    
    ====================================================================================================
    FEATURE INTELLIGENCE ROUND 3/3
    ====================================================================================================
      [I23Sub] mean_local_relevance=0.8141 | nodes=51
      [K99Sub] mean_local_relevance=0.3167 | nodes=45
      [NTD1] mean_local_relevance=0.3359 | nodes=30
      [NTD2] mean_local_relevance=0.7019 | nodes=57
      [WII21] mean_local_relevance=0.7592 | nodes=58
      [I23Sub] selected_features=24 / 51
      [I23Sub] selected_top10=['history', 'id.resp_h', 'cross_conn_state__history', 'cross_conn_state__id.resp_h', 'cross_conn_state__proto', 'cross_history__id.orig_h', 'cross_history__proto', 'cross_history__service', 'cross_id.orig_h__proto', 'cross_id.resp_h__service']...
      [K99Sub] selected_features=22 / 45
      [K99Sub] selected_top10=['count', 'diff_srv_rate', 'dst_bytes', 'dst_host_count', 'dst_host_same_src_port_rate', 'dst_host_same_srv_rate', 'dst_host_serror_rate', 'dst_host_srv_count', 'dst_host_srv_diff_host_rate', 'hot']...
      [NTD1] selected_features=11 / 30
      [NTD1] selected_top10=['Ltime', 'Stime', 'dsport', 'sbytes', 'sport', 'sttl', 'swin', 'proto', 'cross_ackdat__proto', 'cross_proto__synack']...
      [NTD2] selected_features=33 / 57
      [NTD2] selected_top10=['Dload', 'Dpkts', 'Sload', 'Spkts', 'dbytes', 'dsport', 'sbytes', 'smeansz', 'Ltime', 'state']...
      [WII21] selected_features=29 / 58
      [WII21] selected_top10=['DIntPkt', 'DstBytes', 'DstLoad', 'DstPkts', 'Dur', 'IdleTime', 'Min', 'Rate', 'RunTime', 'Sport']...
    
    ----------------------------------------------------------------------------------------------------
    GLOBAL SHARED FEATURE KEYS : 18
    UNIVERSALITY THRESHOLD     : 0.60
    ----------------------------------------------------------------------------------------------------




<style>
  .grip-table-block {
    margin: 12px 0 18px 0;
    border: 1px solid #dbe4ee;
    border-radius: 12px;
    overflow-x: auto;
    background: #ffffff;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.05);
  }
  .grip-table-title {
    padding: 10px 14px;
    font-weight: 700;
    color: #0f172a;
    background: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
    font-size: 15px;
  }
  .grip-table-block table.dataframe {
    border-collapse: separate !important;
    border-spacing: 0;
    width: max-content;
    min-width: 100%;
    margin: 0;
    font-size: 13px;
  }
  .grip-table-block table.dataframe thead th {
    position: sticky;
    top: 0;
    background: #eff6ff;
    color: #0f172a;
    text-align: left;
    padding: 9px 10px;
    border-bottom: 1px solid #cbd5e1;
    white-space: nowrap;
  }
  .grip-table-block table.dataframe tbody td {
    padding: 8px 10px;
    border-bottom: 1px solid #eef2f7;
    vertical-align: top;
    white-space: normal;
    word-break: break-word;
  }
  .grip-table-block table.dataframe tbody tr:nth-child(even) {
    background: #fcfdff;
  }
  .grip-table-block table.dataframe tbody tr:hover {
    background: #f8fbff;
  }
  .grip-note {
    color: #475569;
    font-size: 12px;
    margin: 6px 0 12px 2px;
  }
</style>




<div class="grip-table-block"><div class="grip-table-title">[I23SUB] FINAL ROUTED DATASET</div><table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Field</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>selected_features</td>
      <td>24</td>
    </tr>
    <tr>
      <td>shared_route</td>
      <td>num=0 | cat=0</td>
    </tr>
    <tr>
      <td>private_route</td>
      <td>num=0 | cat=24</td>
    </tr>
    <tr>
      <td>sequence_length</td>
      <td>25</td>
    </tr>
    <tr>
      <td>n_clients</td>
      <td>2</td>
    </tr>
    <tr>
      <td>client_sizes</td>
      <td>[9854, 6347]</td>
    </tr>
    <tr>
      <td>global_relevance_mean</td>
      <td>0.6702</td>
    </tr>
  </tbody>
</table></div><div class="grip-note">Client sizes are wrapped across lines to keep notebook output readable.</div>



<div class="grip-table-block"><div class="grip-table-title">[K99SUB] FINAL ROUTED DATASET</div><table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Field</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>selected_features</td>
      <td>22</td>
    </tr>
    <tr>
      <td>shared_route</td>
      <td>num=0 | cat=1</td>
    </tr>
    <tr>
      <td>private_route</td>
      <td>num=16 | cat=5</td>
    </tr>
    <tr>
      <td>sequence_length</td>
      <td>23</td>
    </tr>
    <tr>
      <td>n_clients</td>
      <td>4</td>
    </tr>
    <tr>
      <td>client_sizes</td>
      <td>[1006, 1816, 8402, 6410]</td>
    </tr>
    <tr>
      <td>global_relevance_mean</td>
      <td>0.6159</td>
    </tr>
  </tbody>
</table></div><div class="grip-note">Client sizes are wrapped across lines to keep notebook output readable.</div>



<div class="grip-table-block"><div class="grip-table-title">[NTD1] FINAL ROUTED DATASET</div><table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Field</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>selected_features</td>
      <td>11</td>
    </tr>
    <tr>
      <td>shared_route</td>
      <td>num=3 | cat=2</td>
    </tr>
    <tr>
      <td>private_route</td>
      <td>num=4 | cat=2</td>
    </tr>
    <tr>
      <td>sequence_length</td>
      <td>12</td>
    </tr>
    <tr>
      <td>n_clients</td>
      <td>10</td>
    </tr>
    <tr>
      <td>client_sizes</td>
      <td>[37754, 2209, 48001, 12872, 62448, 27705, 1357, 5166,<br> 5839, 54260]</td>
    </tr>
    <tr>
      <td>global_relevance_mean</td>
      <td>0.6055</td>
    </tr>
  </tbody>
</table></div><div class="grip-note">Client sizes are wrapped across lines to keep notebook output readable.</div>



<div class="grip-table-block"><div class="grip-table-title">[NTD2] FINAL ROUTED DATASET</div><table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Field</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>selected_features</td>
      <td>33</td>
    </tr>
    <tr>
      <td>shared_route</td>
      <td>num=8 | cat=0</td>
    </tr>
    <tr>
      <td>private_route</td>
      <td>num=0 | cat=25</td>
    </tr>
    <tr>
      <td>sequence_length</td>
      <td>34</td>
    </tr>
    <tr>
      <td>n_clients</td>
      <td>12</td>
    </tr>
    <tr>
      <td>client_sizes</td>
      <td>[2230, 3379, 1789, 142, 2426, 1, 4736, 3988,<br> 4083, 4239, 8696, 3369]</td>
    </tr>
    <tr>
      <td>global_relevance_mean</td>
      <td>0.7384</td>
    </tr>
  </tbody>
</table></div><div class="grip-note">Client sizes are wrapped across lines to keep notebook output readable.</div>



<div class="grip-table-block"><div class="grip-table-title">[WII21] FINAL ROUTED DATASET</div><table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Field</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>selected_features</td>
      <td>29</td>
    </tr>
    <tr>
      <td>shared_route</td>
      <td>num=0 | cat=0</td>
    </tr>
    <tr>
      <td>private_route</td>
      <td>num=17 | cat=12</td>
    </tr>
    <tr>
      <td>sequence_length</td>
      <td>30</td>
    </tr>
    <tr>
      <td>n_clients</td>
      <td>46</td>
    </tr>
    <tr>
      <td>client_sizes</td>
      <td>[8354, 25751, 42801, 15271, 17243, 44654, 47255, 457,<br> 12388, 12876, 6523, 6709, 3239, 259, 8000, 44321,<br> 87021, 6986, 9329, 37143, 34755, 7268, 15886, 18446,<br> 21907, 15749, 41900, 22675, 22659, 6231, 11135, 10805,<br> 11918, 12890, 5447, 743, 52117, 3378, 7628, 10187,<br> 2932, 36349, 5003, 14026, 4514, 2996]</td>
    </tr>
    <tr>
      <td>global_relevance_mean</td>
      <td>0.6486</td>
    </tr>
  </tbody>
</table></div><div class="grip-note">Client sizes are wrapped across lines to keep notebook output readable.</div>



<div class="grip-table-block"><div class="grip-table-title">GRIP-DFFI DATASET TASK / ENCODING SUMMARY</div><table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>dataset</th>
      <th>task</th>
      <th>family</th>
      <th>normal_idx</th>
      <th>bin_view</th>
      <th>selected_feats</th>
      <th>shared_feats</th>
      <th>private_feats</th>
      <th>num_feats</th>
      <th>cat_feats</th>
      <th>seq_len</th>
      <th>vocab_size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>I23Sub</td>
      <td>binary</td>
      <td>binary</td>
      <td>1</td>
      <td>True</td>
      <td>24</td>
      <td>0</td>
      <td>24</td>
      <td>0</td>
      <td>24</td>
      <td>25</td>
      <td>726</td>
    </tr>
    <tr>
      <td>K99Sub</td>
      <td>binary</td>
      <td>binary</td>
      <td>1</td>
      <td>True</td>
      <td>22</td>
      <td>1</td>
      <td>21</td>
      <td>16</td>
      <td>6</td>
      <td>23</td>
      <td>4,613</td>
    </tr>
    <tr>
      <td>NTD1</td>
      <td>binary</td>
      <td>binary</td>
      <td>1</td>
      <td>True</td>
      <td>11</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>4</td>
      <td>12</td>
      <td>4,137</td>
    </tr>
    <tr>
      <td>NTD2</td>
      <td>binary</td>
      <td>binary</td>
      <td>1</td>
      <td>True</td>
      <td>33</td>
      <td>8</td>
      <td>25</td>
      <td>8</td>
      <td>25</td>
      <td>34</td>
      <td>534,061</td>
    </tr>
    <tr>
      <td>WII21</td>
      <td>binary</td>
      <td>binary</td>
      <td>1</td>
      <td>True</td>
      <td>29</td>
      <td>0</td>
      <td>29</td>
      <td>17</td>
      <td>12</td>
      <td>30</td>
      <td>232,441</td>
    </tr>
  </tbody>
</table></div><div class="grip-note">Column labels are shortened only for display. Full names are preserved in artifacts/dataset_task_summary.csv.</div>


    
    GRIP-DFFI OVERALL TASK NATURE: ALL_BINARY
    ####################################################################################################
    
    ====================================================================================================
    FEDERATED ROUND 1/15
    ====================================================================================================
    
    [I23Sub] 2 clients
      client=00 | n= 9854 | train_acc=0.9969 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9972 | time=2.8s
      client=01 | n= 6347 | train_acc=0.9983 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9974 | time=0.8s
    
    [K99Sub] 4 clients
      client=00 | n= 1006 | train_acc=0.8986 | val_acc=0.8018 | val_f1=0.7992 | val_auc=0.9790 | time=0.3s
      client=01 | n= 1816 | train_acc=0.9802 | val_acc=0.8963 | val_f1=0.8933 | val_auc=0.9768 | time=0.3s
      client=02 | n= 8402 | train_acc=0.9395 | val_acc=0.9460 | val_f1=0.9459 | val_auc=0.9900 | time=1.0s
      client=03 | n= 6410 | train_acc=0.9733 | val_acc=0.9513 | val_f1=0.9513 | val_auc=0.9953 | time=0.8s
    
    [NTD1] 10 clients
      client=00 | n=37754 | train_acc=0.9592 | val_acc=0.9727 | val_f1=0.9039 | val_auc=0.9889 | time=2.9s
      client=01 | n= 2209 | train_acc=0.9959 | val_acc=0.7480 | val_f1=0.6214 | val_auc=0.9634 | time=0.3s
      client=02 | n=48001 | train_acc=0.9896 | val_acc=0.9150 | val_f1=0.4778 | val_auc=0.9852 | time=3.6s
      client=03 | n=12872 | train_acc=0.9396 | val_acc=0.9476 | val_f1=0.8565 | val_auc=0.9860 | time=1.0s
      client=04 | n=62448 | train_acc=0.9939 | val_acc=0.9150 | val_f1=0.4778 | val_auc=0.9819 | time=4.2s
      client=05 | n=27705 | train_acc=0.9767 | val_acc=0.9669 | val_f1=0.8786 | val_auc=0.9841 | time=2.0s
      client=06 | n= 1357 | train_acc=0.9027 | val_acc=0.8062 | val_f1=0.6736 | val_auc=0.9691 | time=0.3s
      client=07 | n= 5166 | train_acc=0.9439 | val_acc=0.8504 | val_f1=0.7206 | val_auc=0.9838 | time=0.5s
      client=08 | n= 5839 | train_acc=0.9233 | val_acc=0.9423 | val_f1=0.8462 | val_auc=0.9850 | time=0.6s
      client=09 | n=54260 | train_acc=0.9779 | val_acc=0.9669 | val_f1=0.8786 | val_auc=0.9878 | time=3.9s
    
    [NTD2] 12 clients
      client=00 | n= 2230 | train_acc=0.9520 | val_acc=0.7946 | val_f1=0.7905 | val_auc=0.9634 | time=0.7s
      client=01 | n= 3379 | train_acc=0.9458 | val_acc=0.7910 | val_f1=0.7865 | val_auc=0.9560 | time=1.1s
      client=02 | n= 1789 | train_acc=0.9285 | val_acc=0.8863 | val_f1=0.8838 | val_auc=0.9633 | time=0.6s
      client=03 | n=  142 | train_acc=0.9437 | val_acc=0.4707 | val_f1=0.3201 | val_auc=0.7566 | time=0.2s
      client=04 | n= 2426 | train_acc=0.8784 | val_acc=0.8291 | val_f1=0.8273 | val_auc=0.9779 | time=0.8s
      client=05 | n=    1 | train_acc=1.0000 | val_acc=0.5293 | val_f1=0.3461 | val_auc=0.1172 | time=0.3s
      client=06 | n= 4736 | train_acc=0.9130 | val_acc=0.8893 | val_f1=0.8867 | val_auc=0.9777 | time=1.6s
      client=07 | n= 3988 | train_acc=0.9902 | val_acc=0.7601 | val_f1=0.7519 | val_auc=0.9627 | time=1.4s
      client=08 | n= 4083 | train_acc=0.8726 | val_acc=0.8861 | val_f1=0.8836 | val_auc=0.9749 | time=1.4s
      client=09 | n= 4239 | train_acc=0.9521 | val_acc=0.8891 | val_f1=0.8864 | val_auc=0.9787 | time=1.5s
      client=10 | n= 8696 | train_acc=0.9872 | val_acc=0.5293 | val_f1=0.3461 | val_auc=0.9694 | time=2.8s
      client=11 | n= 3369 | train_acc=0.9659 | val_acc=0.7915 | val_f1=0.7871 | val_auc=0.9629 | time=1.3s
    
    [WII21] 46 clients
      client=00 | n= 8354 | train_acc=0.9998 | val_acc=0.9999 | val_f1=0.9996 | val_auc=1.0000 | time=1.8s
      client=01 | n=25751 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.3s
      client=02 | n=42801 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9998 | val_auc=1.0000 | time=8.7s
      client=03 | n=15271 | train_acc=0.9999 | val_acc=1.0000 | val_f1=0.9998 | val_auc=1.0000 | time=3.3s
      client=04 | n=17243 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.7s
      client=05 | n=44654 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=0.9999 | time=9.1s
      client=06 | n=47255 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.6s
      client=07 | n=  457 | train_acc=0.9278 | val_acc=0.9861 | val_f1=0.9451 | val_auc=0.9765 | time=0.4s
      client=08 | n=12388 | train_acc=0.9993 | val_acc=0.9271 | val_f1=0.4811 | val_auc=0.9985 | time=2.7s
      client=09 | n=12876 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9998 | val_auc=1.0000 | time=2.9s
      client=10 | n= 6523 | train_acc=0.9998 | val_acc=0.9997 | val_f1=0.9989 | val_auc=1.0000 | time=1.6s
      client=11 | n= 6709 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=12 | n= 3239 | train_acc=0.9988 | val_acc=0.9994 | val_f1=0.9977 | val_auc=1.0000 | time=1.0s
      client=13 | n=  259 | train_acc=0.9961 | val_acc=0.9814 | val_f1=0.9383 | val_auc=0.9984 | time=0.5s
      client=14 | n= 8000 | train_acc=1.0000 | val_acc=0.9999 | val_f1=0.9997 | val_auc=1.0000 | time=1.9s
      client=15 | n=44321 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=9.2s
      client=16 | n=87021 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=17.6s
      client=17 | n= 6986 | train_acc=0.9997 | val_acc=0.9271 | val_f1=0.4811 | val_auc=0.9951 | time=1.7s
      client=18 | n= 9329 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.2s
      client=19 | n=37143 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=7.7s
      client=20 | n=34755 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=7.3s
      client=21 | n= 7268 | train_acc=1.0000 | val_acc=0.9999 | val_f1=0.9995 | val_auc=1.0000 | time=1.8s
      client=22 | n=15886 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.5s
      client=23 | n=18446 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.1s
      client=24 | n=21907 | train_acc=1.0000 | val_acc=0.9997 | val_f1=0.9989 | val_auc=1.0000 | time=4.7s
      client=25 | n=15749 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.5s
      client=26 | n=41900 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=8.7s
      client=27 | n=22675 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.9s
      client=28 | n=22659 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.9s
      client=29 | n= 6231 | train_acc=0.9994 | val_acc=0.9973 | val_f1=0.9898 | val_auc=0.9999 | time=1.7s
      client=30 | n=11135 | train_acc=1.0000 | val_acc=0.9991 | val_f1=0.9966 | val_auc=1.0000 | time=2.6s
      client=31 | n=10805 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=2.6s
      client=32 | n=11918 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.8s
      client=33 | n=12890 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.0s
      client=34 | n= 5447 | train_acc=0.9998 | val_acc=0.9999 | val_f1=0.9998 | val_auc=1.0000 | time=1.5s
      client=35 | n=  743 | train_acc=0.9717 | val_acc=0.9816 | val_f1=0.9380 | val_auc=0.9989 | time=0.6s
      client=36 | n=52117 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=10.8s
      client=37 | n= 3378 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.2s
      client=38 | n= 7628 | train_acc=1.0000 | val_acc=0.9995 | val_f1=0.9981 | val_auc=1.0000 | time=2.0s
      client=39 | n=10187 | train_acc=0.9999 | val_acc=0.9997 | val_f1=0.9990 | val_auc=1.0000 | time=2.5s
      client=40 | n= 2932 | train_acc=0.9952 | val_acc=0.9815 | val_f1=0.9386 | val_auc=1.0000 | time=1.1s
      client=41 | n=36349 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=7.7s
      client=42 | n= 5003 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=1.5s
      client=43 | n=14026 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.3s
      client=44 | n= 4514 | train_acc=0.9987 | val_acc=0.9979 | val_f1=0.9922 | val_auc=1.0000 | time=1.4s
      client=45 | n= 2996 | train_acc=0.9983 | val_acc=0.9946 | val_f1=0.9795 | val_auc=0.9992 | time=1.1s
      [VAL] I23Sub | acc=0.9755 | f1=0.9069 | auc=0.9955 | logloss=0.0994
      [VAL] K99Sub | acc=0.8907 | f1=0.8874 | auc=0.9794 | logloss=0.3494
      [VAL] NTD1 | acc=0.9669 | f1=0.8786 | auc=0.9815 | logloss=0.1373
      [VAL] NTD2 | acc=0.8313 | f1=0.8236 | auc=0.9144 | logloss=0.5630
      [VAL] WII21 | acc=1.0000 | f1=1.0000 | auc=1.0000 | logloss=0.0000
    ----------------------------------------------------------------------------------------------------
    GLOBAL VAL | acc=0.9850 | f1=0.9643 | logloss=0.0559 | round_time=789.4s
    
    ====================================================================================================
    FEDERATED ROUND 2/15
    ====================================================================================================
    
    [I23Sub] 2 clients
      client=00 | n= 9854 | train_acc=0.9969 | val_acc=0.9980 | val_f1=0.9933 | val_auc=0.9973 | time=1.8s
      client=01 | n= 6347 | train_acc=0.9976 | val_acc=0.9968 | val_f1=0.9894 | val_auc=0.9971 | time=1.4s
    
    [K99Sub] 4 clients
      client=00 | n= 1006 | train_acc=0.9026 | val_acc=0.8613 | val_f1=0.8613 | val_auc=0.9644 | time=0.7s
      client=01 | n= 1816 | train_acc=0.9741 | val_acc=0.9087 | val_f1=0.9069 | val_auc=0.9828 | time=0.9s
      client=02 | n= 8402 | train_acc=0.9755 | val_acc=0.9627 | val_f1=0.9623 | val_auc=0.9946 | time=1.5s
      client=03 | n= 6410 | train_acc=0.9657 | val_acc=0.9595 | val_f1=0.9593 | val_auc=0.9942 | time=1.3s
    
    [NTD1] 10 clients
      client=00 | n=37754 | train_acc=0.9583 | val_acc=0.9722 | val_f1=0.9030 | val_auc=0.9899 | time=3.2s
      client=01 | n= 2209 | train_acc=0.9919 | val_acc=0.0850 | val_f1=0.0783 | val_auc=0.8912 | time=0.9s
      client=02 | n=48001 | train_acc=0.9902 | val_acc=0.9665 | val_f1=0.8776 | val_auc=0.9840 | time=3.9s
      client=03 | n=12872 | train_acc=0.9518 | val_acc=0.9486 | val_f1=0.8663 | val_auc=0.9847 | time=1.6s
      client=04 | n=62448 | train_acc=0.9939 | val_acc=0.9150 | val_f1=0.4778 | val_auc=0.9824 | time=4.9s
      client=05 | n=27705 | train_acc=0.9759 | val_acc=0.9659 | val_f1=0.8760 | val_auc=0.9838 | time=2.5s
      client=06 | n= 1357 | train_acc=0.9219 | val_acc=0.8634 | val_f1=0.7348 | val_auc=0.9846 | time=0.7s
      client=07 | n= 5166 | train_acc=0.9444 | val_acc=0.8414 | val_f1=0.7110 | val_auc=0.9818 | time=1.0s
      client=08 | n= 5839 | train_acc=0.9210 | val_acc=0.9476 | val_f1=0.8566 | val_auc=0.9851 | time=1.1s
      client=09 | n=54260 | train_acc=0.9779 | val_acc=0.9669 | val_f1=0.8786 | val_auc=0.9865 | time=4.3s
    
    [NTD2] 12 clients
      client=00 | n= 2230 | train_acc=0.9520 | val_acc=0.7957 | val_f1=0.7917 | val_auc=0.9497 | time=1.2s
      client=01 | n= 3379 | train_acc=0.9435 | val_acc=0.7677 | val_f1=0.7606 | val_auc=0.9568 | time=1.6s
      client=02 | n= 1789 | train_acc=0.9268 | val_acc=0.8848 | val_f1=0.8823 | val_auc=0.9652 | time=1.0s
      client=03 | n=  142 | train_acc=0.9437 | val_acc=0.4707 | val_f1=0.3201 | val_auc=0.8876 | time=0.7s
      client=04 | n= 2426 | train_acc=0.8520 | val_acc=0.8045 | val_f1=0.8013 | val_auc=0.9763 | time=1.3s
      client=05 | n=    1 | train_acc=1.0000 | val_acc=0.5301 | val_f1=0.3480 | val_auc=0.6030 | time=0.7s
      client=06 | n= 4736 | train_acc=0.9082 | val_acc=0.8857 | val_f1=0.8832 | val_auc=0.9741 | time=2.0s
      client=07 | n= 3988 | train_acc=0.9847 | val_acc=0.7910 | val_f1=0.7865 | val_auc=0.9608 | time=1.8s
      client=08 | n= 4083 | train_acc=0.8864 | val_acc=0.8276 | val_f1=0.8257 | val_auc=0.9779 | time=1.7s
      client=09 | n= 4239 | train_acc=0.9521 | val_acc=0.8893 | val_f1=0.8866 | val_auc=0.9819 | time=1.8s
      client=10 | n= 8696 | train_acc=0.9872 | val_acc=0.5293 | val_f1=0.3461 | val_auc=0.9749 | time=3.2s
      client=11 | n= 3369 | train_acc=0.9694 | val_acc=0.7602 | val_f1=0.7521 | val_auc=0.9594 | time=1.6s
    
    [WII21] 46 clients
      client=00 | n= 8354 | train_acc=0.9999 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.2s
      client=01 | n=25751 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.7s
      client=02 | n=42801 | train_acc=1.0000 | val_acc=0.9999 | val_f1=0.9998 | val_auc=1.0000 | time=9.1s
      client=03 | n=15271 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9999 | time=3.7s
      client=04 | n=17243 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.1s
      client=05 | n=44654 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.5s
      client=06 | n=47255 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=10.0s
      client=07 | n=  457 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=0.8s
      client=08 | n=12388 | train_acc=0.9998 | val_acc=0.9620 | val_f1=0.8133 | val_auc=0.9823 | time=3.1s
      client=09 | n=12876 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9998 | time=3.2s
      client=10 | n= 6523 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.0s
      client=11 | n= 6709 | train_acc=0.9996 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.0s
      client=12 | n= 3239 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9997 | time=1.4s
      client=13 | n=  259 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=0.8s
      client=14 | n= 8000 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=15 | n=44321 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.5s
      client=16 | n=87021 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=17.9s
      client=17 | n= 6986 | train_acc=1.0000 | val_acc=0.9992 | val_f1=0.9970 | val_auc=0.9989 | time=2.1s
      client=18 | n= 9329 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9999 | time=2.6s
      client=19 | n=37143 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.2s
      client=20 | n=34755 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=7.7s
      client=21 | n= 7268 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.2s
      client=22 | n=15886 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=4.0s
      client=23 | n=18446 | train_acc=0.9999 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9999 | time=4.5s
      client=24 | n=21907 | train_acc=1.0000 | val_acc=0.9995 | val_f1=0.9982 | val_auc=1.0000 | time=5.1s
      client=25 | n=15749 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.9s
      client=26 | n=41900 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=9.1s
      client=27 | n=22675 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.2s
      client=28 | n=22659 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.3s
      client=29 | n= 6231 | train_acc=1.0000 | val_acc=0.9999 | val_f1=0.9997 | val_auc=0.9997 | time=2.0s
      client=30 | n=11135 | train_acc=0.9999 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.0s
      client=31 | n=10805 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.0s
      client=32 | n=11918 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.2s
      client=33 | n=12890 | train_acc=0.9998 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.3s
      client=34 | n= 5447 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.9s
      client=35 | n=  743 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=0.9s
      client=36 | n=52117 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=11.2s
      client=37 | n= 3378 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=1.5s
      client=38 | n= 7628 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=39 | n=10187 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.8s
      client=40 | n= 2932 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.4s
      client=41 | n=36349 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.1s
      client=42 | n= 5003 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9999 | time=1.9s
      client=43 | n=14026 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.7s
      client=44 | n= 4514 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9998 | time=1.8s
      client=45 | n= 2996 | train_acc=1.0000 | val_acc=0.9999 | val_f1=0.9995 | val_auc=0.9997 | time=1.6s
      [VAL] I23Sub | acc=0.9692 | f1=0.8791 | auc=0.9952 | logloss=0.1019
      [VAL] K99Sub | acc=0.9045 | f1=0.9045 | auc=0.9847 | logloss=0.6217
      [VAL] NTD1 | acc=0.9669 | f1=0.8786 | auc=0.9822 | logloss=0.1252
      [VAL] NTD2 | acc=0.7458 | f1=0.7233 | auc=0.9008 | logloss=0.5373
      [VAL] WII21 | acc=1.0000 | f1=1.0000 | auc=1.0000 | logloss=0.0000
    ----------------------------------------------------------------------------------------------------
    GLOBAL VAL | acc=0.9823 | f1=0.9608 | logloss=0.0565 | round_time=876.3s
    
    ====================================================================================================
    FEDERATED ROUND 3/15
    ====================================================================================================
    
    [I23Sub] 2 clients
      client=00 | n= 9854 | train_acc=0.9969 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9973 | time=2.0s
      client=01 | n= 6347 | train_acc=0.9973 | val_acc=0.9968 | val_f1=0.9894 | val_auc=0.9971 | time=1.7s
    
    [K99Sub] 4 clients
      client=00 | n= 1006 | train_acc=0.9215 | val_acc=0.8338 | val_f1=0.8322 | val_auc=0.9860 | time=1.0s
      client=01 | n= 1816 | train_acc=0.9714 | val_acc=0.9643 | val_f1=0.9641 | val_auc=0.9854 | time=1.1s
      client=02 | n= 8402 | train_acc=0.9794 | val_acc=0.9717 | val_f1=0.9715 | val_auc=0.9959 | time=1.8s
      client=03 | n= 6410 | train_acc=0.9822 | val_acc=0.9675 | val_f1=0.9674 | val_auc=0.9967 | time=1.6s
    
    [NTD1] 10 clients
      client=00 | n=37754 | train_acc=0.9596 | val_acc=0.9719 | val_f1=0.9032 | val_auc=0.9882 | time=3.5s
      client=01 | n= 2209 | train_acc=0.9950 | val_acc=0.4434 | val_f1=0.3984 | val_auc=0.8933 | time=1.1s
      client=02 | n=48001 | train_acc=0.9905 | val_acc=0.9669 | val_f1=0.8786 | val_auc=0.9857 | time=4.2s
      client=03 | n=12872 | train_acc=0.9510 | val_acc=0.9478 | val_f1=0.8642 | val_auc=0.9880 | time=1.8s
      client=04 | n=62448 | train_acc=0.9939 | val_acc=0.9150 | val_f1=0.4778 | val_auc=0.9831 | time=5.2s
      client=05 | n=27705 | train_acc=0.9767 | val_acc=0.9669 | val_f1=0.8786 | val_auc=0.9857 | time=2.9s
      client=06 | n= 1357 | train_acc=0.9204 | val_acc=0.8614 | val_f1=0.7324 | val_auc=0.9841 | time=1.0s
      client=07 | n= 5166 | train_acc=0.9441 | val_acc=0.8391 | val_f1=0.7086 | val_auc=0.9826 | time=1.3s
      client=08 | n= 5839 | train_acc=0.9613 | val_acc=0.9450 | val_f1=0.8607 | val_auc=0.9831 | time=1.4s
      client=09 | n=54260 | train_acc=0.9779 | val_acc=0.9669 | val_f1=0.8786 | val_auc=0.9887 | time=4.6s
    
    [NTD2] 12 clients
      client=00 | n= 2230 | train_acc=0.9511 | val_acc=0.7898 | val_f1=0.7852 | val_auc=0.9629 | time=1.5s
      client=01 | n= 3379 | train_acc=0.9461 | val_acc=0.7908 | val_f1=0.7863 | val_auc=0.9617 | time=1.9s
      client=02 | n= 1789 | train_acc=0.9257 | val_acc=0.8860 | val_f1=0.8835 | val_auc=0.9632 | time=1.4s
      client=03 | n=  142 | train_acc=0.9437 | val_acc=0.4707 | val_f1=0.3201 | val_auc=0.3000 | time=1.0s
      client=04 | n= 2426 | train_acc=0.8660 | val_acc=0.8600 | val_f1=0.8580 | val_auc=0.9688 | time=1.6s
      client=05 | n=    1 | train_acc=1.0000 | val_acc=0.5330 | val_f1=0.3547 | val_auc=0.6867 | time=0.9s
      client=06 | n= 4736 | train_acc=0.9065 | val_acc=0.8846 | val_f1=0.8822 | val_auc=0.9735 | time=2.3s
      client=07 | n= 3988 | train_acc=0.9900 | val_acc=0.7601 | val_f1=0.7519 | val_auc=0.9618 | time=2.1s
      client=08 | n= 4083 | train_acc=0.8895 | val_acc=0.8838 | val_f1=0.8814 | val_auc=0.9747 | time=2.0s
      client=09 | n= 4239 | train_acc=0.9519 | val_acc=0.8892 | val_f1=0.8865 | val_auc=0.9755 | time=2.1s
      client=10 | n= 8696 | train_acc=0.9872 | val_acc=0.5293 | val_f1=0.3461 | val_auc=0.9510 | time=3.4s
      client=11 | n= 3369 | train_acc=0.9685 | val_acc=0.7601 | val_f1=0.7519 | val_auc=0.9594 | time=1.9s
    
    [WII21] 46 clients
      client=00 | n= 8354 | train_acc=0.9999 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.6s
      client=01 | n=25751 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=6.1s
      client=02 | n=42801 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.4s
      client=03 | n=15271 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.9s
      client=04 | n=17243 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=4.3s
      client=05 | n=44654 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.8s
      client=06 | n=47255 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=10.3s
      client=07 | n=  457 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.0s
      client=08 | n=12388 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.4s
      client=09 | n=12876 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.4s
      client=10 | n= 6523 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.2s
      client=11 | n= 6709 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=12 | n= 3239 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      client=13 | n=  259 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.0s
      client=14 | n= 8000 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.6s
      client=15 | n=44321 | train_acc=1.0000 | val_acc=0.9999 | val_f1=0.9998 | val_auc=0.9997 | time=9.7s
      client=16 | n=87021 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=18.2s
      client=17 | n= 6986 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=18 | n= 9329 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=2.8s
      client=19 | n=37143 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.3s
      client=20 | n=34755 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=7.8s
      client=21 | n= 7268 | train_acc=0.9999 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9999 | time=2.4s
      client=22 | n=15886 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.1s
      client=23 | n=18446 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.6s
      client=24 | n=21907 | train_acc=0.9981 | val_acc=0.9411 | val_f1=0.6457 | val_auc=0.9942 | time=5.3s
      client=25 | n=15749 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.0s
      client=26 | n=41900 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.2s
      client=27 | n=22675 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.4s
      client=28 | n=22659 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9998 | val_auc=1.0000 | time=5.4s
      client=29 | n= 6231 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.1s
      client=30 | n=11135 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9998 | time=3.1s
      client=31 | n=10805 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.1s
      client=32 | n=11918 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.3s
      client=33 | n=12890 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.5s
      client=34 | n= 5447 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.1s
      client=35 | n=  743 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=36 | n=52117 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=11.3s
      client=37 | n= 3378 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=38 | n= 7628 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.5s
      client=39 | n=10187 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.0s
      client=40 | n= 2932 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      client=41 | n=36349 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.2s
      client=42 | n= 5003 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.0s
      client=43 | n=14026 | train_acc=0.9995 | val_acc=0.9995 | val_f1=0.9983 | val_auc=0.9985 | time=3.8s
      client=44 | n= 4514 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.9s
      client=45 | n= 2996 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      [VAL] I23Sub | acc=0.9919 | f1=0.9725 | auc=0.9956 | logloss=0.0441
      [VAL] K99Sub | acc=0.8044 | f1=0.7900 | auc=0.9785 | logloss=0.4837
      [VAL] NTD1 | acc=0.9669 | f1=0.8786 | auc=0.9841 | logloss=0.1284
      [VAL] NTD2 | acc=0.5596 | f1=0.4294 | auc=0.9331 | logloss=0.7480
      [VAL] WII21 | acc=1.0000 | f1=1.0000 | auc=1.0000 | logloss=0.0000
    ----------------------------------------------------------------------------------------------------
    GLOBAL VAL | acc=0.9749 | f1=0.9505 | logloss=0.0613 | round_time=925.3s
    
    ====================================================================================================
    FEDERATED ROUND 4/15
    ====================================================================================================
    
    [I23Sub] 2 clients
      client=00 | n= 9854 | train_acc=0.9969 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9973 | time=2.2s
      client=01 | n= 6347 | train_acc=0.9975 | val_acc=0.9968 | val_f1=0.9894 | val_auc=0.9972 | time=1.8s
    
    [K99Sub] 4 clients
      client=00 | n= 1006 | train_acc=0.8946 | val_acc=0.9301 | val_f1=0.9292 | val_auc=0.9308 | time=1.1s
      client=01 | n= 1816 | train_acc=0.9725 | val_acc=0.8759 | val_f1=0.8712 | val_auc=0.9911 | time=1.2s
      client=02 | n= 8402 | train_acc=0.9800 | val_acc=0.9733 | val_f1=0.9731 | val_auc=0.9971 | time=1.9s
      client=03 | n= 6410 | train_acc=0.9680 | val_acc=0.9378 | val_f1=0.9378 | val_auc=0.9966 | time=1.8s
    
    [NTD1] 10 clients
      client=00 | n=37754 | train_acc=0.9596 | val_acc=0.9730 | val_f1=0.9047 | val_auc=0.9885 | time=3.7s
      client=01 | n= 2209 | train_acc=0.9964 | val_acc=0.7449 | val_f1=0.6188 | val_auc=0.9688 | time=1.2s
      client=02 | n=48001 | train_acc=0.9896 | val_acc=0.9150 | val_f1=0.4778 | val_auc=0.9875 | time=4.5s
      client=03 | n=12872 | train_acc=0.9539 | val_acc=0.9475 | val_f1=0.8652 | val_auc=0.9875 | time=1.9s
      client=04 | n=62448 | train_acc=0.9939 | val_acc=0.9150 | val_f1=0.4778 | val_auc=0.9838 | time=5.4s
      client=05 | n=27705 | train_acc=0.9767 | val_acc=0.9669 | val_f1=0.8786 | val_auc=0.9868 | time=3.0s
      client=06 | n= 1357 | train_acc=0.9528 | val_acc=0.9479 | val_f1=0.8645 | val_auc=0.9843 | time=1.2s
      client=07 | n= 5166 | train_acc=0.9684 | val_acc=0.9439 | val_f1=0.8585 | val_auc=0.9852 | time=1.5s
      client=08 | n= 5839 | train_acc=0.9611 | val_acc=0.9453 | val_f1=0.8612 | val_auc=0.9852 | time=1.4s
      client=09 | n=54260 | train_acc=0.9779 | val_acc=0.9669 | val_f1=0.8786 | val_auc=0.9859 | time=4.9s
    
    [NTD2] 12 clients
      client=00 | n= 2230 | train_acc=0.9507 | val_acc=0.7956 | val_f1=0.7915 | val_auc=0.9594 | time=1.6s
      client=01 | n= 3379 | train_acc=0.9450 | val_acc=0.7957 | val_f1=0.7917 | val_auc=0.9627 | time=2.0s
      client=02 | n= 1789 | train_acc=0.9273 | val_acc=0.8862 | val_f1=0.8837 | val_auc=0.9632 | time=1.5s
      client=03 | n=  142 | train_acc=0.9437 | val_acc=0.4791 | val_f1=0.3391 | val_auc=0.9152 | time=1.1s
      client=04 | n= 2426 | train_acc=0.8619 | val_acc=0.8599 | val_f1=0.8579 | val_auc=0.9681 | time=1.7s
      client=05 | n=    1 | train_acc=1.0000 | val_acc=0.5293 | val_f1=0.3461 | val_auc=0.9083 | time=1.1s
      client=06 | n= 4736 | train_acc=0.9069 | val_acc=0.8848 | val_f1=0.8822 | val_auc=0.9757 | time=2.4s
      client=07 | n= 3988 | train_acc=0.9812 | val_acc=0.4707 | val_f1=0.3201 | val_auc=0.9562 | time=2.2s
      client=08 | n= 4083 | train_acc=0.8619 | val_acc=0.8288 | val_f1=0.8270 | val_auc=0.9727 | time=2.2s
      client=09 | n= 4239 | train_acc=0.9523 | val_acc=0.8889 | val_f1=0.8862 | val_auc=0.9669 | time=2.2s
      client=10 | n= 8696 | train_acc=0.9872 | val_acc=0.5293 | val_f1=0.3461 | val_auc=0.9628 | time=3.5s
      client=11 | n= 3369 | train_acc=0.9659 | val_acc=0.7953 | val_f1=0.7913 | val_auc=0.9630 | time=2.0s
    
    [WII21] 46 clients
      client=00 | n= 8354 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.7s
      client=01 | n=25751 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9998 | val_auc=0.9999 | time=6.1s
      client=02 | n=42801 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9998 | val_auc=0.9998 | time=9.5s
      client=03 | n=15271 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=4.1s
      client=04 | n=17243 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.4s
      client=05 | n=44654 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.9s
      client=06 | n=47255 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=10.4s
      client=07 | n=  457 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=08 | n=12388 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.4s
      client=09 | n=12876 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.6s
      client=10 | n= 6523 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=11 | n= 6709 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=12 | n= 3239 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=13 | n=  259 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=14 | n= 8000 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.7s
      client=15 | n=44321 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=9.9s
      client=16 | n=87021 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=18.3s
      client=17 | n= 6986 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=18 | n= 9329 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.8s
      client=19 | n=37143 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.4s
      client=20 | n=34755 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9999 | time=7.9s
      client=21 | n= 7268 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=22 | n=15886 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.2s
      client=23 | n=18446 | train_acc=1.0000 | val_acc=0.9999 | val_f1=0.9998 | val_auc=1.0000 | time=4.7s
      client=24 | n=21907 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9998 | val_auc=0.9997 | time=5.4s
      client=25 | n=15749 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9998 | val_auc=0.9997 | time=4.2s
      client=26 | n=41900 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=9.3s
      client=27 | n=22675 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.6s
      client=28 | n=22659 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9999 | time=5.5s
      client=29 | n= 6231 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.2s
      client=30 | n=11135 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9998 | val_auc=0.9998 | time=3.2s
      client=31 | n=10805 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.2s
      client=32 | n=11918 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.4s
      client=33 | n=12890 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.6s
      client=34 | n= 5447 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.1s
      client=35 | n=  743 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.2s
      client=36 | n=52117 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=11.4s
      client=37 | n= 3378 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=38 | n= 7628 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.5s
      client=39 | n=10187 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.0s
      client=40 | n= 2932 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      client=41 | n=36349 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.2s
      client=42 | n= 5003 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.0s
      client=43 | n=14026 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.8s
      client=44 | n= 4514 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.9s
      client=45 | n= 2996 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      [VAL] I23Sub | acc=0.9957 | f1=0.9856 | auc=0.9971 | logloss=0.0249
      [VAL] K99Sub | acc=0.8529 | f1=0.8529 | auc=0.9745 | logloss=0.3920
      [VAL] NTD1 | acc=0.9669 | f1=0.8786 | auc=0.9838 | logloss=0.1158
      [VAL] NTD2 | acc=0.5517 | f1=0.3970 | auc=0.9418 | logloss=0.8029
      [VAL] WII21 | acc=1.0000 | f1=1.0000 | auc=1.0000 | logloss=0.0000
    ----------------------------------------------------------------------------------------------------
    GLOBAL VAL | acc=0.9754 | f1=0.9506 | logloss=0.0587 | round_time=943.3s
    
    ====================================================================================================
    FEDERATED ROUND 5/15
    ====================================================================================================
    
    [I23Sub] 2 clients
      client=00 | n= 9854 | train_acc=0.9969 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9971 | time=2.2s
      client=01 | n= 6347 | train_acc=0.9983 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9971 | time=1.8s
    
    [K99Sub] 4 clients
      client=00 | n= 1006 | train_acc=0.9404 | val_acc=0.8828 | val_f1=0.8826 | val_auc=0.9912 | time=1.1s
      client=01 | n= 1816 | train_acc=0.9736 | val_acc=0.8833 | val_f1=0.8793 | val_auc=0.9679 | time=1.2s
      client=02 | n= 8402 | train_acc=0.9817 | val_acc=0.9772 | val_f1=0.9771 | val_auc=0.9975 | time=1.9s
      client=03 | n= 6410 | train_acc=0.9752 | val_acc=0.9492 | val_f1=0.9492 | val_auc=0.9968 | time=1.8s
    
    [NTD1] 10 clients
      client=00 | n=37754 | train_acc=0.9606 | val_acc=0.9735 | val_f1=0.9071 | val_auc=0.9893 | time=3.7s
      client=01 | n= 2209 | train_acc=0.9986 | val_acc=0.8387 | val_f1=0.7082 | val_auc=0.9661 | time=1.2s
      client=02 | n=48001 | train_acc=0.9905 | val_acc=0.9669 | val_f1=0.8786 | val_auc=0.9835 | time=4.3s
      client=03 | n=12872 | train_acc=0.9490 | val_acc=0.9437 | val_f1=0.8580 | val_auc=0.9860 | time=2.0s
      client=04 | n=62448 | train_acc=0.9939 | val_acc=0.9150 | val_f1=0.4778 | val_auc=0.9844 | time=5.3s
      client=05 | n=27705 | train_acc=0.9767 | val_acc=0.9669 | val_f1=0.8786 | val_auc=0.9869 | time=2.9s
      client=06 | n= 1357 | train_acc=0.9639 | val_acc=0.9300 | val_f1=0.8334 | val_auc=0.9849 | time=1.2s
      client=07 | n= 5166 | train_acc=0.9632 | val_acc=0.9484 | val_f1=0.8665 | val_auc=0.9881 | time=1.4s
      client=08 | n= 5839 | train_acc=0.9584 | val_acc=0.9396 | val_f1=0.8506 | val_auc=0.9849 | time=1.5s
      client=09 | n=54260 | train_acc=0.9811 | val_acc=0.9733 | val_f1=0.9068 | val_auc=0.9876 | time=4.8s
    
    [NTD2] 12 clients
      client=00 | n= 2230 | train_acc=0.9462 | val_acc=0.9110 | val_f1=0.9110 | val_auc=0.9591 | time=1.6s
      client=01 | n= 3379 | train_acc=0.9455 | val_acc=0.7957 | val_f1=0.7917 | val_auc=0.9603 | time=2.0s
      client=02 | n= 1789 | train_acc=0.9279 | val_acc=0.8862 | val_f1=0.8837 | val_auc=0.9199 | time=1.5s
      client=03 | n=  142 | train_acc=0.9507 | val_acc=0.7941 | val_f1=0.7900 | val_auc=0.9615 | time=1.1s
      client=04 | n= 2426 | train_acc=0.8603 | val_acc=0.8513 | val_f1=0.8496 | val_auc=0.9607 | time=1.7s
      client=05 | n=    1 | train_acc=1.0000 | val_acc=0.5293 | val_f1=0.3461 | val_auc=0.9443 | time=1.1s
      client=06 | n= 4736 | train_acc=0.9143 | val_acc=0.8893 | val_f1=0.8866 | val_auc=0.9766 | time=2.4s
      client=07 | n= 3988 | train_acc=0.9812 | val_acc=0.4707 | val_f1=0.3201 | val_auc=0.9353 | time=2.2s
      client=08 | n= 4083 | train_acc=0.8773 | val_acc=0.8594 | val_f1=0.8574 | val_auc=0.9633 | time=2.2s
      client=09 | n= 4239 | train_acc=0.9410 | val_acc=0.8842 | val_f1=0.8816 | val_auc=0.9727 | time=2.2s
      client=10 | n= 8696 | train_acc=0.9872 | val_acc=0.5293 | val_f1=0.3461 | val_auc=0.9654 | time=3.6s
      client=11 | n= 3369 | train_acc=0.9662 | val_acc=0.7950 | val_f1=0.7909 | val_auc=0.9582 | time=2.1s
    
    [WII21] 46 clients
      client=00 | n= 8354 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.6s
      client=01 | n=25751 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=6.1s
      client=02 | n=42801 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.5s
      client=03 | n=15271 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.0s
      client=04 | n=17243 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.4s
      client=05 | n=44654 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.9s
      client=06 | n=47255 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=10.4s
      client=07 | n=  457 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=08 | n=12388 | train_acc=1.0000 | val_acc=0.9946 | val_f1=0.9794 | val_auc=0.9995 | time=3.5s
      client=09 | n=12876 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9998 | val_auc=0.9997 | time=3.6s
      client=10 | n= 6523 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=11 | n= 6709 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=12 | n= 3239 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      client=13 | n=  259 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=14 | n= 8000 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.6s
      client=15 | n=44321 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=9.8s
      client=16 | n=87021 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=18.3s
      client=17 | n= 6986 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=18 | n= 9329 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9998 | val_auc=0.9997 | time=2.9s
      client=19 | n=37143 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.4s
      client=20 | n=34755 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=7.9s
      client=21 | n= 7268 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=22 | n=15886 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.2s
      client=23 | n=18446 | train_acc=0.9999 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.7s
      client=24 | n=21907 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=5.4s
      client=25 | n=15749 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.1s
      client=26 | n=41900 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.3s
      client=27 | n=22675 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.6s
      client=28 | n=22659 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9999 | time=5.6s
      client=29 | n= 6231 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=30 | n=11135 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.2s
      client=31 | n=10805 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.2s
      client=32 | n=11918 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.4s
      client=33 | n=12890 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.6s
      client=34 | n= 5447 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.1s
      client=35 | n=  743 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=36 | n=52117 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=11.3s
      client=37 | n= 3378 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=38 | n= 7628 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.5s
      client=39 | n=10187 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.0s
      client=40 | n= 2932 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      client=41 | n=36349 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.2s
      client=42 | n= 5003 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.0s
      client=43 | n=14026 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.8s
      client=44 | n= 4514 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.9s
      client=45 | n= 2996 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      [VAL] I23Sub | acc=0.9957 | f1=0.9856 | auc=0.9971 | logloss=0.0243
      [VAL] K99Sub | acc=0.5369 | f1=0.3550 | auc=0.9952 | logloss=0.9487
      [VAL] NTD1 | acc=0.9669 | f1=0.8786 | auc=0.9840 | logloss=0.1171
      [VAL] NTD2 | acc=0.5570 | f1=0.4082 | auc=0.9448 | logloss=0.8301
      [VAL] WII21 | acc=1.0000 | f1=1.0000 | auc=1.0000 | logloss=0.0000
    ----------------------------------------------------------------------------------------------------
    GLOBAL VAL | acc=0.9708 | f1=0.9434 | logloss=0.0683 | round_time=945.2s
    
    ====================================================================================================
    FEDERATED ROUND 6/15
    ====================================================================================================
    
    [I23Sub] 2 clients
      client=00 | n= 9854 | train_acc=0.9971 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9972 | time=2.2s
      client=01 | n= 6347 | train_acc=0.9983 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9972 | time=1.8s
    
    [K99Sub] 4 clients
      client=00 | n= 1006 | train_acc=0.9433 | val_acc=0.9005 | val_f1=0.9005 | val_auc=0.9889 | time=1.2s
      client=01 | n= 1816 | train_acc=0.9829 | val_acc=0.9325 | val_f1=0.9315 | val_auc=0.9946 | time=1.2s
      client=02 | n= 8402 | train_acc=0.9800 | val_acc=0.9720 | val_f1=0.9718 | val_auc=0.9969 | time=1.9s
      client=03 | n= 6410 | train_acc=0.9810 | val_acc=0.9730 | val_f1=0.9729 | val_auc=0.9967 | time=1.7s
    
    [NTD1] 10 clients
      client=00 | n=37754 | train_acc=0.9601 | val_acc=0.9723 | val_f1=0.9043 | val_auc=0.9880 | time=3.8s
      client=01 | n= 2209 | train_acc=0.9977 | val_acc=0.8406 | val_f1=0.7102 | val_auc=0.9686 | time=1.3s
      client=02 | n=48001 | train_acc=0.9905 | val_acc=0.9669 | val_f1=0.8786 | val_auc=0.9839 | time=4.8s
      client=03 | n=12872 | train_acc=0.9497 | val_acc=0.9445 | val_f1=0.8586 | val_auc=0.9852 | time=2.0s
      client=04 | n=62448 | train_acc=0.9939 | val_acc=0.9150 | val_f1=0.4778 | val_auc=0.9830 | time=5.3s
      client=05 | n=27705 | train_acc=0.9792 | val_acc=0.9714 | val_f1=0.9002 | val_auc=0.9873 | time=3.2s
      client=06 | n= 1357 | train_acc=0.9654 | val_acc=0.9231 | val_f1=0.8222 | val_auc=0.9842 | time=1.2s
      client=07 | n= 5166 | train_acc=0.9543 | val_acc=0.9475 | val_f1=0.8636 | val_auc=0.9862 | time=1.5s
      client=08 | n= 5839 | train_acc=0.9613 | val_acc=0.9453 | val_f1=0.8612 | val_auc=0.9878 | time=1.5s
      client=09 | n=54260 | train_acc=0.9801 | val_acc=0.9707 | val_f1=0.8950 | val_auc=0.9883 | time=5.0s
    
    [NTD2] 12 clients
      client=00 | n= 2230 | train_acc=0.9529 | val_acc=0.7957 | val_f1=0.7917 | val_auc=0.9601 | time=1.6s
      client=01 | n= 3379 | train_acc=0.9455 | val_acc=0.7960 | val_f1=0.7920 | val_auc=0.9619 | time=2.1s
      client=02 | n= 1789 | train_acc=0.9027 | val_acc=0.8624 | val_f1=0.8604 | val_auc=0.9653 | time=1.5s
      client=03 | n=  142 | train_acc=0.9507 | val_acc=0.8489 | val_f1=0.8473 | val_auc=0.9595 | time=1.2s
      client=04 | n= 2426 | train_acc=0.8504 | val_acc=0.8609 | val_f1=0.8588 | val_auc=0.9602 | time=1.8s
      client=05 | n=    1 | train_acc=1.0000 | val_acc=0.5294 | val_f1=0.3464 | val_auc=0.9452 | time=1.1s
      client=06 | n= 4736 | train_acc=0.9134 | val_acc=0.8898 | val_f1=0.8871 | val_auc=0.9741 | time=2.5s
      client=07 | n= 3988 | train_acc=0.9812 | val_acc=0.4707 | val_f1=0.3201 | val_auc=0.9150 | time=2.2s
      client=08 | n= 4083 | train_acc=0.8548 | val_acc=0.8610 | val_f1=0.8590 | val_auc=0.9670 | time=2.3s
      client=09 | n= 4239 | train_acc=0.9519 | val_acc=0.8888 | val_f1=0.8861 | val_auc=0.9661 | time=2.3s
      client=10 | n= 8696 | train_acc=0.9872 | val_acc=0.5293 | val_f1=0.3461 | val_auc=0.9644 | time=3.6s
      client=11 | n= 3369 | train_acc=0.9679 | val_acc=0.7601 | val_f1=0.7519 | val_auc=0.9612 | time=2.1s
    
    [WII21] 46 clients
      client=00 | n= 8354 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.6s
      client=01 | n=25751 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9998 | val_auc=1.0000 | time=6.2s
      client=02 | n=42801 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=9.6s
      client=03 | n=15271 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.0s
      client=04 | n=17243 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=4.5s
      client=05 | n=44654 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.9s
      client=06 | n=47255 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=10.4s
      client=07 | n=  457 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=08 | n=12388 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.5s
      client=09 | n=12876 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.6s
      client=10 | n= 6523 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=2.3s
      client=11 | n= 6709 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=2.4s
      client=12 | n= 3239 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      client=13 | n=  259 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=14 | n= 8000 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.6s
      client=15 | n=44321 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.9s
      client=16 | n=87021 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=18.3s
      client=17 | n= 6986 | train_acc=1.0000 | val_acc=0.9999 | val_f1=0.9998 | val_auc=1.0000 | time=2.4s
      client=18 | n= 9329 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.9s
      client=19 | n=37143 | train_acc=0.9999 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=8.4s
      client=20 | n=34755 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=7.9s
      client=21 | n= 7268 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=22 | n=15886 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.2s
      client=23 | n=18446 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.7s
      client=24 | n=21907 | train_acc=1.0000 | val_acc=0.9999 | val_f1=0.9998 | val_auc=0.9997 | time=5.3s
      client=25 | n=15749 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9998 | val_auc=0.9999 | time=4.1s
      client=26 | n=41900 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=9.3s
      client=27 | n=22675 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.5s
      client=28 | n=22659 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.5s
      client=29 | n= 6231 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.2s
      client=30 | n=11135 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.2s
      client=31 | n=10805 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.1s
      client=32 | n=11918 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.4s
      client=33 | n=12890 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.6s
      client=34 | n= 5447 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.1s
      client=35 | n=  743 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.2s
      client=36 | n=52117 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=11.4s
      client=37 | n= 3378 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.8s
      client=38 | n= 7628 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.6s
      client=39 | n=10187 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.0s
      client=40 | n= 2932 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      client=41 | n=36349 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.2s
      client=42 | n= 5003 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.0s
      client=43 | n=14026 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.8s
      client=44 | n= 4514 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.0s
      client=45 | n= 2996 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      [VAL] I23Sub | acc=0.9957 | f1=0.9856 | auc=0.9970 | logloss=0.0224
      [VAL] K99Sub | acc=0.9140 | f1=0.9138 | auc=0.9867 | logloss=0.2177
      [VAL] NTD1 | acc=0.9669 | f1=0.8786 | auc=0.9843 | logloss=0.1117
      [VAL] NTD2 | acc=0.6777 | f1=0.6247 | auc=0.9499 | logloss=0.7330
      [VAL] WII21 | acc=1.0000 | f1=1.0000 | auc=1.0000 | logloss=0.0000
    ----------------------------------------------------------------------------------------------------
    GLOBAL VAL | acc=0.9805 | f1=0.9591 | logloss=0.0528 | round_time=952.7s
    
    ====================================================================================================
    FEDERATED ROUND 7/15
    ====================================================================================================
    
    [I23Sub] 2 clients
      client=00 | n= 9854 | train_acc=0.9969 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9975 | time=2.2s
      client=01 | n= 6347 | train_acc=0.9983 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9974 | time=1.7s
    
    [K99Sub] 4 clients
      client=00 | n= 1006 | train_acc=0.9523 | val_acc=0.9547 | val_f1=0.9545 | val_auc=0.9933 | time=1.1s
      client=01 | n= 1816 | train_acc=0.9719 | val_acc=0.9598 | val_f1=0.9596 | val_auc=0.9938 | time=1.2s
      client=02 | n= 8402 | train_acc=0.9845 | val_acc=0.9783 | val_f1=0.9782 | val_auc=0.9977 | time=2.0s
      client=03 | n= 6410 | train_acc=0.9808 | val_acc=0.9619 | val_f1=0.9618 | val_auc=0.9963 | time=1.7s
    
    [NTD1] 10 clients
      client=00 | n=37754 | train_acc=0.9577 | val_acc=0.9717 | val_f1=0.8995 | val_auc=0.9878 | time=3.7s
      client=01 | n= 2209 | train_acc=0.9959 | val_acc=0.4704 | val_f1=0.4179 | val_auc=0.9704 | time=1.2s
      client=02 | n=48001 | train_acc=0.9911 | val_acc=0.9739 | val_f1=0.9087 | val_auc=0.9888 | time=4.3s
      client=03 | n=12872 | train_acc=0.9520 | val_acc=0.9487 | val_f1=0.8666 | val_auc=0.9870 | time=1.9s
      client=04 | n=62448 | train_acc=0.9939 | val_acc=0.9150 | val_f1=0.4778 | val_auc=0.9878 | time=5.5s
      client=05 | n=27705 | train_acc=0.9767 | val_acc=0.9669 | val_f1=0.8786 | val_auc=0.9869 | time=3.1s
      client=06 | n= 1357 | train_acc=0.9234 | val_acc=0.8471 | val_f1=0.7169 | val_auc=0.9860 | time=1.2s
      client=07 | n= 5166 | train_acc=0.9704 | val_acc=0.9265 | val_f1=0.8279 | val_auc=0.9852 | time=1.5s
      client=08 | n= 5839 | train_acc=0.9575 | val_acc=0.9385 | val_f1=0.8486 | val_auc=0.9866 | time=1.5s
      client=09 | n=54260 | train_acc=0.9808 | val_acc=0.9719 | val_f1=0.9003 | val_auc=0.9886 | time=4.9s
    
    [NTD2] 12 clients
      client=00 | n= 2230 | train_acc=0.9520 | val_acc=0.7957 | val_f1=0.7917 | val_auc=0.9612 | time=1.6s
      client=01 | n= 3379 | train_acc=0.9461 | val_acc=0.7959 | val_f1=0.7919 | val_auc=0.9643 | time=2.1s
      client=02 | n= 1789 | train_acc=0.9352 | val_acc=0.8893 | val_f1=0.8866 | val_auc=0.9619 | time=1.6s
      client=03 | n=  142 | train_acc=0.9507 | val_acc=0.7958 | val_f1=0.7918 | val_auc=0.9627 | time=1.1s
      client=04 | n= 2426 | train_acc=0.8640 | val_acc=0.8602 | val_f1=0.8581 | val_auc=0.9655 | time=1.8s
      client=05 | n=    1 | train_acc=1.0000 | val_acc=0.5294 | val_f1=0.3464 | val_auc=0.9445 | time=1.1s
      client=06 | n= 4736 | train_acc=0.9141 | val_acc=0.8892 | val_f1=0.8865 | val_auc=0.9763 | time=2.4s
      client=07 | n= 3988 | train_acc=0.9812 | val_acc=0.4707 | val_f1=0.3201 | val_auc=0.9654 | time=2.2s
      client=08 | n= 4083 | train_acc=0.8729 | val_acc=0.8861 | val_f1=0.8836 | val_auc=0.9677 | time=2.2s
      client=09 | n= 4239 | train_acc=0.9514 | val_acc=0.8882 | val_f1=0.8856 | val_auc=0.9819 | time=2.3s
      client=10 | n= 8696 | train_acc=0.9872 | val_acc=0.5293 | val_f1=0.3461 | val_auc=0.9619 | time=3.6s
      client=11 | n= 3369 | train_acc=0.9671 | val_acc=0.7907 | val_f1=0.7862 | val_auc=0.9614 | time=2.1s
    
    [WII21] 46 clients
      client=00 | n= 8354 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.7s
      client=01 | n=25751 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=6.1s
      client=02 | n=42801 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.5s
      client=03 | n=15271 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.0s
      client=04 | n=17243 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.5s
      client=05 | n=44654 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.9s
      client=06 | n=47255 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=10.4s
      client=07 | n=  457 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.2s
      client=08 | n=12388 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9998 | val_auc=1.0000 | time=3.5s
      client=09 | n=12876 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.6s
      client=10 | n= 6523 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=11 | n= 6709 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=12 | n= 3239 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=13 | n=  259 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=14 | n= 8000 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.7s
      client=15 | n=44321 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=9.8s
      client=16 | n=87021 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=18.3s
      client=17 | n= 6986 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=18 | n= 9329 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.8s
      client=19 | n=37143 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.4s
      client=20 | n=34755 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=7.9s
      client=21 | n= 7268 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=22 | n=15886 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.1s
      client=23 | n=18446 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.7s
      client=24 | n=21907 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=5.3s
      client=25 | n=15749 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.1s
      client=26 | n=41900 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.3s
      client=27 | n=22675 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.5s
      client=28 | n=22659 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.5s
      client=29 | n= 6231 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.2s
      client=30 | n=11135 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.2s
      client=31 | n=10805 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.2s
      client=32 | n=11918 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.4s
      client=33 | n=12890 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.6s
      client=34 | n= 5447 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.1s
      client=35 | n=  743 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=36 | n=52117 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=11.4s
      client=37 | n= 3378 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=38 | n= 7628 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.5s
      client=39 | n=10187 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.0s
      client=40 | n= 2932 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      client=41 | n=36349 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.2s
      client=42 | n= 5003 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.0s
      client=43 | n=14026 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.8s
      client=44 | n= 4514 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.9s
      client=45 | n= 2996 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      [VAL] I23Sub | acc=0.9971 | f1=0.9905 | auc=0.9971 | logloss=0.0264
      [VAL] K99Sub | acc=0.9328 | f1=0.9318 | auc=0.9929 | logloss=0.2034
      [VAL] NTD1 | acc=0.9669 | f1=0.8786 | auc=0.9851 | logloss=0.1085
      [VAL] NTD2 | acc=0.6792 | f1=0.6270 | auc=0.9589 | logloss=0.7598
      [VAL] WII21 | acc=1.0000 | f1=1.0000 | auc=1.0000 | logloss=0.0000
    ----------------------------------------------------------------------------------------------------
    GLOBAL VAL | acc=0.9809 | f1=0.9595 | logloss=0.0529 | round_time=948.9s
    
    ====================================================================================================
    FEDERATED ROUND 8/15
    ====================================================================================================
    
    [I23Sub] 2 clients
      client=00 | n= 9854 | train_acc=0.9971 | val_acc=0.9980 | val_f1=0.9933 | val_auc=0.9971 | time=2.2s
      client=01 | n= 6347 | train_acc=0.9981 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9973 | time=1.8s
    
    [K99Sub] 4 clients
      client=00 | n= 1006 | train_acc=0.9622 | val_acc=0.9577 | val_f1=0.9575 | val_auc=0.9941 | time=1.1s
      client=01 | n= 1816 | train_acc=0.9829 | val_acc=0.9063 | val_f1=0.9039 | val_auc=0.9958 | time=1.3s
      client=02 | n= 8402 | train_acc=0.9762 | val_acc=0.9722 | val_f1=0.9721 | val_auc=0.9966 | time=2.0s
      client=03 | n= 6410 | train_acc=0.9835 | val_acc=0.9672 | val_f1=0.9671 | val_auc=0.9973 | time=1.8s
    
    [NTD1] 10 clients
      client=00 | n=37754 | train_acc=0.9616 | val_acc=0.9625 | val_f1=0.8918 | val_auc=0.9900 | time=3.8s
      client=01 | n= 2209 | train_acc=0.9986 | val_acc=0.8390 | val_f1=0.7085 | val_auc=0.9660 | time=1.3s
      client=02 | n=48001 | train_acc=0.9896 | val_acc=0.9150 | val_f1=0.4778 | val_auc=0.9880 | time=4.5s
      client=03 | n=12872 | train_acc=0.9520 | val_acc=0.9501 | val_f1=0.8689 | val_auc=0.9889 | time=2.1s
      client=04 | n=62448 | train_acc=0.9922 | val_acc=0.9669 | val_f1=0.8786 | val_auc=0.9891 | time=5.6s
      client=05 | n=27705 | train_acc=0.9796 | val_acc=0.9718 | val_f1=0.8998 | val_auc=0.9877 | time=3.1s
      client=06 | n= 1357 | train_acc=0.9646 | val_acc=0.9438 | val_f1=0.8583 | val_auc=0.9852 | time=1.2s
      client=07 | n= 5166 | train_acc=0.9686 | val_acc=0.9409 | val_f1=0.8530 | val_auc=0.9870 | time=1.5s
      client=08 | n= 5839 | train_acc=0.9587 | val_acc=0.9461 | val_f1=0.8627 | val_auc=0.9879 | time=1.5s
      client=09 | n=54260 | train_acc=0.9810 | val_acc=0.9730 | val_f1=0.9047 | val_auc=0.9920 | time=4.9s
    
    [NTD2] 12 clients
      client=00 | n= 2230 | train_acc=0.9453 | val_acc=0.7601 | val_f1=0.7519 | val_auc=0.9627 | time=1.6s
      client=01 | n= 3379 | train_acc=0.9453 | val_acc=0.7959 | val_f1=0.7919 | val_auc=0.9453 | time=2.1s
      client=02 | n= 1789 | train_acc=0.9363 | val_acc=0.8883 | val_f1=0.8856 | val_auc=0.9648 | time=1.5s
      client=03 | n=  142 | train_acc=0.9507 | val_acc=0.8497 | val_f1=0.8480 | val_auc=0.9641 | time=1.1s
      client=04 | n= 2426 | train_acc=0.8656 | val_acc=0.8605 | val_f1=0.8585 | val_auc=0.9667 | time=1.8s
      client=05 | n=    1 | train_acc=1.0000 | val_acc=0.5300 | val_f1=0.3478 | val_auc=0.9569 | time=1.1s
      client=06 | n= 4736 | train_acc=0.9139 | val_acc=0.8885 | val_f1=0.8857 | val_auc=0.9657 | time=2.4s
      client=07 | n= 3988 | train_acc=0.9812 | val_acc=0.4707 | val_f1=0.3201 | val_auc=0.9614 | time=2.3s
      client=08 | n= 4083 | train_acc=0.8922 | val_acc=0.8861 | val_f1=0.8836 | val_auc=0.9765 | time=2.2s
      client=09 | n= 4239 | train_acc=0.9516 | val_acc=0.8889 | val_f1=0.8863 | val_auc=0.9649 | time=2.3s
      client=10 | n= 8696 | train_acc=0.9872 | val_acc=0.5293 | val_f1=0.3461 | val_auc=0.9656 | time=3.6s
      client=11 | n= 3369 | train_acc=0.9632 | val_acc=0.7927 | val_f1=0.7884 | val_auc=0.9616 | time=2.1s
    
    [WII21] 46 clients
      client=00 | n= 8354 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.7s
      client=01 | n=25751 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=6.1s
      client=02 | n=42801 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9998 | val_auc=0.9999 | time=9.5s
      client=03 | n=15271 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.0s
      client=04 | n=17243 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9999 | time=4.5s
      client=05 | n=44654 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.9s
      client=06 | n=47255 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=10.4s
      client=07 | n=  457 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=08 | n=12388 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9998 | val_auc=0.9999 | time=3.5s
      client=09 | n=12876 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.6s
      client=10 | n= 6523 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=11 | n= 6709 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=12 | n= 3239 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=13 | n=  259 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=14 | n= 8000 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.6s
      client=15 | n=44321 | train_acc=0.9999 | val_acc=0.9996 | val_f1=0.9986 | val_auc=0.9999 | time=9.9s
      client=16 | n=87021 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=18.3s
      client=17 | n= 6986 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=2.5s
      client=18 | n= 9329 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.9s
      client=19 | n=37143 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.4s
      client=20 | n=34755 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9998 | time=7.9s
      client=21 | n= 7268 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.5s
      client=22 | n=15886 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.2s
      client=23 | n=18446 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.8s
      client=24 | n=21907 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9999 | time=5.4s
      client=25 | n=15749 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9999 | time=4.2s
      client=26 | n=41900 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9998 | val_auc=0.9998 | time=9.3s
      client=27 | n=22675 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.5s
      client=28 | n=22659 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9999 | time=5.5s
      client=29 | n= 6231 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=30 | n=11135 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.2s
      client=31 | n=10805 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.2s
      client=32 | n=11918 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.4s
      client=33 | n=12890 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.6s
      client=34 | n= 5447 | train_acc=0.9998 | val_acc=0.9999 | val_f1=0.9995 | val_auc=1.0000 | time=2.1s
      client=35 | n=  743 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=36 | n=52117 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=11.3s
      client=37 | n= 3378 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=38 | n= 7628 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.5s
      client=39 | n=10187 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.1s
      client=40 | n= 2932 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      client=41 | n=36349 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.2s
      client=42 | n= 5003 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.0s
      client=43 | n=14026 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9999 | time=3.8s
      client=44 | n= 4514 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.9s
      client=45 | n= 2996 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      [VAL] I23Sub | acc=0.9968 | f1=0.9894 | auc=0.9972 | logloss=0.0206
      [VAL] K99Sub | acc=0.9532 | f1=0.9530 | auc=0.9943 | logloss=0.1350
      [VAL] NTD1 | acc=0.9669 | f1=0.8786 | auc=0.9863 | logloss=0.1051
      [VAL] NTD2 | acc=0.8545 | f1=0.8516 | auc=0.9603 | logloss=0.4107
      [VAL] WII21 | acc=1.0000 | f1=1.0000 | auc=1.0000 | logloss=0.0000
    ----------------------------------------------------------------------------------------------------
    GLOBAL VAL | acc=0.9871 | f1=0.9674 | logloss=0.0393 | round_time=953.9s
    
    ====================================================================================================
    FEDERATED ROUND 9/15
    ====================================================================================================
    
    [I23Sub] 2 clients
      client=00 | n= 9854 | train_acc=0.9971 | val_acc=0.9980 | val_f1=0.9933 | val_auc=0.9970 | time=2.2s
      client=01 | n= 6347 | train_acc=0.9983 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9972 | time=1.8s
    
    [K99Sub] 4 clients
      client=00 | n= 1006 | train_acc=0.9672 | val_acc=0.9481 | val_f1=0.9481 | val_auc=0.9932 | time=1.1s
      client=01 | n= 1816 | train_acc=0.9769 | val_acc=0.8952 | val_f1=0.8921 | val_auc=0.9898 | time=1.3s
      client=02 | n= 8402 | train_acc=0.9820 | val_acc=0.9757 | val_f1=0.9755 | val_auc=0.9975 | time=2.0s
      client=03 | n= 6410 | train_acc=0.9761 | val_acc=0.9518 | val_f1=0.9518 | val_auc=0.9972 | time=1.7s
    
    [NTD1] 10 clients
      client=00 | n=37754 | train_acc=0.9562 | val_acc=0.9700 | val_f1=0.8944 | val_auc=0.9848 | time=3.9s
      client=01 | n= 2209 | train_acc=0.9986 | val_acc=0.8300 | val_f1=0.6987 | val_auc=0.9823 | time=1.3s
      client=02 | n=48001 | train_acc=0.9902 | val_acc=0.9665 | val_f1=0.8776 | val_auc=0.9856 | time=4.4s
      client=03 | n=12872 | train_acc=0.9544 | val_acc=0.9490 | val_f1=0.8682 | val_auc=0.9884 | time=2.0s
      client=04 | n=62448 | train_acc=0.9939 | val_acc=0.9150 | val_f1=0.4778 | val_auc=0.9860 | time=5.6s
      client=05 | n=27705 | train_acc=0.9811 | val_acc=0.9739 | val_f1=0.9087 | val_auc=0.9880 | time=3.1s
      client=06 | n= 1357 | train_acc=0.9234 | val_acc=0.8412 | val_f1=0.7108 | val_auc=0.9840 | time=1.2s
      client=07 | n= 5166 | train_acc=0.9665 | val_acc=0.9437 | val_f1=0.8580 | val_auc=0.9864 | time=1.4s
      client=08 | n= 5839 | train_acc=0.9596 | val_acc=0.9364 | val_f1=0.8445 | val_auc=0.9858 | time=1.4s
      client=09 | n=54260 | train_acc=0.9814 | val_acc=0.9738 | val_f1=0.9088 | val_auc=0.9897 | time=4.8s
    
    [NTD2] 12 clients
      client=00 | n= 2230 | train_acc=0.9525 | val_acc=0.7959 | val_f1=0.7919 | val_auc=0.9630 | time=1.6s
      client=01 | n= 3379 | train_acc=0.9453 | val_acc=0.7960 | val_f1=0.7920 | val_auc=0.9655 | time=2.0s
      client=02 | n= 1789 | train_acc=0.9251 | val_acc=0.8851 | val_f1=0.8827 | val_auc=0.9650 | time=1.5s
      client=03 | n=  142 | train_acc=0.9437 | val_acc=0.4759 | val_f1=0.3321 | val_auc=0.9609 | time=1.1s
      client=04 | n= 2426 | train_acc=0.8751 | val_acc=0.8858 | val_f1=0.8834 | val_auc=0.9769 | time=1.7s
      client=05 | n=    1 | train_acc=1.0000 | val_acc=0.6183 | val_f1=0.5286 | val_auc=0.9561 | time=1.0s
      client=06 | n= 4736 | train_acc=0.9077 | val_acc=0.8845 | val_f1=0.8821 | val_auc=0.9740 | time=2.5s
      client=07 | n= 3988 | train_acc=0.9812 | val_acc=0.4707 | val_f1=0.3201 | val_auc=0.9475 | time=2.2s
      client=08 | n= 4083 | train_acc=0.8969 | val_acc=0.8894 | val_f1=0.8868 | val_auc=0.9794 | time=2.2s
      client=09 | n= 4239 | train_acc=0.9533 | val_acc=0.8881 | val_f1=0.8854 | val_auc=0.9781 | time=2.3s
      client=10 | n= 8696 | train_acc=0.9872 | val_acc=0.5293 | val_f1=0.3461 | val_auc=0.9234 | time=3.5s
      client=11 | n= 3369 | train_acc=0.9653 | val_acc=0.7910 | val_f1=0.7865 | val_auc=0.9639 | time=2.0s
    
    [WII21] 46 clients
      client=00 | n= 8354 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.7s
      client=01 | n=25751 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=6.1s
      client=02 | n=42801 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=9.6s
      client=03 | n=15271 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.0s
      client=04 | n=17243 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.4s
      client=05 | n=44654 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.9s
      client=06 | n=47255 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=10.4s
      client=07 | n=  457 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=08 | n=12388 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.5s
      client=09 | n=12876 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.6s
      client=10 | n= 6523 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=11 | n= 6709 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=12 | n= 3239 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=13 | n=  259 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=14 | n= 8000 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.6s
      client=15 | n=44321 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.8s
      client=16 | n=87021 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=18.2s
      client=17 | n= 6986 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=18 | n= 9329 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.9s
      client=19 | n=37143 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.4s
      client=20 | n=34755 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9997 | time=7.9s
      client=21 | n= 7268 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.5s
      client=22 | n=15886 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.2s
      client=23 | n=18446 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.7s
      client=24 | n=21907 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.4s
      client=25 | n=15749 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9998 | time=4.2s
      client=26 | n=41900 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.3s
      client=27 | n=22675 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.5s
      client=28 | n=22659 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.5s
      client=29 | n= 6231 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=30 | n=11135 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.3s
      client=31 | n=10805 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.3s
      client=32 | n=11918 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.4s
      client=33 | n=12890 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.6s
      client=34 | n= 5447 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.1s
      client=35 | n=  743 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=36 | n=52117 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=11.4s
      client=37 | n= 3378 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=38 | n= 7628 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.5s
      client=39 | n=10187 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.0s
      client=40 | n= 2932 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      client=41 | n=36349 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.2s
      client=42 | n= 5003 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.1s
      client=43 | n=14026 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.8s
      client=44 | n= 4514 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.9s
      client=45 | n= 2996 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      [VAL] I23Sub | acc=0.9980 | f1=0.9933 | auc=0.9972 | logloss=0.0173
      [VAL] K99Sub | acc=0.9720 | f1=0.9718 | auc=0.9953 | logloss=0.1035
      [VAL] NTD1 | acc=0.9669 | f1=0.8786 | auc=0.9868 | logloss=0.1032
      [VAL] NTD2 | acc=0.7882 | f1=0.7722 | auc=0.9653 | logloss=0.6132
      [VAL] WII21 | acc=1.0000 | f1=1.0000 | auc=1.0000 | logloss=0.0000
    ----------------------------------------------------------------------------------------------------
    GLOBAL VAL | acc=0.9851 | f1=0.9651 | logloss=0.0451 | round_time=949.0s
    
    ====================================================================================================
    FEDERATED ROUND 10/15
    ====================================================================================================
    
    [I23Sub] 2 clients
      client=00 | n= 9854 | train_acc=0.9971 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9973 | time=2.2s
      client=01 | n= 6347 | train_acc=0.9981 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9972 | time=1.7s
    
    [K99Sub] 4 clients
      client=00 | n= 1006 | train_acc=0.9652 | val_acc=0.9640 | val_f1=0.9639 | val_auc=0.9967 | time=1.1s
      client=01 | n= 1816 | train_acc=0.9752 | val_acc=0.9685 | val_f1=0.9683 | val_auc=0.9916 | time=1.2s
      client=02 | n= 8402 | train_acc=0.9836 | val_acc=0.9794 | val_f1=0.9793 | val_auc=0.9974 | time=1.9s
      client=03 | n= 6410 | train_acc=0.9817 | val_acc=0.9635 | val_f1=0.9634 | val_auc=0.9972 | time=1.7s
    
    [NTD1] 10 clients
      client=00 | n=37754 | train_acc=0.9598 | val_acc=0.9732 | val_f1=0.9055 | val_auc=0.9913 | time=3.6s
      client=01 | n= 2209 | train_acc=0.9986 | val_acc=0.8360 | val_f1=0.7052 | val_auc=0.9663 | time=1.2s
      client=02 | n=48001 | train_acc=0.9909 | val_acc=0.9704 | val_f1=0.8940 | val_auc=0.9891 | time=4.3s
      client=03 | n=12872 | train_acc=0.9529 | val_acc=0.9612 | val_f1=0.8890 | val_auc=0.9884 | time=2.0s
      client=04 | n=62448 | train_acc=0.9939 | val_acc=0.9150 | val_f1=0.4778 | val_auc=0.9863 | time=5.3s
      client=05 | n=27705 | train_acc=0.9801 | val_acc=0.9730 | val_f1=0.9051 | val_auc=0.9886 | time=3.0s
      client=06 | n= 1357 | train_acc=0.9654 | val_acc=0.9265 | val_f1=0.8276 | val_auc=0.9856 | time=1.2s
      client=07 | n= 5166 | train_acc=0.9694 | val_acc=0.9276 | val_f1=0.8293 | val_auc=0.9862 | time=1.5s
      client=08 | n= 5839 | train_acc=0.9575 | val_acc=0.9388 | val_f1=0.8493 | val_auc=0.9847 | time=1.5s
      client=09 | n=54260 | train_acc=0.9799 | val_acc=0.9703 | val_f1=0.8934 | val_auc=0.9874 | time=4.7s
    
    [NTD2] 12 clients
      client=00 | n= 2230 | train_acc=0.9520 | val_acc=0.7960 | val_f1=0.7920 | val_auc=0.9623 | time=1.6s
      client=01 | n= 3379 | train_acc=0.9447 | val_acc=0.8044 | val_f1=0.8012 | val_auc=0.9669 | time=2.1s
      client=02 | n= 1789 | train_acc=0.9352 | val_acc=0.8897 | val_f1=0.8870 | val_auc=0.9645 | time=1.5s
      client=03 | n=  142 | train_acc=0.9507 | val_acc=0.8498 | val_f1=0.8481 | val_auc=0.9655 | time=1.1s
      client=04 | n= 2426 | train_acc=0.8846 | val_acc=0.8846 | val_f1=0.8822 | val_auc=0.9775 | time=1.8s
      client=05 | n=    1 | train_acc=1.0000 | val_acc=0.5297 | val_f1=0.3472 | val_auc=0.9639 | time=1.1s
      client=06 | n= 4736 | train_acc=0.9130 | val_acc=0.8883 | val_f1=0.8856 | val_auc=0.9784 | time=2.4s
      client=07 | n= 3988 | train_acc=0.9812 | val_acc=0.4707 | val_f1=0.3201 | val_auc=0.9173 | time=2.2s
      client=08 | n= 4083 | train_acc=0.8893 | val_acc=0.8303 | val_f1=0.8286 | val_auc=0.9759 | time=2.2s
      client=09 | n= 4239 | train_acc=0.9521 | val_acc=0.8893 | val_f1=0.8866 | val_auc=0.9813 | time=2.3s
      client=10 | n= 8696 | train_acc=0.9872 | val_acc=0.5293 | val_f1=0.3461 | val_auc=0.9685 | time=3.6s
      client=11 | n= 3369 | train_acc=0.9650 | val_acc=0.7942 | val_f1=0.7901 | val_auc=0.9649 | time=2.0s
    
    [WII21] 46 clients
      client=00 | n= 8354 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.7s
      client=01 | n=25751 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=6.1s
      client=02 | n=42801 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.5s
      client=03 | n=15271 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.1s
      client=04 | n=17243 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.4s
      client=05 | n=44654 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.8s
      client=06 | n=47255 | train_acc=0.9999 | val_acc=0.9999 | val_f1=0.9998 | val_auc=1.0000 | time=10.4s
      client=07 | n=  457 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=08 | n=12388 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.5s
      client=09 | n=12876 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.6s
      client=10 | n= 6523 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=11 | n= 6709 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=12 | n= 3239 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=13 | n=  259 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=14 | n= 8000 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.7s
      client=15 | n=44321 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.9s
      client=16 | n=87021 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=18.3s
      client=17 | n= 6986 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=18 | n= 9329 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.8s
      client=19 | n=37143 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.4s
      client=20 | n=34755 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=7.9s
      client=21 | n= 7268 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.5s
      client=22 | n=15886 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.2s
      client=23 | n=18446 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.7s
      client=24 | n=21907 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.4s
      client=25 | n=15749 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.1s
      client=26 | n=41900 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.3s
      client=27 | n=22675 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.5s
      client=28 | n=22659 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.4s
      client=29 | n= 6231 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.2s
      client=30 | n=11135 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.2s
      client=31 | n=10805 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.2s
      client=32 | n=11918 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.4s
      client=33 | n=12890 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.6s
      client=34 | n= 5447 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.2s
      client=35 | n=  743 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.2s
      client=36 | n=52117 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=11.4s
      client=37 | n= 3378 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.8s
      client=38 | n= 7628 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.5s
      client=39 | n=10187 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.1s
      client=40 | n= 2932 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      client=41 | n=36349 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.3s
      client=42 | n= 5003 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.0s
      client=43 | n=14026 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.8s
      client=44 | n= 4514 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.9s
      client=45 | n= 2996 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      [VAL] I23Sub | acc=0.9983 | f1=0.9943 | auc=0.9972 | logloss=0.0212
      [VAL] K99Sub | acc=0.9653 | f1=0.9652 | auc=0.9965 | logloss=0.1113
      [VAL] NTD1 | acc=0.9669 | f1=0.8786 | auc=0.9878 | logloss=0.0990
      [VAL] NTD2 | acc=0.7915 | f1=0.7763 | auc=0.9662 | logloss=0.5804
      [VAL] WII21 | acc=1.0000 | f1=1.0000 | auc=1.0000 | logloss=0.0000
    ----------------------------------------------------------------------------------------------------
    GLOBAL VAL | acc=0.9852 | f1=0.9651 | logloss=0.0433 | round_time=947.4s
    
    ====================================================================================================
    FEDERATED ROUND 11/15
    ====================================================================================================
    
    [I23Sub] 2 clients
      client=00 | n= 9854 | train_acc=0.9971 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9971 | time=2.3s
      client=01 | n= 6347 | train_acc=0.9980 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9970 | time=1.8s
    
    [K99Sub] 4 clients
      client=00 | n= 1006 | train_acc=0.9722 | val_acc=0.9500 | val_f1=0.9499 | val_auc=0.9956 | time=1.2s
      client=01 | n= 1816 | train_acc=0.9780 | val_acc=0.9693 | val_f1=0.9692 | val_auc=0.9954 | time=1.3s
      client=02 | n= 8402 | train_acc=0.9791 | val_acc=0.9709 | val_f1=0.9707 | val_auc=0.9978 | time=2.0s
      client=03 | n= 6410 | train_acc=0.9858 | val_acc=0.9743 | val_f1=0.9743 | val_auc=0.9968 | time=1.8s
    
    [NTD1] 10 clients
      client=00 | n=37754 | train_acc=0.9572 | val_acc=0.9714 | val_f1=0.8982 | val_auc=0.9870 | time=3.8s
      client=01 | n= 2209 | train_acc=0.9986 | val_acc=0.8320 | val_f1=0.7008 | val_auc=0.9802 | time=1.3s
      client=02 | n=48001 | train_acc=0.9908 | val_acc=0.9724 | val_f1=0.9026 | val_auc=0.9888 | time=4.5s
      client=03 | n=12872 | train_acc=0.9528 | val_acc=0.9609 | val_f1=0.8879 | val_auc=0.9885 | time=2.0s
      client=04 | n=62448 | train_acc=0.9939 | val_acc=0.9150 | val_f1=0.4778 | val_auc=0.9887 | time=5.3s
      client=05 | n=27705 | train_acc=0.9808 | val_acc=0.9734 | val_f1=0.9064 | val_auc=0.9897 | time=2.9s
      client=06 | n= 1357 | train_acc=0.9639 | val_acc=0.9455 | val_f1=0.8616 | val_auc=0.9872 | time=1.2s
      client=07 | n= 5166 | train_acc=0.9386 | val_acc=0.9431 | val_f1=0.8523 | val_auc=0.9840 | time=1.5s
      client=08 | n= 5839 | train_acc=0.9563 | val_acc=0.9445 | val_f1=0.8586 | val_auc=0.9874 | time=1.5s
      client=09 | n=54260 | train_acc=0.9812 | val_acc=0.9734 | val_f1=0.9064 | val_auc=0.9908 | time=5.0s
    
    [NTD2] 12 clients
      client=00 | n= 2230 | train_acc=0.9525 | val_acc=0.7960 | val_f1=0.7921 | val_auc=0.9668 | time=1.7s
      client=01 | n= 3379 | train_acc=0.9447 | val_acc=0.8045 | val_f1=0.8013 | val_auc=0.9671 | time=2.1s
      client=02 | n= 1789 | train_acc=0.9363 | val_acc=0.8898 | val_f1=0.8871 | val_auc=0.9657 | time=1.5s
      client=03 | n=  142 | train_acc=0.9577 | val_acc=0.8503 | val_f1=0.8486 | val_auc=0.9657 | time=1.1s
      client=04 | n= 2426 | train_acc=0.8714 | val_acc=0.8872 | val_f1=0.8855 | val_auc=0.9770 | time=1.7s
      client=05 | n=    1 | train_acc=1.0000 | val_acc=0.6057 | val_f1=0.5060 | val_auc=0.9679 | time=1.1s
      client=06 | n= 4736 | train_acc=0.9145 | val_acc=0.8889 | val_f1=0.8863 | val_auc=0.9791 | time=2.4s
      client=07 | n= 3988 | train_acc=0.9812 | val_acc=0.4707 | val_f1=0.3201 | val_auc=0.9659 | time=2.2s
      client=08 | n= 4083 | train_acc=0.9025 | val_acc=0.8886 | val_f1=0.8859 | val_auc=0.9774 | time=2.2s
      client=09 | n= 4239 | train_acc=0.9523 | val_acc=0.8889 | val_f1=0.8862 | val_auc=0.9706 | time=2.3s
      client=10 | n= 8696 | train_acc=0.9872 | val_acc=0.5293 | val_f1=0.3461 | val_auc=0.9570 | time=3.6s
      client=11 | n= 3369 | train_acc=0.9650 | val_acc=0.7958 | val_f1=0.7918 | val_auc=0.9647 | time=2.0s
    
    [WII21] 46 clients
      client=00 | n= 8354 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.6s
      client=01 | n=25751 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=6.1s
      client=02 | n=42801 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.5s
      client=03 | n=15271 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.0s
      client=04 | n=17243 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.5s
      client=05 | n=44654 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.9s
      client=06 | n=47255 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=10.4s
      client=07 | n=  457 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=08 | n=12388 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.5s
      client=09 | n=12876 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.6s
      client=10 | n= 6523 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=11 | n= 6709 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=12 | n= 3239 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=13 | n=  259 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=14 | n= 8000 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.6s
      client=15 | n=44321 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.9s
      client=16 | n=87021 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=18.3s
      client=17 | n= 6986 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=18 | n= 9329 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.9s
      client=19 | n=37143 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.4s
      client=20 | n=34755 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=7.9s
      client=21 | n= 7268 | train_acc=0.9999 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.5s
      client=22 | n=15886 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.2s
      client=23 | n=18446 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.7s
      client=24 | n=21907 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.4s
      client=25 | n=15749 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.1s
      client=26 | n=41900 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.3s
      client=27 | n=22675 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.5s
      client=28 | n=22659 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.5s
      client=29 | n= 6231 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.2s
      client=30 | n=11135 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.2s
      client=31 | n=10805 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.2s
      client=32 | n=11918 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.5s
      client=33 | n=12890 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.6s
      client=34 | n= 5447 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.1s
      client=35 | n=  743 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=36 | n=52117 | train_acc=0.9999 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=11.4s
      client=37 | n= 3378 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=38 | n= 7628 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.5s
      client=39 | n=10187 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.0s
      client=40 | n= 2932 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      client=41 | n=36349 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.2s
      client=42 | n= 5003 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.0s
      client=43 | n=14026 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.8s
      client=44 | n= 4514 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.9s
      client=45 | n= 2996 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      [VAL] I23Sub | acc=0.9968 | f1=0.9894 | auc=0.9974 | logloss=0.0268
      [VAL] K99Sub | acc=0.9354 | f1=0.9354 | auc=0.9951 | logloss=0.1482
      [VAL] NTD1 | acc=0.9695 | f1=0.8900 | auc=0.9879 | logloss=0.0930
      [VAL] NTD2 | acc=0.8861 | f1=0.8832 | auc=0.9659 | logloss=0.4301
      [VAL] WII21 | acc=1.0000 | f1=1.0000 | auc=1.0000 | logloss=0.0000
    ----------------------------------------------------------------------------------------------------
    GLOBAL VAL | acc=0.9884 | f1=0.9707 | logloss=0.0376 | round_time=947.9s
    
    ====================================================================================================
    FEDERATED ROUND 12/15
    ====================================================================================================
    
    [I23Sub] 2 clients
      client=00 | n= 9854 | train_acc=0.9971 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9974 | time=2.2s
      client=01 | n= 6347 | train_acc=0.9983 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9972 | time=1.8s
    
    [K99Sub] 4 clients
      client=00 | n= 1006 | train_acc=0.9722 | val_acc=0.9492 | val_f1=0.9492 | val_auc=0.9968 | time=1.1s
      client=01 | n= 1816 | train_acc=0.9895 | val_acc=0.9418 | val_f1=0.9409 | val_auc=0.9961 | time=1.3s
      client=02 | n= 8402 | train_acc=0.9848 | val_acc=0.9799 | val_f1=0.9798 | val_auc=0.9979 | time=2.0s
      client=03 | n= 6410 | train_acc=0.9841 | val_acc=0.9717 | val_f1=0.9716 | val_auc=0.9971 | time=1.7s
    
    [NTD1] 10 clients
      client=00 | n=37754 | train_acc=0.9610 | val_acc=0.9622 | val_f1=0.8909 | val_auc=0.9930 | time=3.9s
      client=01 | n= 2209 | train_acc=0.9986 | val_acc=0.8373 | val_f1=0.7066 | val_auc=0.9582 | time=1.3s
      client=02 | n=48001 | train_acc=0.9896 | val_acc=0.9150 | val_f1=0.4778 | val_auc=0.9870 | time=4.6s
      client=03 | n=12872 | train_acc=0.9559 | val_acc=0.9529 | val_f1=0.8752 | val_auc=0.9883 | time=2.1s
      client=04 | n=62448 | train_acc=0.9939 | val_acc=0.9150 | val_f1=0.4778 | val_auc=0.9858 | time=5.3s
      client=05 | n=27705 | train_acc=0.9803 | val_acc=0.9726 | val_f1=0.9030 | val_auc=0.9906 | time=3.2s
      client=06 | n= 1357 | train_acc=0.9550 | val_acc=0.9444 | val_f1=0.8581 | val_auc=0.9867 | time=1.2s
      client=07 | n= 5166 | train_acc=0.9700 | val_acc=0.9221 | val_f1=0.8204 | val_auc=0.9891 | time=1.5s
      client=08 | n= 5839 | train_acc=0.9558 | val_acc=0.9431 | val_f1=0.8558 | val_auc=0.9870 | time=1.5s
      client=09 | n=54260 | train_acc=0.9811 | val_acc=0.9728 | val_f1=0.9040 | val_auc=0.9892 | time=5.1s
    
    [NTD2] 12 clients
      client=00 | n= 2230 | train_acc=0.9511 | val_acc=0.7964 | val_f1=0.7924 | val_auc=0.9629 | time=1.7s
      client=01 | n= 3379 | train_acc=0.9494 | val_acc=0.8292 | val_f1=0.8275 | val_auc=0.9766 | time=2.1s
      client=02 | n= 1789 | train_acc=0.9251 | val_acc=0.8850 | val_f1=0.8825 | val_auc=0.9664 | time=1.6s
      client=03 | n=  142 | train_acc=0.9507 | val_acc=0.7959 | val_f1=0.7919 | val_auc=0.9641 | time=1.1s
      client=04 | n= 2426 | train_acc=0.8615 | val_acc=0.8291 | val_f1=0.8273 | val_auc=0.9752 | time=1.7s
      client=05 | n=    1 | train_acc=1.0000 | val_acc=0.7440 | val_f1=0.7169 | val_auc=0.9653 | time=1.1s
      client=06 | n= 4736 | train_acc=0.9126 | val_acc=0.8888 | val_f1=0.8861 | val_auc=0.9798 | time=2.4s
      client=07 | n= 3988 | train_acc=0.9812 | val_acc=0.4707 | val_f1=0.3201 | val_auc=0.9657 | time=2.2s
      client=08 | n= 4083 | train_acc=0.9052 | val_acc=0.8876 | val_f1=0.8850 | val_auc=0.9789 | time=2.2s
      client=09 | n= 4239 | train_acc=0.9490 | val_acc=0.8887 | val_f1=0.8860 | val_auc=0.9799 | time=2.3s
      client=10 | n= 8696 | train_acc=0.9872 | val_acc=0.5293 | val_f1=0.3461 | val_auc=0.9667 | time=3.6s
      client=11 | n= 3369 | train_acc=0.9673 | val_acc=0.8291 | val_f1=0.8274 | val_auc=0.9758 | time=2.1s
    
    [WII21] 46 clients
      client=00 | n= 8354 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.7s
      client=01 | n=25751 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=6.2s
      client=02 | n=42801 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.5s
      client=03 | n=15271 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.0s
      client=04 | n=17243 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9999 | time=4.4s
      client=05 | n=44654 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.9s
      client=06 | n=47255 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=10.4s
      client=07 | n=  457 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.2s
      client=08 | n=12388 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.5s
      client=09 | n=12876 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.6s
      client=10 | n= 6523 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=11 | n= 6709 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=12 | n= 3239 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=13 | n=  259 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=14 | n= 8000 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.7s
      client=15 | n=44321 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=9.9s
      client=16 | n=87021 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=18.3s
      client=17 | n= 6986 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=18 | n= 9329 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.9s
      client=19 | n=37143 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.4s
      client=20 | n=34755 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=7.9s
      client=21 | n= 7268 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=22 | n=15886 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.2s
      client=23 | n=18446 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.7s
      client=24 | n=21907 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.4s
      client=25 | n=15749 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.1s
      client=26 | n=41900 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.3s
      client=27 | n=22675 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.5s
      client=28 | n=22659 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.5s
      client=29 | n= 6231 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=30 | n=11135 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.2s
      client=31 | n=10805 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.2s
      client=32 | n=11918 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.4s
      client=33 | n=12890 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.6s
      client=34 | n= 5447 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.1s
      client=35 | n=  743 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=36 | n=52117 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=11.3s
      client=37 | n= 3378 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=38 | n= 7628 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.5s
      client=39 | n=10187 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.0s
      client=40 | n= 2932 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      client=41 | n=36349 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.2s
      client=42 | n= 5003 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.0s
      client=43 | n=14026 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.8s
      client=44 | n= 4514 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.9s
      client=45 | n= 2996 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      [VAL] I23Sub | acc=0.9968 | f1=0.9894 | auc=0.9975 | logloss=0.0235
      [VAL] K99Sub | acc=0.9481 | f1=0.9481 | auc=0.9948 | logloss=0.1253
      [VAL] NTD1 | acc=0.9692 | f1=0.8886 | auc=0.9883 | logloss=0.0932
      [VAL] NTD2 | acc=0.8769 | f1=0.8733 | auc=0.9689 | logloss=0.4527
      [VAL] WII21 | acc=1.0000 | f1=1.0000 | auc=1.0000 | logloss=0.0000
    ----------------------------------------------------------------------------------------------------
    GLOBAL VAL | acc=0.9882 | f1=0.9702 | logloss=0.0380 | round_time=950.8s
    
    ====================================================================================================
    FEDERATED ROUND 13/15
    ====================================================================================================
    
    [I23Sub] 2 clients
      client=00 | n= 9854 | train_acc=0.9969 | val_acc=0.9980 | val_f1=0.9933 | val_auc=0.9972 | time=2.2s
      client=01 | n= 6347 | train_acc=0.9975 | val_acc=0.9968 | val_f1=0.9894 | val_auc=0.9975 | time=1.8s
    
    [K99Sub] 4 clients
      client=00 | n= 1006 | train_acc=0.9702 | val_acc=0.9558 | val_f1=0.9558 | val_auc=0.9957 | time=1.1s
      client=01 | n= 1816 | train_acc=0.9895 | val_acc=0.9547 | val_f1=0.9543 | val_auc=0.9963 | time=1.2s
      client=02 | n= 8402 | train_acc=0.9783 | val_acc=0.9640 | val_f1=0.9637 | val_auc=0.9969 | time=2.0s
      client=03 | n= 6410 | train_acc=0.9833 | val_acc=0.9714 | val_f1=0.9714 | val_auc=0.9977 | time=1.8s
    
    [NTD1] 10 clients
      client=00 | n=37754 | train_acc=0.9593 | val_acc=0.9729 | val_f1=0.9045 | val_auc=0.9908 | time=3.8s
      client=01 | n= 2209 | train_acc=0.9986 | val_acc=0.8382 | val_f1=0.7076 | val_auc=0.9312 | time=1.2s
      client=02 | n=48001 | train_acc=0.9910 | val_acc=0.9712 | val_f1=0.8974 | val_auc=0.9870 | time=4.4s
      client=03 | n=12872 | train_acc=0.9562 | val_acc=0.9631 | val_f1=0.8942 | val_auc=0.9899 | time=2.0s
      client=04 | n=62448 | train_acc=0.9939 | val_acc=0.9150 | val_f1=0.4778 | val_auc=0.9471 | time=5.6s
      client=05 | n=27705 | train_acc=0.9796 | val_acc=0.9717 | val_f1=0.8995 | val_auc=0.9899 | time=3.1s
      client=06 | n= 1357 | train_acc=0.9646 | val_acc=0.9397 | val_f1=0.8510 | val_auc=0.9870 | time=1.2s
      client=07 | n= 5166 | train_acc=0.9706 | val_acc=0.9224 | val_f1=0.8211 | val_auc=0.9892 | time=1.5s
      client=08 | n= 5839 | train_acc=0.9558 | val_acc=0.9313 | val_f1=0.8353 | val_auc=0.9869 | time=1.5s
      client=09 | n=54260 | train_acc=0.9803 | val_acc=0.9717 | val_f1=0.8996 | val_auc=0.9845 | time=5.1s
    
    [NTD2] 12 clients
      client=00 | n= 2230 | train_acc=0.9556 | val_acc=0.8208 | val_f1=0.8184 | val_auc=0.9757 | time=1.7s
      client=01 | n= 3379 | train_acc=0.9491 | val_acc=0.8286 | val_f1=0.8270 | val_auc=0.9744 | time=2.1s
      client=02 | n= 1789 | train_acc=0.9352 | val_acc=0.8867 | val_f1=0.8838 | val_auc=0.9816 | time=1.5s
      client=03 | n=  142 | train_acc=0.9507 | val_acc=0.7962 | val_f1=0.7922 | val_auc=0.9657 | time=1.1s
      client=04 | n= 2426 | train_acc=0.8916 | val_acc=0.8880 | val_f1=0.8853 | val_auc=0.9772 | time=1.7s
      client=05 | n=    1 | train_acc=1.0000 | val_acc=0.6895 | val_f1=0.6422 | val_auc=0.9657 | time=1.1s
      client=06 | n= 4736 | train_acc=0.9136 | val_acc=0.8892 | val_f1=0.8865 | val_auc=0.9831 | time=2.4s
      client=07 | n= 3988 | train_acc=0.9872 | val_acc=0.7601 | val_f1=0.7519 | val_auc=0.9667 | time=2.2s
      client=08 | n= 4083 | train_acc=0.8844 | val_acc=0.8894 | val_f1=0.8868 | val_auc=0.9778 | time=2.3s
      client=09 | n= 4239 | train_acc=0.9509 | val_acc=0.8880 | val_f1=0.8854 | val_auc=0.9808 | time=2.3s
      client=10 | n= 8696 | train_acc=0.9872 | val_acc=0.5293 | val_f1=0.3461 | val_auc=0.9367 | time=3.6s
      client=11 | n= 3369 | train_acc=0.9697 | val_acc=0.8292 | val_f1=0.8275 | val_auc=0.9757 | time=2.1s
    
    [WII21] 46 clients
      client=00 | n= 8354 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.7s
      client=01 | n=25751 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=6.2s
      client=02 | n=42801 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.6s
      client=03 | n=15271 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.1s
      client=04 | n=17243 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.5s
      client=05 | n=44654 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.9s
      client=06 | n=47255 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=10.4s
      client=07 | n=  457 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.2s
      client=08 | n=12388 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9998 | val_auc=0.9997 | time=3.5s
      client=09 | n=12876 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9998 | time=3.6s
      client=10 | n= 6523 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=11 | n= 6709 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=2.4s
      client=12 | n= 3239 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=13 | n=  259 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=14 | n= 8000 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.6s
      client=15 | n=44321 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.8s
      client=16 | n=87021 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=18.3s
      client=17 | n= 6986 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9999 | time=2.4s
      client=18 | n= 9329 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=2.9s
      client=19 | n=37143 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.4s
      client=20 | n=34755 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=7.9s
      client=21 | n= 7268 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=22 | n=15886 | train_acc=0.9999 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.2s
      client=23 | n=18446 | train_acc=0.9999 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.7s
      client=24 | n=21907 | train_acc=1.0000 | val_acc=0.9999 | val_f1=0.9998 | val_auc=0.9999 | time=5.3s
      client=25 | n=15749 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=4.1s
      client=26 | n=41900 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.3s
      client=27 | n=22675 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.5s
      client=28 | n=22659 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9998 | time=5.5s
      client=29 | n= 6231 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.2s
      client=30 | n=11135 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9998 | val_auc=0.9997 | time=3.2s
      client=31 | n=10805 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.2s
      client=32 | n=11918 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.4s
      client=33 | n=12890 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.6s
      client=34 | n= 5447 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.1s
      client=35 | n=  743 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=36 | n=52117 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=11.4s
      client=37 | n= 3378 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=38 | n= 7628 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.6s
      client=39 | n=10187 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.0s
      client=40 | n= 2932 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      client=41 | n=36349 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=8.2s
      client=42 | n= 5003 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.0s
      client=43 | n=14026 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=0.9999 | time=3.8s
      client=44 | n= 4514 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.0s
      client=45 | n= 2996 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      [VAL] I23Sub | acc=0.9983 | f1=0.9943 | auc=0.9974 | logloss=0.0240
      [VAL] K99Sub | acc=0.9537 | f1=0.9536 | auc=0.9944 | logloss=0.1191
      [VAL] NTD1 | acc=0.9707 | f1=0.8949 | auc=0.9882 | logloss=0.0891
      [VAL] NTD2 | acc=0.8235 | f1=0.8138 | auc=0.9790 | logloss=0.5366
      [VAL] WII21 | acc=1.0000 | f1=1.0000 | auc=1.0000 | logloss=0.0000
    ----------------------------------------------------------------------------------------------------
    GLOBAL VAL | acc=0.9869 | f1=0.9698 | logloss=0.0398 | round_time=952.1s
    
    ====================================================================================================
    FEDERATED ROUND 14/15
    ====================================================================================================
    
    [I23Sub] 2 clients
      client=00 | n= 9854 | train_acc=0.9971 | val_acc=0.9980 | val_f1=0.9933 | val_auc=0.9973 | time=2.2s
      client=01 | n= 6347 | train_acc=0.9983 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9972 | time=1.9s
    
    [K99Sub] 4 clients
      client=00 | n= 1006 | train_acc=0.9712 | val_acc=0.9696 | val_f1=0.9695 | val_auc=0.9963 | time=1.2s
      client=01 | n= 1816 | train_acc=0.9818 | val_acc=0.9730 | val_f1=0.9729 | val_auc=0.9955 | time=1.3s
      client=02 | n= 8402 | train_acc=0.9844 | val_acc=0.9807 | val_f1=0.9806 | val_auc=0.9978 | time=1.9s
      client=03 | n= 6410 | train_acc=0.9832 | val_acc=0.9727 | val_f1=0.9727 | val_auc=0.9975 | time=1.8s
    
    [NTD1] 10 clients
      client=00 | n=37754 | train_acc=0.9604 | val_acc=0.9735 | val_f1=0.9066 | val_auc=0.9903 | time=3.8s
      client=01 | n= 2209 | train_acc=0.9977 | val_acc=0.8408 | val_f1=0.7104 | val_auc=0.9602 | time=1.3s
      client=02 | n=48001 | train_acc=0.9896 | val_acc=0.9150 | val_f1=0.4778 | val_auc=0.9911 | time=4.7s
      client=03 | n=12872 | train_acc=0.9601 | val_acc=0.9569 | val_f1=0.8847 | val_auc=0.9895 | time=2.0s
      client=04 | n=62448 | train_acc=0.9939 | val_acc=0.9150 | val_f1=0.4778 | val_auc=0.9922 | time=5.6s
      client=05 | n=27705 | train_acc=0.9788 | val_acc=0.9709 | val_f1=0.8960 | val_auc=0.9888 | time=3.0s
      client=06 | n= 1357 | train_acc=0.9624 | val_acc=0.9451 | val_f1=0.8605 | val_auc=0.9873 | time=1.2s
      client=07 | n= 5166 | train_acc=0.9681 | val_acc=0.9400 | val_f1=0.8512 | val_auc=0.9879 | time=1.5s
      client=08 | n= 5839 | train_acc=0.9568 | val_acc=0.9444 | val_f1=0.8584 | val_auc=0.9860 | time=1.5s
      client=09 | n=54260 | train_acc=0.9811 | val_acc=0.9733 | val_f1=0.9060 | val_auc=0.9896 | time=5.1s
    
    [NTD2] 12 clients
      client=00 | n= 2230 | train_acc=0.9561 | val_acc=0.8204 | val_f1=0.8181 | val_auc=0.9710 | time=1.6s
      client=01 | n= 3379 | train_acc=0.9453 | val_acc=0.8053 | val_f1=0.8022 | val_auc=0.9752 | time=2.1s
      client=02 | n= 1789 | train_acc=0.9346 | val_acc=0.8898 | val_f1=0.8871 | val_auc=0.9652 | time=1.5s
      client=03 | n=  142 | train_acc=0.9577 | val_acc=0.8290 | val_f1=0.8272 | val_auc=0.9776 | time=1.1s
      client=04 | n= 2426 | train_acc=0.8702 | val_acc=0.8854 | val_f1=0.8829 | val_auc=0.9763 | time=1.8s
      client=05 | n=    1 | train_acc=1.0000 | val_acc=0.6476 | val_f1=0.5779 | val_auc=0.9667 | time=1.1s
      client=06 | n= 4736 | train_acc=0.9134 | val_acc=0.8893 | val_f1=0.8866 | val_auc=0.9837 | time=2.5s
      client=07 | n= 3988 | train_acc=0.9812 | val_acc=0.4707 | val_f1=0.3201 | val_auc=0.9753 | time=2.3s
      client=08 | n= 4083 | train_acc=0.9015 | val_acc=0.8879 | val_f1=0.8852 | val_auc=0.9777 | time=2.3s
      client=09 | n= 4239 | train_acc=0.9509 | val_acc=0.8888 | val_f1=0.8862 | val_auc=0.9835 | time=2.4s
      client=10 | n= 8696 | train_acc=0.9872 | val_acc=0.5293 | val_f1=0.3461 | val_auc=0.9656 | time=3.7s
      client=11 | n= 3369 | train_acc=0.9697 | val_acc=0.8291 | val_f1=0.8274 | val_auc=0.9756 | time=2.1s
    
    [WII21] 46 clients
      client=00 | n= 8354 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.7s
      client=01 | n=25751 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=6.1s
      client=02 | n=42801 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.6s
      client=03 | n=15271 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.1s
      client=04 | n=17243 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.5s
      client=05 | n=44654 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.9s
      client=06 | n=47255 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=10.4s
      client=07 | n=  457 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=08 | n=12388 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.5s
      client=09 | n=12876 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.6s
      client=10 | n= 6523 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=11 | n= 6709 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=12 | n= 3239 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=13 | n=  259 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=14 | n= 8000 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.6s
      client=15 | n=44321 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.9s
      client=16 | n=87021 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=18.3s
      client=17 | n= 6986 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=18 | n= 9329 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.9s
      client=19 | n=37143 | train_acc=0.9999 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.4s
      client=20 | n=34755 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=7.9s
      client=21 | n= 7268 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.5s
      client=22 | n=15886 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.2s
      client=23 | n=18446 | train_acc=0.9999 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.7s
      client=24 | n=21907 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.4s
      client=25 | n=15749 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.1s
      client=26 | n=41900 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.3s
      client=27 | n=22675 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.5s
      client=28 | n=22659 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=5.5s
      client=29 | n= 6231 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=30 | n=11135 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.3s
      client=31 | n=10805 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.2s
      client=32 | n=11918 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.4s
      client=33 | n=12890 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.6s
      client=34 | n= 5447 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.1s
      client=35 | n=  743 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=36 | n=52117 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=11.4s
      client=37 | n= 3378 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.8s
      client=38 | n= 7628 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.6s
      client=39 | n=10187 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.1s
      client=40 | n= 2932 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=41 | n=36349 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.3s
      client=42 | n= 5003 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.0s
      client=43 | n=14026 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.8s
      client=44 | n= 4514 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.9s
      client=45 | n= 2996 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      [VAL] I23Sub | acc=0.9968 | f1=0.9894 | auc=0.9974 | logloss=0.0189
      [VAL] K99Sub | acc=0.9420 | f1=0.9420 | auc=0.9940 | logloss=0.1396
      [VAL] NTD1 | acc=0.9715 | f1=0.8985 | auc=0.9885 | logloss=0.0877
      [VAL] NTD2 | acc=0.8830 | f1=0.8799 | auc=0.9777 | logloss=0.4026
      [VAL] WII21 | acc=1.0000 | f1=1.0000 | auc=1.0000 | logloss=0.0000
    ----------------------------------------------------------------------------------------------------
    GLOBAL VAL | acc=0.9889 | f1=0.9725 | logloss=0.0352 | round_time=955.9s
    
    ====================================================================================================
    FEDERATED ROUND 15/15
    ====================================================================================================
    
    [I23Sub] 2 clients
      client=00 | n= 9854 | train_acc=0.9971 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9973 | time=2.2s
      client=01 | n= 6347 | train_acc=0.9983 | val_acc=0.9983 | val_f1=0.9943 | val_auc=0.9974 | time=1.8s
    
    [K99Sub] 4 clients
      client=00 | n= 1006 | train_acc=0.9662 | val_acc=0.9296 | val_f1=0.9296 | val_auc=0.9962 | time=1.2s
      client=01 | n= 1816 | train_acc=0.9719 | val_acc=0.9365 | val_f1=0.9357 | val_auc=0.9940 | time=1.3s
      client=02 | n= 8402 | train_acc=0.9818 | val_acc=0.9796 | val_f1=0.9795 | val_auc=0.9981 | time=2.0s
      client=03 | n= 6410 | train_acc=0.9853 | val_acc=0.9762 | val_f1=0.9761 | val_auc=0.9977 | time=1.7s
    
    [NTD1] 10 clients
      client=00 | n=37754 | train_acc=0.9560 | val_acc=0.9665 | val_f1=0.8923 | val_auc=0.9903 | time=3.7s
      client=01 | n= 2209 | train_acc=0.9986 | val_acc=0.8374 | val_f1=0.7067 | val_auc=0.8867 | time=1.3s
      client=02 | n=48001 | train_acc=0.9906 | val_acc=0.9683 | val_f1=0.8850 | val_auc=0.9877 | time=4.7s
      client=03 | n=12872 | train_acc=0.9545 | val_acc=0.9509 | val_f1=0.8725 | val_auc=0.9908 | time=2.0s
      client=04 | n=62448 | train_acc=0.9939 | val_acc=0.9150 | val_f1=0.4778 | val_auc=0.9897 | time=5.3s
      client=05 | n=27705 | train_acc=0.9743 | val_acc=0.9682 | val_f1=0.8932 | val_auc=0.9888 | time=3.0s
      client=06 | n= 1357 | train_acc=0.9609 | val_acc=0.9503 | val_f1=0.8708 | val_auc=0.9874 | time=1.3s
      client=07 | n= 5166 | train_acc=0.9692 | val_acc=0.9292 | val_f1=0.8321 | val_auc=0.9882 | time=1.5s
      client=08 | n= 5839 | train_acc=0.9604 | val_acc=0.9391 | val_f1=0.8495 | val_auc=0.9904 | time=1.5s
      client=09 | n=54260 | train_acc=0.9816 | val_acc=0.9738 | val_f1=0.9082 | val_auc=0.9925 | time=4.8s
    
    [NTD2] 12 clients
      client=00 | n= 2230 | train_acc=0.9565 | val_acc=0.8205 | val_f1=0.8182 | val_auc=0.9730 | time=1.7s
      client=01 | n= 3379 | train_acc=0.9482 | val_acc=0.8325 | val_f1=0.8310 | val_auc=0.9776 | time=2.1s
      client=02 | n= 1789 | train_acc=0.9335 | val_acc=0.8894 | val_f1=0.8868 | val_auc=0.9666 | time=1.5s
      client=03 | n=  142 | train_acc=0.9577 | val_acc=0.8290 | val_f1=0.8272 | val_auc=0.9715 | time=1.1s
      client=04 | n= 2426 | train_acc=0.8920 | val_acc=0.8883 | val_f1=0.8857 | val_auc=0.9784 | time=1.8s
      client=05 | n=    1 | train_acc=1.0000 | val_acc=0.7485 | val_f1=0.7228 | val_auc=0.9665 | time=1.1s
      client=06 | n= 4736 | train_acc=0.9122 | val_acc=0.8883 | val_f1=0.8857 | val_auc=0.9837 | time=2.5s
      client=07 | n= 3988 | train_acc=0.9892 | val_acc=0.7904 | val_f1=0.7858 | val_auc=0.9672 | time=2.3s
      client=08 | n= 4083 | train_acc=0.9084 | val_acc=0.8879 | val_f1=0.8852 | val_auc=0.9822 | time=2.3s
      client=09 | n= 4239 | train_acc=0.9526 | val_acc=0.8898 | val_f1=0.8871 | val_auc=0.9833 | time=2.3s
      client=10 | n= 8696 | train_acc=0.9872 | val_acc=0.5293 | val_f1=0.3461 | val_auc=0.9689 | time=3.6s
      client=11 | n= 3369 | train_acc=0.9668 | val_acc=0.7963 | val_f1=0.7923 | val_auc=0.9755 | time=2.1s
    
    [WII21] 46 clients
      client=00 | n= 8354 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.7s
      client=01 | n=25751 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=6.2s
      client=02 | n=42801 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.6s
      client=03 | n=15271 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.1s
      client=04 | n=17243 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.5s
      client=05 | n=44654 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9999 | time=9.9s
      client=06 | n=47255 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=10.4s
      client=07 | n=  457 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=08 | n=12388 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.5s
      client=09 | n=12876 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.6s
      client=10 | n= 6523 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=11 | n= 6709 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=12 | n= 3239 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=13 | n=  259 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.1s
      client=14 | n= 8000 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.6s
      client=15 | n=44321 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=9.9s
      client=16 | n=87021 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=18.3s
      client=17 | n= 6986 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.4s
      client=18 | n= 9329 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.9s
      client=19 | n=37143 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.4s
      client=20 | n=34755 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=7.9s
      client=21 | n= 7268 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.5s
      client=22 | n=15886 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.2s
      client=23 | n=18446 | train_acc=0.9957 | val_acc=0.9943 | val_f1=0.9781 | val_auc=0.9989 | time=4.7s
      client=24 | n=21907 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=5.4s
      client=25 | n=15749 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=4.2s
      client=26 | n=41900 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=0.9998 | time=9.3s
      client=27 | n=22675 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.5s
      client=28 | n=22659 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=5.5s
      client=29 | n= 6231 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.3s
      client=30 | n=11135 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.3s
      client=31 | n=10805 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.2s
      client=32 | n=11918 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.4s
      client=33 | n=12890 | train_acc=1.0000 | val_acc=1.0000 | val_f1=0.9999 | val_auc=1.0000 | time=3.6s
      client=34 | n= 5447 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.1s
      client=35 | n=  743 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.2s
      client=36 | n=52117 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=11.4s
      client=37 | n= 3378 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.7s
      client=38 | n= 7628 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.6s
      client=39 | n=10187 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.0s
      client=40 | n= 2932 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      client=41 | n=36349 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=8.2s
      client=42 | n= 5003 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=2.1s
      client=43 | n=14026 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=3.8s
      client=44 | n= 4514 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.9s
      client=45 | n= 2996 | train_acc=1.0000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | time=1.6s
      [VAL] I23Sub | acc=0.9968 | f1=0.9894 | auc=0.9973 | logloss=0.0170
      [VAL] K99Sub | acc=0.9643 | f1=0.9642 | auc=0.9972 | logloss=0.1020
      [VAL] NTD1 | acc=0.9728 | f1=0.9040 | auc=0.9885 | logloss=0.0892
      [VAL] NTD2 | acc=0.8832 | f1=0.8802 | auc=0.9786 | logloss=0.4425
      [VAL] WII21 | acc=1.0000 | f1=1.0000 | auc=1.0000 | logloss=0.0000
    ----------------------------------------------------------------------------------------------------
    GLOBAL VAL | acc=0.9895 | f1=0.9741 | logloss=0.0363 | round_time=952.5s
    
    ====================================================================================================
    BEST ROUND: 15 | BEST GLOBAL VAL ACC: 0.9895
    ====================================================================================================



<div class="grip-table-block"><div class="grip-table-title">FINAL REPORT — CORE METRICS</div><table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>split</th>
      <th>dataset</th>
      <th>acc</th>
      <th>prec_macro</th>
      <th>rec_macro</th>
      <th>f1_macro</th>
      <th>logloss</th>
      <th>mcc</th>
      <th>kappa</th>
      <th>auc_roc_macro</th>
      <th>pr_auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VAL</td>
      <td>I23Sub</td>
      <td>0.9968</td>
      <td>0.9983</td>
      <td>0.9809</td>
      <td>0.9894</td>
      <td>0.0170</td>
      <td>0.9790</td>
      <td>0.9788</td>
      <td>0.9973</td>
      <td>0.9847</td>
    </tr>
    <tr>
      <td>TEST</td>
      <td>I23Sub</td>
      <td>0.9957</td>
      <td>0.9977</td>
      <td>0.9740</td>
      <td>0.9855</td>
      <td>0.0219</td>
      <td>0.9714</td>
      <td>0.9710</td>
      <td>0.9957</td>
      <td>0.9750</td>
    </tr>
    <tr>
      <td>VAL</td>
      <td>K99Sub</td>
      <td>0.9643</td>
      <td>0.9641</td>
      <td>0.9662</td>
      <td>0.9642</td>
      <td>0.1020</td>
      <td>0.9302</td>
      <td>0.9285</td>
      <td>0.9972</td>
      <td>0.9976</td>
    </tr>
    <tr>
      <td>TEST</td>
      <td>K99Sub</td>
      <td>0.9643</td>
      <td>0.9639</td>
      <td>0.9660</td>
      <td>0.9642</td>
      <td>0.1010</td>
      <td>0.9299</td>
      <td>0.9285</td>
      <td>0.9971</td>
      <td>0.9975</td>
    </tr>
    <tr>
      <td>VAL</td>
      <td>NTD1</td>
      <td>0.9728</td>
      <td>0.9498</td>
      <td>0.8682</td>
      <td>0.9040</td>
      <td>0.0892</td>
      <td>0.8139</td>
      <td>0.8083</td>
      <td>0.9885</td>
      <td>0.8732</td>
    </tr>
    <tr>
      <td>TEST</td>
      <td>NTD1</td>
      <td>0.9705</td>
      <td>0.9474</td>
      <td>0.8542</td>
      <td>0.8942</td>
      <td>0.0930</td>
      <td>0.7961</td>
      <td>0.7887</td>
      <td>0.9877</td>
      <td>0.8653</td>
    </tr>
    <tr>
      <td>VAL</td>
      <td>NTD2</td>
      <td>0.8832</td>
      <td>0.9030</td>
      <td>0.8769</td>
      <td>0.8802</td>
      <td>0.4425</td>
      <td>0.7795</td>
      <td>0.7628</td>
      <td>0.9786</td>
      <td>0.9737</td>
    </tr>
    <tr>
      <td>TEST</td>
      <td>NTD2</td>
      <td>0.8808</td>
      <td>0.9023</td>
      <td>0.8742</td>
      <td>0.8775</td>
      <td>0.4514</td>
      <td>0.7760</td>
      <td>0.7578</td>
      <td>0.9767</td>
      <td>0.9714</td>
    </tr>
    <tr>
      <td>VAL</td>
      <td>WII21</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <td>TEST</td>
      <td>WII21</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <td>VAL</td>
      <td>global_weighted</td>
      <td>0.9895</td>
      <td>0.9851</td>
      <td>0.9660</td>
      <td>0.9741</td>
      <td>0.0363</td>
      <td>0.9502</td>
      <td>0.9483</td>
      <td>0.9967</td>
      <td>0.9709</td>
    </tr>
    <tr>
      <td>TEST</td>
      <td>global_weighted</td>
      <td>0.9889</td>
      <td>0.9845</td>
      <td>0.9627</td>
      <td>0.9718</td>
      <td>0.0375</td>
      <td>0.9460</td>
      <td>0.9437</td>
      <td>0.9964</td>
      <td>0.9689</td>
    </tr>
  </tbody>
</table></div><div class="grip-note">Full per-split metric export with every recorded column is saved to artifacts/final_report.csv.</div>



<div class="grip-table-block"><div class="grip-table-title">FINAL REPORT — EXTENDED METRICS</div><table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>split</th>
      <th>dataset</th>
      <th>prec_weighted</th>
      <th>rec_weighted</th>
      <th>f1_weighted</th>
      <th>auc_roc_micro</th>
      <th>auc_roc_weighted</th>
      <th>ppv_macro</th>
      <th>npv_macro</th>
      <th>ppv_weighted</th>
      <th>npv_weighted</th>
      <th>ppv_positive</th>
      <th>npv_negative</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VAL</td>
      <td>I23Sub</td>
      <td>0.9968</td>
      <td>0.9968</td>
      <td>0.9968</td>
      <td>0.9973</td>
      <td>0.9973</td>
      <td>0.9983</td>
      <td>0.9983</td>
      <td>0.9968</td>
      <td>0.9997</td>
      <td>1.0000</td>
      <td>0.9966</td>
    </tr>
    <tr>
      <td>TEST</td>
      <td>I23Sub</td>
      <td>0.9957</td>
      <td>0.9957</td>
      <td>0.9956</td>
      <td>0.9957</td>
      <td>0.9957</td>
      <td>0.9977</td>
      <td>0.9977</td>
      <td>0.9957</td>
      <td>0.9996</td>
      <td>1.0000</td>
      <td>0.9953</td>
    </tr>
    <tr>
      <td>VAL</td>
      <td>K99Sub</td>
      <td>0.9661</td>
      <td>0.9643</td>
      <td>0.9643</td>
      <td>0.9972</td>
      <td>0.9972</td>
      <td>0.9641</td>
      <td>0.9641</td>
      <td>0.9661</td>
      <td>0.9620</td>
      <td>0.9947</td>
      <td>0.9334</td>
    </tr>
    <tr>
      <td>TEST</td>
      <td>K99Sub</td>
      <td>0.9658</td>
      <td>0.9643</td>
      <td>0.9643</td>
      <td>0.9971</td>
      <td>0.9971</td>
      <td>0.9639</td>
      <td>0.9639</td>
      <td>0.9658</td>
      <td>0.9620</td>
      <td>0.9916</td>
      <td>0.9362</td>
    </tr>
    <tr>
      <td>VAL</td>
      <td>NTD1</td>
      <td>0.9719</td>
      <td>0.9728</td>
      <td>0.9715</td>
      <td>0.9885</td>
      <td>0.9885</td>
      <td>0.9498</td>
      <td>0.9498</td>
      <td>0.9719</td>
      <td>0.9277</td>
      <td>0.9231</td>
      <td>0.9765</td>
    </tr>
    <tr>
      <td>TEST</td>
      <td>NTD1</td>
      <td>0.9695</td>
      <td>0.9705</td>
      <td>0.9688</td>
      <td>0.9877</td>
      <td>0.9877</td>
      <td>0.9474</td>
      <td>0.9474</td>
      <td>0.9695</td>
      <td>0.9254</td>
      <td>0.9209</td>
      <td>0.9740</td>
    </tr>
    <tr>
      <td>VAL</td>
      <td>NTD2</td>
      <td>0.8986</td>
      <td>0.8832</td>
      <td>0.8813</td>
      <td>0.9786</td>
      <td>0.9786</td>
      <td>0.9030</td>
      <td>0.9030</td>
      <td>0.8986</td>
      <td>0.9075</td>
      <td>0.9790</td>
      <td>0.8271</td>
    </tr>
    <tr>
      <td>TEST</td>
      <td>NTD2</td>
      <td>0.8976</td>
      <td>0.8808</td>
      <td>0.8787</td>
      <td>0.9767</td>
      <td>0.9767</td>
      <td>0.9023</td>
      <td>0.9023</td>
      <td>0.8976</td>
      <td>0.9069</td>
      <td>0.9817</td>
      <td>0.8229</td>
    </tr>
    <tr>
      <td>VAL</td>
      <td>WII21</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <td>TEST</td>
      <td>WII21</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <td>VAL</td>
      <td>global_weighted</td>
      <td>0.9899</td>
      <td>0.9895</td>
      <td>0.9891</td>
      <td>0.9967</td>
      <td>0.9967</td>
      <td>0.9851</td>
      <td>0.9851</td>
      <td>0.9899</td>
      <td>0.9803</td>
      <td>0.9822</td>
      <td>0.9880</td>
    </tr>
    <tr>
      <td>TEST</td>
      <td>global_weighted</td>
      <td>0.9892</td>
      <td>0.9889</td>
      <td>0.9884</td>
      <td>0.9964</td>
      <td>0.9964</td>
      <td>0.9845</td>
      <td>0.9845</td>
      <td>0.9892</td>
      <td>0.9798</td>
      <td>0.9818</td>
      <td>0.9873</td>
    </tr>
  </tbody>
</table></div>



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_11.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_12.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_13.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_14.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_15.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_16.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_17.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_18.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_19.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_20.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_21.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_22.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_23.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_24.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_25.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_26.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_27.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_28.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_29.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_30.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_31.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_32.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_33.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_34.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_35.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_36.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_37.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_38.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_39.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_40.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_41.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_42.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_43.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_44.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_45.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_46.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_47.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_48.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_49.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_50.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_51.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_52.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_53.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_54.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_55.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_56.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_57.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_58.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_59.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_60.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_61.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_62.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_63.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_64.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_65.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_66.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_67.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_68.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_69.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_70.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_71.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_72.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_73.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_74.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_75.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_76.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_77.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_78.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_79.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_80.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_81.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_82.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_83.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_84.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_85.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_86.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_87.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_88.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_89.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_90.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_91.png)
    



    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_0_92.png)
    


    
    Saved artifacts:
      - artifacts/final_report.csv
      - artifacts/round_history.json
      - artifacts/dataset_meta.json
      - artifacts/dataset_task_summary.csv
      - artifacts/roc_data.json
      - artifacts/pr_data.json
      - artifacts/calibration_data.json
      - artifacts/error_analysis_val_samples.csv
      - artifacts/error_analysis_test_samples.csv
      - artifacts/error_analysis_summary_test.csv
      - artifacts/error_analysis_by_class_test.csv
      - artifacts/error_analysis_top_confusions_test.csv
      - artifacts/checkpoints/shared_backbone_best.pth
      - artifacts/checkpoints/private_<dataset>_best.pth
      - artifacts/plots/*.png
      - added global multi-metric / error / calibration profile plots
    
    Done.



```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix counts
cm = np.array([
    [174641, 2418],
    [360, 72579]
])

labels = ["Normal", "Attack"]

# Row-normalized matrix
cm_norm = cm / cm.sum(axis=1, keepdims=True)

# Plot
fig, axes = plt.subplots(
    1, 2,
    figsize=(16, 6),
    gridspec_kw={"wspace": 0.45}
)

title_color = "#102A43"

# Counts heatmap
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    ax=axes[0],
    cbar=True,
    linewidths=0
)

axes[0].set_title(
    "GRIP-DFFI - Global Test Binary Confusion Matrix\nCounts",
    fontsize=18,
    fontweight="bold",
    color=title_color,
    pad=18
)
axes[0].set_xlabel("Predicted", fontsize=16, color=title_color)
axes[0].set_ylabel("True", fontsize=16, color=title_color)

# Row-normalized heatmap
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Greens",
    xticklabels=labels,
    yticklabels=labels,
    ax=axes[1],
    cbar=True,
    linewidths=0,
    vmin=0,
    vmax=1
)

axes[1].set_title(
    "GRIP-DFFI - Global Test Binary Confusion Matrix\nRow-normalized",
    fontsize=18,
    fontweight="bold",
    color=title_color,
    pad=18
)
axes[1].set_xlabel("Predicted", fontsize=16, color=title_color)
axes[1].set_ylabel("True", fontsize=16, color=title_color)

# Tick styling
for ax in axes:
    ax.tick_params(axis="x", labelrotation=45, labelsize=14, colors=title_color)
    ax.tick_params(axis="y", labelrotation=0, labelsize=14, colors=title_color)

# Extra spacing to prevent any title overlap
plt.subplots_adjust(top=0.82, wspace=0.45)

plt.show()
```


    
![png](1_grip_dffi_main_OUTPUT_VIEW_files/1_grip_dffi_main_OUTPUT_VIEW_1_0.png)
    

