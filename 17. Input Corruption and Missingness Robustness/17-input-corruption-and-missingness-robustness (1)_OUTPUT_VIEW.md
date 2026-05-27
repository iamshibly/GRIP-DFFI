```python
def qprint(*args, **kwargs):
    pass

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
import io
import contextlib
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
qprint(f"DEVICE: {DEVICE} | PIN_MEMORY: {PIN_MEM}")

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
        qprint('\n' + '=' * 100)
        qprint(title)
        qprint('-' * 100)
        try:
            qprint(view.to_markdown(index=index))
        except Exception:
            qprint(view.to_string(index=index))
        if note:
            qprint(note)
        qprint('=' * 100)


def show_key_value_table(title, items, note=None):
    df = pd.DataFrame({
        'Field': list(items.keys()),
        'Value': [wrap_sequence_for_display(v) if isinstance(v, (list, tuple, np.ndarray)) else v for v in items.values()],
    })
    show_table(title, df, index=False, float_decimals=4, note=note)


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
        qprint(f"[HF cached] {hf_id} -> {out_dir}")
        return out_dir

    ds_obj = load_dataset(hf_id)
    if hasattr(ds_obj, "keys"):
        split_names = list(ds_obj.keys())
        for split in split_names:
            df = ds_obj[split].to_pandas()
            out_path = os.path.join(out_dir, f"{split}.parquet")
            df.to_parquet(out_path, index=False)
            qprint(f"[HF materialized] {hf_id}::{split} -> {out_path} | shape={df.shape}")
    else:
        df = ds_obj.to_pandas()
        out_path = os.path.join(out_dir, "train.parquet")
        df.to_parquet(out_path, index=False)
        qprint(f"[HF materialized] {hf_id} -> {out_path} | shape={df.shape}")
    return out_dir


_download_cache = {}
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    for spec in DATASET_SPECS:
        source_type = spec.get("source_type", "kaggle")
        if source_type == "huggingface":
            hf_id = spec["hf_id"]
            cache_key = ("hf", hf_id)
            if cache_key not in _download_cache:
                _download_cache[cache_key] = materialize_hf_dataset(hf_id, spec["name"])
            spec["path"] = _download_cache[cache_key]
            qprint(f"[{spec['name']}] HF dataset path -> {spec['path']}")
        else:
            slug = spec["slug"]
            cache_key = ("kaggle", slug)
            if cache_key not in _download_cache:
                _download_cache[cache_key] = kagglehub.dataset_download(slug)
                qprint(f"[{slug}] -> {_download_cache[cache_key]}")
            spec["path"] = _download_cache[cache_key]

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
                qprint(f"    loaded {os.path.basename(fp)}: {d.shape}")
        except Exception as e:
            qprint(f"    [WARN] failed {fp}: {e}")
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
        qprint(f"  [INFO] dropped verified target-sibling columns: {safe_drop}")
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
        qprint(f"  [INFO] verified target-sibling columns (dropped): {sibling_report}")
    else:
        qprint("  [INFO] no deterministic target-sibling feature columns detected")
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
        qprint(f"  [WARN] dropping rare classes before split: {dropped.tolist()}")
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
        qprint(f"  [INFO] dropped ID/high-uniqueness columns: {drop[:12]}{'...' if len(drop) > 12 else ''}")
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


def prepare_dataset_for_feature_intelligence(spec, mi_pre=160):
    files_all = list_table_files(spec["path"])
    if not files_all:
        raise FileNotFoundError(f"{spec['name']}: no files found under {spec['path']}")

    selected_files = choose_files_for_spec(spec, files_all)
    groups = categorize_split_files(selected_files)
    has_named_splits = len(groups["train"]) > 0 or len(groups["test"]) > 0 or len(groups["val"]) > 0
    force_cols = UNSW_COLS if spec.get("unsw_nb15_raw", False) else None

    qprint("\n" + "=" * 100)
    qprint(f"[{spec['name'].upper()}] DATASET BUILD")
    qprint("-" * 100)
    qprint(f"  path              : {spec['path']}")
    qprint(f"  files_found       : {len(files_all)}")
    qprint(f"  files_selected    : {[os.path.basename(f) for f in selected_files]}")
    qprint(f"  split_hint        : {'named train/val/test files detected' if has_named_splits else 'no named split files'}")
    qprint(f"  split_policy      : {SPLIT_POLICY}")
    if not PRESERVE_OFFICIAL_BENCHMARK_SPLITS:
        qprint("  benchmark_note    : provided split files are treated as labeled sources, then re-split fresh")

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
        qprint(f"  target_col        : {target_col}")
        qprint(f"  task_kind         : {task_kind}")
        merge_parts = []
        if target_col in df_train.columns:
            merge_parts.append(("train", df_train))
        if df_test is not None and target_col in df_test.columns:
            if target_compatible(df_train[target_col], df_test[target_col], spec):
                qprint("  [INFO] test target is compatible with train target -> merge train+test")
                merge_parts.append(("test", df_test))
            else:
                qprint("  [INFO] test target incompatible -> ignore test for fresh split")
        if df_val is not None and target_col in df_val.columns:
            if target_compatible(df_train[target_col], df_val[target_col], spec):
                qprint("  [INFO] val target is compatible with train target -> merge val too")
                merge_parts.append(("val", df_val))
            else:
                qprint("  [INFO] val target incompatible -> ignore val for fresh split")
        if merge_parts:
            source_parts_used = [n for n, _ in merge_parts]
            merged_source = pd.concat([d for _, d in merge_parts], axis=0, ignore_index=True)
            qprint(f"  split_mode        : fresh-70/15/15 from merged labeled pieces {source_parts_used}")
        elif df_other is not None:
            merged_source = df_other
            source_parts_used = ["other"]
            target_col, task_kind = detect_target_column(merged_source, spec)
            qprint(f"  target_col        : {target_col}")
            qprint(f"  task_kind         : {task_kind}")
            qprint("  split_mode        : fresh-70/15/15 from unlabeled-split fallback table set")
        else:
            raise RuntimeError(f"{spec['name']}: no usable labeled table found")
    else:
        merged_source = combine_tables(selected_files, force_cols=force_cols)
        if merged_source is None:
            raise RuntimeError(f"{spec['name']}: failed to load selected files")
        source_parts_used = ["combined_tables"]
        target_col, task_kind = detect_target_column(merged_source, spec)
        qprint(f"  target_col        : {target_col}")
        qprint(f"  task_kind         : {task_kind}")
        qprint("  split_mode        : fresh-70/15/15 from combined tables")

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

    qprint(f"  train/val/test    : {len(Xtr)} / {len(Xva)} / {len(Xte)}")
    qprint(f"  n_classes         : {n_classes}")
    qprint(f"  class_names       : {class_names.tolist()}")
    qprint(f"  normal_class_idx  : {normal_index}")

    num_cols, cat_cols = infer_types(Xtr)
    means, stds = fit_num_stats(Xtr, num_cols)
    Xtr, ni_tr, ci_tr = apply_num_cat_preproc(Xtr, num_cols, cat_cols, means, stds)
    Xva, ni_va, ci_va = apply_num_cat_preproc(Xva, num_cols, cat_cols, means, stds)
    Xte, ni_te, ci_te = apply_num_cat_preproc(Xte, num_cols, cat_cols, means, stds)
    qprint(f"  initial_features  : {len(keep_cols)} | numeric={len(num_cols)} | categorical={len(cat_cols)}")
    qprint(f"  impute_counts     : train(n={ni_tr}, c={ci_tr}) | val(n={ni_va}, c={ci_va}) | test(n={ni_te}, c={ci_te})")

    plan = cross_plan(Xtr, ytr, cat_cols)
    Xtr, new_crosses = apply_crosses(Xtr, plan)
    Xva, _ = apply_crosses(Xva, plan)
    Xte, _ = apply_crosses(Xte, plan)
    qprint(f"  cross_features    : use_triples={plan['use_triples']} | base={plan['base_cols']} | new={len(new_crosses)}")

    num2, cat2 = infer_types(Xtr)
    feat_names, mi_scores = compute_mi(Xtr, ytr, num2, cat2)
    mi_pre_dyn = max(mi_pre, int(0.80 * len(feat_names))) if feat_names else 0
    if len(feat_names) > mi_pre_dyn > 0:
        ord_ = np.argsort(mi_scores)[::-1][:mi_pre_dyn]
        feat_names = [feat_names[i] for i in ord_]
        mi_scores = mi_scores[ord_]
    qprint(f"  mi_candidates     : {len(feat_names)}")

    fn_sel = [f for f in feat_names if f in Xtr.columns]
    if len(fn_sel) == 0:
        fn_sel = list(Xtr.columns)
        mi_scores = np.ones(len(fn_sel), np.float32)
        qprint(f"  [WARN] MI overlap empty; using all {len(fn_sel)} columns")

    nc_sel = [c for c in num2 if c in fn_sel]
    cc_sel = [c for c in cat2 if c in fn_sel]
    Xnode, A = build_feature_graph(Xtr[fn_sel], ytr, fn_sel, nc_sel, cc_sel)

    feat_keys = [feature_key(f, "num" if f in nc_sel else "cat") for f in fn_sel]
    feat_type_map = {k: {"name": feature_name_from_key(k), "kind": feature_kind_from_key(k)} for k in feat_keys}
    feat_key_to_mi = {feat_keys[i]: float(mi_scores[i]) for i in range(len(feat_keys))}

    qprint(f"  graph_nodes       : {len(fn_sel)}")
    qprint(f"  semantic_protocol : target-resolution + compatibility-gated merge + leakage-safe re-split")
    qprint("=" * 100)

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


def run_federated_feature_intelligence(stage_records):
    qprint("\n" + "#" * 100)
    qprint(f"{PROCESS_NAME} FEDERATED FEATURE INTELLIGENCE")
    qprint("#" * 100)
    global_state = None
    weights = [st["n_train"] for st in stage_records]
    weights = [w / float(sum(weights)) for w in weights]
    for rnd in range(1, FEATURE_INTEL_ROUNDS + 1):
        qprint("\n" + "=" * 100)
        qprint(f"FEATURE INTELLIGENCE ROUND {rnd}/{FEATURE_INTEL_ROUNDS}")
        qprint("=" * 100)
        local_states = []
        for st in stage_records:
            lstate, lscores = train_graph_refined_relevance_net(st["Xnode"], st["A"], init_state=global_state)
            local_states.append(lstate)
            if len(lscores):
                qprint(f"  [{st['spec']['name']}] mean_local_relevance={float(np.mean(lscores)):.4f} | nodes={len(lscores)}")
            else:
                qprint(f"  [{st['spec']['name']}] mean_local_relevance=NA | nodes=0")
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
        qprint(f"  [{st['spec']['name']}] selected_features={len(selected)} / {max(len(feat_names), 1)}")
        qprint(f"  [{st['spec']['name']}] selected_top10={selected[:10]}{'...' if len(selected) > 10 else ''}")

    universality, occurrence = compute_feature_universality(stage_records, min_occ=MIN_SHARED_OCCURRENCE)
    shared_key_set = {k for k, u in universality.items() if u >= UNIVERSALITY_THRESHOLD and occurrence.get(k, 0) >= MIN_SHARED_OCCURRENCE}
    shared_keys_global = sorted(shared_key_set)
    shared_key_to_id = {k: i for i, k in enumerate(shared_keys_global)}

    qprint("\n" + "-" * 100)
    qprint(f"GLOBAL SHARED FEATURE KEYS : {len(shared_keys_global)}")
    qprint(f"UNIVERSALITY THRESHOLD     : {UNIVERSALITY_THRESHOLD:.2f}")
    qprint("-" * 100)
    return global_state, universality, occurrence, shared_keys_global, shared_key_to_id


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
import builtins

D_MODEL = 192
N_BLOCKS = 3
N_HEADS = 8
FF_DIM = 384
DROP = 0.10
ROUNDS = 15
LOCAL_EPOCHS = 1
LR = 1e-3
BATCH_SIZE = 256
CORRUPTION_ARTIFACT_DIR = "artifacts_input_corruption_missingness"
CORRUPTION_CONFIGS = [
    ("numeric_missing_10", "numeric_missing", 0.10),
    ("numeric_missing_20", "numeric_missing", 0.20),
    ("categorical_oob_10", "categorical_oob", 0.10),
    ("feature_dropout_10", "feature_dropout", 0.10),
]
os.makedirs(CORRUPTION_ARTIFACT_DIR, exist_ok=True)


def reset_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def release_memory():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def finalize_stage_record_silent(stage, universality, occurrence, shared_key_to_id):
    Xtr = stage["Xtr"].copy()
    Xva = stage["Xva"].copy()
    Xte = stage["Xte"].copy()
    ytr = stage["ytr"]
    yva = stage["yva"]
    yte = stage["yte"]
    selected_keys = stage.get("selected_feature_keys", [])
    if not selected_keys:
        selected_keys = [feature_key(c, "num" if c in stage["num_cols_post"] else "cat") for c in Xtr.columns]
    selected_keys = [k for k in selected_keys if feature_name_from_key(k) in Xtr.columns]
    selected_cols = [feature_name_from_key(k) for k in selected_keys]
    shared_keys = [
        k for k in selected_keys
        if k in shared_key_to_id
        and universality.get(k, 0.0) >= UNIVERSALITY_THRESHOLD
        and occurrence.get(k, 0) >= MIN_SHARED_OCCURRENCE
    ]
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
    meta = {
        "name": stage["spec"]["name"],
        "domain_id": stage["spec"]["domain_id"],
        "target_col": stage["target_col"],
        "task_kind": stage["task_kind"],
        "class_names": stage["class_names"],
        "normal_index": stage["normal_index"],
        "n_classes": int(stage["n_classes"]),
        "selected_features": [feature_name_from_key(k) for k in selected_keys],
        "selected_feature_keys": selected_keys,
        "num_cols": shared_num_cols + private_num_cols,
        "cat_cols": shared_cat_cols + private_cat_cols,
        "shared_num_cols": shared_num_cols,
        "shared_cat_cols": shared_cat_cols,
        "private_num_cols": private_num_cols,
        "private_cat_cols": private_cat_cols,
        "shared_num_global_ids": [shared_key_to_id[feature_key(c, "num")] for c in shared_num_cols],
        "shared_cat_global_ids": [shared_key_to_id[feature_key(c, "cat")] for c in shared_cat_cols],
        "cards": cards,
        "n_clients": int(n_clients),
        "client_sizes": [len(x) for x in client_idx],
        "split_shapes": {"train": int(len(tr_ds)), "val": int(len(va_ds)), "test": int(len(te_ds))},
    }
    return meta, tr_ds, va_ds, te_ds, client_idx


def new_model_for_corruption_validation(meta, shared_feature_count):
    return GRIPDFFIModel(meta=meta, n_shared_feature_ids=max(1, int(shared_feature_count)), d_model=D_MODEL, n_blocks=N_BLOCKS, n_heads=N_HEADS, ff=FF_DIM, drop=DROP).to(DEVICE)


def weighted_metric(metric_list, ns, key):
    vals_weights = []
    for m, n in zip(metric_list, ns):
        v = m.get(key, np.nan)
        if v == v:
            vals_weights.append((float(v), float(n)))
    if not vals_weights:
        return np.nan
    return float(sum(v * w for v, w in vals_weights) / max(sum(w for _, w in vals_weights), 1.0))


def train_federated_model_for_corruption_validation(meta_list, train_sets, val_sets, client_indices, shared_feature_count):
    seed_models = [new_model_for_corruption_validation(meta, shared_feature_count) for meta in meta_list]
    shared_global_state = get_shared_state(seed_models[0])
    private_dataset_states = [get_private_state(m) for m in seed_models]
    for m in seed_models:
        del m
    release_memory()
    best_bundle = {"global_val_acc": -1.0, "shared_state": shared_global_state, "private_states": private_dataset_states}
    val_loaders = [mk_loader(v, shuffle=False, batch_size=BATCH_SIZE) for v in val_sets]
    for _ in range(1, ROUNDS + 1):
        shared_candidates = []
        shared_sizes = []
        private_candidates = defaultdict(list)
        private_sizes = defaultdict(list)
        for di, (meta, tr_ds, cidx) in enumerate(zip(meta_list, train_sets, client_indices)):
            for idx in cidx:
                local_ds = tr_ds.subset(idx)
                model = new_model_for_corruption_validation(meta, shared_feature_count)
                load_shared_state(model, shared_global_state)
                load_private_state(model, private_dataset_states[di])
                train_client(model, mk_loader(local_ds, shuffle=True, batch_size=BATCH_SIZE), epochs=LOCAL_EPOCHS, lr=LR)
                shared_candidates.append(get_shared_state(model))
                shared_sizes.append(len(local_ds))
                private_candidates[di].append(get_private_state(model))
                private_sizes[di].append(len(local_ds))
                del model, local_ds
                release_memory()
        shared_weights = [n / float(sum(shared_sizes)) for n in shared_sizes]
        shared_global_state = average_state_dicts(shared_candidates, shared_weights)
        next_private_states = []
        for di in range(len(meta_list)):
            plist = private_candidates[di]
            psz = private_sizes[di]
            weights = [n / float(sum(psz)) for n in psz]
            next_private_states.append(average_state_dicts(plist, weights))
        private_dataset_states = next_private_states
        val_ns = []
        val_ms = []
        for di, meta in enumerate(meta_list):
            model = new_model_for_corruption_validation(meta, shared_feature_count)
            load_shared_state(model, shared_global_state)
            load_private_state(model, private_dataset_states[di])
            yv, pv = predict(model, val_loaders[di])
            val_ms.append(compute_metrics(yv, pv))
            val_ns.append(len(yv))
            del model
            release_memory()
        total_val = max(sum(val_ns), 1)
        global_val_acc = sum(m["acc"] * n for m, n in zip(val_ms, val_ns)) / total_val
        if global_val_acc > best_bundle["global_val_acc"]:
            best_bundle = {"global_val_acc": float(global_val_acc), "shared_state": {k: v.clone() for k, v in shared_global_state.items()}, "private_states": [{k: v.clone() for k, v in p.items()} for p in private_dataset_states]}
        release_memory()
    return best_bundle


def _mask_numeric_array(arr, rate, rng):
    if arr.size == 0:
        return 0, 0
    mask = rng.random(arr.shape) < rate
    arr[mask] = 0.0
    return int(mask.sum()), int(mask.size)


def _mask_shared_cat_array(arr, rate, rng):
    if arr.size == 0:
        return 0, 0
    mask = rng.random(arr.shape) < rate
    arr[mask] = max(int(SHARED_HASH_BUCKETS) - 1, 0)
    return int(mask.sum()), int(mask.size)


def _mask_private_cat_array(arr, cards, rate, rng):
    if arr.size == 0:
        return 0, 0
    total_changed = 0
    total_cells = int(arr.size)
    for j in range(arr.shape[1]):
        mask = rng.random(arr.shape[0]) < rate
        oob = int(cards[j] - 1) if j < len(cards) and cards[j] > 0 else 0
        arr[mask, j] = oob
        total_changed += int(mask.sum())
    return total_changed, total_cells


def corrupt_route_dataset(ds, meta, corruption_type, rate, seed):
    rng = np.random.default_rng(seed)
    Xsn = ds.Xsn.cpu().numpy().copy()
    Xsc = ds.Xsc.cpu().numpy().copy()
    Xpn = ds.Xpn.cpu().numpy().copy()
    Xpc = ds.Xpc.cpu().numpy().copy()
    y = ds.y.cpu().numpy().copy()
    changed = 0
    cells = 0
    if corruption_type == "numeric_missing":
        c, n = _mask_numeric_array(Xsn, rate, rng); changed += c; cells += n
        c, n = _mask_numeric_array(Xpn, rate, rng); changed += c; cells += n
    elif corruption_type == "categorical_oob":
        c, n = _mask_shared_cat_array(Xsc, rate, rng); changed += c; cells += n
        c, n = _mask_private_cat_array(Xpc, meta.get("cards", []), rate, rng); changed += c; cells += n
    elif corruption_type == "feature_dropout":
        c, n = _mask_numeric_array(Xsn, rate, rng); changed += c; cells += n
        c, n = _mask_numeric_array(Xpn, rate, rng); changed += c; cells += n
        c, n = _mask_shared_cat_array(Xsc, rate, rng); changed += c; cells += n
        c, n = _mask_private_cat_array(Xpc, meta.get("cards", []), rate, rng); changed += c; cells += n
    return RouteTabDataset(Xsn, Xsc, Xpn, Xpc, y), int(changed), int(cells), float(changed / max(cells, 1))


def evaluate_corruption_variants(meta_list, val_sets, test_sets, best_bundle, shared_feature_count):
    rows = []
    for variant_index, (variant_name, corruption_type, rate) in enumerate(CORRUPTION_CONFIGS):
        val_metrics_each = []
        test_metrics_each = []
        val_ns = []
        test_ns = []
        changed_val = []
        cells_val = []
        changed_test = []
        cells_test = []
        for di, meta in enumerate(meta_list):
            cva, cv, nv, arv = corrupt_route_dataset(val_sets[di], meta, corruption_type, rate, SEED + 10000 * (variant_index + 1) + 101 * di + 7)
            cte, ct, nt, art = corrupt_route_dataset(test_sets[di], meta, corruption_type, rate, SEED + 10000 * (variant_index + 1) + 101 * di + 17)
            model = new_model_for_corruption_validation(meta, shared_feature_count)
            load_shared_state(model, best_bundle["shared_state"])
            load_private_state(model, best_bundle["private_states"][di])
            yv, pv = predict(model, mk_loader(cva, shuffle=False, batch_size=BATCH_SIZE))
            yt, pt = predict(model, mk_loader(cte, shuffle=False, batch_size=BATCH_SIZE))
            mv = compute_metrics(yv, pv)
            mt = compute_metrics(yt, pt)
            rows.append({"variant": variant_name, "corruption_type": corruption_type, "target_rate": float(rate), "actual_rate": float(arv), "corrupted_cells": int(cv), "split": "VAL", "dataset": meta["name"], **mv})
            rows.append({"variant": variant_name, "corruption_type": corruption_type, "target_rate": float(rate), "actual_rate": float(art), "corrupted_cells": int(ct), "split": "TEST", "dataset": meta["name"], **mt})
            val_metrics_each.append(mv); test_metrics_each.append(mt); val_ns.append(len(yv)); test_ns.append(len(yt))
            changed_val.append(cv); cells_val.append(nv); changed_test.append(ct); cells_test.append(nt)
            del model, cva, cte
            release_memory()
        all_metric_keys = sorted(set().union(*[m.keys() for m in val_metrics_each], *[m.keys() for m in test_metrics_each]))
        rows.append({"variant": variant_name, "corruption_type": corruption_type, "target_rate": float(rate), "actual_rate": float(sum(changed_val) / max(sum(cells_val), 1)), "corrupted_cells": int(sum(changed_val)), "split": "VAL", "dataset": "global_weighted", **{k: weighted_metric(val_metrics_each, val_ns, k) for k in all_metric_keys}})
        rows.append({"variant": variant_name, "corruption_type": corruption_type, "target_rate": float(rate), "actual_rate": float(sum(changed_test) / max(sum(cells_test), 1)), "corrupted_cells": int(sum(changed_test)), "split": "TEST", "dataset": "global_weighted", **{k: weighted_metric(test_metrics_each, test_ns, k) for k in all_metric_keys}})
    return rows


def add_main_deltas(report):
    candidates = ["artifacts/final_report.csv", "/kaggle/working/artifacts/final_report.csv"]
    ref = None
    for path in candidates:
        if os.path.exists(path):
            ref = pd.read_csv(path)
            break
    if ref is None or "dataset" not in ref.columns or "split" not in ref.columns:
        return report
    cols = [c for c in ["acc", "f1_macro", "auc_roc_macro_ovr"] if c in ref.columns and c in report.columns]
    if not cols:
        return report
    ref_small = ref[["split", "dataset"] + cols].rename(columns={c: f"main_{c}" for c in cols})
    out = report.merge(ref_small, on=["split", "dataset"], how="left")
    for c in cols:
        out[f"delta_{c}_vs_main"] = pd.to_numeric(out[c], errors="coerce") - pd.to_numeric(out[f"main_{c}"], errors="coerce")
    return out


def render_final_table(df):
    rename_map = {"precision_macro": "prec_macro", "recall_macro": "rec_macro", "auc_roc_macro_ovr": "auc_roc_macro", "pr_auc_macro": "pr_auc", "precision_weighted": "prec_weighted", "recall_weighted": "rec_weighted", "auc_roc_micro_ovr": "auc_roc_micro", "auc_roc_weighted_ovr": "auc_roc_weighted", "ppv_macro_ovr": "ppv_macro", "npv_macro_ovr": "npv_macro", "ppv_weighted_ovr": "ppv_weighted", "npv_weighted_ovr": "npv_weighted", "delta_auc_roc_macro_ovr_vs_main": "delta_auc_vs_main", "delta_f1_macro_vs_main": "delta_f1_vs_main", "delta_acc_vs_main": "delta_acc_vs_main"}
    view = df.rename(columns=rename_map).copy()
    column_order = ["variant", "corruption_type", "target_rate", "actual_rate", "split", "dataset", "corrupted_cells", "acc", "prec_macro", "rec_macro", "f1_macro", "logloss", "mcc", "kappa", "auc_roc_macro", "pr_auc", "delta_acc_vs_main", "delta_f1_vs_main", "delta_auc_vs_main", "prec_weighted", "rec_weighted", "f1_weighted", "auc_roc_micro", "auc_roc_weighted", "ppv_macro", "npv_macro", "ppv_weighted", "npv_weighted", "ppv_positive", "npv_negative"]
    view = view[[c for c in column_order if c in view.columns]]
    for c in view.columns:
        if c not in ["variant", "corruption_type", "split", "dataset"]:
            if c == "corrupted_cells":
                view[c] = pd.to_numeric(view[c], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{int(x)}")
            else:
                view[c] = pd.to_numeric(view[c], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    view.to_csv(os.path.join(CORRUPTION_ARTIFACT_DIR, "input_corruption_missingness_final_report.csv"), index=False)
    if IN_NOTEBOOK and HTML is not None:
        table_html = view.to_html(index=False, escape=False, border=0)
        html = '<div style="margin:12px 0 18px 0;border:1px solid #dbe4ee;border-radius:12px;overflow-x:auto;background:#ffffff;">'
        html += '<div style="padding:10px 14px;font-weight:700;color:#0f172a;background:#f8fafc;border-bottom:1px solid #e2e8f0;font-size:15px;">FINAL REPORT - INPUT CORRUPTION AND MISSINGNESS ROBUSTNESS VALIDATION</div>'
        html += table_html + '</div>'
        display(HTML(html))
    else:
        builtins.print(view.to_string(index=False))


def run_input_corruption_missingness_validation():
    reset_all_seeds(SEED)
    stage_records = []
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for spec in DATASET_SPECS:
            stage_records.append(prepare_dataset_for_feature_intelligence(spec, mi_pre=120))
        _, universality, occurrence, shared_keys_global, shared_key_to_id = run_federated_feature_intelligence(stage_records)
    all_meta = []
    all_tr = []
    all_va = []
    all_te = []
    all_cidx = []
    for st in stage_records:
        meta, tr_ds, va_ds, te_ds, cidx = finalize_stage_record_silent(st, universality, occurrence, shared_key_to_id)
        all_meta.append(meta)
        all_tr.append(tr_ds)
        all_va.append(va_ds)
        all_te.append(te_ds)
        all_cidx.append(cidx)
    shared_feature_count = max(1, len(shared_keys_global))
    best_bundle = train_federated_model_for_corruption_validation(all_meta, all_tr, all_va, all_cidx, shared_feature_count)
    rows = evaluate_corruption_variants(all_meta, all_va, all_te, best_bundle, shared_feature_count)
    report = pd.DataFrame(rows)
    report.to_csv(os.path.join(CORRUPTION_ARTIFACT_DIR, "input_corruption_missingness_final_report_raw.csv"), index=False)
    report = add_main_deltas(report)
    report.to_csv(os.path.join(CORRUPTION_ARTIFACT_DIR, "input_corruption_missingness_final_report_with_deltas.csv"), index=False)
    render_final_table(report)
    del stage_records, all_meta, all_tr, all_va, all_te, all_cidx, best_bundle
    release_memory()
    return report


if __name__ == "__main__":
    run_input_corruption_missingness_validation()
```


<div style="margin:12px 0 18px 0;border:1px solid #dbe4ee;border-radius:12px;overflow-x:auto;background:#ffffff;"><div style="padding:10px 14px;font-weight:700;color:#0f172a;background:#f8fafc;border-bottom:1px solid #e2e8f0;font-size:15px;">FINAL REPORT - INPUT CORRUPTION AND MISSINGNESS ROBUSTNESS VALIDATION</div><table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>variant</th>
      <th>corruption_type</th>
      <th>target_rate</th>
      <th>actual_rate</th>
      <th>split</th>
      <th>dataset</th>
      <th>corrupted_cells</th>
      <th>acc</th>
      <th>prec_macro</th>
      <th>rec_macro</th>
      <th>f1_macro</th>
      <th>logloss</th>
      <th>mcc</th>
      <th>kappa</th>
      <th>auc_roc_macro</th>
      <th>pr_auc</th>
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
      <td>numeric_missing_10</td>
      <td>numeric_missing</td>
      <td>0.1000</td>
      <td>0.0000</td>
      <td>VAL</td>
      <td>I23Sub</td>
      <td>0</td>
      <td>0.9185</td>
      <td>0.9592</td>
      <td>0.5087</td>
      <td>0.4958</td>
      <td>0.1088</td>
      <td>0.1263</td>
      <td>0.0314</td>
      <td>0.9955</td>
      <td>0.9831</td>
      <td>0.9251</td>
      <td>0.9185</td>
      <td>0.8809</td>
      <td>0.9955</td>
      <td>0.9955</td>
      <td>0.9592</td>
      <td>0.9592</td>
      <td>0.9251</td>
      <td>0.9932</td>
      <td>1.0000</td>
      <td>0.9184</td>
    </tr>
    <tr>
      <td>numeric_missing_10</td>
      <td>numeric_missing</td>
      <td>0.1000</td>
      <td>0.0000</td>
      <td>TEST</td>
      <td>I23Sub</td>
      <td>0</td>
      <td>0.9188</td>
      <td>0.9593</td>
      <td>0.5121</td>
      <td>0.5024</td>
      <td>0.1111</td>
      <td>0.1492</td>
      <td>0.0435</td>
      <td>0.9930</td>
      <td>0.9730</td>
      <td>0.9254</td>
      <td>0.9188</td>
      <td>0.8818</td>
      <td>0.9930</td>
      <td>0.9930</td>
      <td>0.9593</td>
      <td>0.9593</td>
      <td>0.9254</td>
      <td>0.9932</td>
      <td>1.0000</td>
      <td>0.9186</td>
    </tr>
    <tr>
      <td>numeric_missing_10</td>
      <td>numeric_missing</td>
      <td>0.1000</td>
      <td>0.1000</td>
      <td>VAL</td>
      <td>K99Sub</td>
      <td>6046</td>
      <td>0.9198</td>
      <td>0.9276</td>
      <td>0.9156</td>
      <td>0.9186</td>
      <td>0.3090</td>
      <td>0.8432</td>
      <td>0.8376</td>
      <td>0.9781</td>
      <td>0.9763</td>
      <td>0.9247</td>
      <td>0.9198</td>
      <td>0.9192</td>
      <td>0.9781</td>
      <td>0.9781</td>
      <td>0.9276</td>
      <td>0.9276</td>
      <td>0.9247</td>
      <td>0.9306</td>
      <td>0.8844</td>
      <td>0.9709</td>
    </tr>
    <tr>
      <td>numeric_missing_10</td>
      <td>numeric_missing</td>
      <td>0.1000</td>
      <td>0.0975</td>
      <td>TEST</td>
      <td>K99Sub</td>
      <td>5896</td>
      <td>0.9217</td>
      <td>0.9306</td>
      <td>0.9172</td>
      <td>0.9204</td>
      <td>0.3102</td>
      <td>0.8477</td>
      <td>0.8413</td>
      <td>0.9780</td>
      <td>0.9741</td>
      <td>0.9274</td>
      <td>0.9217</td>
      <td>0.9211</td>
      <td>0.9780</td>
      <td>0.9780</td>
      <td>0.9306</td>
      <td>0.9306</td>
      <td>0.9274</td>
      <td>0.9338</td>
      <td>0.8833</td>
      <td>0.9778</td>
    </tr>
    <tr>
      <td>numeric_missing_10</td>
      <td>numeric_missing</td>
      <td>0.1000</td>
      <td>0.1000</td>
      <td>VAL</td>
      <td>NTD1</td>
      <td>38660</td>
      <td>0.9526</td>
      <td>0.9331</td>
      <td>0.7430</td>
      <td>0.8062</td>
      <td>0.1648</td>
      <td>0.6488</td>
      <td>0.6148</td>
      <td>0.9778</td>
      <td>0.7996</td>
      <td>0.9510</td>
      <td>0.9526</td>
      <td>0.9460</td>
      <td>0.9778</td>
      <td>0.9778</td>
      <td>0.9331</td>
      <td>0.9331</td>
      <td>0.9510</td>
      <td>0.9153</td>
      <td>0.9116</td>
      <td>0.9546</td>
    </tr>
    <tr>
      <td>numeric_missing_10</td>
      <td>numeric_missing</td>
      <td>0.1000</td>
      <td>0.0992</td>
      <td>TEST</td>
      <td>NTD1</td>
      <td>38322</td>
      <td>0.9515</td>
      <td>0.9341</td>
      <td>0.7345</td>
      <td>0.7989</td>
      <td>0.1714</td>
      <td>0.6381</td>
      <td>0.6005</td>
      <td>0.9768</td>
      <td>0.7947</td>
      <td>0.9499</td>
      <td>0.9515</td>
      <td>0.9443</td>
      <td>0.9768</td>
      <td>0.9768</td>
      <td>0.9341</td>
      <td>0.9341</td>
      <td>0.9499</td>
      <td>0.9183</td>
      <td>0.9151</td>
      <td>0.9531</td>
    </tr>
    <tr>
      <td>numeric_missing_10</td>
      <td>numeric_missing</td>
      <td>0.1000</td>
      <td>0.0987</td>
      <td>VAL</td>
      <td>NTD2</td>
      <td>6612</td>
      <td>0.7940</td>
      <td>0.8460</td>
      <td>0.7823</td>
      <td>0.7808</td>
      <td>0.5796</td>
      <td>0.6251</td>
      <td>0.5773</td>
      <td>0.9211</td>
      <td>0.9203</td>
      <td>0.8390</td>
      <td>0.7940</td>
      <td>0.7840</td>
      <td>0.9211</td>
      <td>0.9211</td>
      <td>0.8460</td>
      <td>0.8460</td>
      <td>0.8390</td>
      <td>0.8531</td>
      <td>0.9663</td>
      <td>0.7257</td>
    </tr>
    <tr>
      <td>numeric_missing_10</td>
      <td>numeric_missing</td>
      <td>0.1000</td>
      <td>0.1010</td>
      <td>TEST</td>
      <td>NTD2</td>
      <td>6766</td>
      <td>0.7923</td>
      <td>0.8464</td>
      <td>0.7804</td>
      <td>0.7786</td>
      <td>0.5817</td>
      <td>0.6233</td>
      <td>0.5737</td>
      <td>0.9179</td>
      <td>0.9166</td>
      <td>0.8392</td>
      <td>0.7923</td>
      <td>0.7818</td>
      <td>0.9179</td>
      <td>0.9179</td>
      <td>0.8464</td>
      <td>0.8464</td>
      <td>0.8392</td>
      <td>0.8536</td>
      <td>0.9693</td>
      <td>0.7234</td>
    </tr>
    <tr>
      <td>numeric_missing_10</td>
      <td>numeric_missing</td>
      <td>0.1000</td>
      <td>0.1000</td>
      <td>VAL</td>
      <td>WII21</td>
      <td>304684</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.9999</td>
      <td>1.0000</td>
      <td>0.0001</td>
      <td>0.9999</td>
      <td>0.9999</td>
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
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <td>numeric_missing_10</td>
      <td>numeric_missing</td>
      <td>0.1000</td>
      <td>0.1004</td>
      <td>TEST</td>
      <td>WII21</td>
      <td>305754</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.9999</td>
      <td>1.0000</td>
      <td>0.0001</td>
      <td>0.9999</td>
      <td>0.9999</td>
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
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <td>numeric_missing_10</td>
      <td>numeric_missing</td>
      <td>0.1000</td>
      <td>0.1000</td>
      <td>VAL</td>
      <td>global_weighted</td>
      <td>356002</td>
      <td>0.9803</td>
      <td>0.9784</td>
      <td>0.9278</td>
      <td>0.9416</td>
      <td>0.0620</td>
      <td>0.8953</td>
      <td>0.8848</td>
      <td>0.9921</td>
      <td>0.9525</td>
      <td>0.9816</td>
      <td>0.9803</td>
      <td>0.9780</td>
      <td>0.9921</td>
      <td>0.9921</td>
      <td>0.9784</td>
      <td>0.9784</td>
      <td>0.9816</td>
      <td>0.9752</td>
      <td>0.9776</td>
      <td>0.9792</td>
    </tr>
    <tr>
      <td>numeric_missing_10</td>
      <td>numeric_missing</td>
      <td>0.1000</td>
      <td>0.1002</td>
      <td>TEST</td>
      <td>global_weighted</td>
      <td>356738</td>
      <td>0.9800</td>
      <td>0.9787</td>
      <td>0.9259</td>
      <td>0.9400</td>
      <td>0.0636</td>
      <td>0.8933</td>
      <td>0.8818</td>
      <td>0.9917</td>
      <td>0.9511</td>
      <td>0.9814</td>
      <td>0.9800</td>
      <td>0.9775</td>
      <td>0.9917</td>
      <td>0.9917</td>
      <td>0.9787</td>
      <td>0.9787</td>
      <td>0.9814</td>
      <td>0.9760</td>
      <td>0.9784</td>
      <td>0.9789</td>
    </tr>
    <tr>
      <td>numeric_missing_20</td>
      <td>numeric_missing</td>
      <td>0.2000</td>
      <td>0.0000</td>
      <td>VAL</td>
      <td>I23Sub</td>
      <td>0</td>
      <td>0.9185</td>
      <td>0.9592</td>
      <td>0.5087</td>
      <td>0.4958</td>
      <td>0.1088</td>
      <td>0.1263</td>
      <td>0.0314</td>
      <td>0.9955</td>
      <td>0.9831</td>
      <td>0.9251</td>
      <td>0.9185</td>
      <td>0.8809</td>
      <td>0.9955</td>
      <td>0.9955</td>
      <td>0.9592</td>
      <td>0.9592</td>
      <td>0.9251</td>
      <td>0.9932</td>
      <td>1.0000</td>
      <td>0.9184</td>
    </tr>
    <tr>
      <td>numeric_missing_20</td>
      <td>numeric_missing</td>
      <td>0.2000</td>
      <td>0.0000</td>
      <td>TEST</td>
      <td>I23Sub</td>
      <td>0</td>
      <td>0.9188</td>
      <td>0.9593</td>
      <td>0.5121</td>
      <td>0.5024</td>
      <td>0.1111</td>
      <td>0.1492</td>
      <td>0.0435</td>
      <td>0.9930</td>
      <td>0.9730</td>
      <td>0.9254</td>
      <td>0.9188</td>
      <td>0.8818</td>
      <td>0.9930</td>
      <td>0.9930</td>
      <td>0.9593</td>
      <td>0.9593</td>
      <td>0.9254</td>
      <td>0.9932</td>
      <td>1.0000</td>
      <td>0.9186</td>
    </tr>
    <tr>
      <td>numeric_missing_20</td>
      <td>numeric_missing</td>
      <td>0.2000</td>
      <td>0.1986</td>
      <td>VAL</td>
      <td>K99Sub</td>
      <td>12010</td>
      <td>0.9166</td>
      <td>0.9213</td>
      <td>0.9133</td>
      <td>0.9156</td>
      <td>0.3354</td>
      <td>0.8346</td>
      <td>0.8315</td>
      <td>0.9726</td>
      <td>0.9720</td>
      <td>0.9192</td>
      <td>0.9166</td>
      <td>0.9162</td>
      <td>0.9726</td>
      <td>0.9726</td>
      <td>0.9213</td>
      <td>0.9213</td>
      <td>0.9192</td>
      <td>0.9234</td>
      <td>0.8908</td>
      <td>0.9519</td>
    </tr>
    <tr>
      <td>numeric_missing_20</td>
      <td>numeric_missing</td>
      <td>0.2000</td>
      <td>0.2015</td>
      <td>TEST</td>
      <td>K99Sub</td>
      <td>12183</td>
      <td>0.9166</td>
      <td>0.9196</td>
      <td>0.9140</td>
      <td>0.9158</td>
      <td>0.3405</td>
      <td>0.8336</td>
      <td>0.8318</td>
      <td>0.9718</td>
      <td>0.9709</td>
      <td>0.9181</td>
      <td>0.9166</td>
      <td>0.9164</td>
      <td>0.9718</td>
      <td>0.9718</td>
      <td>0.9196</td>
      <td>0.9196</td>
      <td>0.9181</td>
      <td>0.9211</td>
      <td>0.8973</td>
      <td>0.9420</td>
    </tr>
    <tr>
      <td>numeric_missing_20</td>
      <td>numeric_missing</td>
      <td>0.2000</td>
      <td>0.1985</td>
      <td>VAL</td>
      <td>NTD1</td>
      <td>76714</td>
      <td>0.9416</td>
      <td>0.9299</td>
      <td>0.6703</td>
      <td>0.7345</td>
      <td>0.1812</td>
      <td>0.5412</td>
      <td>0.4756</td>
      <td>0.9748</td>
      <td>0.8035</td>
      <td>0.9403</td>
      <td>0.9416</td>
      <td>0.9291</td>
      <td>0.9748</td>
      <td>0.9748</td>
      <td>0.9299</td>
      <td>0.9299</td>
      <td>0.9403</td>
      <td>0.9196</td>
      <td>0.9175</td>
      <td>0.9424</td>
    </tr>
    <tr>
      <td>numeric_missing_20</td>
      <td>numeric_missing</td>
      <td>0.2000</td>
      <td>0.2002</td>
      <td>TEST</td>
      <td>NTD1</td>
      <td>77346</td>
      <td>0.9392</td>
      <td>0.9277</td>
      <td>0.6555</td>
      <td>0.7175</td>
      <td>0.1879</td>
      <td>0.5157</td>
      <td>0.4431</td>
      <td>0.9736</td>
      <td>0.7929</td>
      <td>0.9378</td>
      <td>0.9392</td>
      <td>0.9252</td>
      <td>0.9736</td>
      <td>0.9736</td>
      <td>0.9277</td>
      <td>0.9277</td>
      <td>0.9378</td>
      <td>0.9175</td>
      <td>0.9154</td>
      <td>0.9399</td>
    </tr>
    <tr>
      <td>numeric_missing_20</td>
      <td>numeric_missing</td>
      <td>0.2000</td>
      <td>0.1978</td>
      <td>VAL</td>
      <td>NTD2</td>
      <td>13248</td>
      <td>0.7940</td>
      <td>0.8460</td>
      <td>0.7823</td>
      <td>0.7808</td>
      <td>0.5796</td>
      <td>0.6251</td>
      <td>0.5773</td>
      <td>0.9227</td>
      <td>0.9217</td>
      <td>0.8390</td>
      <td>0.7940</td>
      <td>0.7840</td>
      <td>0.9227</td>
      <td>0.9227</td>
      <td>0.8460</td>
      <td>0.8460</td>
      <td>0.8390</td>
      <td>0.8531</td>
      <td>0.9663</td>
      <td>0.7257</td>
    </tr>
    <tr>
      <td>numeric_missing_20</td>
      <td>numeric_missing</td>
      <td>0.2000</td>
      <td>0.1993</td>
      <td>TEST</td>
      <td>NTD2</td>
      <td>13354</td>
      <td>0.7923</td>
      <td>0.8464</td>
      <td>0.7804</td>
      <td>0.7786</td>
      <td>0.5817</td>
      <td>0.6233</td>
      <td>0.5737</td>
      <td>0.9200</td>
      <td>0.9185</td>
      <td>0.8392</td>
      <td>0.7923</td>
      <td>0.7818</td>
      <td>0.9200</td>
      <td>0.9200</td>
      <td>0.8464</td>
      <td>0.8464</td>
      <td>0.8392</td>
      <td>0.8536</td>
      <td>0.9693</td>
      <td>0.7234</td>
    </tr>
    <tr>
      <td>numeric_missing_20</td>
      <td>numeric_missing</td>
      <td>0.2000</td>
      <td>0.2000</td>
      <td>VAL</td>
      <td>WII21</td>
      <td>609108</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.9999</td>
      <td>1.0000</td>
      <td>0.0001</td>
      <td>0.9999</td>
      <td>0.9999</td>
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
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <td>numeric_missing_20</td>
      <td>numeric_missing</td>
      <td>0.2000</td>
      <td>0.2000</td>
      <td>TEST</td>
      <td>WII21</td>
      <td>609305</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.9999</td>
      <td>0.9999</td>
      <td>0.0001</td>
      <td>0.9999</td>
      <td>0.9999</td>
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
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <td>numeric_missing_20</td>
      <td>numeric_missing</td>
      <td>0.2000</td>
      <td>0.1998</td>
      <td>VAL</td>
      <td>global_weighted</td>
      <td>711080</td>
      <td>0.9778</td>
      <td>0.9776</td>
      <td>0.9117</td>
      <td>0.9257</td>
      <td>0.0661</td>
      <td>0.8714</td>
      <td>0.8540</td>
      <td>0.9914</td>
      <td>0.9533</td>
      <td>0.9791</td>
      <td>0.9778</td>
      <td>0.9742</td>
      <td>0.9914</td>
      <td>0.9914</td>
      <td>0.9776</td>
      <td>0.9776</td>
      <td>0.9791</td>
      <td>0.9761</td>
      <td>0.9790</td>
      <td>0.9762</td>
    </tr>
    <tr>
      <td>numeric_missing_20</td>
      <td>numeric_missing</td>
      <td>0.2000</td>
      <td>0.2001</td>
      <td>TEST</td>
      <td>global_weighted</td>
      <td>712188</td>
      <td>0.9772</td>
      <td>0.9771</td>
      <td>0.9084</td>
      <td>0.9220</td>
      <td>0.0677</td>
      <td>0.8660</td>
      <td>0.8468</td>
      <td>0.9910</td>
      <td>0.9507</td>
      <td>0.9786</td>
      <td>0.9772</td>
      <td>0.9733</td>
      <td>0.9910</td>
      <td>0.9910</td>
      <td>0.9771</td>
      <td>0.9771</td>
      <td>0.9786</td>
      <td>0.9756</td>
      <td>0.9787</td>
      <td>0.9755</td>
    </tr>
    <tr>
      <td>categorical_oob_10</td>
      <td>categorical_oob</td>
      <td>0.1000</td>
      <td>0.0973</td>
      <td>VAL</td>
      <td>I23Sub</td>
      <td>8109</td>
      <td>0.9234</td>
      <td>0.9614</td>
      <td>0.5382</td>
      <td>0.5509</td>
      <td>0.1242</td>
      <td>0.2655</td>
      <td>0.1317</td>
      <td>0.9947</td>
      <td>0.9837</td>
      <td>0.9293</td>
      <td>0.9234</td>
      <td>0.8921</td>
      <td>0.9947</td>
      <td>0.9947</td>
      <td>0.9614</td>
      <td>0.9614</td>
      <td>0.9293</td>
      <td>0.9936</td>
      <td>1.0000</td>
      <td>0.9229</td>
    </tr>
    <tr>
      <td>categorical_oob_10</td>
      <td>categorical_oob</td>
      <td>0.1000</td>
      <td>0.0990</td>
      <td>TEST</td>
      <td>I23Sub</td>
      <td>8250</td>
      <td>0.9240</td>
      <td>0.9617</td>
      <td>0.5433</td>
      <td>0.5597</td>
      <td>0.1256</td>
      <td>0.2826</td>
      <td>0.1479</td>
      <td>0.9923</td>
      <td>0.9761</td>
      <td>0.9298</td>
      <td>0.9240</td>
      <td>0.8935</td>
      <td>0.9923</td>
      <td>0.9923</td>
      <td>0.9617</td>
      <td>0.9617</td>
      <td>0.9298</td>
      <td>0.9936</td>
      <td>1.0000</td>
      <td>0.9234</td>
    </tr>
    <tr>
      <td>categorical_oob_10</td>
      <td>categorical_oob</td>
      <td>0.1000</td>
      <td>0.1006</td>
      <td>VAL</td>
      <td>K99Sub</td>
      <td>2282</td>
      <td>0.9159</td>
      <td>0.9264</td>
      <td>0.9109</td>
      <td>0.9143</td>
      <td>0.2920</td>
      <td>0.8372</td>
      <td>0.8294</td>
      <td>0.9812</td>
      <td>0.9796</td>
      <td>0.9229</td>
      <td>0.9159</td>
      <td>0.9151</td>
      <td>0.9812</td>
      <td>0.9812</td>
      <td>0.9264</td>
      <td>0.9264</td>
      <td>0.9229</td>
      <td>0.9300</td>
      <td>0.8741</td>
      <td>0.9788</td>
    </tr>
    <tr>
      <td>categorical_oob_10</td>
      <td>categorical_oob</td>
      <td>0.1000</td>
      <td>0.0963</td>
      <td>TEST</td>
      <td>K99Sub</td>
      <td>2183</td>
      <td>0.9196</td>
      <td>0.9312</td>
      <td>0.9144</td>
      <td>0.9180</td>
      <td>0.2935</td>
      <td>0.8455</td>
      <td>0.8368</td>
      <td>0.9801</td>
      <td>0.9773</td>
      <td>0.9274</td>
      <td>0.9196</td>
      <td>0.9188</td>
      <td>0.9801</td>
      <td>0.9801</td>
      <td>0.9312</td>
      <td>0.9312</td>
      <td>0.9274</td>
      <td>0.9350</td>
      <td>0.8752</td>
      <td>0.9873</td>
    </tr>
    <tr>
      <td>categorical_oob_10</td>
      <td>categorical_oob</td>
      <td>0.1000</td>
      <td>0.1000</td>
      <td>VAL</td>
      <td>NTD1</td>
      <td>22074</td>
      <td>0.9574</td>
      <td>0.9386</td>
      <td>0.7720</td>
      <td>0.8319</td>
      <td>0.1581</td>
      <td>0.6908</td>
      <td>0.6654</td>
      <td>0.9771</td>
      <td>0.8026</td>
      <td>0.9560</td>
      <td>0.9574</td>
      <td>0.9525</td>
      <td>0.9771</td>
      <td>0.9771</td>
      <td>0.9386</td>
      <td>0.9386</td>
      <td>0.9560</td>
      <td>0.9212</td>
      <td>0.9176</td>
      <td>0.9596</td>
    </tr>
    <tr>
      <td>categorical_oob_10</td>
      <td>categorical_oob</td>
      <td>0.1000</td>
      <td>0.1006</td>
      <td>TEST</td>
      <td>NTD1</td>
      <td>22213</td>
      <td>0.9550</td>
      <td>0.9367</td>
      <td>0.7565</td>
      <td>0.8186</td>
      <td>0.1659</td>
      <td>0.6694</td>
      <td>0.6393</td>
      <td>0.9759</td>
      <td>0.7871</td>
      <td>0.9535</td>
      <td>0.9550</td>
      <td>0.9491</td>
      <td>0.9759</td>
      <td>0.9759</td>
      <td>0.9367</td>
      <td>0.9367</td>
      <td>0.9535</td>
      <td>0.9200</td>
      <td>0.9166</td>
      <td>0.9569</td>
    </tr>
    <tr>
      <td>categorical_oob_10</td>
      <td>categorical_oob</td>
      <td>0.1000</td>
      <td>0.1007</td>
      <td>VAL</td>
      <td>NTD2</td>
      <td>21076</td>
      <td>0.7374</td>
      <td>0.8168</td>
      <td>0.7221</td>
      <td>0.7107</td>
      <td>0.6040</td>
      <td>0.5305</td>
      <td>0.4574</td>
      <td>0.9008</td>
      <td>0.9024</td>
      <td>0.8083</td>
      <td>0.7374</td>
      <td>0.7159</td>
      <td>0.9008</td>
      <td>0.9008</td>
      <td>0.8168</td>
      <td>0.8168</td>
      <td>0.8083</td>
      <td>0.8252</td>
      <td>0.9614</td>
      <td>0.6722</td>
    </tr>
    <tr>
      <td>categorical_oob_10</td>
      <td>categorical_oob</td>
      <td>0.1000</td>
      <td>0.0998</td>
      <td>TEST</td>
      <td>NTD2</td>
      <td>20883</td>
      <td>0.7350</td>
      <td>0.8193</td>
      <td>0.7193</td>
      <td>0.7068</td>
      <td>0.6055</td>
      <td>0.5293</td>
      <td>0.4520</td>
      <td>0.9000</td>
      <td>0.8980</td>
      <td>0.8106</td>
      <td>0.7350</td>
      <td>0.7121</td>
      <td>0.9000</td>
      <td>0.9000</td>
      <td>0.8193</td>
      <td>0.8193</td>
      <td>0.8106</td>
      <td>0.8281</td>
      <td>0.9695</td>
      <td>0.6692</td>
    </tr>
    <tr>
      <td>categorical_oob_10</td>
      <td>categorical_oob</td>
      <td>0.1000</td>
      <td>0.0997</td>
      <td>VAL</td>
      <td>WII21</td>
      <td>214375</td>
      <td>0.9996</td>
      <td>0.9975</td>
      <td>0.9996</td>
      <td>0.9985</td>
      <td>0.0020</td>
      <td>0.9971</td>
      <td>0.9971</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.9996</td>
      <td>0.9996</td>
      <td>0.9996</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.9975</td>
      <td>0.9975</td>
      <td>0.9996</td>
      <td>0.9954</td>
      <td>1.0000</td>
      <td>0.9950</td>
    </tr>
    <tr>
      <td>categorical_oob_10</td>
      <td>categorical_oob</td>
      <td>0.1000</td>
      <td>0.0998</td>
      <td>TEST</td>
      <td>WII21</td>
      <td>214480</td>
      <td>0.9996</td>
      <td>0.9974</td>
      <td>0.9996</td>
      <td>0.9985</td>
      <td>0.0021</td>
      <td>0.9969</td>
      <td>0.9969</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.9996</td>
      <td>0.9996</td>
      <td>0.9996</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.9974</td>
      <td>0.9974</td>
      <td>0.9996</td>
      <td>0.9951</td>
      <td>1.0000</td>
      <td>0.9947</td>
    </tr>
    <tr>
      <td>categorical_oob_10</td>
      <td>categorical_oob</td>
      <td>0.1000</td>
      <td>0.0997</td>
      <td>VAL</td>
      <td>global_weighted</td>
      <td>267916</td>
      <td>0.9792</td>
      <td>0.9769</td>
      <td>0.9323</td>
      <td>0.9446</td>
      <td>0.0627</td>
      <td>0.9012</td>
      <td>0.8912</td>
      <td>0.9913</td>
      <td>0.9526</td>
      <td>0.9814</td>
      <td>0.9792</td>
      <td>0.9769</td>
      <td>0.9913</td>
      <td>0.9913</td>
      <td>0.9769</td>
      <td>0.9769</td>
      <td>0.9814</td>
      <td>0.9723</td>
      <td>0.9786</td>
      <td>0.9751</td>
    </tr>
    <tr>
      <td>categorical_oob_10</td>
      <td>categorical_oob</td>
      <td>0.1000</td>
      <td>0.0998</td>
      <td>TEST</td>
      <td>global_weighted</td>
      <td>268009</td>
      <td>0.9786</td>
      <td>0.9765</td>
      <td>0.9289</td>
      <td>0.9417</td>
      <td>0.0646</td>
      <td>0.8967</td>
      <td>0.8855</td>
      <td>0.9909</td>
      <td>0.9489</td>
      <td>0.9810</td>
      <td>0.9786</td>
      <td>0.9761</td>
      <td>0.9909</td>
      <td>0.9909</td>
      <td>0.9765</td>
      <td>0.9765</td>
      <td>0.9810</td>
      <td>0.9720</td>
      <td>0.9786</td>
      <td>0.9744</td>
    </tr>
    <tr>
      <td>feature_dropout_10</td>
      <td>feature_dropout</td>
      <td>0.1000</td>
      <td>0.1003</td>
      <td>VAL</td>
      <td>I23Sub</td>
      <td>8361</td>
      <td>0.9254</td>
      <td>0.9624</td>
      <td>0.5503</td>
      <td>0.5719</td>
      <td>0.1237</td>
      <td>0.3052</td>
      <td>0.1704</td>
      <td>0.9954</td>
      <td>0.9837</td>
      <td>0.9310</td>
      <td>0.9254</td>
      <td>0.8964</td>
      <td>0.9954</td>
      <td>0.9954</td>
      <td>0.9624</td>
      <td>0.9624</td>
      <td>0.9310</td>
      <td>0.9938</td>
      <td>1.0000</td>
      <td>0.9248</td>
    </tr>
    <tr>
      <td>feature_dropout_10</td>
      <td>feature_dropout</td>
      <td>0.1000</td>
      <td>0.1013</td>
      <td>TEST</td>
      <td>I23Sub</td>
      <td>8440</td>
      <td>0.9248</td>
      <td>0.9621</td>
      <td>0.5484</td>
      <td>0.5686</td>
      <td>0.1249</td>
      <td>0.2992</td>
      <td>0.1644</td>
      <td>0.9922</td>
      <td>0.9753</td>
      <td>0.9305</td>
      <td>0.9248</td>
      <td>0.8954</td>
      <td>0.9922</td>
      <td>0.9922</td>
      <td>0.9621</td>
      <td>0.9621</td>
      <td>0.9305</td>
      <td>0.9937</td>
      <td>1.0000</td>
      <td>0.9242</td>
    </tr>
    <tr>
      <td>feature_dropout_10</td>
      <td>feature_dropout</td>
      <td>0.1000</td>
      <td>0.1019</td>
      <td>VAL</td>
      <td>K99Sub</td>
      <td>8473</td>
      <td>0.9143</td>
      <td>0.9215</td>
      <td>0.9101</td>
      <td>0.9130</td>
      <td>0.3187</td>
      <td>0.8315</td>
      <td>0.8264</td>
      <td>0.9762</td>
      <td>0.9744</td>
      <td>0.9187</td>
      <td>0.9143</td>
      <td>0.9137</td>
      <td>0.9762</td>
      <td>0.9762</td>
      <td>0.9215</td>
      <td>0.9215</td>
      <td>0.9187</td>
      <td>0.9242</td>
      <td>0.8808</td>
      <td>0.9621</td>
    </tr>
    <tr>
      <td>feature_dropout_10</td>
      <td>feature_dropout</td>
      <td>0.1000</td>
      <td>0.0991</td>
      <td>TEST</td>
      <td>K99Sub</td>
      <td>8243</td>
      <td>0.9177</td>
      <td>0.9237</td>
      <td>0.9140</td>
      <td>0.9166</td>
      <td>0.3186</td>
      <td>0.8377</td>
      <td>0.8335</td>
      <td>0.9758</td>
      <td>0.9750</td>
      <td>0.9213</td>
      <td>0.9177</td>
      <td>0.9172</td>
      <td>0.9758</td>
      <td>0.9758</td>
      <td>0.9237</td>
      <td>0.9237</td>
      <td>0.9213</td>
      <td>0.9262</td>
      <td>0.8874</td>
      <td>0.9601</td>
    </tr>
    <tr>
      <td>feature_dropout_10</td>
      <td>feature_dropout</td>
      <td>0.1000</td>
      <td>0.0997</td>
      <td>VAL</td>
      <td>NTD1</td>
      <td>60524</td>
      <td>0.9453</td>
      <td>0.9309</td>
      <td>0.6945</td>
      <td>0.7601</td>
      <td>0.1716</td>
      <td>0.5790</td>
      <td>0.5251</td>
      <td>0.9741</td>
      <td>0.8031</td>
      <td>0.9438</td>
      <td>0.9453</td>
      <td>0.9350</td>
      <td>0.9741</td>
      <td>0.9741</td>
      <td>0.9309</td>
      <td>0.9309</td>
      <td>0.9438</td>
      <td>0.9181</td>
      <td>0.9155</td>
      <td>0.9464</td>
    </tr>
    <tr>
      <td>feature_dropout_10</td>
      <td>feature_dropout</td>
      <td>0.1000</td>
      <td>0.0999</td>
      <td>TEST</td>
      <td>NTD1</td>
      <td>60636</td>
      <td>0.9437</td>
      <td>0.9301</td>
      <td>0.6843</td>
      <td>0.7495</td>
      <td>0.1780</td>
      <td>0.5631</td>
      <td>0.5045</td>
      <td>0.9738</td>
      <td>0.8003</td>
      <td>0.9422</td>
      <td>0.9437</td>
      <td>0.9325</td>
      <td>0.9738</td>
      <td>0.9738</td>
      <td>0.9301</td>
      <td>0.9301</td>
      <td>0.9422</td>
      <td>0.9180</td>
      <td>0.9155</td>
      <td>0.9447</td>
    </tr>
    <tr>
      <td>feature_dropout_10</td>
      <td>feature_dropout</td>
      <td>0.1000</td>
      <td>0.1002</td>
      <td>VAL</td>
      <td>NTD2</td>
      <td>27698</td>
      <td>0.7348</td>
      <td>0.8172</td>
      <td>0.7192</td>
      <td>0.7069</td>
      <td>0.6043</td>
      <td>0.5274</td>
      <td>0.4516</td>
      <td>0.9045</td>
      <td>0.9049</td>
      <td>0.8085</td>
      <td>0.7348</td>
      <td>0.7122</td>
      <td>0.9045</td>
      <td>0.9045</td>
      <td>0.8172</td>
      <td>0.8172</td>
      <td>0.8085</td>
      <td>0.8258</td>
      <td>0.9649</td>
      <td>0.6695</td>
    </tr>
    <tr>
      <td>feature_dropout_10</td>
      <td>feature_dropout</td>
      <td>0.1000</td>
      <td>0.0999</td>
      <td>TEST</td>
      <td>NTD2</td>
      <td>27612</td>
      <td>0.7308</td>
      <td>0.8161</td>
      <td>0.7150</td>
      <td>0.7016</td>
      <td>0.6066</td>
      <td>0.5213</td>
      <td>0.4432</td>
      <td>0.9023</td>
      <td>0.9012</td>
      <td>0.8073</td>
      <td>0.7308</td>
      <td>0.7070</td>
      <td>0.9023</td>
      <td>0.9023</td>
      <td>0.8161</td>
      <td>0.8161</td>
      <td>0.8073</td>
      <td>0.8249</td>
      <td>0.9663</td>
      <td>0.6659</td>
    </tr>
    <tr>
      <td>feature_dropout_10</td>
      <td>feature_dropout</td>
      <td>0.1000</td>
      <td>0.0999</td>
      <td>VAL</td>
      <td>WII21</td>
      <td>518842</td>
      <td>0.9996</td>
      <td>0.9978</td>
      <td>0.9993</td>
      <td>0.9985</td>
      <td>0.0019</td>
      <td>0.9971</td>
      <td>0.9971</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.9996</td>
      <td>0.9996</td>
      <td>0.9996</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.9978</td>
      <td>0.9978</td>
      <td>0.9996</td>
      <td>0.9960</td>
      <td>0.9999</td>
      <td>0.9956</td>
    </tr>
    <tr>
      <td>feature_dropout_10</td>
      <td>feature_dropout</td>
      <td>0.1000</td>
      <td>0.1000</td>
      <td>TEST</td>
      <td>WII21</td>
      <td>519571</td>
      <td>0.9994</td>
      <td>0.9966</td>
      <td>0.9992</td>
      <td>0.9979</td>
      <td>0.0029</td>
      <td>0.9958</td>
      <td>0.9958</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.9994</td>
      <td>0.9994</td>
      <td>0.9994</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.9966</td>
      <td>0.9966</td>
      <td>0.9994</td>
      <td>0.9938</td>
      <td>0.9999</td>
      <td>0.9933</td>
    </tr>
    <tr>
      <td>feature_dropout_10</td>
      <td>feature_dropout</td>
      <td>0.1000</td>
      <td>0.0999</td>
      <td>VAL</td>
      <td>global_weighted</td>
      <td>623898</td>
      <td>0.9764</td>
      <td>0.9753</td>
      <td>0.9150</td>
      <td>0.9289</td>
      <td>0.0660</td>
      <td>0.8769</td>
      <td>0.8605</td>
      <td>0.9907</td>
      <td>0.9527</td>
      <td>0.9787</td>
      <td>0.9764</td>
      <td>0.9730</td>
      <td>0.9907</td>
      <td>0.9907</td>
      <td>0.9753</td>
      <td>0.9753</td>
      <td>0.9787</td>
      <td>0.9720</td>
      <td>0.9783</td>
      <td>0.9724</td>
    </tr>
    <tr>
      <td>feature_dropout_10</td>
      <td>feature_dropout</td>
      <td>0.1000</td>
      <td>0.1000</td>
      <td>TEST</td>
      <td>global_weighted</td>
      <td>624502</td>
      <td>0.9758</td>
      <td>0.9743</td>
      <td>0.9126</td>
      <td>0.9259</td>
      <td>0.0683</td>
      <td>0.8723</td>
      <td>0.8548</td>
      <td>0.9905</td>
      <td>0.9519</td>
      <td>0.9782</td>
      <td>0.9758</td>
      <td>0.9722</td>
      <td>0.9905</td>
      <td>0.9905</td>
      <td>0.9743</td>
      <td>0.9743</td>
      <td>0.9782</td>
      <td>0.9704</td>
      <td>0.9785</td>
      <td>0.9701</td>
    </tr>
  </tbody>
</table></div>



```python

```
