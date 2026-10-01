"""Train one survival graph attention network per cross-validation fold on the per-slide cell graphs.
Inputs: --graph-dir (per-slide graphs) and --ai-score-dir (cancer_dataset.csv, fold_indices.pkl from 16_ai_score_cv.py --dry-run).
Outputs: per-fold embedding CSVs, gnn_embeddings_per_fold.csv, gnn_embeddings_oof.csv (out-of-fold) and run summaries."""

import argparse
import gc
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import DataLoader as PyGDataLoader
    from torch_geometric.nn import GATConv, global_mean_pool
except ImportError as e:
    print(f"Error: {e}\npip install torch torch_geometric")
    sys.exit(1)

try:
    from sksurv.metrics import concordance_index_censored
except ImportError:
    print("Error: pip install scikit-survival")
    sys.exit(1)

from sklearn.model_selection import StratifiedShuffleSplit

# some graph files use a different separator than slide_name in cancer_dataset.csv
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.graph_io import resolve_graph_path


SEED = 42


def cox_loss(log_hazard, time, event):
    # Cox partial likelihood: learns a risk ordering, not survival times (what the C-index measures)
    order = torch.argsort(time, descending=True)
    lh = log_hazard[order]
    ev = event[order]
    log_cumsum = torch.logcumsumexp(lh, dim=0)
    mask = ev.bool()
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=lh.device)
    return -torch.mean((lh - log_cumsum)[mask])


class SurvivalGAT(nn.Module):
    def __init__(self, in_channels=57, hidden_dim=32, embedding_dim=32,
                 heads=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_dim, heads=heads,
                             dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads,
                             dropout=dropout, concat=True)
        self.conv3 = GATConv(hidden_dim * heads, embedding_dim,
                             heads=1, dropout=dropout, concat=False)
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.embedding_dim = embedding_dim

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        # mean pooling gives a fixed-width vector whatever the number of cells in the slide
        emb = global_mean_pool(x, batch)
        out = F.dropout(F.relu(self.fc1(emb)), p=self.dropout, training=self.training)
        return self.fc2(out), emb


def compute_cindex(model, loader, device):
    model.eval()
    hazards, times, events = [], [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            lh, _ = model(b.x, b.edge_index, b.batch)
            hazards.append(lh.squeeze(-1).cpu().numpy())
            times.append(b.y_time.cpu().numpy())
            events.append(b.y_event.cpu().numpy())
    h = np.concatenate(hazards)
    t = np.concatenate(times)
    e = np.concatenate(events).astype(bool)
    if e.sum() < 2:
        return 0.5
    return float(concordance_index_censored(e, t, h)[0])


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for b in loader:
        b = b.to(device)
        optimizer.zero_grad()
        lh, _ = model(b.x, b.edge_index, b.batch)
        loss = cox_loss(lh.squeeze(-1), b.y_time, b.y_event)
        if torch.isnan(loss) or torch.isinf(loss):
            torch.cuda.empty_cache()
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item(); n += 1
        torch.cuda.empty_cache()
    return total / max(n, 1)


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embs, names = [], []
    for b in loader:
        b = b.to(device)
        _, e = model(b.x, b.edge_index, b.batch)
        embs.append(e.cpu().numpy())
        if isinstance(b.slide_name, (list, tuple)):
            names.extend(b.slide_name)
        else:
            names.append(b.slide_name)
    return np.vstack(embs), names


def load_graphs_for_slides(graph_dir, slide_names, surv_lookup):
    """Load graphs by slide name and attach (months_fu, death) targets from surv_lookup."""
    graph_dir = Path(graph_dir)
    graphs = {}
    missing = 0
    missing_names = []
    for name in tqdm(slide_names, desc="Loading graphs", leave=False):
        try:
            pt = resolve_graph_path(name, graph_dir)
        except FileNotFoundError:
            missing += 1
            missing_names.append(name)
            continue
        try:
            g = torch.load(pt, map_location="cpu", weights_only=False)
            months_fu, death = surv_lookup[name]
            g.y_time = torch.tensor([months_fu], dtype=torch.float)
            g.y_event = torch.tensor([death], dtype=torch.float)
            g.slide_name = name
            graphs[name] = g
        except Exception as e:
            print(f"  Warning: {name}.pt — {e}")
    if missing:
        preview = ", ".join(missing_names[:5])
        more = f" (+{missing - 5} more)" if missing > 5 else ""
        print(f"  Warning: {missing} slides without graph files: {preview}{more}")
    print(f"  Loaded {len(graphs)} graphs")
    return graphs


def train_inner_fold(inner_train_set, inner_val_set, args, device, label):
    """Train with early stopping on inner_val, a split of the fold's training slides (never its test fold)."""
    train_loader = PyGDataLoader(inner_train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(inner_val_set, batch_size=args.batch_size, shuffle=False)

    model = SurvivalGAT(args.in_channels, args.hidden_dim,
                        args.embedding_dim, args.heads, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    best_val, best_epoch, patience, best_state = 0.0, 0, 0, None
    history = {"train_loss": [], "train_ci": [], "val_ci": []}

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        train_ci = compute_cindex(model, train_loader, device)
        val_ci = compute_cindex(model, val_loader, device)
        history["train_loss"].append(loss)
        history["train_ci"].append(train_ci)
        history["val_ci"].append(val_ci)
        scheduler.step()
        if val_ci > best_val:
            best_val, best_epoch, patience = val_ci, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            improved = "*"
        else:
            patience += 1
            improved = ""
        print(
            f"    {label} epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={loss:.4f} train_CI={train_ci:.4f} val_CI={val_ci:.4f} "
            f"best_epoch={best_epoch} patience={patience}/{args.patience}{improved}",
            flush=True,
        )
        if patience >= args.patience:
            print(
                f"    {label} early stopping at epoch {epoch} "
                f"(best epoch {best_epoch}, best val_CI={best_val:.4f})",
                flush=True,
            )
            break

    train_ci_at_best = history["train_ci"][best_epoch - 1] if best_epoch > 0 else 0.5

    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return best_val, best_epoch, best_state, train_ci_at_best, history


def run_one_fold(repeat, fold, train_slides, test_slides, graph_lookup, args, device, folds_dir):
    """Train the GNN for one (repeat, fold) and write its train and test embeddings."""
    t0 = time.time()
    label = f"R{repeat}_F{fold}"
    out_csv = folds_dir / f"r{repeat}_f{fold}.csv"

    if out_csv.exists() and not args.force:
        print(f"[{label}] Cached — skipping")
        return {"repeat": repeat, "fold": fold, "status": "cached",
                "val_cindex": None, "best_epoch": None, "train_cindex": None,
                "n_train": None, "n_test": None, "seconds": 0.0}

    train_graphs = [graph_lookup[s] for s in train_slides if s in graph_lookup]
    test_graphs = [graph_lookup[s] for s in test_slides if s in graph_lookup]
    n_missing_train = len(train_slides) - len(train_graphs)
    n_missing_test = len(test_slides) - len(test_graphs)

    if len(train_graphs) < 10:
        print(f"[{label}] SKIP — only {len(train_graphs)} train graphs")
        return {"repeat": repeat, "fold": fold, "status": "too_few_graphs",
                "val_cindex": None, "best_epoch": None, "train_cindex": None,
                "n_train": len(train_graphs), "n_test": len(test_graphs), "seconds": 0.0}

    # inner split of the training graphs, used only for early stopping
    train_events = np.array([int(g.y_event.item()) for g in train_graphs])
    inner_seed = args.seed + repeat * 100 + fold
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.inner_val_frac,
                                     random_state=inner_seed)
        inner_tr_idx, inner_val_idx = next(sss.split(np.zeros(len(train_graphs)), train_events))
    except ValueError:
        # only one event class present: fall back to a random split
        rng = np.random.default_rng(inner_seed)
        n = len(train_graphs)
        n_val = max(2, int(round(n * args.inner_val_frac)))
        perm = rng.permutation(n)
        inner_val_idx = perm[:n_val]
        inner_tr_idx = perm[n_val:]

    inner_train_graphs = [train_graphs[i] for i in inner_tr_idx]
    inner_val_graphs = [train_graphs[i] for i in inner_val_idx]

    best_val, best_epoch, best_state, train_ci, history = train_inner_fold(
        inner_train_graphs, inner_val_graphs, args, device, label)

    model = SurvivalGAT(args.in_channels, args.hidden_dim, args.embedding_dim,
                        args.heads, args.dropout).to(device)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # train rows are in-sample (to fit the Cox model), test rows are out-of-fold (to evaluate it)
    all_fold_graphs = train_graphs + test_graphs
    splits_list = ["train"] * len(train_graphs) + ["test"] * len(test_graphs)
    loader = PyGDataLoader(all_fold_graphs, batch_size=args.batch_size, shuffle=False)
    embs, names = extract_embeddings(model, loader, device)

    name_to_split = {n: s for n, s in zip(
        [g.slide_name for g in all_fold_graphs], splits_list)}
    split_column = [name_to_split[n] for n in names]

    df_emb = pd.DataFrame(embs, columns=[f"emb_{i}" for i in range(embs.shape[1])])
    df_emb.insert(0, "slide_name", names)
    df_emb.insert(1, "repeat", repeat)
    df_emb.insert(2, "fold", fold)
    df_emb.insert(3, "split", split_column)
    df_emb.to_csv(out_csv, index=False)

    if args.save_curves:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        ep = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(ep, history["train_loss"], color="#2166AC")
        axes[0].set(xlabel="Epoch", ylabel="Loss", title=f"{label} — Loss")
        axes[1].plot(ep, history["train_ci"], color="#2166AC", label="Inner train")
        axes[1].plot(ep, history["val_ci"], color="#B2182B", ls="--", label="Inner val")
        axes[1].axhline(0.5, color="grey", lw=0.5, ls="--", alpha=0.5)
        axes[1].set(xlabel="Epoch", ylabel="C-index", title=f"{label} — C-index",
                    ylim=(0.35, 0.85))
        axes[1].legend()
        plt.tight_layout()
        fig.savefig(folds_dir / f"training_{label}.png", dpi=150)
        plt.close(fig)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    elapsed = time.time() - t0
    print(f"[{label}] inner_val_ci={best_val:.4f}  best_epoch={best_epoch}  "
          f"n_train={len(train_graphs)} (missing={n_missing_train})  "
          f"n_test={len(test_graphs)} (missing={n_missing_test})  "
          f"{elapsed/60:.1f}min")

    return {"repeat": repeat, "fold": fold, "status": "trained",
            "val_cindex": float(best_val), "best_epoch": int(best_epoch),
            "train_cindex": float(train_ci),
            "n_train": len(train_graphs), "n_test": len(test_graphs),
            "n_missing_train": n_missing_train, "n_missing_test": n_missing_test,
            "seconds": float(elapsed)}


def parse_args():
    p = argparse.ArgumentParser(
        description="C3PRO step 15 — survival GNN, one network per cross-validation fold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ai-score-dir", required=True, type=Path,
                   help="Directory with cancer_dataset.csv + fold_indices.pkl "
                        "(emitted by 16_ai_score_cv.py --dry-run)")
    p.add_argument("--graph-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)

    p.add_argument("--in-channels", type=int, default=57)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--embedding-dim", type=int, default=32)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--inner-val-frac", type=float, default=0.1,
                   help="Fraction of train set held out for early stopping")

    p.add_argument("--folds-subset", type=str, default=None,
                   help="Comma-separated list of 'r,f' pairs to run (e.g. '0,0;0,1'). "
                        "If omitted, runs all folds from fold_indices.pkl.")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if a per-fold CSV already exists")
    p.add_argument("--save-curves", action="store_true",
                   help="Save per-fold training curve PNGs")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    global SEED
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds_dir = args.output_dir / "folds"
    folds_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else args.device if args.device != "auto" else "cpu")

    print("=" * 70)
    print("C3PRO — GNN SURVIVAL NESTED CV (Script 15)")
    print("=" * 70)
    print(f"  Device       : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  AI score dir : {args.ai_score_dir}")
    print(f"  Graph dir    : {args.graph_dir}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Arch         : GAT hidden={args.hidden_dim}x{args.heads}h "
          f"-> emb={args.embedding_dim}  dropout={args.dropout}")
    print(f"  Inner val    : {args.inner_val_frac*100:.0f}% held-out (stratified by event)")

    fold_pkl = args.ai_score_dir / "fold_indices.pkl"
    if not fold_pkl.exists():
        print(f"ERROR: {fold_pkl} not found. Run 16_ai_score_cv.py --dry-run first.")
        sys.exit(1)
    with open(fold_pkl, "rb") as f:
        fold_indices = pickle.load(f)
    print(f"  Loaded {len(fold_indices)} folds from fold_indices.pkl")

    if args.folds_subset:
        want = set()
        for token in args.folds_subset.split(";"):
            r, f = token.strip().split(",")
            want.add((int(r), int(f)))
        fold_indices = {k: v for k, v in fold_indices.items() if k in want}
        print(f"  Filtering to subset: {sorted(fold_indices.keys())}")

    cancer_csv = args.ai_score_dir / "cancer_dataset.csv"
    cancer_df = pd.read_csv(cancer_csv)
    surv_mask = (cancer_df["DEATH"].notna() & cancer_df["MONTHS_FU"].notna()
                 & (cancer_df["MONTHS_FU"] > 0))
    cancer_df = cancer_df[surv_mask].copy()
    cancer_df["DEATH"] = cancer_df["DEATH"].astype(int)
    cancer_df["MONTHS_FU"] = cancer_df["MONTHS_FU"].astype(float)
    surv_lookup = {row["slide_name"]: (row["MONTHS_FU"], row["DEATH"])
                   for _, row in cancer_df.iterrows()}
    print(f"  Survival slides available: {len(surv_lookup)}  "
          f"events: {int(cancer_df['DEATH'].sum())}  "
          f"({100 * cancer_df['DEATH'].mean():.1f}%)")

    # load every graph once and share it across folds (a few hundred MB)
    all_slides_needed = set()
    for payload in fold_indices.values():
        all_slides_needed.update(payload["train"])
        all_slides_needed.update(payload["test"])
    slides_to_load = [s for s in all_slides_needed if s in surv_lookup]
    print(f"  Loading {len(slides_to_load)} graphs "
          f"({len(all_slides_needed) - len(slides_to_load)} excluded — no survival data)")
    graph_lookup = load_graphs_for_slides(args.graph_dir, slides_to_load, surv_lookup)

    if not graph_lookup:
        print("ERROR: no graphs loaded. Check --graph-dir and cancer_dataset.csv.")
        sys.exit(1)

    # --in-channels is overridden by the feature width of the graphs
    first_graph = next(iter(graph_lookup.values()))
    args.in_channels = first_graph.x.shape[1]
    print(f"  in_channels  : {args.in_channels} (inferred from graph features)")

    n_params = sum(p.numel() for p in SurvivalGAT(
        args.in_channels, args.hidden_dim, args.embedding_dim,
        args.heads, args.dropout).parameters() if p.requires_grad)
    print(f"  Model params : {n_params:,}")
    print()

    fold_records = []
    t_all = time.time()

    ordered = sorted(fold_indices.items())
    for i, ((repeat, fold), payload) in enumerate(ordered):
        print(f"─── Fold {i+1}/{len(ordered)}: R{repeat}_F{fold} ───")
        train_slides = [s for s in payload["train"] if s in graph_lookup]
        test_slides = [s for s in payload["test"] if s in graph_lookup]
        rec = run_one_fold(repeat, fold, train_slides, test_slides,
                           graph_lookup, args, device, folds_dir)
        fold_records.append(rec)

    total_min = (time.time() - t_all) / 60
    print(f"\n{'=' * 70}")
    print(f"Trained {sum(1 for r in fold_records if r['status'] == 'trained')} folds, "
          f"{sum(1 for r in fold_records if r['status'] == 'cached')} cached, "
          f"total {total_min:.1f} min")

    all_dfs = []
    for (r, f) in sorted(fold_indices.keys()):
        p = folds_dir / f"r{r}_f{f}.csv"
        if p.exists():
            all_dfs.append(pd.read_csv(p))
    if not all_dfs:
        print("ERROR: no per-fold CSVs found. Training must have failed.")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    per_fold_path = args.output_dir / "gnn_embeddings_per_fold.csv"
    combined.to_csv(per_fold_path, index=False)
    print(f"  Wrote {combined.shape} -> {per_fold_path}")

    # test rows only: the out-of-fold embeddings used by step 16
    oof = combined[combined["split"] == "test"].drop(columns=["split"])
    oof_path = args.output_dir / "gnn_embeddings_oof.csv"
    oof.to_csv(oof_path, index=False)
    print(f"  Wrote {oof.shape} -> {oof_path}")

    summary_df = pd.DataFrame(fold_records)
    summary_df.to_csv(args.output_dir / "nested_cv_summary.csv", index=False)

    trained = summary_df[summary_df["status"] == "trained"]
    summary = {
        "n_folds_total": len(fold_records),
        "n_folds_trained": int(len(trained)),
        "n_folds_cached": int((summary_df["status"] == "cached").sum()),
        "mean_inner_val_cindex": float(trained["val_cindex"].mean())
            if len(trained) else None,
        "std_inner_val_cindex": float(trained["val_cindex"].std())
            if len(trained) > 1 else None,
        "total_minutes": float(total_min),
        "config": {k: getattr(args, k.replace("-", "_"))
                   for k in ["in_channels", "hidden_dim", "embedding_dim",
                             "heads", "dropout", "epochs", "lr",
                             "weight_decay", "batch_size", "patience",
                             "inner_val_frac"]},
        "note": ("Inner val C-index is measured on a 10% held-out of each outer "
                 "train set, used only for early stopping. It is NOT the test-fold "
                 "C-index — that one is computed downstream by script 16 via Cox on "
                 "the OOF embeddings exported here."),
    }
    with open(args.output_dir / "nested_cv_results.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Wrote nested_cv_results.json")
    print("Done.")


if __name__ == "__main__":
    main()
