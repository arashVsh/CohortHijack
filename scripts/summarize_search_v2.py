from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def final_per_target(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "seed", "classifier", "audit", "target_index"]
    return df.sort_values("repeat").groupby(keys, as_index=False).tail(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="outputs/iccke")
    ap.add_argument("--followup", default="outputs/followup_matched_greedy")
    ap.add_argument("--search-v2", default="outputs/search_v2")
    ap.add_argument("--output", default="outputs/search_v2_summary")
    args = ap.parse_args()
    frames=[]
    for label,path in [("original",args.primary),("matched_pool240",args.followup),("search_v2",args.search_v2)]:
        p=Path(path)/"per_target_results.csv"
        if not p.exists():
            continue
        d=pd.read_csv(p)
        d=d[d.audit.isin(["greedy","multistart_greedy","beam"])].copy()
        d["source"]=label
        frames.append(final_per_target(d))
    if not frames:
        raise FileNotFoundError("No search result files found.")
    all_df=pd.concat(frames,ignore_index=True)
    summary=(all_df.groupby(["source","dataset","classifier","audit"],dropna=False)
             .agg(targets=("target_index","size"), successes=("target_flipped","sum"),
                  success_rate=("target_flipped","mean"),
                  mean_final_removed_fraction=("removed_fraction","mean"),
                  mean_collateral_flip_rate=("collateral_flip_rate","mean"),
                  mean_final_margin=("target_margin_original_class","mean"))
             .reset_index())
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    summary.to_csv(out/"search_comparison.csv",index=False)
    all_df.to_csv(out/"search_final_per_target.csv",index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
