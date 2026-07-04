from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def make_submission(test_df: pd.DataFrame, preds, path=None, clip_negative: bool = True) -> pd.DataFrame:
    preds = np.asarray(preds, dtype=float)
    if len(preds) != len(test_df):
        raise ValueError(f"preds length {len(preds)} != test rows {len(test_df)}")
    if clip_negative:
        preds = np.clip(preds, 0, None)

    dates = pd.to_datetime(test_df["Date"]).dt.strftime("%Y-%m-%d")
    ids = test_df["Store"].astype(str) + "_" + test_df["Dept"].astype(str) + "_" + dates
    sub = pd.DataFrame({"Id": ids.to_numpy(), "Weekly_Sales": preds})

    out_path = path or (config.WORKING_DIR / "submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"Wrote {len(sub):,} rows -> {out_path}")
    return sub
