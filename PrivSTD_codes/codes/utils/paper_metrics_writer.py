
import os
from typing import Dict, Any, Optional, Sequence, List, Tuple
import pandas as pd

def _to_row_range(epoch: int, res: Dict[str, Any]) -> dict:
    return {
        "epoch": epoch,
        "RE@sm_mean": res.get("RE@sm_mean", float("nan")),
    }

def _to_rows_hotspots(epoch: int, res: Dict[str, Any]) -> dict:
    return {
        "epoch": epoch,
        "dist_mae_m_mean": res.get("dist_mae_m_mean", float("nan")),
        "regret_abs_mean": res.get("regret_abs_mean", float("nan")),
        "regret_rel_mean": res.get("regret_rel_mean", float("nan")),
        "topk": res.get("config", {}).get("topk", None),
        "meters_per_cell": res.get("config", {}).get("meters_per_cell", None),
        "match_mode": res.get("config", {}).get("match_mode", None),
    }

def _to_rows_forecast(epoch: int, res: Dict[str, Any]) -> dict:
    return {
        "epoch": epoch,
        "sMAPE_mean": res.get("sMAPE_mean", float("nan")),
        "horizon": res.get("config", {}).get("horizon", None),
        "num_cells": res.get("config", {}).get("num_cells", None),
    }

def _to_df_range_by_window(epoch: int, items: List[Tuple[tuple, float]]) -> pd.DataFrame:
    rows = []
    for (wx, wy, wt), val in items:
        rows.append({"epoch": epoch, "window": f"{wx}x{wy}x{wt}", "RE@sm_mean": val})
    return pd.DataFrame(rows)

def write_metrics_xlsx(
    xlsx_path: str,
    epoch: int,
    range_re: Dict[str, Any],
    hotspots: Dict[str, Any],
    forecast: Dict[str, Any],
    range_by_window: Optional[List[Tuple[tuple, float]]] = None,
    engine: str = "openpyxl",
) -> str:
    """
    Append metrics to an Excel workbook with three sheets:
      - 'range_re' (RE@sm_mean)
      - 'hotspots' (distance/regret means)
      - 'forecast' (sMAPE mean)
    And optionally 'range_by_window' (per-window RE means).

    If the file doesn't exist, it's created with headers.
    Returns the absolute path to the written file.
    """
    os.makedirs(os.path.dirname(xlsx_path), exist_ok=True)

    # Prepare rows
    row_range = _to_row_range(epoch, range_re)
    row_hot = _to_rows_hotspots(epoch, hotspots)
    row_fc  = _to_rows_forecast(epoch, forecast)

    # Existing file?
    file_exists = os.path.isfile(xlsx_path)

    if not file_exists:
        with pd.ExcelWriter(xlsx_path, engine=engine, mode="w") as writer:
            pd.DataFrame([row_range]).to_excel(writer, sheet_name="range_re", index=False)
            pd.DataFrame([row_hot]).to_excel(writer, sheet_name="hotspots", index=False)
            pd.DataFrame([row_fc]).to_excel(writer, sheet_name="forecast", index=False)
            if range_by_window is not None:
                _to_df_range_by_window(epoch, range_by_window).to_excel(writer, sheet_name="range_by_window", index=False)
    else:
        with pd.ExcelWriter(xlsx_path, engine=engine, mode="a", if_sheet_exists="overlay") as writer:
            # range_re
            try:
                existing = pd.read_excel(xlsx_path, sheet_name="range_re")
                df = pd.concat([existing, pd.DataFrame([row_range])], ignore_index=True)
            except Exception:
                df = pd.DataFrame([row_range])
            df.to_excel(writer, sheet_name="range_re", index=False)

            # hotspots
            try:
                existing = pd.read_excel(xlsx_path, sheet_name="hotspots")
                df = pd.concat([existing, pd.DataFrame([row_hot])], ignore_index=True)
            except Exception:
                df = pd.DataFrame([row_hot])
            df.to_excel(writer, sheet_name="hotspots", index=False)

            # forecast
            try:
                existing = pd.read_excel(xlsx_path, sheet_name="forecast")
                df = pd.concat([existing, pd.DataFrame([row_fc])], ignore_index=True)
            except Exception:
                df = pd.DataFrame([row_fc])
            df.to_excel(writer, sheet_name="forecast", index=False)

            # range_by_window
            if range_by_window is not None:
                try:
                    existing = pd.read_excel(xlsx_path, sheet_name="range_by_window")
                    df = pd.concat([existing, _to_df_range_by_window(epoch, range_by_window)], ignore_index=True)
                except Exception:
                    df = _to_df_range_by_window(epoch, range_by_window)
                df.to_excel(writer, sheet_name="range_by_window", index=False)

    return os.path.abspath(xlsx_path)
