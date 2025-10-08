import json
from pathlib import Path

import pandas as pd

from models.meta import stack_scores, train_stacker


def test_train_stacker_linear_weights(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL"],
            "equities_xs": [0.1, -0.2, 0.3, -0.1, 0.4],
            "intraday_xs": [0.05, -0.05, 0.08, -0.02, 0.1],
        }
    )
    y = pd.Series([0.12, -0.15, 0.25, -0.05, 0.3])

    cfg = {
        "algorithm": "linear_weights",
        "artifact_dir": tmp_path.as_posix(),
        "linear_weights": {"equities_xs": 0.7, "intraday_xs": 0.3},
    }
    result = train_stacker(df, y, cfg)
    metadata_path = Path(result["metadata_path"])
    assert metadata_path.exists()
    meta = json.loads(metadata_path.read_text())
    assert meta["weights"]["equities_xs"] == 0.7

    stacked = stack_scores(df[["equities_xs", "intraday_xs"]], meta["weights"])
    assert len(stacked) == len(df)
    assert float(stacked.mean()) != 0.0
