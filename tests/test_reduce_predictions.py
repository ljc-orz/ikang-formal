from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from scripts.infer.reduce_predictions import (
    discover_layout,
    load_prediction_cube,
    parse_args,
    parse_seed_selection,
    reduce_pca,
    run,
)


def write_prediction(
    root: Path,
    indicator: str,
    seed: int,
    rows: list[int],
    values: dict[int, float],
) -> None:
    directory = root / indicator
    directory.mkdir(parents=True, exist_ok=True)
    logits = torch.tensor(
        [[values[row], values[row] + 2.0] for row in rows], dtype=torch.float32
    )
    torch.save(
        {
            "source_row": torch.tensor(rows, dtype=torch.int64),
            "logits": logits,
            "eye_order": ("left", "right"),
            "seed": seed,
            "target": f"result_{indicator}",
            "split": "external_validation",
            "transform": "train",
            "checkpoint": f"/models/{indicator}/best.pt",
            "source_parquet": "/data/source.parquet",
        },
        directory / f"result_{indicator}.external_validation.seed-{seed}.pt",
    )


def build_predictions(root: Path) -> None:
    write_prediction(
        root,
        "alt",
        2026,
        [5, 2, 9],
        {2: 2.0, 5: 5.0, 9: 9.0},
    )
    write_prediction(
        root,
        "bmi",
        2026,
        [9, 5, 2],
        {2: 20.0, 5: 50.0, 9: 90.0},
    )
    write_prediction(
        root,
        "alt",
        2027,
        [9, 2, 5],
        {2: 102.0, 5: 105.0, 9: 109.0},
    )
    write_prediction(
        root,
        "bmi",
        2027,
        [2, 9, 5],
        {2: 120.0, 5: 150.0, 9: 190.0},
    )


class ReducePredictionsTests(unittest.TestCase):
    def test_seed_sequence_is_inclusive(self) -> None:
        self.assertEqual(
            parse_seed_selection(("seq", "2026", "2029")),
            (2026, 2027, 2028, 2029),
        )

    def test_load_cube_aligns_patients_by_source_row(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            build_predictions(root)
            layout = discover_layout(
                root, indicators=None, split=None, seeds=None
            )
            loaded = load_prediction_cube(layout, eye="mean")

        self.assertEqual(layout.indicators, ("alt", "bmi"))
        self.assertEqual(layout.seeds, (2026, 2027))
        self.assertTrue(torch.equal(loaded.source_rows, torch.tensor([2, 5, 9])))
        self.assertTrue(
            torch.equal(
                loaded.logits[0],
                torch.tensor([[3.0, 21.0], [6.0, 51.0], [10.0, 91.0]]),
            )
        )

    def test_indicator_argument_controls_feature_order(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            build_predictions(root)
            layout = discover_layout(
                root,
                indicators=("bmi", "alt"),
                split="external_validation",
                seeds=(2026,),
            )
            loaded = load_prediction_cube(layout, eye="left")

        self.assertEqual(layout.indicators, ("bmi", "alt"))
        self.assertTrue(torch.equal(loaded.logits[0, :, 0], torch.tensor([20, 50, 90])))
        self.assertTrue(torch.equal(loaded.logits[0, :, 1], torch.tensor([2, 5, 9])))

    def test_joint_and_per_seed_pca_shapes(self) -> None:
        logits = torch.arange(60, dtype=torch.float32).reshape(2, 10, 3)

        joint = reduce_pca(logits, n_components=2, fit_scope="joint")
        per_seed = reduce_pca(logits, n_components=2, fit_scope="per-seed")

        self.assertEqual(joint.values.shape, (2, 10, 2))
        self.assertEqual(joint.components.shape, (2, 3))
        self.assertEqual(per_seed.values.shape, (2, 10, 2))
        self.assertEqual(per_seed.components.shape, (2, 2, 3))

    def test_run_saves_all_seeds_in_one_weights_only_file(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "predictions"
            build_predictions(root)
            output = Path(directory) / "embeddings" / "pca.pt"
            args = parse_args(
                [
                    "--input-dir",
                    str(root),
                    "--output-file",
                    str(output),
                    "--eye",
                    "right",
                    "--n-components",
                    "2",
                    "--pca-fit",
                    "mean",
                ]
            )

            run(args)
            result = torch.load(output, map_location="cpu", weights_only=True)

        self.assertEqual(result["X_prime"].shape, (2, 3, 2))
        self.assertTrue(torch.equal(result["seeds"], torch.tensor([2026, 2027])))
        self.assertTrue(torch.equal(result["source_row"], torch.tensor([2, 5, 9])))
        self.assertEqual(result["indicators"], ("alt", "bmi"))
        self.assertEqual(result["eye"], "right")
        self.assertEqual(result["reducer"]["fit_scope"], "mean")


if __name__ == "__main__":
    unittest.main()
