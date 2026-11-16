import json

import numpy as np
import pytest
import torch

from common.provenance import digest
from generator.conditioned import checkpoint_payload, create_model
from generator.conditioned_cli import sample_recipe


def test_recipe_cli_conditions_on_pose_and_writes_native_sample(tmp_path):
    torch.set_num_threads(1)
    checkpoint, out = tmp_path / "model.pt", tmp_path / "sample.npz"
    model = create_model("residual", config=dict(d=16, heads=2, layers=1, maxlen=80))
    stats = dict(MU=[0.] * 9, SD=[1.] * 9, PMU=[0.] * 8, PSD=[1.] * 8)
    torch.save(checkpoint_payload(model, stats), checkpoint)
    origin = dict(roll=0., pitch=-4., yaw=10., body=5., right=-10., left=10., x=.001, y=0., z=0.)
    result = sample_recipe(checkpoint, "go 1 p=6 E=0 | hold .5 E=0", origin, out, steps=2)
    with np.load(out, allow_pickle=False) as sample:
        assert sample["trajectory"].shape[1] == 9 and np.isfinite(sample["trajectory"]).all()
        assert sample["origin"][2] == 10.
    receipt = json.loads(out.with_suffix(".json").read_text())
    assert receipt == result and receipt["trajectory_sha256"] == digest(out)
    assert receipt["origin"] == origin and not receipt["anchored_to_origin"]
    with pytest.raises(FileExistsError):
        sample_recipe(checkpoint, "hold 1 E=0", origin, out, steps=2)
    with pytest.raises(ValueError, match="nine"):
        sample_recipe(checkpoint, "hold 1 E=0", {}, tmp_path / "invalid.npz", steps=2)
