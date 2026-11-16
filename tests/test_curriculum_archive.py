import io
import json
import tarfile

import numpy as np
import pytest
import torch

from common.provenance import digest
from generator.curriculum import extract_archive
from generator.losses import ClearanceSurrogate


def test_archive_extraction_checks_all_members_and_preserves_existing_data(tmp_path):
    source = tmp_path / "one.npz"
    np.savez(source, X=np.zeros((2, 9)), Q=np.zeros((2, 8)), B=np.zeros((2, 9)))
    hashes = {"arrays/one.npz": digest(source)}
    (tmp_path / "manifest.json").write_text(json.dumps(dict(files=hashes)))
    (tmp_path / "trajectory-hashes.json").write_text(json.dumps(hashes))
    with tarfile.open(tmp_path / "trajectories.tar.gz", "w:gz") as archive:
        archive.add(source, arcname="arrays/one.npz")
    assert extract_archive(tmp_path) == 1
    assert digest(tmp_path / "arrays/one.npz") == digest(source)
    with pytest.raises(FileExistsError):
        extract_archive(tmp_path)


@pytest.mark.parametrize("case", ["traversal", "symlink", "changed", "extra"])
def test_unsafe_or_unbound_archive_rejects_atomically(tmp_path, case):
    name = "../escape.npz" if case == "traversal" else "arrays/one.npz"
    hashes = {name: "0" * 64}
    (tmp_path / "manifest.json").write_text(json.dumps(dict(files=hashes)))
    (tmp_path / "trajectory-hashes.json").write_text(json.dumps(hashes))
    with tarfile.open(tmp_path / "trajectories.tar.gz", "w:gz") as archive:
        entry = tarfile.TarInfo(name)
        if case == "symlink":
            entry.type, entry.linkname = tarfile.SYMTYPE, "/tmp/escape"
            archive.addfile(entry)
        else:
            entry.size = 3
            archive.addfile(entry, io.BytesIO(b"bad"))
        if case == "extra":
            archive.addfile(tarfile.TarInfo("arrays/extra.npz"))
    with pytest.raises(ValueError):
        extract_archive(tmp_path)
    assert not (tmp_path / "arrays").exists()


def test_external_clearance_table_is_required_validated_and_differentiable(tmp_path):
    path = tmp_path / "table.npz"
    axes = {"rel_yaw": [0., 90.], "pitch": [-20., 20.], "roll": [-20., 20.],
            "z_mm": [-10., 0.], "x_mm": [-10., 10.]}
    np.savez(path, **axes, gap_decimm=np.full((2,) * 5, 50.))
    table = ClearanceSurrogate(path)
    world = torch.tensor([[0., 0., 2., 0., -10., 10., .001, .001, -.001]], requires_grad=True)
    gap, domain = table.gap(world)
    torch.testing.assert_close(gap, torch.tensor([4.8]))
    assert domain.item() == 0
    gap.sum().backward()
    assert torch.isfinite(world.grad).all() and world.grad.abs().sum() > 0
    np.savez(path, **{**axes, "pitch": [20., -20.]}, gap_decimm=np.full((2,) * 5, 50.))
    with pytest.raises(ValueError, match="uniform increasing"):
        ClearanceSurrogate(path)
