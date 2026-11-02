"""Archive provenance, native image ordering and immutable cache behavior."""

import hashlib
import importlib
import io
import json
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch


def api():
    try:
        return importlib.import_module("scripts.tiny_imagenet_data")
    except ModuleNotFoundError:
        pytest.fail("TinyImageNet data implementation is missing")


def toy_archive(tmp_path, *, bad_size=False, missing_annotation=False):
    path = tmp_path / "tiny.zip"
    with zipfile.ZipFile(path, "w") as archive:
        prefix = "tiny-imagenet-200/"
        archive.writestr(prefix + "wnids.txt", "n9\nn1\n")
        for wnid, base in [("n9", 10), ("n1", 30)]:
            for name, offset in [("b", 1), ("a", 0)]:
                stream = io.BytesIO()
                size = (7, 8) if bad_size and wnid == "n9" and name == "a" else (8, 8)
                Image.new("L" if name == "a" else "RGB", size, base + offset).save(stream, format="PNG")
                archive.writestr(prefix + f"train/{wnid}/images/{name}.JPEG", stream.getvalue())
        for name, value in [("val_b", 50), ("val_a", 60)]:
            stream = io.BytesIO()
            Image.new("L", (8, 8), value).save(stream, format="PNG")
            archive.writestr(prefix + f"val/images/{name}.JPEG", stream.getvalue())
        rows = "val_b.JPEG\tn9\t0\t0\t7\t7\n"
        if not missing_annotation:
            rows += "val_a.JPEG\tn1\t0\t0\t7\t7\n"
        archive.writestr(prefix + "val/val_annotations.txt", rows)
        archive.writestr(prefix + "test/images/unused.JPEG", b"unused official test")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(tmp_path, **kwargs):
    archive, digest = toy_archive(tmp_path, **kwargs)
    output = tmp_path / "cache"
    manifest = api().prepare_archive(archive, output, expected_sha256=digest,
                                    classes=2, train_per_class=2,
                                    validation_per_class=1, image_size=8)
    return archive, digest, output, manifest


def test_native_memmaps_follow_wnids_train_names_and_validation_annotation(tmp_path):
    _, digest, output, manifest = prepare(tmp_path)
    assert manifest["archive_sha256"] == digest
    assert manifest["wnids"] == ["n9", "n1"]
    assert manifest["classes"] == 2
    assert manifest["counts"] == {"train": 4, "validation": 2}
    assert manifest["image_size"] == 8
    expected_files = {"train_images.npy", "train_labels.npy", "validation_images.npy", "validation_labels.npy"}
    assert set(manifest["files"]) == expected_files
    assert {p.name for p in output.iterdir()} == expected_files | {"manifest.json"}
    assert json.loads((output / "manifest.json").read_text()) == manifest
    for name, checksum in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == checksum
    train = api().load_split(output, "train")
    validation = api().load_split(output, "validation")
    assert train.images.shape == (4, 3, 8, 8)
    assert train.images.dtype == torch.uint8
    assert train.labels.dtype == torch.int64
    assert train.labels.tolist() == [0, 0, 1, 1]
    assert train.images[:, 0, 0, 0].tolist() == [10, 11, 30, 31]
    assert torch.equal(train.images[0, 0], train.images[0, 1])
    assert validation.labels.tolist() == [1, 0]
    assert validation.images[:, 0, 0, 0].tolist() == [60, 50]
    assert train.mean == (0., 0., 0.) and train.std == (1., 1., 1.)
    # writable tensors must be backed by copy-on-write mappings, never change cache.
    train.images[0, 0, 0, 0] = 99
    train.labels[0] = 1
    assert api().load_split(output, "train").images[0, 0, 0, 0].item() == 10
    assert api().load_split(output, "train").labels[0].item() == 0


def test_reuse_verifies_provenance_configuration_and_every_array(tmp_path):
    archive, digest, output, manifest = prepare(tmp_path)
    manifest_mtime = (output / "manifest.json").stat().st_mtime_ns
    assert api().prepare_archive(archive, output, expected_sha256=digest, classes=2,
                                 train_per_class=2, validation_per_class=1, image_size=8) == manifest
    assert (output / "manifest.json").stat().st_mtime_ns == manifest_mtime
    with pytest.raises(ValueError, match="configuration|contract"):
        api().prepare_archive(archive, output, expected_sha256=digest, classes=2,
                              train_per_class=3, validation_per_class=1, image_size=8)
    with (output / "train_images.npy").open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"X")
    corrupted = (output / "train_images.npy").read_bytes()
    with pytest.raises(ValueError, match="checksum|corrupt"):
        api().load_split(output, "train")
    with pytest.raises(ValueError, match="checksum|corrupt"):
        api().prepare_archive(archive, output, expected_sha256=digest, classes=2,
                              train_per_class=2, validation_per_class=1, image_size=8)
    assert (output / "train_images.npy").read_bytes() == corrupted


def test_checksum_failure_has_no_completed_cache(tmp_path):
    archive, _ = toy_archive(tmp_path)
    output = tmp_path / "cache"
    with pytest.raises(ValueError, match="checksum"):
        api().prepare_archive(archive, output, expected_sha256="0" * 64,
                              classes=2, train_per_class=2, validation_per_class=1, image_size=8)
    assert not output.exists()


@pytest.mark.parametrize("kwargs,match", [({"bad_size": True}, "size|shape"),
                                          ({"missing_annotation": True}, "annotation")])
def test_invalid_native_image_or_annotation_never_completes(tmp_path, kwargs, match):
    archive, digest = toy_archive(tmp_path, **kwargs)
    output = tmp_path / "cache"
    with pytest.raises(ValueError, match=match):
        api().prepare_archive(archive, output, expected_sha256=digest, classes=2,
                              train_per_class=2, validation_per_class=1, image_size=8)
    assert not output.exists()
    assert not list(tmp_path.glob(".cache.*"))


def test_load_rejects_incomplete_cache_and_unknown_split(tmp_path):
    output = tmp_path / "cache"
    output.mkdir()
    with pytest.raises(ValueError, match="manifest|incomplete"):
        api().load_split(output, "train")
    with pytest.raises(ValueError, match="split"):
        api().load_split(output, "test")
