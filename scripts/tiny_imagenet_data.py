"""Checksum-verified TinyImageNet arrays, decoded directly from the official ZIP.

The cache is published by one directory rename only after every array is flushed
and hashed. Existing caches are verified and never silently rebuilt. No archive
members are extracted, and no image is resized.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from PIL import Image
import torch

from wnet.data import ImageData


_FILES = tuple(f"{split}_{kind}.npy" for split in ("train", "validation")
               for kind in ("images", "labels"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(output):
    try:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        classes = manifest["classes"]
        image_size = manifest["image_size"]
        wnids = manifest["wnids"]
        if (manifest["version"] != 1 or not isinstance(classes, int) or classes < 1
                or not isinstance(image_size, int) or image_size < 1
                or not isinstance(wnids, list) or len(wnids) != classes
                or any(not isinstance(value, str) or not value for value in wnids)
                or len(set(wnids)) != classes or set(manifest["files"]) != set(_FILES)
                or len(manifest["archive_sha256"]) != 64):
            raise ValueError("Invalid cache manifest contract")
        for split in ("train", "validation"):
            count = manifest[f"{split}_per_class"]
            if not isinstance(count, int) or count < 1 or manifest["counts"][split] != classes * count:
                raise ValueError("Invalid cache manifest counts")
        return manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Missing, incomplete or corrupt cache manifest: {output}") from error


def _verify_cache(output, manifest, *, verify):
    """Validate array layout/label balance as well as their persisted hashes."""
    for name in _FILES:
        path = output / name
        if not path.is_file():
            raise ValueError(f"Incomplete cache: missing {name}")
        if verify and _sha256(path) != manifest["files"][name]:
            raise ValueError(f"Cache checksum mismatch: {name}")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            try:
                split = name.split("_", 1)[0]
                count = manifest["counts"][split]
                if name.endswith("images.npy"):
                    expected = (count, 3, manifest["image_size"], manifest["image_size"])
                    valid = array.dtype == np.uint8 and array.shape == expected
                else:
                    valid = (array.dtype == np.int64 and array.shape == (count,)
                             and bool(np.all((array >= 0) & (array < manifest["classes"]))))
                    if valid:
                        valid = bool(np.all(np.bincount(array, minlength=manifest["classes"])
                                            == manifest[f"{split}_per_class"]))
                if not valid:
                    raise ValueError(f"Corrupt cache array shape, dtype or labels: {name}")
            finally:
                array._mmap.close()
        except (OSError, EOFError, TypeError) as error:
            raise ValueError(f"Corrupt cache array: {name}") from error


def _archive_index(archive, *, classes, train_per_class, validation_per_class):
    names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate ZIP member names")
    roots = [name[:-len("wnids.txt")] for name in names
             if name == "wnids.txt" or name.endswith("/wnids.txt")]
    if len(roots) != 1:
        raise ValueError("Archive must contain one wnids.txt")
    root = roots[0]
    wnids = archive.read(root + "wnids.txt").decode("utf-8").splitlines()
    if (len(wnids) != classes or len(set(wnids)) != classes
            or any(not name or "/" in name or "\\" in name for name in wnids)):
        raise ValueError("Unexpected wnids class mapping")
    mapping = {wnid: index for index, wnid in enumerate(wnids)}
    train = []
    train_prefix = root + "train/"
    image_members = {name for name in names if name.startswith(train_prefix)
                     and "/images/" in name}
    for wnid in wnids:
        prefix = f"{train_prefix}{wnid}/images/"
        members = sorted(name for name in image_members if name.startswith(prefix))
        if len(members) != train_per_class or any("/" in name[len(prefix):] for name in members):
            raise ValueError(f"Unexpected train image count/layout for {wnid}")
        train.extend((name, mapping[wnid]) for name in members)
    if len(image_members) != len(train):
        raise ValueError("Unexpected train image classes")
    annotations = {}
    for row in archive.read(root + "val/val_annotations.txt").decode("utf-8").splitlines():
        fields = row.split()
        if len(fields) != 6 or fields[0] in annotations or fields[1] not in mapping:
            raise ValueError("Invalid or duplicate validation annotation")
        annotations[fields[0]] = mapping[fields[1]]
    prefix = root + "val/images/"
    members = sorted(name for name in names if name.startswith(prefix))
    if (any("/" in name[len(prefix):] for name in members)
            or {name[len(prefix):] for name in members} != set(annotations)):
        raise ValueError("Validation images and annotations do not match")
    validation = [(name, annotations[name[len(prefix):]]) for name in members]
    if (len(validation) != classes * validation_per_class
            or np.bincount([label for _, label in validation], minlength=classes).tolist()
            != [validation_per_class] * classes):
        raise ValueError("Unexpected validation annotation class counts")
    return wnids, {"train": train, "validation": validation}


def _write_split(archive, destination, split, entries, image_size):
    images = np.lib.format.open_memmap(destination / f"{split}_images.npy", mode="w+",
                                       dtype=np.uint8, shape=(len(entries), 3, image_size, image_size))
    labels = None
    try:
        labels = np.lib.format.open_memmap(destination / f"{split}_labels.npy", mode="w+",
                                           dtype=np.int64, shape=(len(entries),))
        for index, (name, label) in enumerate(entries):
            with archive.open(name) as member, Image.open(member) as original:
                if original.size != (image_size, image_size):
                    raise ValueError(f"Unexpected native image size {original.size}: {name}")
                images[index] = np.asarray(original.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)
            labels[index] = label
            if (index + 1) % 10000 == 0 or index + 1 == len(entries):
                print(f"TinyImageNet {split}: {index + 1}/{len(entries)} decoded", flush=True)
        images.flush()
        labels.flush()
    finally:
        images._mmap.close()
        if labels is not None:
            labels._mmap.close()


def prepare_archive(archive, output, *, expected_sha256, classes=200,
                    train_per_class=500, validation_per_class=50, image_size=64):
    """Verify/decode official ZIP to an atomic, reusable native-pixel cache.

    A completed corrupt cache raises instead of overwriting evidence. An existing
    incomplete output directory also raises; the caller can explicitly relocate
    it before retrying. Temporary staging directories are cleaned on failure.
    """
    archive, output = Path(archive), Path(output)
    contract = {"classes": classes, "train_per_class": train_per_class,
                "validation_per_class": validation_per_class, "image_size": image_size}
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
           for value in contract.values()):
        raise ValueError("Cache configuration must contain positive integers")
    actual_sha256 = _sha256(archive)
    if actual_sha256 != expected_sha256:
        raise ValueError("TinyImageNet archive checksum mismatch")
    if output.exists():
        manifest = _read_manifest(output)
        if (manifest["archive_sha256"] != actual_sha256
                or any(manifest[key] != value for key, value in contract.items())):
            raise ValueError("Existing cache provenance/configuration contract mismatch")
        _verify_cache(output, manifest, verify=True)
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as staging:
        destination = Path(staging) / "complete"
        destination.mkdir()
        with zipfile.ZipFile(archive) as source:
            wnids, splits = _archive_index(source, classes=classes, train_per_class=train_per_class,
                                           validation_per_class=validation_per_class)
            for split, entries in splits.items():
                _write_split(source, destination, split, entries, image_size)
        manifest = {"version": 1, "archive_sha256": actual_sha256, "wnids": wnids,
                    **contract, "counts": {split: len(entries) for split, entries in splits.items()},
                    "files": {name: _sha256(destination / name) for name in _FILES}}
        with (destination / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _verify_cache(destination, manifest, verify=True)
        destination.rename(output)
    return manifest


def load_split(output, split, verify=True):
    """Return writable copy-on-write mmap tensors with neutral normalization."""
    if split not in ("train", "validation"):
        raise ValueError("split must be train or validation")
    output = Path(output)
    manifest = _read_manifest(output)
    _verify_cache(output, manifest, verify=verify)
    images = np.load(output / f"{split}_images.npy", mmap_mode="c", allow_pickle=False)
    labels = np.load(output / f"{split}_labels.npy", mmap_mode="c", allow_pickle=False)
    return ImageData(torch.from_numpy(images), torch.from_numpy(labels),
                     f"tiny_imagenet_official_{split}", mean=(0., 0., 0.), std=(1., 1., 1.))
