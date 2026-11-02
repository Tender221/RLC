"""Fixed training/audit splits and index-only, replayable reservoir memory."""

import copy
import hashlib
import os
import pickle
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

CIFAR_ORIGINAL_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_URL = "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz"
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"


@dataclass
class ImageData:
    images: torch.Tensor
    labels: torch.Tensor
    source: str
    mean: tuple = (0.4914, 0.4822, 0.4465)
    std: tuple = (0.2470, 0.2435, 0.2616)

    def fingerprint(self):
        digest = hashlib.sha256()
        digest.update(self.images.numpy().tobytes())
        digest.update(self.labels.numpy().tobytes())
        return digest.hexdigest()


def read_cifar_archive(path, expected_md5=CIFAR_MD5):
    """Read training entries only, after validating the trusted archive digest.

    CIFAR's official Python archive uses pickle. Do not pass arbitrary untrusted
    archives or a digest taken from an untrusted supplier. No files are extracted.
    """
    path = Path(path)
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_md5:
        raise ValueError("CIFAR archive checksum mismatch")
    images, labels = [], []
    with tarfile.open(path, "r:gz") as archive:
        for number in range(1, 6):
            with archive.extractfile(f"cifar-10-batches-py/data_batch_{number}") as handle:
                batch = pickle.load(handle, encoding="latin1")
            images.append(np.asarray(batch["data"], dtype=np.uint8).reshape(-1, 3, 32, 32))
            labels.extend(batch["labels"])
    return ImageData(torch.from_numpy(np.concatenate(images)), torch.tensor(labels, dtype=torch.long),
                     "cifar10_official_train_only")


def load_cifar(root, download=False):
    root = Path(root)
    path = root / "cifar-10-python.tar.gz"
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Missing {path}; use --download to fetch the official archive")
        root.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".gz.part")
        print("Downloading CIFAR-10 archive from a mirror; the standard archive checksum is required", flush=True)
        with urllib.request.urlopen(CIFAR_URL, timeout=45) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        # Validation precedes promotion to the stable cache filename.
        bundle = read_cifar_archive(temporary)
        os.replace(temporary, path)
        return bundle
    return read_cifar_archive(path)


def load_cifar100(root):
    """Read only official fine-label training data, validating archive and member."""
    import io
    path = Path(root) / "cifar-100-python.tar.gz"
    if hashlib.md5(path.read_bytes()).hexdigest() != "eb9058c3a382ffc7106e4002c42a8d85":
        raise ValueError("CIFAR100 archive checksum mismatch")
    with tarfile.open(path, "r:gz") as archive:
        raw = archive.extractfile("cifar-100-python/train").read()
    if hashlib.md5(raw).hexdigest() != "16019d7e3df5f24257cddd939b257f8d":
        raise ValueError("CIFAR100 train member checksum mismatch")
    batch = pickle.load(io.BytesIO(raw), encoding="latin1")
    images = np.asarray(batch['data'], dtype=np.uint8).reshape(-1, 3, 32, 32)
    labels = torch.tensor(batch['fine_labels'], dtype=torch.long)
    if images.shape != (50000, 3, 32, 32) or not torch.equal(torch.bincount(labels), torch.full((100,), 500)):
        raise ValueError("Unexpected CIFAR100 fine-label training data")
    return ImageData(torch.from_numpy(images), labels, "cifar100_official_train_only")


def training_normalization(bundle, train, batch_size=256):
    """Population channel statistics from training ids; integer sums are chunk invariant."""
    if batch_size < 1 or len(train) == 0:
        raise ValueError("Normalization needs nonempty training ids and a positive batch size")
    total = torch.zeros(3, dtype=torch.int64)
    squares = torch.zeros(3, dtype=torch.int64)
    count = 0
    for chunk in train.split(batch_size):
        pixels = bundle.images[chunk].to(torch.int64)
        total += pixels.sum((0, 2, 3))
        squares += (pixels * pixels).sum((0, 2, 3))
        count += len(chunk) * pixels.shape[2] * pixels.shape[3]
    mean = total.double() / count / 255.
    std = (squares.double() / count / 255.**2 - mean.square()).clamp_min(0).sqrt()
    if bool((std == 0).any()):
        raise ValueError("Training channel has zero variance")
    return tuple(mean.tolist()), tuple(std.tolist())


def synthetic_data(per_class=16, num_classes=10):
    generator = torch.Generator().manual_seed(918)
    labels = torch.arange(num_classes).repeat_interleave(per_class)
    images = torch.randint(0, 256, (len(labels), 3, 32, 32), dtype=torch.uint8, generator=generator)
    return ImageData(images, labels, "synthetic_technical_only")


def split_indices(labels, audit_per_class, seed):
    generator = np.random.default_rng(seed)
    train, audit = [], []
    if audit_per_class < 1:
        raise ValueError("audit_per_class must be positive")
    for label in sorted(labels.unique().tolist()):
        ids = torch.where(labels == label)[0].numpy().copy()
        if len(ids) <= audit_per_class:
            raise ValueError("Each class must retain training examples after the audit split")
        generator.shuffle(ids)
        audit.extend(ids[:audit_per_class].tolist())
        train.extend(ids[audit_per_class:].tolist())
    return torch.tensor(train, dtype=torch.long), torch.tensor(audit, dtype=torch.long)


def make_stream(labels, train_indices, order, batch_size, seed, task_count=5, max_per_task=0, classes_per_task=2):
    if (sorted(order) != list(range(len(order))) or classes_per_task < 1
            or len(order) % classes_per_task or not 1 <= task_count <= len(order)//classes_per_task or batch_size < 1):
        raise ValueError("Expected a class permutation, valid equal task groups and positive batch size")
    generator = np.random.default_rng(seed)
    stream = []
    for task in range(task_count):
        classes = torch.tensor(order[task * classes_per_task:(task + 1) * classes_per_task])
        ids = train_indices[torch.isin(labels[train_indices], classes)].numpy().copy()
        generator.shuffle(ids)
        if max_per_task > 0:
            ids = ids[:max_per_task]
        stream.extend((task, ids[start:start + batch_size].tolist())
                      for start in range(0, len(ids), batch_size))
    return stream


class IndexReplay:
    """Reservoir contents and sampling use a private, checkpointed RNG."""

    def __init__(self, capacity, seed):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.slots = []
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def add(self, indices):
        for index in indices:
            self.seen += 1
            if len(self.slots) < self.capacity:
                self.slots.append(int(index))
            else:
                position = int(self.rng.integers(self.seen))
                if position < self.capacity:
                    self.slots[position] = int(index)

    def sample(self, count):
        count = min(count, len(self.slots))
        if count <= 0:
            return []
        positions = self.rng.choice(len(self.slots), size=count, replace=False)
        return [self.slots[int(position)] for position in positions]

    def state_dict(self):
        return copy.deepcopy({"capacity": self.capacity, "slots": self.slots,
                              "seen": self.seen, "rng": self.rng.bit_generator.state})

    def load_state_dict(self, state):
        self.capacity = state["capacity"]
        self.slots = list(state["slots"])
        self.seen = state["seen"]
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
