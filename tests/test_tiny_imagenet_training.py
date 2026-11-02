import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from tests.test_paired import assert_tree_equal
from wnet.data import IndexReplay, make_stream, split_indices, synthetic_data


def trainer():
    import scripts.tiny_imagenet_training as module
    return module


def config(**changes):
    return trainer().TinyConfig(
        dataset='synthetic', num_classes=4, classes_per_task=2, task_count=2,
        synthetic_per_class=8, audit_per_class=2, audit_probe_per_class=2,
        batch_size=4, replay_batch_size=4, memory_size=7, model='small-cnn',
        device='cpu', image_size=32, padding=4, checkpoint_every=2, threads=1,
        **changes,
    )


def load(path):
    return torch.load(path / 'checkpoint.pt', map_location='cpu', weights_only=False)


def test_tiny_defaults_are_frozen():
    cfg = trainer().TinyConfig()
    expected = dict(dataset='tinyimagenet', num_classes=200, classes_per_task=10,
                    task_count=20, audit_per_class=25, memory_size=500,
                    model='resnet18', device='cuda', seed=5110, order_seed=511,
                    learning_rate=.05, momentum=.9, weight_decay=0.,
                    max_diagnostics=0, checkpoint_every=300, threads=2,
                    image_size=64, padding=8, evaluation_batch_size=128,
                    boundary_correction=False)
    assert all(getattr(cfg, key) == value for key, value in expected.items())
    with pytest.raises(FrozenInstanceError):
        cfg.image_size = 32


def test_views_match_frozen_cifar_and_native_64_edge_crops():
    from scripts.augmented_policy import apply_views as old_views, view_parameters as old_parameters
    module = trainer()
    cfg = config()
    x = torch.arange(3 * 3 * 32 * 32, dtype=torch.float32).reshape(3, 3, 32, 32)
    before_torch, before_numpy = torch.get_rng_state(), np.random.get_state()
    parameters = module.view_parameters(cfg, 7, len(x))
    assert parameters == old_parameters(cfg, 7, len(x))
    assert_tree_equal(module.apply_views(x, parameters, image_size=32, padding=4), old_views(x, parameters))
    x = torch.arange(2 * 3 * 64 * 64, dtype=torch.float32).reshape(2, 3, 64, 64)
    padded = F.pad(x, (8, 8, 8, 8), mode='reflect')
    expected = torch.stack([padded[0, :, 0:64, 16:80], padded[1, :, 16:80, 0:64].flip(-1)])
    assert_tree_equal(module.apply_views(x, [[0, 16, 0], [16, 0, 1]]), expected)
    parameters = np.asarray(module.view_parameters(replace(cfg, image_size=64, padding=8), 3, 1000))
    assert parameters[:, :2].min() == 0 and parameters[:, :2].max() == 16
    assert set(parameters[:, 2]) == {0, 1}
    assert_tree_equal(before_torch, torch.get_rng_state())
    assert_tree_equal(before_numpy, np.random.get_state())
    with pytest.raises(ValueError):
        module.apply_views(x, [[17, 0, 0], [0, 0, 0]])


def test_synthetic_32_training_exactly_matches_existing_augmented_er(tmp_path):
    from scripts.augmented_policy import AugmentedPolicy
    from wnet.pilot import run
    module, cfg = trainer(), config()
    bundle = synthetic_data(cfg.synthetic_per_class, cfg.num_classes)
    old, new = tmp_path / 'old', tmp_path / 'new'
    run(cfg, old, policy=AugmentedPolicy())
    summary = module.train_er(cfg, bundle, new)
    a, b = load(old), load(new)
    for key in ('training_state', 'records', 'replay'):
        assert_tree_equal(a[key], b[key])
    assert summary['complete'] and summary['steps'] == len(b['records'])
    assert summary['training_seconds'] > 0 and summary['evaluation_seconds'] >= 0
    assert summary['peak_cuda_bytes'] is None


def test_native64_resume_is_exact_and_saves_final_preinsertion_memory(tmp_path):
    module = trainer()
    cfg = replace(config(), image_size=64, padding=8)
    bundle = synthetic_data(cfg.synthetic_per_class, cfg.num_classes)
    bundle.images = bundle.images.repeat_interleave(2, -1).repeat_interleave(2, -2)
    bundle.mean, bundle.std = (.2, .3, .4), (.8, .7, .6)
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    module.train_er(cfg, bundle, full)
    partial = module.train_er(cfg, bundle, resumed, stop_after=3)
    assert not partial['complete'] and partial['steps'] == 3
    same = module.train_er(cfg, bundle, resumed, stop_after=3, resume=True)
    assert same['steps'] == 3
    module.train_er(cfg, bundle, resumed, resume=True)
    a, b = load(full), load(resumed)
    for key in ('training_state', 'replay', 'records', 'pre_final_memory_indices', 'metadata'):
        assert_tree_equal(a[key], b[key])
    assert a['metadata'] == json.loads((full / 'metadata.json').read_text())
    assert a['metadata']['normalization']['mean'] == list(bundle.mean)
    train, _ = split_indices(bundle.labels, cfg.audit_per_class, cfg.split_seed)
    stream = make_stream(bundle.labels, train, a['metadata']['class_order'], cfg.batch_size,
                         cfg.seed, cfg.task_count, cfg.max_train_per_task, cfg.classes_per_task)
    memory = IndexReplay(cfg.memory_size, cfg.seed + 700)
    for _, ids in stream[:-1]:
        memory.sample(cfg.replay_batch_size)
        memory.add(ids)
    assert a['pre_final_memory_indices'] == memory.slots
    memory.sample(cfg.replay_batch_size)
    memory.add(stream[-1][1])
    assert_tree_equal(memory.state_dict(), a['replay'])


def test_supplied_fingerprint_avoids_whole_dataset_copy(tmp_path, monkeypatch):
    module, cfg = trainer(), config()
    bundle = synthetic_data(cfg.synthetic_per_class, cfg.num_classes)
    def forbidden():
        raise AssertionError('Provided fingerprint must bypass ImageData.fingerprint')
    monkeypatch.setattr(bundle, 'fingerprint', forbidden)
    module.train_er(cfg, bundle, tmp_path, stop_after=0, data_fingerprint='prepared-array-sha')
    assert load(tmp_path)['metadata']['data_fingerprint'] == 'prepared-array-sha'


@pytest.mark.parametrize('field', ['config', 'data_fingerprint', 'environment', 'training_code_sha256', 'normalization'])
def test_resume_rejects_changed_metadata(tmp_path, field):
    module, cfg = trainer(), config()
    bundle = synthetic_data(cfg.synthetic_per_class, cfg.num_classes)
    module.train_er(cfg, bundle, tmp_path, stop_after=1, data_fingerprint='frozen-data')
    saved = load(tmp_path)
    saved['metadata'][field] = {'changed': True}
    torch.save(saved, tmp_path / 'checkpoint.pt')
    with pytest.raises(ValueError, match='Resume'):
        module.train_er(cfg, bundle, tmp_path, resume=True, data_fingerprint='frozen-data')


def test_nondefault_config_change_rejected_and_existing_checkpoint_protected(tmp_path):
    module, cfg = trainer(), config()
    bundle = synthetic_data(cfg.synthetic_per_class, cfg.num_classes)
    module.train_er(cfg, bundle, tmp_path, stop_after=1)
    with pytest.raises(ValueError, match='Resume'):
        module.train_er(replace(cfg, learning_rate=.1), bundle, tmp_path, resume=True)
    with pytest.raises(FileExistsError):
        module.train_er(cfg, bundle, tmp_path)


def test_boundary_32_exactly_matches_existing_policy_and_event_fits(tmp_path):
    from scripts.boundary_bias_policy import BoundaryBiasPolicy
    from wnet.pilot import run
    module = trainer()
    cfg = replace(config(), boundary_correction=True, num_classes=6, task_count=3,
                  memory_size=12)
    bundle = synthetic_data(cfg.synthetic_per_class, cfg.num_classes)
    old, new = tmp_path / 'old', tmp_path / 'new'
    run(cfg, old, policy=BoundaryBiasPolicy(False))
    summary = module.train_er(cfg, bundle, new)
    a, b = load(old), load(new)
    for key in ('training_state', 'records', 'replay'):
        assert_tree_equal(a[key], b[key])
    old_events = [r for r in a['policy_records'] if 'calibration' in r]
    assert len(b['calibrations']) == len(old_events) == 2
    for event, old_event in zip(b['calibrations'], old_events):
        assert event['step'] == old_event['step'] and event['task'] == old_event['task']
        assert_tree_equal(event['memory_indices'], old_event['calibration']['memory_indices'])
        assert_tree_equal(event['direction'], old_event['calibration']['direction'])
        expected_fit = {key: value for key, value in old_event['calibration'].items()
                        if key not in ('memory_indices', 'direction')}
        assert_tree_equal(event['fit'], expected_fit)
        assert event['fit']['velocity_removed_norm'] == 0.
    assert summary['calibration_events'] == 2
    assert 0 < summary['calibration_seconds'] <= summary['training_seconds']


def test_native64_boundary_resume_after_a_correction_is_exact_and_config_bound(tmp_path):
    module = trainer()
    cfg = replace(config(), boundary_correction=True, num_classes=6, task_count=3,
                  memory_size=12, image_size=64, padding=8)
    bundle = synthetic_data(cfg.synthetic_per_class, cfg.num_classes)
    bundle.images = bundle.images.repeat_interleave(2, -1).repeat_interleave(2, -2)
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    module.train_er(cfg, bundle, full)
    partial = module.train_er(cfg, bundle, resumed, stop_after=6)
    assert partial['calibration_events'] == 1
    with pytest.raises(ValueError, match='Resume.*config'):
        module.train_er(replace(cfg, boundary_correction=False), bundle, resumed, resume=True)
    summary = module.train_er(cfg, bundle, resumed, resume=True)
    a, b = load(full), load(resumed)
    for key in ('training_state', 'replay', 'records', 'calibrations',
                'pre_final_memory_indices', 'metadata'):
        assert_tree_equal(a[key], b[key])
    assert summary['calibration_events'] == 2
    assert b['calibrations'][-1]['memory_indices'] == b['pre_final_memory_indices']
    assert b['timing']['calibration_seconds'] > 0


@pytest.mark.parametrize('helper', ['boundary_bias_policy.py', 'replay_bias.py'])
def test_training_hash_binds_reused_boundary_helpers(monkeypatch, helper):
    from pathlib import Path
    module = trainer()
    before = module._training_code_hash()
    original = Path.read_bytes
    def changed(path):
        value = original(path)
        return value + b'\n# simulated source edit\n' if path.name == helper else value
    monkeypatch.setattr(Path, 'read_bytes', changed)
    assert module._training_code_hash() != before


def test_metadata_records_installed_training_and_calibration_library_versions(tmp_path):
    from importlib.metadata import version
    module, cfg = trainer(), config()
    bundle = synthetic_data(cfg.synthetic_per_class, cfg.num_classes)
    module.train_er(cfg, bundle, tmp_path, stop_after=0)
    environment = load(tmp_path)['metadata']['environment']
    for package in ('torchvision', 'scipy', 'Pillow'):
        assert environment.get(package) == version(package)


def test_resume_rejects_changed_installed_library_version(tmp_path, monkeypatch):
    from importlib import metadata
    module, cfg = trainer(), config()
    bundle = synthetic_data(cfg.synthetic_per_class, cfg.num_classes)
    module.train_er(cfg, bundle, tmp_path, stop_after=0)
    original = metadata.version
    def changed(package):
        value = original(package)
        return value + '+simulated-change' if package == 'torchvision' else value
    monkeypatch.setattr(metadata, 'version', changed)
    with pytest.raises(ValueError, match='Resume.*environment'):
        module.train_er(cfg, bundle, tmp_path, resume=True)
