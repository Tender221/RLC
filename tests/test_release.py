import json
from pathlib import Path

import pytest
import torch


def test_portable_config_contract():
    from rlc.experiment import make_config
    for name, classes, tasks, size, padding, heldout in [
        ('cifar10', 10, 5, 32, 4, 500), ('cifar100', 100, 10, 32, 4, 50),
        ('tinyimagenet', 200, 20, 64, 8, 25),
    ]:
        cfg = make_config(name, 1810, 181, 'cpu', 500)
        assert (cfg.num_classes, cfg.task_count, cfg.image_size, cfg.padding, cfg.audit_per_class) == (
            classes, tasks, size, padding, heldout)
        assert cfg.model == 'resnet18'
        assert (cfg.learning_rate, cfg.momentum, cfg.weight_decay) == (.05, .9, 0.)
        assert (cfg.batch_size, cfg.replay_batch_size, cfg.memory_size) == (32, 32, 500)


def test_all_calibrators_writeback_matches_cached_predictions():
    from rlc import fit_methods, apply_cached, write_calibrator
    torch.manual_seed(80)
    model = torch.nn.Linear(5, 4).double()
    x = torch.randn(40, 5, dtype=torch.float64)
    z = model(x).detach()
    y = torch.arange(4).repeat(10)
    fits = fit_methods(z, y, [0, 1, 2, 3], 2)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    for fit in fits.values():
        model.load_state_dict(initial)
        write_calibrator(model, fit)
        torch.testing.assert_close(model(x), apply_cached(z, fit), rtol=1e-10, atol=1e-10)
    assert fits['rlc']['identifiable']
    assert abs(fits['rlc']['gradient_after']) < 1e-10
    assert fits['rlc']['loss_after'] <= fits['rlc']['loss_before']
    assert fits['bic_memory']['loss_after'] <= fits['bic_memory']['loss_before']


def test_metrics_known_balanced_example():
    from rlc import prediction_metrics
    z = torch.tensor([[0., 0.], [0., 0.]])
    m = prediction_metrics(z, torch.tensor([0, 1]), [0, 1], 1)
    assert m['accuracy_percent'] == 50.
    assert m['ece_percent'] == 0.
    assert m['nll'] == pytest.approx(0.6931471805599453)


def test_synthetic_train_fit_evaluate_and_checkpoint_resume(tmp_path):
    from rlc.experiment import make_config, load_data, execute
    cfg = make_config('synthetic', 10, 1, 'cpu', 12)
    train, benchmark = load_data(cfg, tmp_path / 'data')
    output = tmp_path / 'run'
    result = execute(cfg, train, benchmark, output)
    assert result['complete']
    assert set(result['metrics']) == {'raw', 'rlc', 'bic_memory', 'la_memory', 'la_mixture', 'group_la'}
    saved = torch.load(output / 'checkpoint.pt', weights_only=False)
    assert len(saved['pre_final_memory_indices']) == 12
    assert set(saved['pre_final_memory_indices']) < set(range(len(train.labels)))
    assert json.loads((output / 'evaluation.json').read_text())['memory_timing'] == 'pre_final_insertion'
    repeated = execute(cfg, train, benchmark, output, resume=True)
    assert repeated['metrics'] == result['metrics']


def test_reject_existing_evaluation_before_training(tmp_path):
    from rlc.experiment import make_config, load_data, execute
    cfg = make_config('synthetic', 10, 1, 'cpu', 12)
    train, benchmark = load_data(cfg, tmp_path / 'data')
    (tmp_path / 'evaluation.json').write_text('{}')
    with pytest.raises(FileExistsError):
        execute(cfg, train, benchmark, tmp_path)


def test_saved_logit_workflow_never_fits_evaluation_labels(tmp_path):
    import numpy as np
    from rlc.calibrate import calibrate_files
    rng = np.random.default_rng(13)
    z = rng.normal(size=(40, 4))
    labels = np.arange(4).repeat(10)
    memory = tmp_path / 'memory.npz'
    evaluation = tmp_path / 'evaluation.npz'
    np.savez(memory, logits=z, labels=labels, order=np.arange(4), classes_per_task=2)
    np.savez(evaluation, logits=z, labels=labels)
    a = calibrate_files(memory, evaluation, tmp_path / 'a')
    np.savez(evaluation, logits=z, labels=labels[::-1])
    b = calibrate_files(memory, evaluation, tmp_path / 'b')
    assert a['fits'] == b['fits']
    assert a['metrics'] != b['metrics']


def test_main_model_manifest_has_30_unique_pairs_and_explicit_scope():
    rows = json.loads((Path(__file__).parents[1] / 'configs' / 'main_models.json').read_text())
    assert len(rows) == 30
    assert len({(r['dataset'], r['learner'], r['seed'], r['order_seed']) for r in rows}) == 30
    assert sum(r['workflow'] == 'fresh_er_training' for r in rows) == 25
    assert sum(r['workflow'] == 'external_mir_logits' for r in rows) == 5
