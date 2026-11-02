import json

import numpy as np


def test_optional_same_cache_fits_are_independent_of_evaluation_labels(tmp_path):
    from rlc.calibrate import calibrate_files
    logits = np.array([[2.,0.],[0.,1.],[1.,0.],[0.,2.]])
    labels = np.array([0,1,0,1], dtype=np.int64)
    memory = tmp_path/'memory.npz'
    np.savez(memory, logits=logits, labels=labels, order=np.array([0,1]), classes_per_task=1)
    fits = []
    for i, y in enumerate((labels, 1-labels)):
        benchmark = tmp_path/f'benchmark{i}.npz'
        np.savez(benchmark, logits=logits, labels=y)
        result = calibrate_files(memory, benchmark, tmp_path/f'out{i}', alternative_fits=True)
        saved = json.loads((tmp_path/f'out{i}'/'fitting.json').read_text())
        assert set(saved['fits']) == {'raw','rlc','bic_memory','la_memory','la_mixture','group_la',
                                     'auto_mixture','temperature'}
        assert saved['fits']['temperature']['beta'] > 0
        assert result['metrics']['temperature']['accuracy_percent'] == result['metrics']['raw']['accuracy_percent']
        fits.append(saved)
    assert fits[0] == fits[1]
