"""Replay Likelihood Correction and same-cache comparison operators."""
from scripts.replay_bias import fit_bias_shift
from scripts.boundary_bias_policy import bias_direction, apply_bias_calibration
from scripts.posthoc_calibration import (
    apply_cached, write_calibrator, prediction_metrics, frequency_priors,
    fit_methods as _fit_methods,
)

FREQUENCY_SELECTION = {'variants': {
    name: {'tau': tau} for name, tau in [('la_memory', 0.), ('la_mixture', 1.), ('group_la', 1.)]
}}


def fit_methods(logits, labels, order, classes_per_task, selection=None):
    """Fit all six methods using memory only; default strengths are paper-fixed."""
    return _fit_methods(logits, labels, order, classes_per_task,
                        FREQUENCY_SELECTION if selection is None else selection)
