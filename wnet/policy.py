"""Causal, budgeted update policies; audit evaluations stay outside decisions."""

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np

from .pilot import _batch, _features, _sync, run


@dataclass(frozen=True)
class PolicyConfig:
    method: str = "none"
    decision_every: int = 5
    max_fraction: float = .1
    factor: float = 0.
    predictor: dict | None = None
    matched_counts: tuple = ()
    random_seed: int = 419


class UpdatePolicy:
    def __init__(self, spec):
        if spec.method not in {"none", "observe", "risk", "instant", "random"}:
            raise ValueError("Unknown policy method")
        if spec.decision_every < 1 or not 0 <= spec.max_fraction <= 1 or not 0 <= spec.factor <= 1:
            raise ValueError("Invalid policy configuration")
        if spec.method in {"risk", "instant"} and spec.predictor is None:
            raise ValueError("A frozen predictor is required")
        self.spec = spec
        self.records = []
        self.feature_seconds = 0.

    def configuration(self):
        return asdict(self.spec)

    def initialize(self, stream, cfg):
        if self.spec.predictor and self.spec.predictor["model"] != cfg.model:
            raise ValueError("Predictor model differs from training model")
        self.limits = {task: int(sum(t == task for t, _ in stream) * self.spec.max_fraction)
                       for task in range(cfg.task_count)}
        if self.spec.matched_counts:
            if (len(self.spec.matched_counts) != cfg.task_count
                    or any(n < 0 or n > self.limits[t] for t, n in enumerate(self.spec.matched_counts))):
                raise ValueError("Matched counts must respect per-task budgets")
            self.limits = dict(enumerate(self.spec.matched_counts))
        self.random_steps = set()
        if self.spec.method == "random":
            if len(self.spec.matched_counts) != cfg.task_count:
                raise ValueError("Random policy needs per-task matched counts")
            rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, cfg.order_seed, self.spec.random_seed]))
            for task in range(1, cfg.task_count):
                eligible = [step for step, (t, _) in enumerate(stream)
                            if t == task and step % self.spec.decision_every == 0]
                count = self.spec.matched_counts[task]
                if count > len(eligible) or count > self.limits[task]:
                    raise ValueError("Matched random budget exceeds available decisions")
                self.random_steps.update(rng.choice(eligible, count, replace=False).tolist())

    def decide(self, model, opt, candidate, bundle, indices, memory, order, task, step, cfg, loss_fn):
        if task == 0 or step % self.spec.decision_every or self.spec.method == "none":
            return 1.
        _sync(cfg.device)
        started = time.perf_counter()
        features, score, factor = {}, None, 1.
        used = sum(r["task"] == task and r["factor"] < 1 for r in self.records)
        if self.spec.method == "random":
            factor = self.spec.factor if step in self.random_steps else 1.
        else:
            old_classes = order[:task * cfg.classes_per_task]
            slots = [i for i in memory.slots if int(bundle.labels[i]) in old_classes][:128]
            if slots:
                features = _features(model, opt, candidate, _batch(bundle, indices, cfg.device),
                                     _batch(bundle, slots, cfg.device), loss_fn)
                features.update(step=step, task=task)
                predictor = self.spec.predictor
                if self.spec.method == "risk":
                    x = np.array([features[k] for k in predictor["features"]])
                    score = float(np.dot((x - predictor["mean"]) / predictor["scale"], predictor["coef"])
                                  + predictor["intercept"])
                    threshold = predictor["risk_threshold"]
                elif self.spec.method == "instant":
                    score, threshold = features["instant_probe_change"], predictor["instant_threshold"]
                else:
                    threshold = float("inf")
                if score is not None and score > threshold and used < self.limits[task]:
                    factor = self.spec.factor
        _sync(cfg.device)
        self.feature_seconds += time.perf_counter() - started
        self.records.append({"step": step, "task": task, "features": features,
                             "score": score, "factor": factor})
        return factor

    def state(self):
        return copy.deepcopy({"records": self.records, "feature_seconds": self.feature_seconds})

    def load(self, state):
        self.records = copy.deepcopy(state["records"])
        self.feature_seconds = state["feature_seconds"]

    def summary(self):
        return {"policy_feature_seconds": self.feature_seconds,
                "interventions": sum(r["factor"] < 1 for r in self.records),
                "decisions": len(self.records), "policy_method": self.spec.method}


def run_policy(cfg, spec, output, stop_after=None, resume=False):
    if cfg.max_diagnostics:
        raise ValueError("Online experiments require max_diagnostics=0")
    return run(cfg, output, stop_after=stop_after, resume=resume, policy=UpdatePolicy(spec))
