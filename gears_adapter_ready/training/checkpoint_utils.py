"""Complete restart state; save locally and copy completed checkpoint objects."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from common import digest, publish, timestamp, write_json


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state())


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def state_dict(model, optimizer, scheduler, **extra):
    return dict(
        model=model.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), rng=rng_state(), created=timestamp(), **extra,
    )


def save_checkpoint(path, state, publish_dir=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    record = dict(filename=path.name, bytes=path.stat().st_size, sha256=digest(path),
                  epoch=state["epoch"], cursor=state.get("cursor", 0), created=timestamp())
    write_json(path.with_suffix(".json"), record)
    if publish_dir:
        publish(path, Path(publish_dir) / path.name)
        publish(path.with_suffix(".json"), Path(publish_dir) / path.with_suffix(".json").name)
    return record


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"])
    return state
