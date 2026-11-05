"""Actual-data inference/schema smoke test; no measured test outcomes are used."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from common import load_prepared, write_json
from inference.predict_effects import GEARSPredictor
from adapter_interface.validation import validate_bundle
from adapter_interface.schema import ExternalEmbedding, align_external_embedding


def run(checkpoint, prepared, output):
    meta, obs, var, roles, graph, x = load_prepared(prepared)
    predictor = GEARSPredictor(checkpoint)
    target = meta.get("frozen_split", predictor.metadata["frozen_split"])["conditions"]["test"][0]
    assert target.replace("+ctrl","") not in predictor.training_targets
    controls = x[roles["fit_control"][:16]].toarray()
    first = predictor.predict_effects([target], controls)
    second = predictor.predict_effects([target], controls)
    validate_bundle(first)
    np.testing.assert_allclose(first["predicted_effects"], second["predicted_effects"], rtol=0, atol=1e-6)
    try:
        predictor.predict_effects(["NOT_A_VALID_TARGET"], controls)
        raise AssertionError("Unsupported target was not rejected")
    except ValueError:
        pass
    repeated = np.repeat(controls[:1],16,axis=0)
    comparison = predictor.predict_effects([target], repeated)["predicted_effects"][0]
    batch_slot_error = float(np.max(np.abs(comparison[0]-comparison[1])))
    dummy_external = ExternalEmbedding(
        identifiers=[predictor.genes[0]], values=np.ones((1,3),np.float32),
        provenance="explicitly synthetic schema-only fixture, not biological results")
    alignment = align_external_embedding(dummy_external,predictor.genes[:2])
    assert alignment["coverage"].tolist() == [True,False]
    report = dict(
        checkpoint_epoch=predictor.state["epoch"], unseen_target=target,
        prediction_shape=list(first["predicted_effects"].shape),
        repeated_fixed_batch_predictions_equal_within_1e_minus6=True,
        unsupported_target_rejected=True, adapter_bundle_valid=True,
        upstream_batch_slot_max_abs_effect_difference=batch_slot_error,
        measured_test_outcomes_used=False, external_alignment_schema_test=True,
        controls_used=obs.index[roles["fit_control"][:16]].tolist(),
    )
    write_json(output, report)
    print(report)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--prepared",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    run(args.checkpoint,args.prepared,args.output)
