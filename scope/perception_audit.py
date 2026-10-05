"""Measured M4 replay audit; independent toggles, no robot connection."""

import json
from pathlib import Path
import time

from .backends import hardware_report
from .human_cli import summary
from .human_pipeline import HumanPipeline
from .human_synthetic import HumanCameraSample
from .human_viewer import HumanRecorder
from .mapping import MapConfig
from .runtime import ModuleConfig, distribution


def audit(dataset, output, frames=6, start_frame=355, width=320):
    from .detector import TorchvisionMaskDetector
    from .pose_detector import TorchvisionPoseDetector
    from .tum import TumRgbd
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory")
    if frames < 1 or start_frame < 0 or width < 64:
        raise ValueError("Positive frames, nonnegative start, width at least 64 required")
    source = TumRgbd(Path(dataset), frames, 1, width)
    source.selected = source.rgb[start_frame:start_frame+frames]
    iterator, inputs, decode = iter(source), [], []
    while True:
        start = time.perf_counter()
        try:
            frame = next(iterator)
        except StopIteration:
            break
        decode.append((time.perf_counter()-start)*1000)
        inputs.append(frame)
    if not inputs:
        raise ValueError("No aligned frames at this start index")
    output.mkdir(parents=True, exist_ok=True)
    objects, humans = TorchvisionMaskDetector(), TorchvisionPoseDetector()
    transforms = {"objects": [], "humans": []}
    for name, model in (("objects", objects.model), ("humans", humans.model)):
        original = model.transform.forward
        def measured(*args, _original=original, _name=name, **kwargs):
            start = time.perf_counter()
            value = _original(*args, **kwargs)
            transforms[_name].append((time.perf_counter()-start)*1000)
            return value
        model.transform.forward = measured
    objects.detect(inputs[0])
    humans.detect(inputs[0], "front")
    cases = {}
    for name, disabled in (("all", ()), ("no_rerun", ("rerun",)),
                           ("no_mapping", ("mapping",)), ("no_semantics", ("semantic_detection",)),
                           ("no_humans", ("human_pose",)),
                           ("core_only", ("mapping", "semantic_detection", "human_pose", "rerun"))):
        pipeline = HumanPipeline(("front",), {n: ModuleConfig(enabled=False) for n in disabled},
                                 humans, objects, map_config=MapConfig.around_frames(inputs))
        recorder = HumanRecorder(output/name, ("front",))
        (output/name).mkdir(exist_ok=True)
        ticks, serialization = [], []
        transforms = {"objects": [], "humans": []}
        for i, frame in enumerate(inputs):
            start = time.perf_counter()
            snapshot = pipeline.step([HumanCameraSample("front", frame, [], [])], frame.timestamp_s)
            if "rerun" not in disabled:
                recorder.log(pipeline, snapshot, i)
            stamp = time.perf_counter()
            json.dumps(summary(snapshot), allow_nan=False)
            serialization.append((time.perf_counter()-stamp)*1000)
            ticks.append((time.perf_counter()-start)*1000)
        cases[name] = {"latency_ms": distribution(ticks), "effective_hz": 1000/(sum(ticks)/len(ticks)),
                       "modules": pipeline.runtime.snapshot(inputs[-1].timestamp_s),
                       "serialization_ms": distribution(serialization),
                       "model_transform_ms": {k: distribution(v) for k, v in transforms.items()}}
        pipeline.close()
        print(name, cases[name]["latency_ms"], flush=True)
    result = {"hardware": hardware_report(), "frames": len(inputs), "input_width": width,
              "object_model_short_edge": 480, "pose_model_short_edge": 640, "device": "cpu",
              "decode_ms": distribution(decode), "cases": cases,
              "timing_notes": "Warm models; first Rerun startup included. Transform timings are nested in inference. Memory collection excludes final persistence. Sequential short cases reflect host variation."}
    (output/"audit.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result
