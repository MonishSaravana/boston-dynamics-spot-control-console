"""Operator-selectable perception models for the live console.

Every choice names its measured cost and accuracy so the tradeoff is visible
in the app. Numbers come from docs/milestone-4.5.md (held-out NYU rooms and a
six-query timing subset on the development Mac), not from Spot cameras.
"""

import copy
import threading

from .open_vocab import VerifiedQueryDetector


SOURCE_NOTE = ("Measured on public indoor photos (NYU Depth V2) and a Mac with an Apple GPU, "
               "not on Spot's cameras. Expect different numbers on the robot; the live "
               "timings at the top of this dialog are measured on this connection.")

OPTIONS = [
    {"key": "finder", "title": "Object finder",
     "help": "Which models must agree before a typed object becomes a candidate.",
     "choices": [
         {"value": "checked", "label": "Checked: Grounding DINO + OWLv2",
          "summary": "A second model has to agree on the same region. Fewer wrong targets, "
                     "misses a few more. Use this before GO.",
          "measured": "Found 16 of 25 objects. Accepted 6 of 119 objects that were not there. "
                      "About 170 ms extra per search."},
         {"value": "fast", "label": "Fast: Grounding DINO only",
          "summary": "One model, no second opinion. Finds a little more and answers sooner, "
                     "but about 1 in 6 searches for an absent object still returns a box. "
                     "Check the box carefully before Confirm.",
          "measured": "Found 18 of 25 objects. Accepted 19 of 119 objects that were not there."},
     ]},
    {"key": "confidence", "title": "Confidence needed",
     "help": "Minimum Grounding DINO score for a candidate.",
     "choices": [
         {"value": "loose", "label": "Loose · 0.35",
          "summary": "Picks up distant, dark or partly hidden objects, with more wrong boxes.",
          "measured": "Not separately measured. A trash can at 0.41 was refused at the default."},
         {"value": "standard", "label": "Standard · 0.45",
          "summary": "The tuned default.",
          "measured": "Tuned on two development rooms; the held-out results above use it."},
         {"value": "strict", "label": "Strict · 0.55",
          "summary": "Fewer wrong boxes, but more real objects are refused.",
          "measured": "Not separately measured."},
     ]},
    {"key": "detail", "title": "Image detail",
     "help": "Pixels the finder models see. More detail is slower.",
     "choices": [
         {"value": "standard", "label": "Standard · 512 px",
          "summary": "Fast enough to re-check the target about once a second.",
          "measured": "Grounding DINO 384 ms, OWLv2 168 ms on the Apple GPU."},
         {"value": "high", "label": "High · DINO 800 px, OWLv2 768 px",
          "summary": "Tighter OWLv2 boxes on small objects. Each search takes more than "
                     "twice as long, so the target updates less often.",
          "measured": "Grounding DINO 843 ms, OWLv2 446 ms. Pillow and lamp boxes improved "
                      "from below 0.5 to about 0.70 overlap. DINO found no extra objects."},
     ]},
    {"key": "outline", "title": "Object outline",
     "help": "How the object is cut out of its box before depth gives it a 3D position.",
     "choices": [
         {"value": "grabcut", "label": "GrabCut (OpenCV, CPU)",
          "summary": "No model to load. It separates the object by color and brightness "
                     "edges, so expect weaker outlines on Spot's grayscale cameras.",
          "measured": "Outline overlap 0.72. 13 of 18 found objects got a 3D position."},
         {"value": "sam2", "label": "SAM 2.1 Tiny",
          "summary": "Learned outlines. More objects get a usable 3D position, so a "
                     "destination preview is available more often. The first use loads "
                     "the model and takes several seconds.",
          "measured": "Outline overlap 0.82. 17 of 18 found objects got a 3D position. "
                      "About 280–300 ms per search once loaded."},
     ]},
    {"key": "refresh", "title": "Re-check the target every",
     "help": "How often an active search runs the finder again. The tracker follows the "
             "object between searches.",
     "choices": [
         {"value": "fast", "label": "1.0 s",
          "summary": "Fresher target evidence. With searches taking 0.5–1 s the finder "
                     "runs almost constantly, which can slow mapping and pose."},
         {"value": "standard", "label": "1.5 s",
          "summary": "The default."},
         {"value": "slow", "label": "3.0 s",
          "summary": "Frees compute. Confirm and GO need evidence under 1.5 s old, so if "
                     "the tracker loses the object you may wait for the next search."},
     ]},
    {"key": "device", "title": "Run models on",
     "help": "Where Grounding DINO, OWLv2 and SAM run. GrabCut always uses the CPU.",
     "choices": [
         {"value": "auto", "label": "Apple GPU when available",
          "summary": "The default.",
          "measured": "Grounding DINO 384 ms, OWLv2 168 ms."},
         {"value": "cpu", "label": "CPU",
          "summary": "About 1.7 times slower. Use it if the GPU reports errors or memory "
                     "runs short.",
          "measured": "Grounding DINO 662 ms, OWLv2 284 ms."},
     ]},
]

DEFAULTS = {"finder": "checked", "confidence": "standard", "detail": "standard",
            "outline": "grabcut", "refresh": "standard", "device": "auto"}

THRESHOLDS = {"loose": .35, "standard": .45, "strict": .55}
SIZES = {"standard": (512, 512), "high": (800, 768)}
REFRESH_S = {"fast": 1., "standard": 1.5, "slow": 3.}


def validate(values):
    """Return a complete settings dict, rejecting unknown keys or choices."""
    if not isinstance(values, dict):
        raise ValueError("Model settings must be an object")
    allowed = {option["key"]: {c["value"] for c in option["choices"]} for option in OPTIONS}
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown model setting: {sorted(unknown)[0]}")
    result = dict(DEFAULTS)
    for key, value in values.items():
        if value not in allowed[key]:
            raise ValueError(f"Unknown choice for {key}")
        result[key] = value
    return result


class ModelLibrary:
    """Load each model's weights once. Settings only change how they are called."""

    def __init__(self):
        self.lock = threading.Lock()
        self.loaded = {}

    def _get(self, key, build):
        with self.lock:
            if key not in self.loaded:
                self.loaded[key] = build()
            return self.loaded[key]

    def detector(self, backend, device):
        from .open_vocab import OpenVocabularyDetector
        return self._get(("detector", backend, device),
                         lambda: OpenVocabularyDetector(backend, device))

    def segmenter(self, device):
        from .query_segmentation import Sam2Segmenter
        return self._get(("sam2", device), lambda: Sam2Segmenter(device))

    def keep_only(self, device):
        """Free weights loaded for a device that is no longer selected."""
        with self.lock:
            for key in [k for k in self.loaded if k[-1] != device]:
                del self.loaded[key]


class CheckedFinder(VerifiedQueryDetector):
    """VerifiedQueryDetector built from already loaded, configured detectors."""

    def __init__(self, primary, verifier):
        self.primary, self.verifier = primary, verifier
        self.name = primary.name+"+owlv2-verification"
        self.metadata = {**primary.metadata,
                         "verification": {**verifier.metadata, "minimum_box_iou": .3}}


def _configured(detector, size, threshold, low_threshold):
    # Shallow copy shares the loaded weights; only per-call parameters differ.
    result = copy.copy(detector)
    result.size, result.threshold, result.low_threshold = size, threshold, low_threshold
    result.metadata = {**detector.metadata, "resolution": size,
                       "acceptance_threshold": threshold, "weak_threshold": low_threshold}
    return result


def _device(values):
    from .backends import resolve_device
    return resolve_device(values["device"])


def query_factory(values, library):
    def build():
        device = _device(values)
        finder_px, checker_px = SIZES[values["detail"]]
        primary = _configured(library.detector("grounding-dino", device), finder_px,
                              THRESHOLDS[values["confidence"]], .15)
        if values["finder"] == "fast":
            return primary
        verifier = _configured(library.detector("owlv2", device), checker_px, .15, .075)
        return CheckedFinder(primary, verifier)
    return build


def segmenter_factory(values, library):
    if values["outline"] == "grabcut":
        from .query_segmentation import GrabCutSegmenter
        return GrabCutSegmenter
    return lambda: library.segmenter(_device(values))


def preload(values, library):
    """Load weights and run one blank pass, so the first real search is not a stall.

    The first inference after loading is several times slower than later ones.
    """
    from types import SimpleNamespace
    import numpy as np
    from .interaction import TextCommand
    from .semantic_query import SemanticQuery
    library.keep_only(_device(values))
    finder = query_factory(values, library)()
    blank = SimpleNamespace(rgb=np.zeros((480, 640, 3), np.uint8), timestamp_s=0.,
                            frame_id="warmup")
    query = SemanticQuery.from_command(TextCommand("chair"))
    # A blank image finds nothing, so warm the checker directly as well.
    for model in (finder.primary, finder.verifier) if isinstance(finder, CheckedFinder) else (finder,):
        model.query(blank, query)
    segmenter = segmenter_factory(values, library)()
    segmenter.segment(blank.rgb, [SimpleNamespace(box_xyxy=(200., 140., 440., 340.),
                                                  evidence={})])
