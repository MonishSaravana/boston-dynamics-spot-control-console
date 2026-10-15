"""Automatic check that registered depth lines up with its camera image.

Object outlines appear both as depth jumps and as brightness edges. When the
depth is aligned, depth jumps land on image edges far more often than when the
same depth is shifted by a few pixels. The check only applies to Spot's own
registered pairs (<camera>_depth_in_visual_frame); calibration, timestamp and
transform checks still run in pair_to_frame. Thresholds were set on synthetic
RGB-D frames and have not been validated on Spot captures, so a failed check
falls back to the manual confirmation in Sources.
"""
import numpy as np

SHIFTS_PX = ((6, 0), (-6, 0), (0, 6), (0, -6), (10, 0), (-10, 0), (0, 10), (0, -10))
MIN_DEPTH_EDGES = 150
MIN_SCORE = .25
MIN_RATIO = 1.3
PASSES_NEEDED = 3
ATTEMPTS_ALLOWED = 20


def registered_depth_for(visual_source):
    if not visual_source.endswith("_fisheye_image"):
        return None
    return visual_source.replace("_fisheye_image", "_depth_in_visual_frame")


def _image_edges(rgb):
    import cv2
    gray = rgb.astype(np.float32).mean(axis=2)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    edges = magnitude > max(np.percentile(magnitude, 85), 1e-3)
    # About two pixels of tolerance for depth quantization and resampling.
    return cv2.dilate(edges.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def _depth_edges(depth_m):
    valid = np.isfinite(depth_m) & (depth_m > 0)
    edges = np.zeros(depth_m.shape, bool)
    for axis in (0, 1):
        a = depth_m.take(range(depth_m.shape[axis]-1), axis=axis)
        b = depth_m.take(range(1, depth_m.shape[axis]), axis=axis)
        both = valid.take(range(depth_m.shape[axis]-1), axis=axis) & \
            valid.take(range(1, depth_m.shape[axis]), axis=axis)
        with np.errstate(invalid="ignore"):
            jump = both & (np.abs(a-b) > np.maximum(.08, .05*np.minimum(a, b)))
        if axis == 0:
            edges[:-1] |= jump
        else:
            edges[:, :-1] |= jump
    return edges


def edge_alignment(rgb, depth_m):
    """Return (aligned_score, best_shifted_score, depth_edge_count)."""
    image = _image_edges(rgb)
    depth = _depth_edges(depth_m)
    margin = max(abs(v) for shift in SHIFTS_PX for v in shift)
    inner = np.zeros(depth.shape, bool)
    inner[margin:-margin, margin:-margin] = True

    def score(dx, dy):
        moved = np.roll(depth, (dy, dx), axis=(0, 1)) & inner
        count = moved.sum()
        return float((moved & image).sum()/count) if count else 0.
    count = int((depth & inner).sum())
    return score(0, 0), max(score(dx, dy) for dx, dy in SHIFTS_PX), count


def passes(aligned, shifted, count):
    return (count >= MIN_DEPTH_EDGES and aligned >= MIN_SCORE and
            aligned >= MIN_RATIO*max(shifted, 1e-6))


class AlignmentCheck:
    """Needs several passing frames; gives up after a bounded number of tries."""

    def __init__(self, clock=None):
        import time
        self.clock = clock or time.monotonic
        self.started_s = self.clock()
        self.passes = 0
        self.attempts = 0
        self.last_frame = None
        self.state = "checking"
        self.detail = "Checking that depth lines up with the camera image"

    def observe(self, frame):
        if self.state != "checking" or frame.frame_id == self.last_frame:
            return self.state
        self.last_frame = frame.frame_id
        self.attempts += 1
        aligned, shifted, count = edge_alignment(frame.rgb, frame.depth_m)
        if passes(aligned, shifted, count):
            self.passes += 1
        self.detail = (f"Depth edges on image edges: {aligned:.2f} aligned vs {shifted:.2f} "
                       f"shifted ({count} depth edges); {self.passes}/{PASSES_NEEDED} frames passed")
        if self.passes >= PASSES_NEEDED:
            self.state = "passed"
        elif self.attempts >= ATTEMPTS_ALLOWED:
            self.state = "failed"
            self.detail = ("Automatic check could not confirm alignment. " + self.detail +
                           ". Check the depth overlay and confirm in Sources.")
        return self.state

    def give_up_if_slow(self, limit_s=20.):
        """Stop waiting when no usable camera + depth pair arrives at all."""
        if self.state == "checking" and self.clock()-self.started_s > limit_s:
            self.state = "failed"
            self.detail = ("No usable camera + depth frames arrived for the automatic check. "
                           "Check the depth overlay and confirm in Sources.")
        return self.state

    def snapshot(self):
        return {"state": self.state, "detail": self.detail}
