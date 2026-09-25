"""Offline-testable gesture gate and conservative visual/depth checks."""

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np


MODEL_DIR = Path(__file__).resolve().parent / 'models'
MIN_APPROACH_DISTANCE = 2.5  # extra 0.5 m margin over the requested 2 m
STABLE_SECONDS = 0.6
NEUTRAL_SECONDS = 0.5
MAX_GAP = 0.30


@dataclass(frozen=True)
class Observation:
    label: str  # '1', '2', 'neutral', or 'unclear'
    distance_m: float | None = None
    reason: str = ''


class GestureGate:
    """One gesture action per neutral open-palm reset."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = 'need_neutral'
        self.candidate = None
        self.since = None
        self.last_at = None
        self.count = 0
        self.backup_until = None

    def stop_and_rearm(self):
        self.reset()

    def observe(self, observation, now):
        label = observation.label
        if self.last_at is not None and now - self.last_at > MAX_GAP:
            self.reset()
        self.last_at = now
        if label == 'unclear':
            self.reset()
            return 0, 'Unclear signal or person lost; show open palm to rearm'
        if label != self.candidate:
            self.candidate, self.since, self.count = label, now, 1
        else:
            self.count += 1
        stable = self.count >= 3 and now - self.since >= (
            NEUTRAL_SECONDS if label == 'neutral' else STABLE_SECONDS)

        if self.state == 'need_neutral':
            if label == 'neutral' and stable:
                self.state = 'ready'
                return 0, 'Gesture ready: hold 1 or 2 steadily'
            return 0, 'Waiting for neutral open palm'
        if self.state == 'ready':
            if label == '1' and stable and observation.distance_m is not None and \
                    observation.distance_m > MIN_APPROACH_DISTANCE:
                self.state = 'approach'
            elif label == '2' and stable:
                self.state = 'backup'
                self.backup_until = now + 0.5
            else:
                return 0, 'Waiting for steady 1 or 2'

        if self.state == 'approach':
            if label != '1' or observation.distance_m is None or \
                    observation.distance_m <= MIN_APPROACH_DISTANCE:
                self.reset()
                return 0, 'Approach stopped; show open palm to rearm'
            return 1, f'Approaching slowly; torso depth {observation.distance_m:.2f} m'
        if self.state == 'backup':
            if label != '2' or now >= self.backup_until:
                self.reset()
                return 0, 'Backup stopped; show open palm to rearm'
            return -1, 'Backing up briefly'
        return 0, 'Waiting for neutral open palm'


def torso_depth(depth, scale, landmarks, person_mask):
    """Use the near side of a well-covered torso patch, never a far background pixel."""
    if not math.isfinite(scale) or scale <= 0 or depth.ndim != 2 or \
            person_mask is None or person_mask.shape != depth.shape:
        return None
    points = [landmarks[i] for i in (11, 12, 23, 24)]
    if any((p.x <= 0.04 or p.x >= 0.96 or p.y <= 0.04 or p.y >= 0.96 or
            (p.visibility or 0) < 0.8 or (p.presence or 0) < 0.8) for p in points):
        return None
    cx = sum(p.x for p in points) / 4
    if abs(cx - 0.5) > 0.18:  # forward command only for a centered person
        return None
    h, w = depth.shape
    x0, x1 = sorted((int(points[0].x * w), int(points[1].x * w)))
    y0 = int(min(points[0].y, points[1].y) * h)
    y1 = int(max(points[2].y, points[3].y) * h)
    # Keep away from the torso outline, where background depth can leak in.
    dx, dy = int((x1 - x0) * .25), int((y1 - y0) * .25)
    patch = depth[y0 + dy:y1 - dy, x0 + dx:x1 - dx]
    mask_patch = person_mask[y0 + dy:y1 - dy, x0 + dx:x1 - dx]
    if patch.size < 100:
        return None
    person_pixels = mask_patch >= 0.8
    if person_pixels.mean() < 0.7:
        return None
    valid = person_pixels & (patch > 0) & (patch < 65535)
    if valid.sum() / person_pixels.sum() < 0.8:
        return None
    meters = patch[valid].astype(np.float32) / scale
    low, high = np.percentile(meters, [10, 90])
    if not np.isfinite(low) or low < 0.3 or high - low > 0.6:
        return None
    # Use the nearest measured person pixel (including an outstretched hand) for
    # the stop boundary. Spurious near pixels only cause an earlier stop.
    whole_person = (person_mask >= 0.8) & (depth > 0) & (depth < 65535)
    if not np.any(whole_person):
        return None
    return float(min(meters.min(), depth[whole_person].min() / scale))


class GestureVision:
    """MediaPipe models run only after the operator opts in."""

    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        gesture_path = MODEL_DIR / 'gesture_recognizer.task'
        pose_path = MODEL_DIR / 'pose_landmarker_lite.task'
        if not gesture_path.is_file() or not pose_path.is_file():
            raise RuntimeError('Gesture model files are missing; see README')
        self.gesture = mp.tasks.vision.GestureRecognizer.create_from_options(
            mp.tasks.vision.GestureRecognizerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(gesture_path),
                                                   delegate=mp.tasks.BaseOptions.Delegate.CPU),
                running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2,
                min_hand_detection_confidence=0.8, min_hand_presence_confidence=0.8))
        self.pose = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_path),
                                                   delegate=mp.tasks.BaseOptions.Delegate.CPU),
                running_mode=mp.tasks.vision.RunningMode.IMAGE, num_poses=2,
                min_pose_detection_confidence=0.8, min_pose_presence_confidence=0.8,
                output_segmentation_masks=True))

    def close(self):
        self.gesture.close()
        self.pose.close()

    def inspect(self, rgb, depth, scale):
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                 data=np.ascontiguousarray(rgb))
        hands = self.gesture.recognize(mp_image)
        bodies = self.pose.detect(mp_image)
        if len(hands.hand_landmarks) != 1 or len(bodies.pose_landmarks) != 1 or \
                len(hands.gestures) != 1 or not hands.gestures[0]:
            return Observation('unclear', reason='Need exactly one visible hand and person')
        pose = bodies.pose_landmarks[0]
        hand = hands.hand_landmarks[0]
        wrist = hand[0]
        if not any(math.hypot(wrist.x - pose[i].x, wrist.y - pose[i].y) < 0.20
                   and (pose[i].visibility or 0) >= 0.8 for i in (15, 16)):
            return Observation('unclear', reason='Hand is not associated with visible person')
        category = hands.gestures[0][0]
        labels = {'Pointing_Up': '1', 'Victory': '2', 'Open_Palm': 'neutral'}
        label = labels.get(category.category_name, 'unclear') if category.score >= 0.85 else 'unclear'
        if label == 'unclear':
            return Observation('unclear', reason='Hand signal is not clear')
        if not bodies.segmentation_masks or len(bodies.segmentation_masks) != 1:
            return Observation('unclear', reason='Person segmentation is unavailable')
        person_mask = np.squeeze(bodies.segmentation_masks[0].numpy_view())
        distance = torso_depth(depth, scale, pose, person_mask)
        if label == '1' and distance is None:
            return Observation('unclear', reason='Torso depth is unreliable')
        return Observation(label, distance)
