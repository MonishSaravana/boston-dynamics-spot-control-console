"""Read-only Spot image/state normalization for SCOPE's existing frame contract.

No lease, power, E-stop, or robot-command service is imported here.
"""
from dataclasses import dataclass
from collections import deque
import io
import time

import numpy as np
from PIL import Image

from .data import Intrinsics, RgbdFrame
from .live_interaction import RobotPose, TimingBook


@dataclass
class CameraPolicy:
    acquire: bool = True
    process: bool = True
    display: bool = True
    max_hz: float = 5.0

    def validate(self):
        if not 0 < self.max_hz <= 30:
            raise ValueError("Camera rate must be 0–30 Hz")


@dataclass
class SpotImage:
    source: str
    pixels: np.ndarray
    acquisition_robot_s: float
    receive_host_s: float
    receive_unix_s: float
    sensor_frame: str
    T_root_sensor: np.ndarray | None
    root_frame: str
    intrinsics: Intrinsics | None
    depth_scale: float | None
    rpc_ms: float
    decode_ms: float
    camera_model: str = "pinhole"
    distortion: tuple[float, ...] = ()


def decode_pixels(image):
    from bosdyn.api import image_pb2
    if image.format == image_pb2.Image.FORMAT_JPEG:
        with Image.open(io.BytesIO(image.data)) as picture:
            picture.load()
            return np.asarray(picture).copy()
    if image.format != image_pb2.Image.FORMAT_RAW:
        raise ValueError("Unsupported image format")
    formats = {
        image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8: (np.uint8, 1),
        image_pb2.Image.PIXEL_FORMAT_RGB_U8: (np.uint8, 3),
        image_pb2.Image.PIXEL_FORMAT_RGBA_U8: (np.uint8, 4),
        image_pb2.Image.PIXEL_FORMAT_DEPTH_U16: (np.dtype("<u2"), 1),
        image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16: (np.dtype("<u2"), 1),
    }
    if image.pixel_format not in formats:
        raise ValueError("Unsupported image pixel format")
    dtype, channels = formats[image.pixel_format]
    array = np.frombuffer(image.data, dtype=dtype)
    if array.size != image.rows * image.cols * channels:
        raise ValueError("Image data size does not match dimensions")
    return array.reshape(image.rows, image.cols, channels).copy()


def _intrinsics(source):
    model = source.WhichOneof("camera_models")
    if model == "pinhole":
        k = source.pinhole.intrinsics
    elif model == "kannala_brandt":
        k = source.kannala_brandt.intrinsics.pinhole_intrinsics
    else:
        return None
    return Intrinsics(source.cols, source.rows, k.focal_length.x, k.focal_length.y,
                      k.principal_point.x, k.principal_point.y)


def _distortion(source):
    if source.WhichOneof("camera_models") != "kannala_brandt":
        return ()
    k = source.kannala_brandt.intrinsics
    return (k.k1, k.k2, k.k3, k.k4)


def normalize_response(response, source, *, root_frame="odom", rpc_ms=0.,
                       receive_host_s=None, receive_unix_s=None):
    from bosdyn.api import image_pb2
    from bosdyn.client.frame_helpers import get_a_tform_b
    if response.status != image_pb2.ImageResponse.STATUS_OK:
        raise ValueError(image_pb2.ImageResponse.Status.Name(response.status))
    shot = response.shot
    start = time.perf_counter()
    pixels = decode_pixels(shot.image)
    decode_ms = (time.perf_counter()-start)*1000
    try:
        transform = get_a_tform_b(shot.transforms_snapshot, root_frame,
                                  shot.frame_name_image_sensor)
    except Exception:
        transform = None
    T = None if transform is None else np.asarray(transform.to_matrix(), dtype=float)
    stamp = shot.acquisition_time
    return SpotImage(source.name, pixels, stamp.seconds+stamp.nanos/1e9,
        time.monotonic() if receive_host_s is None else receive_host_s,
        time.time() if receive_unix_s is None else receive_unix_s,
        shot.frame_name_image_sensor, T, root_frame, _intrinsics(source),
        source.depth_scale or None, rpc_ms, decode_ms,
        source.WhichOneof("camera_models") or "unavailable", _distortion(source))


def pair_to_frame(visual: SpotImage, depth: SpotImage, *, alignment_verified=False,
                  max_pair_offset_s=.05):
    """Only verified registered depth may authorize world geometry."""
    if not alignment_verified:
        raise ValueError("Depth alignment requires physical verification")
    rgb = visual.pixels
    if rgb.ndim == 2 or rgb.ndim == 3 and rgb.shape[2] == 1:
        gray = rgb if rgb.ndim == 2 else rgb[:, :, 0]
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
    elif rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Visual image is not RGB or grayscale U8")
    if depth.pixels.ndim != 3 or depth.pixels.shape[2] != 1 or depth.pixels.dtype != np.uint16:
        raise ValueError("Depth image is not U16")
    if visual.intrinsics is None or visual.T_root_sensor is None:
        raise ValueError("Visual intrinsics or world transform unavailable")
    if depth.T_root_sensor is None:
        raise ValueError("Depth world transform unavailable")
    if visual.root_frame != depth.root_frame or visual.pixels.shape[:2] != depth.pixels.shape[:2]:
        raise ValueError("Image/depth frame or resolution mismatch")
    if (np.linalg.norm(visual.T_root_sensor[:3, 3]-depth.T_root_sensor[:3, 3]) > .03 or
            np.linalg.norm(visual.T_root_sensor[:3, :3]-depth.T_root_sensor[:3, :3]) > .08):
        raise ValueError("Image/depth extrinsics differ")
    if not depth.depth_scale or depth.depth_scale <= 0:
        raise ValueError("Depth scale unavailable")
    raw = depth.pixels[:, :, 0]
    # Spot ImageSource.depth_scale is raw units per meter (image.proto).
    meters = raw.astype(np.float32) / depth.depth_scale
    meters[raw == 0] = np.nan
    K = visual.intrinsics
    if visual.camera_model == "kannala_brandt":
        if depth.camera_model != "kannala_brandt" or depth.distortion != visual.distortion:
            raise ValueError("Registered depth fisheye calibration differs from visual")
        import cv2
        size = (K.width, K.height)
        matrix = np.array([[K.fx, 0, K.cx], [0, K.fy, K.cy], [0, 0, 1]], dtype=np.float64)
        D = np.asarray(visual.distortion, dtype=np.float64)
        new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            matrix, D, size, np.eye(3), balance=0.)
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            matrix, D, np.eye(3), new_matrix, size, cv2.CV_32FC1)
        rgb = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        meters = cv2.remap(meters, map_x, map_y, cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))
        K = Intrinsics(K.width, K.height, float(new_matrix[0, 0]),
                       float(new_matrix[1, 1]), float(new_matrix[0, 2]),
                       float(new_matrix[1, 2]))
    elif visual.camera_model != "pinhole" or depth.camera_model not in ("pinhole", "unavailable"):
        raise ValueError("Unsupported camera projection for RGB-D geometry")
    frame = RgbdFrame(rgb.copy(), meters, K, visual.T_root_sensor,
        visual.acquisition_robot_s, depth.acquisition_robot_s, "SPOT_LIVE",
        f"{visual.source}:{visual.acquisition_robot_s:.6f}")
    frame.validate(max_pair_offset_s=max_pair_offset_s)
    return frame


class SpotReadOnlySource:
    """Independent acquisition, processing, and display policy per listed source."""

    def __init__(self, image_client, state_client=None, robot=None, *, root_frame="odom"):
        from bosdyn.client.image import build_image_request
        self.image_client, self.state_client, self.robot = image_client, state_client, robot
        self._build_request = build_image_request
        self.root_frame = root_frame
        self.sources = {source.name: source for source in
                        image_client.list_image_sources(timeout=3.)}
        self.policy = {name: CameraPolicy() for name in self.sources}
        self.last_request = {}
        self.latest = {}
        self.health = {name: "UNAVAILABLE" for name in self.sources}
        self.errors = {}
        self.timings = TimingBook()
        self.alignment_verified = False
        self.acquisitions = {name: deque(maxlen=60) for name in self.sources}
        self.acquired_count = {name: 0 for name in self.sources}
        self.processed = {name: 0 for name in self.sources}
        self.displayed = {name: 0 for name in self.sources}
        self._last_processed = {}
        self._last_displayed = {}

    @classmethod
    def connect(cls, hostname, *, root_frame="odom"):
        import bosdyn.client
        import bosdyn.client.util
        from bosdyn.client.image import ImageClient
        from bosdyn.client.robot_state import RobotStateClient
        robot = bosdyn.client.create_standard_sdk("scope-read-only-interaction").create_robot(hostname)
        bosdyn.client.util.authenticate(robot)
        robot.time_sync.wait_for_sync(timeout_sec=5.)
        return cls(robot.ensure_client(ImageClient.default_service_name),
                   robot.ensure_client(RobotStateClient.default_service_name),
                   robot, root_frame=root_frame)

    def configure(self, name, *, acquire=None, process=None, display=None, max_hz=None):
        if name not in self.policy:
            raise ValueError("Camera source unavailable")
        policy = self.policy[name]
        for field, value in (("acquire", acquire), ("process", process),
                             ("display", display)):
            if value is not None:
                if not isinstance(value, bool):
                    raise ValueError("Camera toggle must be boolean")
                setattr(policy, field, value)
        if max_hz is not None:
            policy.max_hz = float(max_hz)
        policy.validate()

    def poll(self):
        """A failed source does not prevent reading other enabled cameras."""
        due = []
        for name, source in self.sources.items():
            policy = self.policy[name]
            if not policy.acquire:
                self.health[name] = "DISABLED"
                continue
            now = time.monotonic()
            if now-self.last_request.get(name, -float("inf")) < 1/policy.max_hz:
                continue
            self.last_request[name] = now
            due.append((name, source))
        if not due:
            return self.latest
        if len(due) > 1:
            start = time.perf_counter()
            try:
                responses = self.image_client.get_image(
                    [self._build_request(name) for name, _ in due], timeout=2.)
                receive_host_s = time.monotonic()
                rpc_ms = (time.perf_counter()-start)*1000
                if len(responses) != len(due):
                    raise ValueError("Image response count differs from request count")
                for (name, source), response in zip(due, responses):
                    self._record(name, source, response, receive_host_s, rpc_ms)
                return self.latest
            except Exception:
                # Some source combinations fail as a group. Retry each once so
                # one bad camera does not hide a healthy camera.
                pass
        for name, source in due:
            start = time.perf_counter()
            try:
                response = self.image_client.get_image([self._build_request(name)], timeout=2.)[0]
                receive_host_s = time.monotonic()
                self._record(name, source, response, receive_host_s,
                             (time.perf_counter()-start)*1000)
            except Exception as exc:
                self.health[name] = "FAILED"
                self.errors[name] = f"{type(exc).__name__}: {exc}"
        return self.latest

    def _record(self, name, source, response, receive_host_s, rpc_ms):
        try:
            image = normalize_response(response, source, root_frame=self.root_frame,
                rpc_ms=rpc_ms, receive_host_s=receive_host_s)
            self.latest[name] = image
            self.acquisitions[name].append(receive_host_s)
            self.acquired_count[name] += 1
            self.health[name] = "OK"
            self.errors.pop(name, None)
            self.timings.add(f"sensor_rpc/{name}", rpc_ms)
            self.timings.add(f"decode/{name}", image.decode_ms)
        except Exception as exc:
            self.health[name] = "FAILED"
            self.errors[name] = f"{type(exc).__name__}: {exc}"

    def rgbd(self, visual_name, depth_name):
        if not self.policy[visual_name].process or not self.policy[depth_name].process:
            raise ValueError("Camera processing disabled")
        if self.health[visual_name] != "OK" or self.health[depth_name] != "OK":
            raise ValueError("Visual or depth camera failed")
        visual, depth = self.latest.get(visual_name), self.latest.get(depth_name)
        if visual is None or depth is None:
            raise ValueError("Visual or depth stream unavailable")
        if (time.monotonic()-visual.receive_host_s > .6 or
                time.monotonic()-depth.receive_host_s > .6):
            raise ValueError("Camera observation stale")
        if self.robot is not None:
            stamp = self.robot.time_sync.robot_timestamp_from_local_secs(time.time())
            robot_now = stamp.seconds + stamp.nanos/1e9
            if any(not 0 <= robot_now-image.acquisition_robot_s <= .6
                   for image in (visual, depth)):
                raise ValueError("Camera acquisition stale")
        return pair_to_frame(visual, depth, alignment_verified=self.alignment_verified)

    def robot_pose(self):
        from bosdyn.client.frame_helpers import get_a_tform_b
        from bosdyn.api import robot_state_pb2
        if self.state_client is None or self.robot is None:
            raise ValueError("Robot state client unavailable")
        state = self.state_client.get_robot_state(timeout=.7)
        kin = state.kinematic_state
        robot_now = self.robot.time_sync.robot_timestamp_from_local_secs(time.time())
        age = ((robot_now.seconds-kin.acquisition_timestamp.seconds) +
               (robot_now.nanos-kin.acquisition_timestamp.nanos)/1e9)
        if not 0 <= age <= .6:
            raise ValueError("ROBOT STATE STALE")
        transform = get_a_tform_b(kin.transforms_snapshot, self.root_frame, "body")
        if transform is None:
            raise ValueError("Robot body transform unavailable")
        T = np.asarray(transform.to_matrix(), dtype=float)
        return RobotPose(float(T[0, 3]), float(T[1, 3]),
            float(np.arctan2(T[1, 0], T[0, 0])), time.monotonic(),
            self.root_frame, state.behavior_state.state ==
            robot_state_pb2.BehaviorState.STATE_STANDING)

    def mark_processed(self, name, acquisition_robot_s):
        if self._last_processed.get(name) != acquisition_robot_s:
            self.processed[name] += 1
            self._last_processed[name] = acquisition_robot_s

    def mark_displayed(self, name, acquisition_robot_s):
        if self._last_displayed.get(name) != acquisition_robot_s:
            self.displayed[name] += 1
            self._last_displayed[name] = acquisition_robot_s

    def status(self):
        now = time.monotonic()
        result = {}
        for name in self.sources:
            times = self.acquisitions[name]
            hz = (len(times)-1)/(times[-1]-times[0]) if len(times) > 1 and times[-1] > times[0] else None
            result[name] = {"health": self.health[name], "acquired": self.policy[name].acquire,
            "processed": self.policy[name].process, "displayed": self.policy[name].display,
            "max_hz": self.policy[name].max_hz,
            "acquired_count": self.acquired_count[name], "processed_count": self.processed[name],
            "displayed_count": self.displayed[name], "observed_acquire_hz": hz,
            "age_s": None if name not in self.latest else now-self.latest[name].receive_host_s,
            "error": self.errors.get(name)}
        return result
