"""Spot sensor inspection only: authentication, time sync, image reads.

No lease, E-stop, power, robot-command, or control-console imports.
Images are decoded in memory and never saved by this command.
"""
import io
import time

import numpy as np
from PIL import Image

from .runtime import distribution


def decode_image(image):
    from bosdyn.api import image_pb2
    if image.format==image_pb2.Image.FORMAT_JPEG:
        with Image.open(io.BytesIO(image.data)) as picture:
            picture.load()
            return {"shape":list(np.asarray(picture).shape),"dtype":str(np.asarray(picture).dtype)}
    if image.format!=image_pb2.Image.FORMAT_RAW:
        raise ValueError("Unsupported image encoding")
    formats = {image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:(np.uint8,1),
               image_pb2.Image.PIXEL_FORMAT_RGB_U8:(np.uint8,3),
               image_pb2.Image.PIXEL_FORMAT_RGBA_U8:(np.uint8,4),
               image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:(np.dtype('<u2'),1),
               image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:(np.dtype('<u2'),1)}
    if image.pixel_format not in formats:
        raise ValueError("Unsupported pixel format")
    dtype,channels = formats[image.pixel_format]
    array = np.frombuffer(image.data,dtype=dtype).reshape(image.rows,image.cols,channels)
    return {"shape":list(array.shape),"dtype":str(array.dtype)}


def inspect_sources(client,samples=5,source_names=None):
    from bosdyn.api import image_pb2
    from bosdyn.client.frame_helpers import get_a_tform_b
    from bosdyn.client.image import build_image_request
    if not 1<=samples<=100:
        raise ValueError("Request 1–100 sensor samples")
    available = client.list_image_sources(timeout=3.)
    selected = [s for s in available if source_names is None or s.name in source_names]
    if source_names and set(source_names)-{s.name for s in available}:
        raise ValueError("Requested source not available")
    report = {"mode":"READ_ONLY", "requested_samples_per_source":samples,
              "clock":"acquisition: robot clock; receive: host monotonic + Unix",
              "network_one_way_ms":None,"alignment_verified":False,
              "alignment_note":"Names/resolution/transforms do not prove pixel-level alignment",
              "sources":[]}
    for source in selected:
        entry = {"name":source.name,"resolution":[source.cols,source.rows],
            "image_type":image_pb2.ImageSource.ImageType.Name(source.image_type),
            "advertised_pixel_formats":[image_pb2.Image.PixelFormat.Name(f) for f in source.pixel_formats],
            "advertised_formats":[image_pb2.Image.Format.Name(f) for f in source.image_formats],
            "depth_scale":source.depth_scale or None,
            "calibration_model":source.WhichOneof("camera_models"),
            "calibration":{},"samples":[],"failures":[]}
        if source.HasField("pinhole"):
            k = source.pinhole.intrinsics
            entry["calibration"] = {"fx":k.focal_length.x,"fy":k.focal_length.y,
                                     "cx":k.principal_point.x,"cy":k.principal_point.y}
        timestamps,rpc,decodes = [],[],[]
        for _ in range(samples):
            start = time.perf_counter()
            try:
                response = client.get_image([build_image_request(source.name)],timeout=3.)[0]
                receive = time.monotonic()
                duration = (time.perf_counter()-start)*1000
                if response.status!=image_pb2.ImageResponse.STATUS_OK:
                    raise ValueError(image_pb2.ImageResponse.Status.Name(response.status))
                shot = response.shot
                timestamp = shot.acquisition_time.seconds+shot.acquisition_time.nanos/1e9
                image = shot.image
                decode_start = time.perf_counter()
                decoded = decode_image(image)
                decode_ms = (time.perf_counter()-decode_start)*1000
                transforms = {}
                for root in ("body","odom","vision"):
                    try:
                        transform = get_a_tform_b(shot.transforms_snapshot,root,shot.frame_name_image_sensor)
                        transforms[root] = None if transform is None else transform.to_matrix().tolist()
                    except Exception as exc:
                        transforms[root] = {"error":f"{type(exc).__name__}: {exc}"}
                entry["samples"].append({"acquisition_robot_s":timestamp,
                    "host_receive_monotonic_s":receive,"host_receive_unix_s":time.time(),
                    "rpc_round_trip_ms":duration,"decode_ms":decode_ms,
                    "image_format":image_pb2.Image.Format.Name(image.format),
                    "pixel_format":image_pb2.Image.PixelFormat.Name(image.pixel_format),
                    "resolution":[image.cols,image.rows],"decoded":decoded,
                    "sensor_frame":shot.frame_name_image_sensor,"T_root_camera":transforms})
                timestamps.append(timestamp)
                rpc.append(duration)
                decodes.append(decode_ms)
            except Exception as exc:
                entry["failures"].append(f"{type(exc).__name__}: {exc}")
        unique = sorted(set(timestamps))
        entry.update({"status":"OK" if len(entry["samples"])==samples else
                      "DEGRADED" if entry["samples"] else "FAILED",
            "observed_acquisition_hz":(len(unique)-1)/(unique[-1]-unique[0]) if len(unique)>1 else None,
            "fps_note":"Sampled request-limited rate; not maximum sensor FPS",
            "duplicate_acquisitions":len(timestamps)-len(unique),
            "rpc_latency":distribution(rpc),"decode_latency":distribution(decodes)})
        report["sources"].append(entry)
    return report


def diagnose_spot(hostname,samples=5,source_names=None):
    import bosdyn.client
    import bosdyn.client.util
    from bosdyn.client.image import ImageClient
    robot = bosdyn.client.create_standard_sdk("scope-read-only-sensors").create_robot(hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync(timeout_sec=5.)
    client = robot.ensure_client(ImageClient.default_service_name)
    return inspect_sources(client,samples,source_names)
