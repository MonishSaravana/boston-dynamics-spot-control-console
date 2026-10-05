"""Explicit file, video, webcam, and calibrated RGB-D sources."""

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .data import Intrinsics,RgbdFrame


def image_frame(path,depth_path=None,calibration=None):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    h,w = rgb.shape[:2]
    stamp = time.monotonic()
    if depth_path:
        if not calibration:
            raise ValueError("Depth requires a calibration JSON with intrinsics, depth_scale_m, and T_world_camera")
        params = json.loads(Path(calibration).read_text())
        K = Intrinsics(**params["intrinsics"])
        pose = np.asarray(params["T_world_camera"],float)
        raw = np.load(depth_path) if Path(depth_path).suffix==".npy" else np.asarray(Image.open(depth_path))
        scale = float(params["depth_scale_m"])
        if not np.isfinite(scale) or scale<=0:
            raise ValueError("Positive finite depth_scale_m required")
        depth = raw.astype(np.float32)*scale
        source = "MEASURED_RGBD_FILE"
    else:
        # Contract placeholder only: never used for 3D without depth/calibration.
        K,pose,depth,source = Intrinsics(w,h,1.,1.,0.,0.),np.eye(4),np.full((h,w),np.nan,np.float32),"RGB_FILE"
    f = RgbdFrame(rgb,depth,K,pose,stamp,stamp,source,Path(path).stem)
    f.validate()
    return f


class VideoSource:
    def __init__(self,source,width=640):
        import cv2
        self.cv2,self.width = cv2,width
        self.capture = cv2.VideoCapture(source if isinstance(source,int) else str(source))
        if not self.capture.isOpened():
            raise ValueError("Cannot open the selected camera/video")
        self.is_camera = isinstance(source,int)
        self.index = 0
        self.fps = None if self.is_camera else float(self.capture.get(cv2.CAP_PROP_FPS))
        self.clock_start = time.monotonic()

    def read(self):
        ok,bgr = self.capture.read()
        if not ok:
            return None
        self.index += 1
        rgb = self.cv2.cvtColor(bgr,self.cv2.COLOR_BGR2RGB)
        if rgb.shape[1]>self.width:
            rgb = self.cv2.resize(rgb,(self.width,round(rgb.shape[0]*self.width/rgb.shape[1])))
        h,w = rgb.shape[:2]
        stamp = time.monotonic() if self.is_camera or not self.fps or self.fps<=0 else self.clock_start+(self.index-1)/self.fps
        return RgbdFrame(rgb,np.full((h,w),np.nan,np.float32),Intrinsics(w,h,1.,1.,0.,0.),
            np.eye(4),stamp,stamp,"RGB_WEBCAM_HOST_RECEIVE" if self.is_camera else "RGB_VIDEO",f"video-{self.index:06d}")

    def close(self):
        self.capture.release()


def capture_reality(output,source,frames=100,hz=10,width=640):
    """User-invoked RGB capture/import. It never opens a robot connection."""
    if frames<1 or hz<=0 or width<64:
        raise ValueError("Positive frames/rate and width at least 64 required")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty capture directory")
    output.mkdir(parents=True,exist_ok=True)
    video = VideoSource(source,width)
    rows = []
    try:
        for i in range(frames):
            start = time.monotonic();f = video.read()
            if f is None:break
            decode_ms = (time.monotonic()-start)*1000
            name = f"frame-{i:06d}.png"
            Image.fromarray(f.rgb).save(output/name)
            rows.append({"image":name,"timestamp_s":f.timestamp_s,"host_receive_monotonic_s":time.monotonic(),
                         "decode_ms":decode_ms})
            if video.is_camera:
                time.sleep(max(0.,1/hz-(time.monotonic()-start)))
    finally:
        video.close()
    manifest = {"format_version":1,"source":"webcam" if isinstance(source,int) else "video import",
                "acquisition_timestamp_s":None,"depth":None,"intrinsics":None,"poses":None,
                "frames":rows,"evaluation_cases":[],
                "notes":"RGB only. Add per-frame query/box or mask annotations and room IDs for evaluation. No 3D or measured camera acquisition claim."}
    (output/"reality.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest
