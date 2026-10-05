"""Official labeled NYU RGB-D stills; no invented trajectories or filled depth."""

from pathlib import Path

import numpy as np

from .data import Intrinsics, RgbdFrame


class NyuDataset:
    """Each still defines its own camera-local world. Labels are evaluation only."""
    def __init__(self, path):
        import h5py
        self.file = h5py.File(Path(path), "r")
        def strings(name):
            return ["".join(chr(int(i)) for i in self.file[ref][:].ravel())
                    for ref in self.file[name][:].ravel()]
        self.scenes, self.types, self.names = strings("scenes"), strings("sceneTypes"), strings("names")
        # Official toolbox camera_params.m uses MATLAB's 1-based pixels.
        self.intrinsics = Intrinsics(640, 480, 518.8579011745019, 519.4696111212749,
                                     325.58244941119034-1, 253.73616633400465-1)

    def frame(self, index, timestamp_s=0.):
        if not 0 <= index < len(self.scenes):
            raise ValueError("NYU frame index out of range")
        rgb = self.file["images"][index].transpose(2,1,0)
        # Registered optical Z in meters, BEFORE colorization/inpainting.
        depth = self.file["rawDepths"][index].T.astype(np.float32)
        depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
        pose = np.eye(4)
        pose[:3,:3] = [[1,0,0],[0,0,1],[0,-1,0]]
        return RgbdFrame(rgb, depth, self.intrinsics, pose, timestamp_s, timestamp_s,
                         "MEASURED_NYU_REGISTERED_STILL", f"nyu-{index:04d}")

    def labels(self, index):
        return self.file["labels"][index].T.astype(int), self.file["instances"][index].T.astype(int)

    def metadata(self, index):
        return {"dataset": "NYU Depth V2", "index": index, "scene": self.scenes[index],
                "scene_type": self.types[index], "world_frame": f"camera-local:{index}",
                "acquisition_timestamp_s": None, "source_fps": None,
                "depth": "registered rawDepths in meters, no inpainting",
                "calibration": "official RGB pinhole, 1-based principal point converted to 0-based; RGB distortion not corrected",
                "pose": "declared optical-to-Z-up conversion, not measured room localization"}

    def close(self):
        self.file.close()
