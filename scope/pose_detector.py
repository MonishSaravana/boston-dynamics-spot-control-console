"""Optional maintained TorchVision multi-person pose model; lazy imports."""

import hashlib
from pathlib import Path
import threading

import numpy as np

from .detector import _ensure_official_weights
from .humans import HumanKeypoint2D, JOINTS, PersonObservation2D
from .runtime import Runtime,ModuleConfig


class TorchvisionPoseDetector:
    name = "torchvision-keypointrcnn-resnet50-fpn-coco-v1"

    def __init__(self, model_width=640, score_threshold=.7, joint_score_min=2.,hand_model=None):
        try:
            import torch
            import torchvision
            from torchvision.models.detection import (
                KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn)
        except ImportError as exc:
            raise RuntimeError("Install pose support: pip install -e '.[pose]'") from exc
        self.torch = torch
        self.weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
        _ensure_official_weights(torch,self.weights)
        self.model = keypointrcnn_resnet50_fpn(weights=self.weights,min_size=model_width,
                                              max_size=model_width*2).eval().to("cpu")
        torch.set_num_threads(min(4,torch.get_num_threads()))
        self.score_threshold, self.joint_score_min = score_threshold,joint_score_min
        self.lock = threading.Lock()  # Shared model/hand runtime across camera workers.
        self.hands = None
        self.hand_runtime = Runtime({"hand_pose":ModuleConfig(enabled=bool(hand_model))})
        if hand_model:
            from .hand_pose import HandPoseAdapter
            self.hands = self.hand_runtime.run("hand_pose",0.,lambda:HandPoseAdapter(hand_model))
        self.metadata = {"backend":"torchvision", "model": self.name, "weights": self.weights.name,
                         "torch": torch.__version__, "torchvision": torchvision.__version__,
                         "device": "cpu", "model_width": model_width,
                         "joint_quality": "heatmap score heuristic; not calibrated visibility"}
        if hand_model and Path(hand_model).is_file():
            self.metadata["hand_model"] = {"filename":Path(hand_model).name,
                "sha256":hashlib.sha256(Path(hand_model).read_bytes()).hexdigest()}

    def detect(self, frame, camera_id):
        with self.lock:
            return self._detect(frame,camera_id)

    def _detect(self, frame, camera_id):
        image = self.torch.from_numpy(frame.rgb.copy()).permute(2,0,1).float()/255.
        with self.torch.inference_mode():
            result = self.model([image])[0]
        observations = []
        for i,score in enumerate(result["scores"].numpy()):
            if score<self.score_threshold:
                continue
            keypoints = result["keypoints"][i].numpy()
            scores = result["keypoints_scores"][i].numpy()
            found = {}
            for name,k,raw in zip(JOINTS,keypoints,scores):
                # TorchVision's v output is NOT a measured visibility confidence.
                # Heatmap strength only supplies a conservative heuristic gate.
                quality = float(np.clip(raw/10.,0.,1.)) if raw>=self.joint_score_min else 0.
                found[name] = HumanKeypoint2D(name,k[:2].astype(float),quality,
                                             2.+6.*(1.-quality))
            observations.append(PersonObservation2D(f"{frame.frame_id}:pose-{i}",frame.frame_id,
                camera_id,frame.timestamp_s,found,tuple(result["boxes"][i].numpy().tolist()),
                float(score),self.name))
        if self.hands:
            attached = self.hand_runtime.run("hand_pose",frame.timestamp_s,
                lambda:self.hands.attach(frame,observations))
            return attached if attached is not None else observations
        return observations

    def close(self):
        if self.hands:
            self.hands.close()
