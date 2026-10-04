"""Optional finger landmarks linked geometrically to a body observation.

The model bundle stays outside Git. No gesture-to-robot command connection.
"""
from dataclasses import replace

import numpy as np

from .humans import HumanKeypoint2D


class HandPoseAdapter:
    def __init__(self,model_path):
        import mediapipe as mp
        self.mp = mp
        self.model = mp.tasks.vision.GestureRecognizer.create_from_options(
            mp.tasks.vision.GestureRecognizerOptions(base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path),delegate=mp.tasks.BaseOptions.Delegate.CPU),
                num_hands=4,min_hand_detection_confidence=.7,min_hand_presence_confidence=.7))

    def attach(self,frame,observations):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=np.ascontiguousarray(frame.rgb))
        result = self.model.recognize(image)
        output = list(observations)
        used = set()
        for hand,handed in zip(result.hand_landmarks,result.handedness):
            xyz = np.array([[p.x,p.y,p.z] for p in hand])
            uv = xyz[:,:2]*[frame.intrinsics.width,frame.intrinsics.height]
            candidates = []
            for i,o in enumerate(output):
                for side in ("left","right"):
                    k = o.keypoints.get(f"{side}_wrist")
                    if k is not None and k.confidence>=.35 and (i,side) not in used:
                        candidates.append((np.linalg.norm(k.uv-uv[0]),i,side))
            candidates.sort()
            if not candidates or candidates[0][0]>65 or len(candidates)>1 and candidates[1][0]-candidates[0][0]<12:
                continue
            _,i,side = candidates[0]
            # Normalize X/Y to the same image scale before measuring extension.
            p = xyz*[frame.intrinsics.width,frame.intrinsics.height,frame.intrinsics.width]
            def extension(indices):
                chain = p[indices]
                return np.linalg.norm(chain[-1]-chain[0])/max(np.linalg.norm(np.diff(chain,axis=0),axis=1).sum(),1e-8)
            index = extension([5,6,7,8])
            others = np.mean([extension(ids) for ids in ([9,10,11,12],[13,14,15,16],[17,18,19,20])])
            quality = float(index**6*(1-others**4)*handed[0].score)
            keys = dict(output[i].keypoints)
            for name,j in (("mcp",5),("pip",6),("tip",8)):
                keys[f"{side}_index_{name}"] = HumanKeypoint2D(f"{side}_index_{name}",uv[j],
                    float(handed[0].score),1.5)
            # A separately estimated wrist improves the hand/body association.
            keys[f"{side}_wrist"] = HumanKeypoint2D(f"{side}_wrist",uv[0],float(handed[0].score),1.5)
            output[i] = replace(output[i],keypoints=keys,hand_quality={**output[i].hand_quality,side:quality})
            used.add((i,side))
        return output

    def close(self):
        self.model.close()
