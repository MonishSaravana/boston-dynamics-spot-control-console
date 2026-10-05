"""Bounded 2D target tracking between discovery/verification calls."""

from dataclasses import dataclass, field

import numpy as np

from .semantic_query import ObjectCandidate, QueryState, box_iou


@dataclass
class ObjectTrack:
    track_id: str
    candidate: ObjectCandidate
    queries: set[str]
    verified_s: float
    updated_s: float
    gray: np.ndarray
    histogram: np.ndarray
    points: np.ndarray | None
    state: QueryState = QueryState.FOUND
    reason: str = "Detector initialized mask"
    history: list = field(default_factory=list)


class MaskTracker:
    def __init__(self, max_age_s=3., max_tracks=24):
        if max_age_s <= 0 or max_tracks < 1:
            raise ValueError("Positive age and track capacity required")
        self.max_age_s, self.max_tracks = max_age_s, max_tracks
        self.tracks, self.counter = {}, 0

    @staticmethod
    def _histogram(rgb, mask):
        import cv2
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h = cv2.calcHist([hsv], [0,1], mask.astype(np.uint8), [18,16], [0,180,0,256])
        return cv2.normalize(h, h).astype(np.float32)

    @staticmethod
    def _points(gray, mask):
        import cv2
        return cv2.goodFeaturesToTrack(gray,80,.01,4,mask=mask.astype(np.uint8)*255)

    def initialize(self, frame, query, candidates):
        import cv2
        gray = cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2GRAY)
        initialized, used = [], set()
        for candidate in candidates:
            if candidate.mask is None:
                continue
            matches = sorted(((box_iou(candidate.box_xyxy,t.candidate.box_xyxy),t)
                for t in self.tracks.values() if t.track_id not in used and t.state not in
                (QueryState.TRACK_LOST,QueryState.STALE)), key=lambda x:x[0], reverse=True)
            # Identity requires a unique spatial match and compatible appearance.
            histogram = self._histogram(frame.rgb,candidate.mask)
            match = (matches[0][1] if matches and matches[0][0]>.45 and
                (len(matches)==1 or matches[0][0]-matches[1][0]>.15) and
                cv2.compareHist(histogram,matches[0][1].histogram,cv2.HISTCMP_BHATTACHARYYA)<.45 else None)
            if match:
                identity = match.track_id
                queries = match.queries|{query.normalized}
            else:
                self.counter += 1
                identity,queries = f"target-{self.counter:03d}",{query.normalized}
            candidate.track_id = identity
            if match:
                candidate.entity_id = match.candidate.entity_id
            track = ObjectTrack(identity,candidate,queries,frame.timestamp_s,frame.timestamp_s,
                gray.copy(),histogram,self._points(gray,candidate.mask))
            self.tracks[identity] = track
            used.add(identity)
            initialized.append(track)
        if len(self.tracks)>self.max_tracks:
            for old in sorted(self.tracks.values(),key=lambda t:t.updated_s)[:len(self.tracks)-self.max_tracks]:
                del self.tracks[old.track_id]
        return initialized

    def update(self, frame):
        import cv2
        gray = cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2GRAY)
        for t in self.tracks.values():
            if frame.timestamp_s <= t.updated_s:
                continue
            expired = frame.timestamp_s-t.verified_s > self.max_age_s
            if t.state==QueryState.TRACK_LOST:
                if expired:t.state,t.reason=QueryState.STALE,"Lost track and expired verification"
                continue
            if expired:
                t.state,t.reason=QueryState.STALE,"Detector verification expired; continued flow is only an estimate"
            if gray.shape!=t.gray.shape:
                t.state,t.reason = QueryState.TRACK_LOST,"Image geometry changed"
                continue
            if np.array_equal(gray,t.gray):
                t.updated_s = frame.timestamp_s
                t.reason = "Identical pixels retained; verification age still advances"
                continue
            if t.points is None or len(t.points)<4:
                t.state,t.reason = QueryState.TRACK_LOST,"Insufficient texture for optical flow"
                continue
            nxt,status,_ = cv2.calcOpticalFlowPyrLK(t.gray,gray,t.points,None,winSize=(21,21),maxLevel=3)
            if nxt is None:
                t.state,t.reason = QueryState.TRACK_LOST,"Optical flow unavailable"
                continue
            back,back_status,_ = cv2.calcOpticalFlowPyrLK(gray,t.gray,nxt,None,winSize=(21,21),maxLevel=3)
            if back is None:
                t.state,t.reason = QueryState.TRACK_LOST,"Backward flow unavailable"
                continue
            good = (status.ravel()!=0)&(back_status.ravel()!=0)&(np.linalg.norm(back-t.points,axis=2).ravel()<1.5)
            if good.sum()<4 or good.mean()<.35:
                t.state,t.reason = QueryState.TRACK_LOST,"Forward/backward flow disagreed"
                continue
            matrix,inliers = cv2.estimateAffinePartial2D(t.points[good],nxt[good],method=cv2.RANSAC,
                ransacReprojThreshold=2.,maxIters=200)
            scale = np.hypot(matrix[0,0],matrix[1,0]) if matrix is not None else 0.
            if (matrix is None or inliers.mean()<.6 or not .8<scale<1.25 or
                    np.linalg.norm(matrix[:,2])>max(gray.shape)*.4):
                t.state,t.reason = QueryState.TRACK_LOST,"Motion gate rejected drift"
                continue
            mask = cv2.warpAffine(t.candidate.mask.astype(np.uint8),matrix,(gray.shape[1],gray.shape[0]),
                                  flags=cv2.INTER_NEAREST).astype(bool)
            if mask.sum()<24:
                t.state,t.reason = QueryState.TRACK_LOST,"Mask left the view"
                continue
            hist = self._histogram(frame.rgb,mask)
            disagreement = cv2.compareHist(hist,t.histogram,cv2.HISTCMP_BHATTACHARYYA)
            if disagreement>.55:
                t.state,t.reason = QueryState.TRACK_LOST,"Appearance changed; detector must reacquire"
                continue
            yy,xx = np.nonzero(mask)
            t.candidate.mask = mask
            t.candidate.box_xyxy = (float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1))
            t.candidate.evidence["tracking"] = {"kind":"estimated mask from sparse optical flow",
                "verified_source_s":t.verified_s,"appearance_distance":float(disagreement),
                "flow_inlier_fraction":float(inliers.mean())}
            t.gray,t.points,t.updated_s = gray.copy(),self._points(gray,mask),frame.timestamp_s
            t.reason = "Verification expired; optical flow continues without a fresh object claim" if expired else "Optical-flow estimate; detector score is retained, not recomputed"
            t.history.append({"timestamp_s":frame.timestamp_s,"box_xyxy":t.candidate.box_xyxy})
            t.history = t.history[-60:]
        return list(self.tracks.values())

    def active(self):
        return [t for t in self.tracks.values() if t.state==QueryState.FOUND]
