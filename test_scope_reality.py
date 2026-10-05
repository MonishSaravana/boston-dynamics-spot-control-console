import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from scope.reality_recording import RealityRecording
from scope.perception_sources import image_frame
from scope.reality_benchmark import score_case
from scope.semantic_query import QueryResult,QueryState,ObjectCandidate,SemanticQuery
from scope.interaction import TextCommand


class RealityRecordingTests(unittest.TestCase):
    def test_explicit_absence_and_box_only_truth(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);Image.fromarray(np.zeros((30,40,3),np.uint8)).save(root/'frame.png')
            manifest={'frames':[{'image':'frame.png'}],'evaluation_cases':[
                {'frame':0,'room_id':'new-room','split':'heldout','query':'power strip','present':True,'boxes_xyxy':[[2,3,22,23]]},
                {'frame':0,'room_id':'new-room','split':'heldout','query':'flarblenox','present':False}]}
            path=root/'reality.json';path.write_text(json.dumps(manifest));r=RealityRecording(path)
            f=r.frame(0);self.assertFalse(np.isfinite(f.depth_m).any())
            masks,boxes=r.truth(0,'power strip',f.rgb.shape[:2]);q=SemanticQuery.from_command(TextCommand('power strip'))
            found=QueryResult(q,QueryState.FOUND,'test',[ObjectCandidate((2,3,22,23),.8,'power strip','test')],f.timestamp_s,f.frame_id)
            metric=score_case(f,found,masks,boxes)
            self.assertTrue(metric['success']);self.assertIsNone(metric['best_mask_iou'])
            self.assertEqual(metric['projected_candidates'],0)
            manifest['evaluation_cases'][1]['room_id']='new-room'
            manifest['evaluation_cases'][1]['split']='development';path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'both'):RealityRecording(path)

    def test_unannotated_presence_is_not_treated_as_absence(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'reality.json';p.write_text(json.dumps({'frames':[{'image':'x.png'}],
                'evaluation_cases':[{'frame':0,'room_id':'room','split':'heldout','query':'bag','present':True}]}))
            with self.assertRaisesRegex(ValueError,'require boxes'):RealityRecording(p)


if __name__=='__main__':unittest.main()
