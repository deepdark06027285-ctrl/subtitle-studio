import sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from batch_ocr import predict_frames


class BatchOcrTests(unittest.TestCase):
    def make_pipeline(self):
        def crop(_,polys):
            return [SimpleNamespace(size=1 if poly[2]>0 else 0,shape=(10,poly[2]*10)) for poly in polys]
        return SimpleNamespace(use_doc_preprocessor=False,use_textline_orientation=False,text_type='general',
            text_rec_score_thresh=.75,_sort_boxes=lambda polys:polys,_crop_by_polys=crop)

    def test_cross_frame_crops_share_one_call_and_restore_frame_box_order(self):
        pipeline=self.make_pipeline()
        polys=[[(0,0,2,1),(0,1,9,2)],[],[(0,0,3,1),(0,0,-1,1)]]
        pipeline.text_det_model=Mock(return_value=[{'dt_polys':p} for p in polys])
        pipeline.text_rec_model=Mock(return_value=[{'rec_text':'短い','rec_score':.9},
            {'rec_text':'weak','rec_score':.4},{'rec_text':'長い文章','rec_score':.95}])
        ocr=SimpleNamespace(paddlex_pipeline=pipeline)
        components=SimpleNamespace(convert_points_to_boxes=lambda boxes:boxes)
        with patch.dict(sys.modules,{'paddlex.inference.pipelines.components':components}):
            results=predict_frames(ocr,['frame0','frame1','frame2'],lambda:None)
        self.assertEqual(results[0]['rec_texts'],['短い','長い文章'])
        self.assertEqual(results[0]['rec_boxes'],polys[0])
        self.assertEqual(results[1]['rec_texts'],[]);self.assertEqual(results[2]['rec_texts'],[])
        pipeline.text_rec_model.assert_called_once()
        self.assertEqual([crop.shape[1] for crop in pipeline.text_rec_model.call_args.args[0]],[20,30,90])

    def test_single_frame_uses_original_pipeline(self):
        ocr=SimpleNamespace(predict=Mock(return_value=['original']))
        self.assertEqual(predict_frames(ocr,['frame'],lambda:None),['original'])
        ocr.predict.assert_called_once_with(['frame'])

    def test_incomplete_detection_or_recognition_is_an_error(self):
        pipeline=self.make_pipeline();pipeline.text_det_model=Mock(return_value=[])
        ocr=SimpleNamespace(paddlex_pipeline=pipeline)
        components=SimpleNamespace(convert_points_to_boxes=lambda boxes:boxes)
        with patch.dict(sys.modules,{'paddlex.inference.pipelines.components':components}):
            with self.assertRaisesRegex(RuntimeError,'检测結果|检测结果'):
                predict_frames(ocr,['a','b'],lambda:None)
            pipeline.text_det_model.return_value=[{'dt_polys':[(0,0,1,1)]}]*2
            pipeline.text_rec_model=Mock(return_value=[])
            with self.assertRaisesRegex(RuntimeError,'识别结果'):
                predict_frames(ocr,['a','b'],lambda:None)

    def test_cancel_before_model_call_does_not_run_inference(self):
        pipeline=self.make_pipeline();pipeline.text_det_model=Mock()
        ocr=SimpleNamespace(paddlex_pipeline=pipeline)
        def cancel():raise RuntimeError('cancel')
        components=SimpleNamespace(convert_points_to_boxes=lambda boxes:boxes)
        with patch.dict(sys.modules,{'paddlex.inference.pipelines.components':components}):
            with self.assertRaisesRegex(RuntimeError,'cancel'):
                predict_frames(ocr,['a','b'],cancel)
        pipeline.text_det_model.assert_not_called()
