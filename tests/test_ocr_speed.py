from pathlib import Path
import sys,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import engine
from screen_text import select_ocr_device,scan_progress

class ScanSpeedTests(unittest.TestCase):
    def test_gpu_selection_requires_cuda_build_and_device(self):
        paddle=SimpleNamespace(is_compiled_with_cuda=Mock(return_value=True),
            device=SimpleNamespace(cuda=SimpleNamespace(device_count=Mock(return_value=1))))
        self.assertEqual(select_ocr_device(paddle),'gpu:0')
        self.assertEqual(select_ocr_device(paddle,cpu=True),'cpu')
        paddle.is_compiled_with_cuda.return_value=False
        with self.assertRaisesRegex(RuntimeError,'需要 NVIDIA'):
            select_ocr_device(paddle)
        paddle.is_compiled_with_cuda.return_value=True
        paddle.device.cuda.device_count.return_value=0
        with self.assertRaisesRegex(RuntimeError,'需要 NVIDIA'):
            select_ocr_device(paddle)

    def test_fallback_restarts_in_compatible_process_and_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            job=Path(temp)
            checkpoint={'next_time':145.14,'rows':[{'ja':'原文'}],'complete':False}
            engine.save_json(job/'screen-recognition.json',checkpoint)
            with patch.object(engine,'screen_worker',side_effect=[RuntimeError('decode'),['ok']]) as worker:
                with patch.object(engine,'emit'):
                    self.assertEqual(engine.run_screen_text('video',{},job,None,1000),['ok'])
            self.assertEqual([call.args[-2:] for call in worker.call_args_list],
                [(False,False),(False,True)])
            self.assertEqual(engine.read_json(job/'screen-recognition.json'),checkpoint)
            self.assertEqual(len(engine.read_json(job/'screen-fallback.json')['attempts']),1)

    def test_failed_gpu_does_not_silently_run_cpu(self):
        with tempfile.TemporaryDirectory() as temp:
            job=Path(temp)
            with patch.object(engine,'screen_worker',side_effect=RuntimeError('CUDA unavailable')) as worker:
                with patch.object(engine,'emit'),self.assertRaisesRegex(RuntimeError,'断点已保留'):
                    engine.run_screen_text('video',{},job,None,100)
            self.assertEqual(worker.call_count,2)
            self.assertTrue(all(not call.args[-2] for call in worker.call_args_list))

    def test_scan_eta_is_clamped_and_uses_only_resumed_work(self):
        processed,fraction,eta,speed=scan_progress(20.02,10,20,5,11)
        self.assertEqual((processed,fraction,eta,speed),(20,1,0,2))
        self.assertEqual(scan_progress(15,10,20,2.5,6),(15,.75,2.5,2))

    def test_cpu_choice_does_not_try_gpu_and_cancel_is_not_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            job=Path(temp)
            with patch.object(engine,'screen_worker',return_value=[]) as worker:
                engine.run_screen_text('video',{},job,None,100,cpu=True)
                self.assertEqual(worker.call_args.args[-2:],(True,True))
            with patch.object(engine,'screen_worker',side_effect=engine.Cancelled('stop')) as worker:
                with self.assertRaises(engine.Cancelled):
                    engine.run_screen_text('video',{},job,None,100)
                self.assertEqual(worker.call_count,1)

    def test_acceleration_assets_do_not_invalidate_finished_dialogue(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'movie.mp4';source.write_bytes(b'video')
            cfg={'whisper':temp,'translation':temp}
            key=engine.job_key(source,cfg)
            cfg.update(ocr_gpu_dependencies='new-runtime',ocr_gpu_cuda='new-cuda')
            self.assertEqual(engine.job_key(source,cfg),key)
