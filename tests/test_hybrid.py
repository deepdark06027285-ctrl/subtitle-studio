import json,subprocess,sys,threading,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from frame_pipeline import prepared_frames
import hardware,engine


class HybridTests(unittest.TestCase):
    def test_cpu_prepares_next_frame_while_consumer_is_working(self):
        consuming=threading.Event();prepared=threading.Event()
        def producer():
            yield 0
            self.assertTrue(consuming.wait(1))
            prepared.set();yield 1
        with prepared_frames(producer,lambda:None,True,2) as frames:
            self.assertEqual(next(frames),0)
            consuming.set()
            self.assertTrue(prepared.wait(1))
            self.assertEqual(list(frames),[1])

    def test_prefetch_preserves_order_and_producer_error(self):
        def producer():
            yield 0;yield 1
            raise ValueError('decode failed')
        with prepared_frames(producer,lambda:None,True,2) as frames:
            self.assertEqual(next(frames),0);self.assertEqual(next(frames),1)
            with self.assertRaisesRegex(ValueError,'decode failed'): next(frames)

    def test_cancel_releases_full_queue_without_checkpoint_advance(self):
        cancel=threading.Event();finished=threading.Event()
        def producer():
            try:
                for i in range(1000): yield i
            finally: finished.set()
        def check():
            if cancel.is_set(): raise engine.Cancelled('stop')
        with prepared_frames(producer,check,True,1) as frames:
            self.assertEqual(next(frames),0)
            cancel.set()
            with self.assertRaises(engine.Cancelled): next(frames)
        self.assertTrue(finished.wait(1))

    def test_adapter_inventory_reports_amd_and_virtual_without_fake_cuda(self):
        rows=[{'Name':'AMD Radeon RX 7800 XT','PNPDeviceID':'PCI\\VEN_1002'},
            {'Name':'Remote Display','PNPDeviceID':'ROOT\\DISPLAY'}]
        with patch.object(hardware,'windows_display_devices',return_value=rows):
            adapters=hardware.query_display_adapters([{'name':'NVIDIA RTX 4080'}])
        self.assertTrue(adapters[0]['supported'])
        self.assertFalse(adapters[1]['supported'])
        self.assertIn('虚拟',adapters[2]['note'])

    def test_auto_gpu_selection_handles_other_computer_models(self):
        devices=[{'uuid':'GPU-A','index':0,'name':'NVIDIA RTX 3060','memory_total':6144},
            {'uuid':'GPU-B','index':1,'name':'NVIDIA RTX 4080','memory_total':16384}]
        self.assertEqual(hardware.choose_gpu(devices)['uuid'],'GPU-B')
        self.assertIsNone(hardware.choose_gpu(devices,'cpu'))


if __name__=='__main__': unittest.main()
