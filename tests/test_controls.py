from pathlib import Path
import os,subprocess,sys,tempfile,time,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import engine,hardware
from local_translation import LocalTranslator
from runtime_options import profile


class ControlTests(unittest.TestCase):
    def test_device_inventory_keeps_unavailable_metrics_unknown(self):
        result=subprocess.CompletedProcess([],0,stdout='0, GPU-A, NVIDIA RTX 4070, 8192, 2048, 27, N/A, 12\n')
        with patch.object(hardware.subprocess,'run',return_value=result):
            devices=hardware.query_gpus()
        self.assertEqual(devices[0]['encoder_utilization'],None)
        self.assertEqual(devices[0]['utilization'],27)
        self.assertEqual(hardware.choose_gpu(devices,'GPU-A')['uuid'],'GPU-A')
        with self.assertRaises(RuntimeError):
            hardware.choose_gpu(devices,'GPU-removed')

    def test_selection_routes_by_uuid_without_changing_global_settings(self):
        with patch.dict(os.environ,{},clear=False):
            hardware.route_gpu({'index':5,'uuid':'GPU-B'})
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'GPU-B')
            hardware.route_gpu(None)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'')

    def test_presets_change_real_search_parallelism_and_encoder_parameters(self):
        self.assertLess(profile('fast')['beam_size'],profile('quality')['beam_size'])
        self.assertGreater(profile('fast')['translation_parallel'],profile('quality')['translation_parallel'])
        fast=engine.encoder_args('ff',10,1000000,encoder='hevc_nvenc',speed='fast')
        fine=engine.encoder_args('ff',10,1000000,encoder='hevc_nvenc',speed='quality')
        self.assertEqual(fast[fast.index('-preset')+1],'p3')
        self.assertEqual(fine[fine.index('-preset')+1],'p6')
        self.assertEqual(fast[fast.index('-gpu')+1],'0')
        self.assertEqual(fast[fast.index('-b:v')+1],fine[fine.index('-b:v')+1])

    def test_driver_based_video_processing_maps_uuid_to_driver_ordinal(self):
        devices=[{'uuid':'GPU-A','index':0},{'uuid':'GPU-B','index':2}]
        with patch.object(hardware,'cuda_driver_devices',return_value=devices):
            ordinal=hardware.cuda_driver_index({'uuid':'GPU-B','index':5})
            self.assertEqual(ordinal,2)
            with self.assertRaises(RuntimeError):
                hardware.cuda_driver_index({'uuid':'GPU-removed'})
        args=engine.encoder_args('ff',10,1000000,encoder='hevc_nvenc',device_index=ordinal)
        self.assertEqual(args[args.index('-gpu')+1],'2')

    def test_translation_retries_truncated_output_before_saving(self):
        model=LocalTranslator({'speed':'fast'},Path('.'),lambda:None)
        with patch.object(model,'_request',side_effect=[('半句','length'),('完整译文','stop')]) as request:
            self.assertEqual(model.translate('今日はいい天気ですね'),'完整译文')
            self.assertGreater(request.call_args_list[1].args[1],request.call_args_list[0].args[1])

    def test_parallel_requests_keep_input_order(self):
        model=LocalTranslator({'speed':'fast'},Path('.'),lambda:None)
        def translate(text):
            time.sleep(.04 if text=='first' else .005)
            return text+' translated'
        with patch.object(model,'translate',side_effect=translate):
            self.assertEqual(model.translate_batch(['first','second']),['first translated','second translated'])

    def test_repeated_dialogue_reuses_translation_without_merging_rows(self):
        calls=[]
        class Model:
            parallel=4
            backend={'active_device_type':'gpu','active_device_name':'test GPU'}
            def __init__(self,*args,**kwargs): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def translate_batch(self,texts):
                calls.extend(texts)
                return ['译:'+text for text in texts]
        rows=[{'ja':text,'start':i,'end':i+1,'track':'speech'} for i,text in enumerate(['うん','はい','うん','うん'])]
        with tempfile.TemporaryDirectory() as temp:
            with patch('local_translation.LocalTranslator',Model),patch.object(engine,'emit'):
                result=engine.translate_local(rows,{'speed':'fast'},Path(temp),None,'cuda','translation.json')
            self.assertEqual(calls,['うん','はい'])
            self.assertEqual([r['start'] for r in result],[0,1,2,3])
            self.assertEqual(result[0]['zh'],result[2]['zh'])
            self.assertEqual(engine.read_json(Path(temp)/'translation-performance.json')['reused_rows_this_run'],2)

    def test_equal_length_stale_translation_is_not_reused(self):
        class Model:
            parallel=1
            backend={}
            def __init__(self,*args,**kwargs): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def translate_batch(self,texts): return ['新译文']*len(texts)
        with tempfile.TemporaryDirectory() as temp:
            job=Path(temp)
            engine.save_json(job/'translation.json',[{'ja':'旧原文','start':0,'end':1,'zh':'旧译文'}])
            with patch('local_translation.LocalTranslator',Model),patch.object(engine,'emit'):
                result=engine.translate_local([{'ja':'新原文','start':0,'end':1}],{},job,None,'cuda','translation.json')
            self.assertEqual(result[0]['zh'],'新译文')

    def test_only_repeated_remaining_rows_do_not_reload_model(self):
        first={'ja':'はい','start':0,'end':1,'track':'speech','zh':'是的'}
        second={'ja':'はい','start':2,'end':3,'track':'speech'}
        with tempfile.TemporaryDirectory() as temp:
            job=Path(temp)
            engine.save_json(job/'translation.json',[first])
            with patch('local_translation.LocalTranslator',side_effect=AssertionError('model loaded')):
                with patch.object(engine,'emit'):
                    result=engine.translate_local([first,second],{},job,None,'cuda','translation.json')
            self.assertEqual(result[1]['zh'],'是的')
            self.assertEqual(engine.read_json(job/'translation-performance.json')['model_requests_this_run'],0)


if __name__=='__main__': unittest.main()
