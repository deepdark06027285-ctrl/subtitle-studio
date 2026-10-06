import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess
import os
import threading

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('engine',root/'engine.py')
engine=importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


class SubtitleTests(unittest.TestCase):
    def test_timing_unicode_and_overlap(self):
        rows=[{'start':2,'end':12,'zh':'第二句'},{'start':0,'end':5,'zh':'第一句'},
              {'start':15,'end':16,'zh':'⁇'}]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'字幕 空格.srt'
            self.assertEqual(engine.write_srt(rows,path,20),2)
            text=path.read_text(encoding='utf-8-sig')
            self.assertIn('00:00:00,000 --> 00:00:02,000',text)
            self.assertIn('第一句',text)
            self.assertNotIn('⁇',text)

    def test_checkpoint_cancel(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.json'
            engine.save_json(path,{'next':8,'rows':[{'ja':'ごみ'}]})
            self.assertEqual(engine.read_json(path)['next'],8)
            self.assertFalse(path.with_suffix('.json.tmp').exists())
            stop=Path(temp)/'stop'
            stop.touch()
            with self.assertRaises(engine.Cancelled): engine.check_cancel(stop)

    @unittest.skipUnless(os.name=='nt','Windows file sharing behavior')
    def test_checkpoint_survives_temporary_windows_reader_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.json'
            engine.save_json(path,{'next':1})
            reader=path.open('rb')
            release=threading.Timer(.12,reader.close)
            release.start()
            try:
                engine.save_json(path,{'next':2})
                self.assertEqual(engine.read_json(path),{'next':2})
                self.assertFalse(path.with_suffix('.json.tmp').exists())
            finally:
                release.join();reader.close()

    def test_permanent_checkpoint_failure_preserves_previous_state(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.json'
            engine.save_json(path,{'next':1})
            with patch.object(Path,'replace',side_effect=PermissionError('locked')):
                with patch.object(engine.time,'sleep'),self.assertRaises(PermissionError):
                    engine.save_json(path,{'next':2})
            self.assertEqual(engine.read_json(path),{'next':1})

    def test_noise_filter_and_short_speech(self):
        self.assertTrue(engine.sensible_speech('はい',-.3,.1))
        self.assertFalse(engine.sensible_speech('あああああああああああああああ',-.3,.1))
        self.assertFalse(engine.sensible_speech('英文 nonsense',-2,.1))
        self.assertTrue(engine.sensible_speech('こんにちは',-.3,.8))
        self.assertFalse(engine.sensible_speech('こんにちは',-1.05,.8))
        self.assertFalse(engine.sensible_speech('こんにちは',-.3,.8,soft_voice=False))

    def test_key_changes_when_source_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'movie.mp4';p.write_bytes(b'one')
            cfg={'whisper':temp,'translation':temp}
            first=engine.job_key(p,cfg)
            p.write_bytes(b'two-more')
            self.assertNotEqual(first,engine.job_key(p,cfg))

    def test_nvidia_probe_uses_supported_dimensions(self):
        with patch.object(engine.subprocess,'run',return_value=subprocess.CompletedProcess([],0,stderr='')) as run:
            self.assertEqual(engine.select_encoder('ffmpeg'),('hevc_nvenc',''))
        command=run.call_args.args[0]
        self.assertIn('color=s=256x256:d=0.1',command)
        self.assertEqual(run.call_args.kwargs['timeout'],15)

    def test_cpu_selection_and_hardware_failure_reason(self):
        with patch.object(engine.subprocess,'run') as run:
            self.assertEqual(engine.select_encoder('ffmpeg',False)[0],'libx265')
            run.assert_not_called()
        with patch.object(engine.subprocess,'run',return_value=subprocess.CompletedProcess([],1,stderr='encoder unavailable')):
            self.assertEqual(engine.select_encoder('ffmpeg'),('libx265','encoder unavailable'))
        with patch.object(engine.subprocess,'run',side_effect=subprocess.TimeoutExpired('ffmpeg',15)):
            self.assertEqual(engine.select_encoder('ffmpeg'),('libx265','显卡编码检测超时'))

    def test_cache_separates_models_and_processing_options(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'movie.mp4';p.write_bytes(b'video')
            model=Path(temp)/'translator.gguf';model.write_bytes(b'model')
            cfg={'whisper':temp,'translation':temp,'translation_gguf':str(model)}
            key=engine.job_key(p,cfg)
            self.assertNotEqual(key,engine.job_key(p,cfg,soft_voice=False))
            self.assertEqual(key,engine.job_key(p,cfg,screen_text=False))
            self.assertEqual(key,engine.job_key(p,cfg,screen_position='top'))
            self.assertEqual(key,engine._legacy_job_key(p,cfg,screen_position='near'))
            model.write_bytes(b'new-model')
            self.assertNotEqual(key,engine.job_key(p,cfg))

    def test_old_top_job_is_reused_when_canonical_job_does_not_exist(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'movie.mp4';source.write_bytes(b'video')
            cfg={'whisper':temp,'translation':temp}
            old=root/(source.stem+'-'+engine._legacy_job_key(source,cfg,screen_position='top'))
            old.mkdir();(old/'subtitles.zh.srt').write_text('user edited',encoding='utf-8')
            self.assertEqual(engine.job_directory(source,root,cfg),old)
            canonical=root/(source.stem+'-'+engine.job_key(source,cfg));canonical.mkdir()
            self.assertEqual(engine.job_directory(source,root,cfg),canonical)


if __name__=='__main__': unittest.main()
