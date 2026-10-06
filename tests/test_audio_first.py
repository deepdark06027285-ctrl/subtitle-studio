import sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import engine,hardware


class AudioFirstTests(unittest.TestCase):
    def test_old_audio_only_directory_survives_enabling_ocr(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'movie.mp4';source.write_bytes(b'video')
            cfg={'whisper':temp,'translation':temp}
            old=root/('movie-'+engine._legacy_job_key(source,cfg,screen_text=False))
            old.mkdir();(old/'recognition.json').write_text('cached',encoding='utf-8')
            self.assertEqual(engine.job_directory(source,root,cfg,screen_text=True),old)
            self.assertEqual(engine.job_directory(source,root,cfg,screen_text=False),old)

    def exercise(self,retranslate=False,fail=False):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'movie.mp4';source.write_bytes(b'video')
            cfg={'whisper':temp,'translation':temp,'cpu_threads':4,'translation_backend':'hy-mt2'}
            job=engine.job_directory(source,root,cfg);job.mkdir()
            srt=job/'subtitles.zh.srt'
            original='1\n00:00:00,000 --> 00:00:01,000\n旧字幕（手动编辑）\n'
            srt.write_text(original,encoding='utf-8-sig')
            rows=[{'start':0,'end':1,'ja':'まさか本気で言ってるの？'}]
            engine.save_json(job/'recognition.json',{'complete':True,'rows':rows})
            recognition=(job/'recognition.json').read_bytes()
            engine.save_json(job/'translation.json',[{**rows[0],'track':'speech','zh':'旧译文'}])
            container=SimpleNamespace(duration=2000000,streams=SimpleNamespace(
                video=[SimpleNamespace(codec_context=SimpleNamespace(width=960,height=540))],
                audio=[SimpleNamespace(codec_context=SimpleNamespace(bit_rate=128000,name='aac'))]))
            class Open:
                def __enter__(self):return container
                def __exit__(self,*args):pass
            modules={'av':SimpleNamespace(open=lambda *a:Open(),time_base=1000000),
                'imageio_ffmpeg':SimpleNamespace(get_ffmpeg_exe=lambda:'unused'),
                'ctranslate2':SimpleNamespace(get_cuda_device_count=lambda:0),
                'psutil':SimpleNamespace(virtual_memory=lambda:SimpleNamespace(available=2**30))}
            class Model:
                parallel=4;backend={}
                def __init__(self,*a,**kw):pass
                def __enter__(self):return self
                def __exit__(self,*a):pass
                def translate_batch(self,texts):
                    if fail:raise RuntimeError('test translation failure')
                    return ['你认真的吗？']*len(texts)
            with patch.dict(sys.modules,modules),patch.object(engine,'load_config',return_value=cfg),\
                 patch.object(engine,'configure'),patch.object(hardware,'query_gpus',return_value=[]),\
                 patch.object(engine,'emit'),patch.object(engine,'prepare_speech',side_effect=AssertionError('ASR rerun')),\
                 patch.object(engine,'run_screen_text',side_effect=AssertionError('OCR started')),\
                 patch('local_translation.LocalTranslator',Model):
                if fail:
                    with self.assertRaisesRegex(RuntimeError,'translation failure'):
                        engine.run(source,root,cpu=True,subtitles_only=True,retranslate=retranslate)
                else:
                    engine.run(source,root,cpu=True,subtitles_only=True,retranslate=retranslate)
            self.assertEqual((job/'recognition.json').read_bytes(),recognition)
            self.assertFalse(engine.read_json(job/'processing.json')['translate_screen_text'])
            backups=list(job.glob('subtitles.zh.backup-*.srt'))
            if retranslate and not fail:
                self.assertIn('你认真的吗？',srt.read_text(encoding='utf-8-sig'))
                self.assertEqual(len(backups),1)
                self.assertEqual(backups[0].read_text(encoding='utf-8-sig'),original)
                self.assertEqual(engine.read_json(job/'translation.json')[0]['zh'],'旧译文')
                self.assertEqual(engine.read_json(job/'translation.natural-v1.json')[0]['zh'],'你认真的吗？')
            else:
                self.assertEqual(srt.read_text(encoding='utf-8-sig'),original)
                self.assertFalse(backups)

    def test_default_audio_only_preserves_edited_subtitles(self):self.exercise()
    def test_explicit_natural_retranslation_reuses_asr_and_backs_up(self):self.exercise(retranslate=True)
    def test_translation_failure_preserves_old_subtitles(self):self.exercise(retranslate=True,fail=True)
    def test_invalid_retranslation_does_not_initialize_hardware(self):
        with patch.object(engine,'load_config',side_effect=AssertionError('hardware setup')):
            with self.assertRaisesRegex(ValueError,'清空'):
                engine.run('ignored','ignored',external_srt='existing.srt',retranslate=True)


if __name__=='__main__':unittest.main()
