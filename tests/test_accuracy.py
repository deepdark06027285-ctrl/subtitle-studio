from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from screen_text import track_frame,japanese_text
from subtitle_layout import write_ass,read_srt
from engine import enhance_quiet_audio,load_config,configure
configure(load_config())
import numpy as np

class AccuracyTests(unittest.TestCase):
    def test_quiet_gain_preserves_source_and_does_not_clip(self):
        signal=np.full(16000,.001,dtype=np.float32)
        signal[100]=.15
        original=signal.copy()
        enhanced=enhance_quiet_audio(signal)
        np.testing.assert_array_equal(original,signal)
        self.assertGreater(enhanced[0],signal[0])
        self.assertLessEqual(np.max(np.abs(enhanced)),.981)
        quiet=np.full(16000,.00001,dtype=np.float32)
        np.testing.assert_array_equal(quiet,enhance_quiet_audio(quiet))

    def test_tracking_merges_persistent_text_and_ends_when_gone(self):
        state={}
        detection={'ja':'おはようございます','score':.95,'box':[.1,.2,.6,.3]}
        track_frame(state,[detection],0,1,10)
        track_frame(state,[detection],1,1,10)
        track_frame(state,[],2,1,10)
        self.assertEqual(len(state['rows']),1)
        self.assertEqual((state['rows'][0]['start'],state['rows'][0]['end']),(0,2))
        track_frame(state,[{**detection,'score':.2}],3,1,10)
        self.assertEqual(len(state['active']),0)
        self.assertFalse(japanese_text('12345'))

    def test_ass_keeps_tracks_and_escapes_detected_control_text(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'中文.ass'
            voice=[{'start':1,'end':3,'zh':'对白'}]
            screen=[{'start':1,'end':3,'zh':r'中文{\pos(0,0)}','box':[.2,.2,.6,.3]}]
            write_ass(voice,screen,path,960,540,position='near')
            text=path.read_text(encoding='utf-8-sig')
            self.assertIn(',Speech,',text)
            self.assertIn(r'\pos(384,170)',text)
            self.assertIn('｛＼pos',text)
            write_ass(voice,screen,path,960,540,position='top')
            self.assertIn(r'\pos(480,54)',path.read_text(encoding='utf-8-sig'))

    def test_existing_srt_keeps_unicode_and_multiline(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'中文.srt'
            path.write_text('1\n00:00:01,200 --> 00:00:03,100\n第一行\n第二行\n',encoding='utf-8-sig')
            self.assertEqual(read_srt(path),[{'start':1.2,'end':3.1,'zh':'第一行\n第二行'}])
