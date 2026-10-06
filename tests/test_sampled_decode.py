import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sampled_decode import sampling_filter,PipeReader
from subtitle_layout import normalize_placement,placement_tag,write_ass
import tempfile


class SampledDecodeTests(unittest.TestCase):
    def test_sampling_threshold_matches_old_epsilon_and_resume(self):
        result=sampling_filter(145.14,1,'p010le')
        self.assertIn('145.139900000',result)
        self.assertIn('0.999900000',result)
        self.assertLess(result.index('select='),result.index('hwdownload'))
        self.assertNotIn('scale',result)
        self.assertIn('hwdownload,format=p010le,format=yuv420p10le',result)
        with self.assertRaises(ValueError):sampling_filter(0,0,'nv12')
        with self.assertRaises(ValueError):sampling_filter(0,1,'unsafe')

    def test_live_pipe_does_not_expose_seek(self):
        import io
        reader=PipeReader(io.BytesIO(b'frames'))
        self.assertFalse(hasattr(reader,'seek'));self.assertEqual(reader.read(3),b'fra')

    def test_all_tracks_follow_selected_anchor_and_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'style.ass'
            write_ass([{'start':0,'end':2,'zh':'对白'}],[{'start':0,'end':2,'zh':'译文'}],
                p,1920,1080,anchor=8,x=40,y=8)
            text=p.read_text(encoding='utf-8-sig')
        self.assertIn(r'{\an8\pos(768,86)}译文\N对白',text)
        self.assertEqual(normalize_placement(2,50,95),(2,50,95))
        for values in ((0,50,95),(10,0,0),(2,-1,0),(2,50,101),(2,float('nan'),50)):
            with self.assertRaises(ValueError):normalize_placement(*values)

if __name__=='__main__':unittest.main()
