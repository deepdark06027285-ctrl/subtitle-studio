import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from subtitle_layout import write_ass,normalize_style


class StyleTests(unittest.TestCase):
    def test_bottom_overlap_is_one_block_with_exact_entry_and_exit_times(self):
        speech=[{'start':1,'end':3,'zh':'对白'}]
        screen=[{'start':2,'end':4,'zh':'画面文字','box':[.1,.1,.8,.2]}]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'styled.ass'
            write_ass(speech,screen,path,1920,1080,font_size=60,color='#123456')
            text=path.read_text(encoding='utf-8-sig')
        self.assertIn('Microsoft YaHei,60,&H00563412',text)
        self.assertNotIn(r'\pos',text)
        events=[line for line in text.splitlines() if line.startswith('Dialogue:')]
        self.assertEqual(len(events),3)
        self.assertIn('0:00:01.00,0:00:02.00,Speech',events[0])
        self.assertIn(r'画面文字\N对白',events[1])
        self.assertIn('0:00:03.00,0:00:04.00,Screen',events[2])
        styles=[line.split(',') for line in text.splitlines() if line.startswith('Style:')]
        self.assertTrue(all(row[18]=='2' for row in styles))

    def test_speech_only_exports_configurable_ass_and_scales_with_resolution(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'styled.ass'
            write_ass([{'start':0,'end':2,'zh':'对白{test}'}],[],path,960,540,font_size=48,color='#abcdef')
            text=path.read_text(encoding='utf-8-sig')
        self.assertIn('Microsoft YaHei,24,&H00EFCDAB',text)
        self.assertIn('对白｛test｝',text)

    def test_invalid_style_does_not_overwrite_existing_subtitles(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'styled.ass';path.write_text('existing',encoding='utf-8')
            for size,color in [(0,'#FFFFFF'),(97,'#FFFFFF'),(44,'white'),(44,'#FFF'),(44,'#FF0000,{bad}')]:
                with self.subTest(size=size,color=color),self.assertRaises(ValueError):
                    write_ass([],[],path,1920,1080,font_size=size,color=color)
                self.assertEqual(path.read_text(encoding='utf-8'),'existing')

    def test_equal_concurrent_text_is_not_duplicated_and_zero_duration_is_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'styled.ass'
            write_ass([{'start':0,'end':2,'zh':'相同'}],
                [{'start':0,'end':2,'zh':'相同'},{'start':2,'end':2,'zh':'忽略'}],path,1920,1080)
            events=[line for line in path.read_text(encoding='utf-8-sig').splitlines() if line.startswith('Dialogue:')]
        self.assertEqual(len(events),1)
        self.assertTrue(events[0].endswith('相同'));self.assertNotIn(r'\N',events[0])

if __name__=='__main__':unittest.main()
