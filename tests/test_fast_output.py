import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import engine


class FastOutputTests(unittest.TestCase):
    def test_soft_output_copies_video_audio_and_embeds_ass_with_optional_font(self):
        with tempfile.TemporaryDirectory() as temp:
            font=Path(temp)/'字体.ttc';font.write_bytes(b'font')
            args=engine.soft_subtitle_args('源 视频.mp4','subtitles.zh.ass','result.partial.mkv',font)
            for option in ('-c:v','-c:a'):self.assertEqual(args[args.index(option)+1],'copy')
            self.assertEqual(args[args.index('-c:s')+1],'ass')
            self.assertEqual(args[args.index('-disposition:s:0')+1],'default')
            self.assertEqual(args[args.index('-attach')+1],str(font))
            self.assertNotIn('-vf',args)
            self.assertEqual(args[-1],'result.partial.mkv')

    def test_missing_optional_font_does_not_block_output(self):
        args=engine.soft_subtitle_args('source.mp4','subtitle.ass','partial.mkv','nonexistent-font.ttc')
        self.assertNotIn('-attach',args)


if __name__=='__main__':unittest.main()
