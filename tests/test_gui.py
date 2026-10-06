"""Verify allocated button space, including DPI scaling and wrapped status."""
from pathlib import Path
import sys
import unittest
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import App,ctk,progress_text


class LayoutTests(unittest.TestCase):
    def test_buttons_at_multiple_scales_and_smallest_window(self):
        root=ctk.CTk()
        root.withdraw()
        root.attributes('-alpha',0)
        app=App(root)
        try:
            self.assertTrue(app.soft.get())
            self.assertFalse(app.screen.get())
            self.assertEqual(app.translation_style.get(),'自然口语')
            self.assertFalse(app.retranslate.get())
            self.assertEqual(app.output_mode.get(),'固定在画面')
            self.assertEqual(app.speed.get(),'高速')
            self.assertEqual(app.screen_position.get(),'跟随字幕位置')
            self.assertEqual(app.subtitle_position.get(),'底部中央')
            app.subtitle_position.set('顶部右侧');app.choose_subtitle_position()
            self.assertEqual((app.subtitle_anchor,app.subtitle_x.get(),app.subtitle_y.get()),(9,92,5))
            app.font_size.set(96);app.subtitle_color.set('#52DCC0');app.update_style_preview()
            self.assertEqual(app.preview_label.cget('text_color'),'#52DCC0')
            self.assertEqual(app.size_text.get(),'96')
            app.update_progress({'stage':'screen','progress':47.5,'stage_progress':50,
                'processed_seconds':15,'total_seconds':30,'eta_seconds':25})
            self.assertEqual(app.phase_name.get(),'识别画面日语')
            self.assertEqual(app.number.get(),'50%')
            app.update_progress({'stage':'screen','progress':47.5,'stage_progress':50,
                'processed_seconds':15,'total_seconds':30,'eta_seconds':3,
                'ocr_device':'NVIDIA','ocr_gpu':True,'scan_speed':12.47})
            self.assertEqual(app.phase_name.get(),'识别画面日语')
            self.assertIn('12.5×',app.detail.get())
            app.update_progress({'stage':'screen','ocr_device':'NVIDIA','ocr_gpu':True,
                'ocr_gpu_name':'NVIDIA GeForce RTX 4070 Laptop GPU',
                'active_device_type':'gpu','active_device_name':'NVIDIA GeForce RTX 4070 Laptop GPU',
                'active_gpu_uuid':'GPU-test'})
            self.assertIn('RTX 4070',app.hardware_text.get())
            for scale in (0.8,1.0,1.25):
                with self.subTest(scale=scale):
                    ctk.set_widget_scaling(scale)
                    ctk.set_window_scaling(scale)
                    root.deiconify()
                    root.update()
                    root.geometry(f'900x{app.minimum_height}')
                    root.update_idletasks();app.show_style_options()
                    self.assertEqual(app.workspace_choice.get(),'字幕设计')
                    self.assertTrue(app.style_form.winfo_ismapped())
                    self.assertFalse(app.form.winfo_ismapped())
                    self.assertEqual(app.style_form._parent_canvas.yview()[0],0)
                    app.update_progress({'message':'这是可能换行的较长进度说明。'*8,
                        'stage':'encode','progress':86.5,'stage_progress':55,
                        'processed_seconds':55,'total_seconds':100,'eta_seconds':30,
                        'encoder':'hevc_nvenc','encoding_speed':11.43,
                        'active_device_type':'gpu','active_device_name':'NVIDIA GeForce RTX 4070 Laptop GPU',
                        'active_gpu_uuid':'GPU-test'})
                    # Windows DPI changes settle asynchronously; wait for stable geometry.
                    deadline=time.monotonic()+2
                    last=None
                    stable=0
                    while time.monotonic()<deadline:
                        root.update()
                        dimensions=(root.winfo_height(),root.winfo_width(),
                            root.winfo_rooty(),app.start_button.winfo_rooty())
                        stable=stable+1 if dimensions==last else 0
                        last=dimensions
                        if stable>=3 and root.winfo_height()==round(app.minimum_height*root._get_window_scaling()):
                            break
                        root.after(100,root.quit)
                        root.mainloop()
                    self.assertIn('RTX 4070',app.hardware_text.get())
                    self.assertGreaterEqual(app.hardware_label.winfo_height(),app.hardware_label.winfo_reqheight())
                    app.update_hardware([{'index':0,'uuid':'GPU-test','name':'NVIDIA GeForce RTX 4070 Laptop GPU',
                        'memory_total':8188,'memory_used':8000,'utilization':100,
                        'encoder_utilization':100,'decoder_utilization':100}])
                    app.update_resources({'cpu_percent':100,'ram_used':96*1024**3,'ram_total':128*1024**3,
                        'ram_percent':75,'app_working_set':85*1024**3})
                    root.update_idletasks()
                    self.assertLessEqual(app.speed_note_label._label.winfo_reqheight(),app.speed_note_label.winfo_height())
                    self.assertEqual(app.device_state.get(),'当前正在使用')
                    self.assertIn('RTX 4070',app.device_name.get())
                    self.assertEqual(app.metric_values['gpu'].get(),'100%')
                    for label in list(app.metric_labels.values())+[app.status_label]+app.resource_labels:
                        pixels=root.tk.call('font','measure',label._label.cget('font'),label._label.cget('text'))
                        self.assertLessEqual(pixels,label.winfo_width())
                    for widget in (app.hardware_label,app.gpu_menu):
                        self.assertTrue(widget.winfo_ismapped())
                        self.assertGreaterEqual(widget.winfo_rootx(),root.winfo_rootx())
                        self.assertLessEqual(widget.winfo_rootx()+widget.winfo_width(),root.winfo_rootx()+root.winfo_width())
                    for button,label in ((app.start_button,'开始处理'),
                                         (app.cancel_button,'停止'),
                                         (app.folder_button,'打开结果文件夹')):
                        self.assertEqual(str(button.cget('text')),label)
                        self.assertTrue(button.winfo_ismapped())
                        self.assertGreaterEqual(button.winfo_height(),button.winfo_reqheight())
                        self.assertGreaterEqual(button.winfo_width(),button.winfo_reqwidth())
                        self.assertGreaterEqual(button.winfo_rooty(),root.winfo_rooty())
                        self.assertLessEqual(button.winfo_rooty()+button.winfo_height(),
                                             root.winfo_rooty()+root.winfo_height())
        finally:
            app.destroy()
            ctk.set_widget_scaling(1)
            ctk.set_window_scaling(1)

    def test_progress_uses_actual_stage_and_time(self):
        percent,detail,remaining=progress_text({'stage':'encode','progress':86.5,
            'stage_progress':55,'processed_seconds':55,'total_seconds':100,'eta_seconds':30})
        self.assertEqual(percent,55)
        self.assertEqual(detail,'已处理 00:00:55 / 00:01:40')
        self.assertEqual(remaining,'预计剩余 00:00:30')

    def test_encoding_speed_is_shown_with_video_progress(self):
        value,detail,_=progress_text({'stage':'encode','stage_progress':55,
            'processed_seconds':55,'total_seconds':100,'encoding_speed':11.43})
        self.assertEqual(value,55)
        self.assertEqual(detail,'已处理 00:00:55 / 00:01:40 · 11.4×')

    def test_screen_scan_progress_and_default_choices(self):
        value,detail,remaining=progress_text({'stage':'screen','progress':47.5,'stage_progress':50,
            'processed_seconds':15,'total_seconds':30,'eta_seconds':25})
        self.assertEqual(value,50)
        self.assertEqual(detail,'已处理 00:00:15 / 00:00:30')
        self.assertEqual(remaining,'预计剩余 00:00:25')

    def test_screen_scan_speed_uses_ocr_metric(self):
        value,detail,_=progress_text({'stage':'screen','stage_progress':12.5,
            'processed_seconds':15,'total_seconds':120,'scan_speed':12.47})
        self.assertEqual(value,12.5)
        self.assertEqual(detail,'已处理 00:00:15 / 00:02:00 · 12.5×')

    def test_hardware_monitor_distinguishes_selection_from_actual_work(self):
        root=ctk.CTk();root.withdraw()
        app=App(root)
        gpu={'index':1,'uuid':'GPU-test','name':'NVIDIA RTX 4070','memory_total':8192,
            'memory_used':2048,'utilization':27,'encoder_utilization':None,'decoder_utilization':12}
        try:
            app.update_hardware([gpu])
            name=next(label for label,value in app.gpu_map.items() if value=='GPU-test')
            app.gpu_choice.set(name);app.choose_device()
            self.assertIn('待机',app.hardware_text.get())
            self.assertNotIn('当前工作',app.hardware_text.get())
            app.update_progress({'active_device_type':'gpu','active_device_name':'NVIDIA RTX 4070',
                'active_gpu_uuid':'GPU-test'})
            self.assertIn('当前工作：NVIDIA RTX 4070',app.hardware_text.get())
            self.assertIn('GPU 27%',app.hardware_text.get())
            self.assertIn('编码 —',app.hardware_text.get())
            self.assertEqual(app.metric_values['encoder'].get(),'—')
            self.assertIn('RTX 4070',app.device_name.get())
            self.assertEqual(app.device_state.get(),'当前正在使用')
            app.update_progress({'stage':'encode','encoder':'stream_copy'})
            app.update_progress({'stage':'encode','progress':80,'stage_progress':20})
            self.assertEqual(app.phase_name.get(),'封装字幕视频')
            app.update_progress({'active_device_type':'hybrid','active_device_name':'NVIDIA RTX 4070',
                'active_gpu_uuid':'GPU-test'})
            self.assertEqual(app.device_state.get(),'当前协同工作 · CPU + GPU')
            self.assertIn('CPU + NVIDIA RTX 4070',app.hardware_text.get())
            app.update_hardware([{**gpu,'index':5}])
            self.assertEqual(app.requested_device,'GPU-test')
            self.assertTrue(app.gpu_choice.get().startswith('GPU 5'))
            app.update_progress({'active_device_type':'cpu','active_device_name':'CPU','active_gpu_uuid':None})
            self.assertIn('当前工作：CPU',app.hardware_text.get())
            self.assertNotIn('当前工作：NVIDIA',app.hardware_text.get())
            self.assertEqual(app.device_name.get(),'CPU · 处理器')
            self.assertEqual(app.device_state.get(),'当前正在使用')
            app.gpu_choice.set('CPU（较慢）');app.choose_device()
            self.assertEqual(app.requested_device,'cpu')
            self.assertTrue(app.cpu.get())
            self.assertEqual(app.hybrid_switch.cget('state'),'disabled')
            self.assertTrue(all(value.get()=='—' for value in app.metric_values.values()))
            gpu_label=next(label for label,value in app.gpu_map.items() if value=='GPU-test')
            app.gpu_choice.set(gpu_label);app.choose_device()
            self.assertEqual(app.requested_device,'GPU-test')
            self.assertFalse(app.cpu.get())
            self.assertEqual(app.hybrid_switch.cget('state'),'normal')
            self.assertEqual(app.device_name.get(),'CPU · 处理器')  # Selection cannot relabel active work.
            app.speed.set('精细');app.choose_speed()
            self.assertIn('逐句翻译',app.speed_note.get())
            with tempfile.TemporaryDirectory() as temp:
                folder=Path(temp)
                (folder/'Chinese-subtitled.mp4').touch()
                (folder/'encoding.json').write_text('{"complete":true}',encoding='utf-8')
                app.show_completed(folder)
                self.assertIsNone(app.worker)
                self.assertEqual(app.number.get(),'100%')
                self.assertEqual(app.folder_button.cget('state'),'normal')
                self.assertEqual(app.result,str(folder.resolve()))
                self.assertIn('上次任务已完成',app.status.get())
        finally:
            app.destroy()


if __name__=='__main__': unittest.main()
