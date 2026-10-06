import json
import argparse
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
import uuid
from hardware import query_gpus,choose_gpu,query_display_adapters,query_resources
from runtime_options import PROFILE_LABELS,PROFILE_NOTES

ROOT = Path(__file__).resolve().parent
if (ROOT/'config.json').exists():
    _settings = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
    sys.path.insert(0,str((ROOT/_settings['dependencies']).resolve()))
import customtkinter as ctk
from PIL import Image

ctk.set_appearance_mode('dark')
THEME = {
    'background':'#141312', 'panel':'#201E1B', 'field':'#171613',
    'border':'#3C3831', 'text':'#F5F0E7', 'muted':'#B5ADA0',
    'accent':'#E8B86D', 'hover':'#F5CC89', 'tint':'#352B1D',
    'radius':8, 'font':'Microsoft YaHei UI', 'numbers':'Consolas',
}
BG, PANEL, FIELD, BORDER = (THEME[k] for k in ('background','panel','field','border'))
TEXT, MUTED, ACCENT, HOVER = (THEME[k] for k in ('text','muted','accent','hover'))
TEAL = ACCENT
PLACEMENTS={
    '底部左侧':(1,8,95),'底部中央':(2,50,95),'底部右侧':(3,92,95),
    '中央左侧':(4,8,50),'画面中央':(5,50,50),'中央右侧':(6,92,50),
    '顶部左侧':(7,8,5),'顶部中央':(8,50,5),'顶部右侧':(9,92,5),
}


def clock_text(seconds):
    seconds=max(0,int(seconds))
    return f'{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02}'


def progress_text(event):
    """Use stage progress directly; overall weighting is separate."""
    value=event.get('stage_progress',event.get('progress',0))
    value=max(0,min(100,float(value or 0)))
    if 'processed_seconds' in event and 'total_seconds' in event:
        detail=f"已处理 {clock_text(event['processed_seconds'])} / {clock_text(event['total_seconds'])}"
    elif 'stage_current' in event:
        unit='句' if event.get('stage')=='translate' else '段'
        detail=f"已完成 {event['stage_current']} / {event['stage_total']} {unit}"
    else:
        detail='字幕与成片会保存在独立任务文件夹中'
    speed=event.get('scan_speed') if event.get('stage')=='screen' else event.get('encoding_speed')
    if event.get('stage') in ('encode','screen') and speed is not None and speed>0:
        detail+=f' · {speed:.1f}×'
    eta=event.get('eta_seconds')
    remaining='正在估算剩余时间' if eta is None else f'预计剩余 {clock_text(eta)}'
    return value,detail,remaining


class App:
    def __init__(self, root):
        self.root = root
        root.title('字幕工坊 · 本地中文字幕')
        root.configure(fg_color=BG)
        self.messages = queue.Queue()
        self.worker = None
        self.controls=[]
        self.gpus=[]
        self.gpu_map={'自动选择显卡':'auto','CPU（较慢）':'cpu'}
        self.gpu_choice=tk.StringVar(value='自动选择显卡')
        self.requested_device='auto'
        self.speed=tk.StringVar(value='高速')
        self.speed_note=tk.StringVar(value=PROFILE_NOTES['fast'])
        self.active_device_type='none'
        self.active_device_name='待机'
        self.active_gpu_uuid=None
        self.hardware_text=tk.StringVar(value='正在检测显卡 · 占用率为整卡统计，包含其他程序')
        self._hardware_stop=threading.Event()
        self.result = None
        self.closing = False
        self.video = tk.StringVar()
        self.output = tk.StringVar(value=str(ROOT/'jobs'))
        self.srt = tk.StringVar()
        self.cpu = tk.BooleanVar(value=False)
        self.only = tk.BooleanVar(value=False)
        self.output_mode = tk.StringVar(value='固定在画面')
        self.soft = tk.BooleanVar(value=True)
        self.screen = tk.BooleanVar(value=False)
        self.translation_style = tk.StringVar(value='自然口语')
        self.retranslate = tk.BooleanVar(value=False)
        self.hybrid=tk.BooleanVar(value=True)
        self.hybrid_note=tk.StringVar(value='CPU 并行准备画面 · GPU 识别')
        self.adapters=[]
        self.inventory_text=tk.StringVar(value='正在检测本机所有显卡…')
        self.cpu_text=tk.StringVar(value='CPU —')
        self.ram_text=tk.StringVar(value='电脑内存 —')
        self.working_text=tk.StringVar(value='工具工作集 —')
        self.screen_position = tk.StringVar(value='跟随字幕位置')
        self.subtitle_position=tk.StringVar(value='底部中央')
        self.subtitle_anchor=2
        self.subtitle_x=tk.DoubleVar(value=50)
        self.subtitle_y=tk.DoubleVar(value=95)
        self.x_text=tk.StringVar(value='50%')
        self.y_text=tk.StringVar(value='95%')
        self.font_size=tk.IntVar(value=44)
        self.subtitle_color=tk.StringVar(value='#FFFFFF')
        self.size_text=tk.StringVar(value='44')
        self.status = tk.StringVar(value='请选择一个日语视频。')
        self.percent = tk.DoubleVar(value=0)
        self.display_status = tk.StringVar(value='准备就绪，选择视频后即可开始')
        self.detail = tk.StringVar(value='原视频保留，自动输出中文字幕与成片')
        self.remaining = tk.StringVar(value='等待开始')
        self.phase_name = tk.StringVar(value='准备就绪')
        self.encoding_device = None
        self.ocr_device = None
        self.number = tk.StringVar(value='0%')
        self.badge = tk.StringVar(value='●  本地离线')
        self.device_state=tk.StringVar(value='正在检测设备')
        self.device_name=tk.StringVar(value='—')
        self.metrics_note=tk.StringVar(value='整卡统计 · 包含其他程序')
        self.metric_values={name:tk.StringVar(value='—') for name in ('gpu','encoder','decoder','memory')}
        self.metric_labels={}
        self.metric_bars={}
        self.font=lambda size,weight='normal':ctk.CTkFont(family=THEME['font'],size=size,weight=weight)
        self.build_ui()
        root.protocol('WM_DELETE_WINDOW', self.close)
        self._poll_after_id=root.after(100,self.poll)
        self.minimum_height=640
        root.minsize(900,self.minimum_height)
        available=int((root.winfo_screenheight()-100)/root._get_window_scaling())
        root.geometry(f'1180x{max(self.minimum_height,min(840,available))}')
        icon=ROOT/'assets'/'subtitle-studio.ico'
        if icon.is_file():
            try: root.iconbitmap(str(icon))
            except tk.TclError: pass
        threading.Thread(target=self.monitor_hardware,daemon=True).start()

    def build_ui(self):
        root=self.root
        root.grid_rowconfigure(0,weight=1)
        root.grid_columnconfigure(0,weight=1)
        shell=ctk.CTkFrame(root,fg_color='transparent')
        shell.grid(row=0,column=0,sticky='nsew',padx=24,pady=20)
        shell.grid_columnconfigure(0,weight=1)
        shell.grid_rowconfigure(1,weight=1)
        nav=ctk.CTkFrame(shell,fg_color='transparent',height=50)
        nav.grid(row=0,column=0,sticky='ew')
        nav.grid_columnconfigure(2,weight=1)
        logo=ctk.CTkLabel(nav,text='字',width=46,height=46,corner_radius=6,
            fg_color=ACCENT,text_color=FIELD,font=self.font(26,'bold'))
        logo.grid(row=0,column=0,rowspan=2,padx=(0,14))
        ctk.CTkLabel(nav,text='字幕工坊',height=28,text_color=TEXT,font=self.font(24,'bold')).grid(row=0,column=1,sticky='w')
        ctk.CTkLabel(nav,text='LOCAL SUBTITLE  /  日语 → 简体中文',height=18,text_color=MUTED,font=self.font(10)).grid(row=1,column=1,sticky='w')
        self.badge_label=ctk.CTkLabel(nav,textvariable=self.badge,fg_color=THEME['tint'],text_color=TEAL,
                     corner_radius=8,width=122,height=32,font=self.font(11))
        self.badge_label.grid(row=0,column=3,rowspan=2,sticky='e')
        content=ctk.CTkFrame(shell,fg_color='transparent')
        content.grid(row=1,column=0,sticky='nsew',pady=(20,16))
        content.grid_rowconfigure(0,weight=1)
        content.grid_columnconfigure(0,weight=1)
        content.grid_columnconfigure(1,weight=0,minsize=316)
        workspace=ctk.CTkFrame(content,fg_color=PANEL,corner_radius=8,border_width=1,border_color=BORDER)
        workspace.grid(row=0,column=0,sticky='nsew',padx=(0,16))
        workspace.grid_columnconfigure(0,weight=1);workspace.grid_rowconfigure(1,weight=1)
        self.workspace_choice=tk.StringVar(value='视频任务')
        self.workspace_tabs=ctk.CTkSegmentedButton(workspace,values=['视频任务','翻译选项','字幕设计'],
            variable=self.workspace_choice,command=self.select_workspace,height=42,font=self.font(13,'bold'),
            fg_color=PANEL,text_color=TEXT,selected_color=THEME['tint'],selected_hover_color='#493B27',
            unselected_color=PANEL,unselected_hover_color=BORDER,corner_radius=6,dynamic_resizing=False)
        self.workspace_tabs.grid(row=0,column=0,sticky='ew',padx=16,pady=(12,8))
        self.pages={}
        for name in ('视频任务','翻译选项','字幕设计'):
            page=ctk.CTkScrollableFrame(workspace,fg_color=PANEL,corner_radius=0,
                scrollbar_button_color=BORDER,scrollbar_button_hover_color=MUTED)
            page.grid(row=1,column=0,sticky='nsew',padx=6,pady=(0,8));page.grid_columnconfigure(0,weight=1)
            self.pages[name]=page
        self.form=self.pages['视频任务']
        self.translation_form=self.pages['翻译选项']
        self.style_form=self.pages['字幕设计']
        self.select_workspace('视频任务')
        form=self.form
        form.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(form,text='01  /  SOURCE',height=16,font=self.font(10,'bold'),text_color=TEAL).grid(row=0,column=0,sticky='w',padx=16,pady=(10,4))
        ctk.CTkLabel(form,text='从一段日语视频开始',height=28,font=self.font(21,'bold'),text_color=TEXT).grid(row=1,column=0,sticky='w',padx=16)
        self.style_button=ctk.CTkButton(form,text='字幕设计 →',width=108,height=28,command=self.show_style_options,
            fg_color=ACCENT,hover_color=HOVER,text_color=FIELD,font=self.font(11,'bold'))
        self.style_button.grid(row=1,column=0,sticky='e',padx=16)
        ctk.CTkLabel(form,text='默认只翻译音频对白 · 画面文字翻译按需开启',height=18,font=self.font(11),text_color=TEAL).grid(row=2,column=0,sticky='w',padx=16,pady=(2,16))
        self.field(form,3,'视频文件',self.video,self.choose_video,'选择视频',placeholder='选择日语视频 · MP4 / MKV / MOV')
        self.field(form,4,'保存位置',self.output,self.choose_output,'选择位置')
        self.field(form,5,'已有中文字幕  ·  可选',self.srt,self.choose_srt,'导入字幕',clear=True,placeholder='已有 SRT 可跳过对白识别与翻译')
        divider=ctk.CTkFrame(form,height=1,fg_color=BORDER)
        divider.grid(row=6,column=0,sticky='ew',padx=16,pady=(12,14))
        ctk.CTkLabel(form,text='成片输出方式',height=18,font=self.font(12,'bold'),text_color=TEXT).grid(row=7,column=0,sticky='w',padx=16,pady=(0,8))
        self.output_menu=ctk.CTkOptionMenu(form,variable=self.output_mode,
            values=['固定在画面','可切换字幕 · 极速'],font=self.font(11),text_color=TEXT,
            fg_color=FIELD,button_color=BORDER,button_hover_color=MUTED,height=28,width=178,dynamic_resizing=False)
        self.output_menu.grid(row=7,column=0,sticky='e',padx=16,pady=(0,8))
        self.controls.append(self.output_menu)
        ctk.CTkLabel(form,text='固定在画面：输出 MP4，字幕永久显示。\n可切换字幕：输出 MKV，保留原画质，省去重新压制。',
            font=self.font(12),text_color=MUTED,justify='left',anchor='w',wraplength=410).grid(row=8,column=0,sticky='ew',padx=16,pady=(8,18))
        ctk.CTkLabel(form,text='选择视频 → 调整字幕 → 开始处理',font=self.font(13,'bold'),
            text_color=TEXT,anchor='w').grid(row=9,column=0,sticky='ew',padx=16,pady=(8,8))
        ctk.CTkLabel(self.translation_form,text='02  /  TRANSLATION',font=self.font(10,'bold'),text_color=ACCENT).grid(row=0,column=0,sticky='w',padx=16,pady=(10,4))
        ctk.CTkLabel(self.translation_form,text='听清对白，译成自然中文',font=self.font(21,'bold'),text_color=TEXT).grid(row=1,column=0,sticky='w',padx=16,pady=(0,18))
        opts=ctk.CTkFrame(self.translation_form,fg_color='transparent')
        opts.grid(row=2,column=0,sticky='ew',padx=16)
        opts.grid_columnconfigure((0,1),weight=1,uniform='options')
        descriptions=[('增强轻声识别',self.soft,'保留轻声与短句'),
            ('翻译画面文字 · 可选',self.screen,'额外扫描画面，耗时更长'),
            ('只生成字幕文件',self.only,'跳过压制，生成 SRT / ASS')]
        for index,(text,var,description) in enumerate(descriptions):
            option=ctk.CTkFrame(opts,fg_color='transparent')
            option.grid(row=index//2,column=index%2,sticky='ew',padx=(0,8),pady=(0,10))
            switch=ctk.CTkSwitch(option,text=text,variable=var,font=self.font(11),text_color=TEXT,
                progress_color=ACCENT,button_color=TEXT,button_hover_color=HOVER,
                fg_color=BORDER,switch_width=32,switch_height=18)
            switch.pack(anchor='w')
            ctk.CTkLabel(option,text=description,height=16,font=self.font(10),text_color=MUTED).pack(anchor='w',padx=(42,0),pady=(2,0))
            self.controls.append(switch)
        position=ctk.CTkFrame(opts,fg_color='transparent')
        position.grid(row=1,column=1,sticky='ew',pady=(0,10))
        ctk.CTkLabel(position,text='画面译文布局',height=16,font=self.font(10),text_color=MUTED).pack(anchor='w')
        placement=ctk.CTkOptionMenu(position,variable=self.screen_position,values=['跟随字幕位置','原文附近','画面顶部'],
            font=self.font(11),fg_color=FIELD,button_color=BORDER,button_hover_color=MUTED,
            text_color=TEXT,width=144,height=28,dynamic_resizing=False)
        placement.pack(anchor='w',pady=(3,0))
        self.controls.append(placement)
        translation=ctk.CTkFrame(opts,fg_color='transparent')
        translation.grid(row=2,column=0,columnspan=2,sticky='ew',pady=(0,10))
        ctk.CTkLabel(translation,text='中文翻译风格',font=self.font(11),text_color=TEXT).pack(side='left')
        self.translation_menu=ctk.CTkOptionMenu(translation,variable=self.translation_style,
            values=['自然口语','忠实直译'],font=self.font(11),fg_color=FIELD,button_color=BORDER,
            button_hover_color=MUTED,text_color=TEXT,width=132,height=28,dynamic_resizing=False)
        self.translation_menu.pack(side='left',padx=12)
        self.controls.append(self.translation_menu)
        redo=ctk.CTkFrame(opts,fg_color='transparent')
        redo.grid(row=3,column=0,columnspan=2,sticky='ew',pady=(0,10))
        self.retranslate_switch=ctk.CTkSwitch(redo,text='重新翻译已有对白',variable=self.retranslate,
            font=self.font(11),text_color=TEXT,progress_color=ACCENT,button_color=TEXT,
            button_hover_color=HOVER,fg_color=BORDER,switch_width=32,switch_height=18)
        self.retranslate_switch.pack(anchor='w')
        ctk.CTkLabel(redo,text='应用新风格 · 复用日语识别 · 自动备份旧字幕',
            font=self.font(10),text_color=MUTED).pack(anchor='w',padx=42,pady=(2,0))
        self.controls.append(self.retranslate_switch)
        ctk.CTkLabel(opts,text='极速模式免重编码，输出 MKV；需播放器支持字幕。',font=self.font(10),
            text_color=MUTED,wraplength=340,justify='left').grid(row=4,column=0,columnspan=2,sticky='w',pady=(0,8))
        style_panel=ctk.CTkFrame(self.style_form,fg_color='transparent',corner_radius=0)
        self.style_panel=style_panel
        style_panel.grid(row=0,column=0,sticky='ew',padx=4,pady=(4,10))
        style_panel.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(style_panel,text='设计你的字幕',font=self.font(21,'bold'),text_color=TEXT).grid(row=0,column=0,sticky='w',padx=14,pady=(10,0))
        ctk.CTkLabel(style_panel,text='03  /  APPEARANCE · 同时应用于对白和画面译文',font=self.font(10),text_color=MUTED).grid(row=1,column=0,sticky='w',padx=14)
        size_row=ctk.CTkFrame(style_panel,fg_color='transparent')
        size_row.grid(row=3,column=0,sticky='ew',padx=14,pady=(6,0))
        size_row.grid_columnconfigure(1,weight=1)
        ctk.CTkLabel(size_row,text='大小',font=self.font(11),text_color=TEXT).grid(row=0,column=0,padx=(0,12))
        self.size_slider=ctk.CTkSlider(size_row,from_=16,to=96,number_of_steps=80,variable=self.font_size,
            command=self.update_style_preview,progress_color=ACCENT,button_color=ACCENT,button_hover_color=HOVER)
        self.size_slider.grid(row=0,column=1,sticky='ew')
        ctk.CTkLabel(size_row,textvariable=self.size_text,width=36,font=self.font(11),text_color=TEXT).grid(row=0,column=2,padx=(8,0))
        colors=ctk.CTkFrame(style_panel,fg_color='transparent')
        colors.grid(row=4,column=0,sticky='ew',padx=14,pady=(8,8))
        ctk.CTkLabel(colors,text='颜色',font=self.font(11),text_color=TEXT).pack(side='left',padx=(0,12))
        self.color_button=ctk.CTkButton(colors,text='选择颜色',width=88,height=28,command=self.choose_subtitle_color,
            fg_color=BORDER,hover_color=MUTED,text_color=TEXT,font=self.font(11))
        self.color_button.pack(side='left')
        self.color_entry=ctk.CTkEntry(colors,textvariable=self.subtitle_color,width=100,height=28,
            fg_color=PANEL,border_color=BORDER,text_color=TEXT,font=self.font(11))
        self.color_entry.pack(side='left',padx=(8,0))
        location=ctk.CTkFrame(style_panel,fg_color='transparent')
        location.grid(row=5,column=0,sticky='ew',padx=14,pady=(0,8))
        ctk.CTkLabel(location,text='位置',font=self.font(11),text_color=TEXT).pack(side='left',padx=(0,12))
        self.position_menu=ctk.CTkOptionMenu(location,variable=self.subtitle_position,values=list(PLACEMENTS),
            command=self.choose_subtitle_position,width=160,height=28,fg_color=PANEL,button_color=BORDER,
            button_hover_color=MUTED,text_color=TEXT,font=self.font(11))
        self.position_menu.pack(side='left');self.controls.append(self.position_menu)
        for row,label,var,value in ((6,'横向',self.subtitle_x,self.x_text),(7,'纵向',self.subtitle_y,self.y_text)):
            line=ctk.CTkFrame(style_panel,fg_color='transparent');line.grid(row=row,column=0,sticky='ew',padx=14,pady=(0,8))
            line.grid_columnconfigure(1,weight=1)
            ctk.CTkLabel(line,text=label,font=self.font(11),text_color=TEXT).grid(row=0,column=0,padx=(0,12))
            slider=ctk.CTkSlider(line,from_=0,to=100,number_of_steps=100,variable=var,command=self.update_style_preview,
                progress_color=ACCENT,button_color=ACCENT,button_hover_color=HOVER)
            slider.grid(row=0,column=1,sticky='ew');self.controls.append(slider)
            ctk.CTkLabel(line,textvariable=value,width=40,font=self.font(11),text_color=TEXT).grid(row=0,column=2,padx=(8,0))
        self.preview_frame=ctk.CTkFrame(style_panel,fg_color='#0B0B0A',height=180,corner_radius=6,border_width=1,border_color=BORDER)
        self.preview_frame.grid(row=2,column=0,sticky='ew',padx=14,pady=(12,10))
        self.preview_frame.grid_propagate(False)
        ctk.CTkLabel(self.preview_frame,text='字幕位置示意 / 非视频画面',text_color=MUTED,font=self.font(10),
            fg_color='transparent').place(relx=.04,rely=.05,anchor='nw')
        self.preview_frame.bind('<Configure>',self.update_style_preview)
        self.preview_label=ctk.CTkLabel(self.preview_frame,text='让每一句对白，都能看懂。',width=1,height=32,
            fg_color='transparent',text_color='#FFFFFF',font=self.font(20),justify='center')
        ctk.CTkLabel(style_panel,text='横向从左到右 · 纵向从上到下 · 预览为位置示意',font=self.font(10),text_color=MUTED).grid(row=8,column=0,sticky='w',padx=14,pady=(0,10))
        self.controls.extend([self.size_slider,self.color_button,self.color_entry])
        self.subtitle_color.trace_add('write',lambda *_:self.update_style_preview())
        self.update_style_preview()
        ctk.CTkLabel(form,text='●  原视频保留 · 全部处理在本机完成',height=18,
            font=self.font(10),text_color=TEAL).grid(row=10,column=0,sticky='w',padx=16,pady=(2,14))
        self.performance=ctk.CTkScrollableFrame(content,fg_color=PANEL,corner_radius=8,width=292,
            border_width=1,border_color=BORDER,scrollbar_button_color=BORDER,scrollbar_button_hover_color=MUTED)
        self.performance.grid(row=0,column=1,sticky='nsew')
        sidebar=self.performance
        sidebar.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(sidebar,text='LIVE  /  PERFORMANCE',height=16,font=self.font(10,'bold'),text_color=TEAL).grid(row=0,column=0,sticky='w',padx=14,pady=(10,4))
        ctk.CTkLabel(sidebar,text='计算设备',height=28,font=self.font(19,'bold'),text_color=TEXT).grid(row=1,column=0,sticky='w',padx=14,pady=(0,12))
        device=ctk.CTkFrame(sidebar,fg_color=FIELD,corner_radius=8)
        device.grid(row=2,column=0,sticky='ew',padx=14)
        ctk.CTkLabel(device,textvariable=self.device_state,height=16,font=self.font(10),text_color=TEAL).pack(anchor='w',padx=14,pady=(10,0))
        self.hardware_label=ctk.CTkLabel(device,textvariable=self.device_name,height=26,
            font=self.font(16,'bold'),text_color=TEXT,anchor='w',wraplength=248,justify='left')
        self.hardware_label.pack(fill='x',padx=14,pady=(2,10))
        self.gpu_menu=ctk.CTkOptionMenu(sidebar,variable=self.gpu_choice,values=list(self.gpu_map),
            command=self.choose_device,width=250,height=32,font=self.font(11),fg_color=FIELD,
            button_color=BORDER,button_hover_color=MUTED,text_color=TEXT,dynamic_resizing=False)
        self.gpu_menu.grid(row=3,column=0,sticky='ew',padx=14,pady=(8,12))
        cooperation=ctk.CTkFrame(sidebar,fg_color='transparent')
        cooperation.grid(row=4,column=0,sticky='ew',padx=14,pady=(0,3))
        self.hybrid_switch=ctk.CTkSwitch(cooperation,text='CPU + GPU 协同',variable=self.hybrid,
            font=self.font(11),text_color=TEXT,progress_color=ACCENT,button_color=TEXT,
            fg_color=BORDER,switch_width=32,switch_height=18)
        self.hybrid_switch.pack(anchor='w')
        ctk.CTkLabel(cooperation,textvariable=self.hybrid_note,height=16,font=self.font(9),text_color=MUTED).pack(anchor='w',pady=(3,0))
        self.controls.append(self.hybrid_switch)
        self.inventory_button=ctk.CTkButton(sidebar,textvariable=self.inventory_text,command=self.show_devices,
            height=22,font=self.font(10),text_color=TEAL,fg_color='transparent',hover_color=BORDER,anchor='w')
        self.inventory_button.grid(row=5,column=0,sticky='ew',padx=10,pady=(0,8))
        metrics=ctk.CTkFrame(sidebar,fg_color='transparent')
        metrics.grid(row=8,column=0,sticky='ew',padx=14)
        metrics.grid_columnconfigure((0,1),weight=1,uniform='metrics')
        for index,(name,title) in enumerate([('gpu','计算占用'),('encoder','编码占用'),('decoder','解码占用'),('memory','显存')]):
            tile=ctk.CTkFrame(metrics,fg_color=FIELD,corner_radius=8)
            tile.grid(row=index//2,column=index%2,sticky='ew',padx=(0,6) if index%2==0 else (6,0),pady=(0,8))
            ctk.CTkLabel(tile,text=title,height=14,font=self.font(10),text_color=MUTED).pack(anchor='w',padx=12,pady=(8,0))
            label=ctk.CTkLabel(tile,textvariable=self.metric_values[name],height=24,
                font=ctk.CTkFont(family=THEME['numbers'],size=14 if name=='memory' else 24,weight='bold'),text_color=TEXT,anchor='w')
            label.pack(fill='x',padx=12,pady=(1,4))
            bar=ctk.CTkProgressBar(tile,height=3,fg_color=BORDER,progress_color=ACCENT,corner_radius=2)
            bar.pack(fill='x',padx=12,pady=(0,9));bar.set(0)
            self.metric_labels[name]=label;self.metric_bars[name]=bar
        ctk.CTkLabel(sidebar,textvariable=self.metrics_note,height=16,font=self.font(9),text_color=MUTED).grid(row=9,column=0,sticky='w',padx=14,pady=(0,12))
        ctk.CTkLabel(sidebar,text='处理速度',height=18,font=self.font(12,'bold'),text_color=TEXT).grid(row=6,column=0,sticky='w',padx=14)
        self.speed_buttons=ctk.CTkSegmentedButton(sidebar,values=list(PROFILE_LABELS),variable=self.speed,
            command=self.choose_speed,font=self.font(11),height=34,dynamic_resizing=False,
            text_color=TEXT,text_color_disabled=MUTED,selected_color=THEME['tint'],
            selected_hover_color=HOVER,unselected_color=BORDER,unselected_hover_color='#514A40')
        self.speed_buttons.grid(row=7,column=0,sticky='ew',padx=14,pady=(6,8))
        self.speed_note_label=ctk.CTkLabel(sidebar,textvariable=self.speed_note,wraplength=262,height=44,font=self.font(10),
            text_color=MUTED,justify='left',anchor='nw')
        self.speed_note_label.grid(row=11,column=0,sticky='ew',padx=14,pady=(0,12))
        ctk.CTkLabel(sidebar,text='处理中切换设备或速度：先停止，再选择并继续。',
            height=30,wraplength=260,font=self.font(9),text_color=MUTED,justify='left',anchor='w').grid(row=12,column=0,sticky='ew',padx=14,pady=(0,10))
        self.controls.extend([self.gpu_menu,self.speed_buttons])
        progress=ctk.CTkFrame(shell,fg_color=PANEL,corner_radius=8,border_width=1,border_color=BORDER)
        progress.grid(row=2,column=0,sticky='ew',pady=(0,12))
        progress.grid_columnconfigure(0,weight=1)
        resource_strip=ctk.CTkFrame(progress,fg_color=FIELD,corner_radius=6)
        resource_strip.grid(row=0,column=0,columnspan=2,sticky='ew',padx=12,pady=(10,0))
        resource_strip.grid_columnconfigure((0,1,2),weight=1,uniform='resources')
        self.resource_labels=[]
        for index,variable in enumerate((self.cpu_text,self.ram_text,self.working_text)):
            label=ctk.CTkLabel(resource_strip,textvariable=variable,height=26,font=self.font(10),text_color=TEAL,anchor='w')
            label.grid(row=0,column=index,sticky='ew',padx=10)
            self.resource_labels.append(label)
        resource_strip.bind('<Button-1>',lambda _:messagebox.showinfo('内存统计','CPU 与电脑内存为整机统计。工具工作集是界面及所有处理子进程的驻留内存合计，共享页可能重复计入，不能理解为独占物理内存。'))
        stepper=ctk.CTkFrame(progress,fg_color='transparent')
        stepper.grid(row=1,column=0,sticky='ew',padx=18,pady=(12,5))
        self.steps=[]
        for index,title in enumerate(('对白识别','画面文字','中文翻译','压制成片')):
            item=ctk.CTkFrame(stepper,fg_color='transparent')
            item.grid(row=0,column=index,sticky='w',padx=(0,22))
            circle=ctk.CTkLabel(item,text=f'{index+1:02}',width=22,height=22,corner_radius=7,
                fg_color=BORDER,text_color=MUTED,font=self.font(9,'bold'))
            circle.pack(side='left',padx=(0,6))
            label=ctk.CTkLabel(item,text=title,height=22,font=self.font(10),text_color=MUTED)
            label.pack(side='left')
            self.steps.append((circle,label))
        ctk.CTkLabel(progress,textvariable=self.number,height=38,font=ctk.CTkFont(family=THEME['numbers'],size=36,weight='bold'),text_color=ACCENT).grid(row=1,column=1,rowspan=3,padx=20,sticky='e')
        ctk.CTkLabel(progress,textvariable=self.phase_name,height=24,font=self.font(15,'bold'),text_color=TEXT).grid(row=2,column=0,sticky='w',padx=18)
        self.status_label=ctk.CTkLabel(progress,textvariable=self.display_status,height=18,font=self.font(11),text_color=MUTED,anchor='w')
        self.status_label.grid(row=3,column=0,sticky='ew',padx=18,pady=(0,2))
        self.bar=ctk.CTkProgressBar(progress,height=6,corner_radius=4,fg_color=BORDER,progress_color=ACCENT)
        self.bar.set(0)
        self.bar.grid(row=4,column=0,columnspan=2,sticky='ew',padx=18,pady=(8,8))
        ctk.CTkLabel(progress,textvariable=self.detail,height=16,font=self.font(10),text_color=MUTED).grid(row=5,column=0,sticky='w',padx=18,pady=(0,12))
        ctk.CTkLabel(progress,textvariable=self.remaining,height=16,font=self.font(10),text_color=TEAL).grid(row=5,column=1,sticky='e',padx=18,pady=(0,12))
        # This row never shares leftover height with the content above it.
        buttons=ctk.CTkFrame(shell,fg_color='transparent',height=44)
        buttons.grid(row=3,column=0,sticky='ew')
        buttons.grid_columnconfigure(2,weight=1)
        common={'height':44,'corner_radius':8,'font':self.font(13,'bold'),'text_color_disabled':'#9CACB2'}
        self.start_button=ctk.CTkButton(buttons,text='开始处理',command=self.start,width=182,fg_color=ACCENT,hover_color=HOVER,text_color=FIELD,**common)
        self.start_button.grid(row=0,column=0,sticky='w')
        self.cancel_button=ctk.CTkButton(buttons,text='停止',command=self.cancel,width=98,state='disabled',
            fg_color=PANEL,hover_color=BORDER,text_color=TEXT,border_width=1,border_color=BORDER,**common)
        self.cancel_button.grid(row=0,column=1,padx=(10,0),sticky='w')
        self.folder_button=ctk.CTkButton(buttons,text='打开结果文件夹',command=self.open_folder,width=172,state='disabled',
            fg_color=THEME['tint'],hover_color='#493B27',text_color=TEAL,border_width=1,border_color=BORDER,**common)
        self.folder_button.grid(row=0,column=3,sticky='e')
        self.controls.append(self.start_button)

    def choose_device(self,_=None):
        self.requested_device=self.gpu_map.get(self.gpu_choice.get())
        self.cpu.set(self.requested_device=='cpu')
        self.choose_speed()
        self.hybrid_switch.configure(state='disabled' if self.cpu.get() or self.worker else 'normal')
        self.hybrid_note.set('CPU 模式：不使用显卡' if self.cpu.get() else 'CPU 并行准备画面 · GPU 识别')
        self.render_hardware()

    def choose_speed(self,_=None):
        note=PROFILE_NOTES[PROFILE_LABELS[self.speed.get()]]
        if self.cpu.get():
            note=note.replace('并行翻译','顺序翻译').replace('显存占用稍高，','')
        self.speed_note.set(note)

    def monitor_hardware(self):
        inventoried=0
        sampled=False
        while not self._hardware_stop.is_set():
            devices=query_gpus()
            if self._hardware_stop.is_set():
                return
            self.messages.put({'hardware':devices})
            resources=query_resources()
            if not sampled and resources: resources['cpu_percent']=None
            sampled=True
            self.messages.put({'resources':resources})
            if time.monotonic()-inventoried>60:
                self.messages.put({'adapters':query_display_adapters(devices)})
                inventoried=time.monotonic()
            self._hardware_stop.wait(2)

    def update_hardware(self,devices):
        self.gpus=devices
        labels={'自动选择显卡':'auto'}
        for gpu in devices:
            short=gpu['name'].replace('NVIDIA GeForce ','').replace(' Laptop GPU','').replace(' GPU','')
            labels[f"GPU {gpu['index']} · {short}"]=gpu['uuid']
        labels['CPU（较慢）']='cpu'
        self.gpu_map=labels
        self.gpu_menu.configure(values=list(labels))
        current=next((label for label,value in labels.items() if value==self.requested_device),None)
        if current:
            self.gpu_choice.set(current)
        self.render_hardware()

    def render_hardware(self):
        selected=self.requested_device
        target=self.active_gpu_uuid if self.active_device_type in ('gpu','hybrid','loading') else selected
        try:
            gpu=choose_gpu(self.gpus,target or 'auto')
        except RuntimeError:
            gpu=None
        if self.active_device_type=='cpu':
            prefix='当前工作：CPU'
        elif self.active_device_type=='gpu':
            prefix='当前工作：'+self.active_device_name
        elif self.active_device_type=='hybrid':
            prefix='当前工作：CPU + '+self.active_device_name
        elif self.active_device_type=='loading':
            prefix='加载中：'+self.active_device_name
        elif self.active_device_type=='unknown':
            prefix='当前设备：等待后台确认'
        else:
            prefix='待机' if not self.worker else self.active_device_name
        if gpu:
            if self.active_device_type=='none':
                prefix+=' · 所选：'+gpu['name']
            def pct(value): return '—' if value is None else f'{value:.0f}%'
            used,total=gpu['memory_used'],gpu['memory_total']
            memory='—' if used is None or not total else f'{used/1024:.1f}/{total/1024:.1f} GB'
            self.hardware_text.set(f"{prefix}\n整卡 GPU {pct(gpu['utilization'])} · 编码 {pct(gpu['encoder_utilization'])} · 解码 {pct(gpu['decoder_utilization'])} · 显存 {memory}")
            self.metric_values['gpu'].set(pct(gpu['utilization']))
            self.metric_values['encoder'].set(pct(gpu['encoder_utilization']))
            self.metric_values['decoder'].set(pct(gpu['decoder_utilization']))
            self.metric_values['memory'].set(memory)
            values={'gpu':gpu['utilization'],'encoder':gpu['encoder_utilization'],
                'decoder':gpu['decoder_utilization'],
                'memory':100*used/total if used is not None and total else None}
            self.metrics_note.set(gpu['name'].replace('NVIDIA GeForce ','').replace(' Laptop GPU','')+' · 整卡统计（含其他程序）')
        else:
            self.hardware_text.set(prefix+(' · CPU 模式' if selected=='cpu' else ' · 显卡占用暂不可读取'))
            values={name:None for name in self.metric_values}
            for value in self.metric_values.values(): value.set('—')
            self.metrics_note.set('CPU 模式 · 显卡指标不适用' if selected=='cpu' else '显卡指标暂不可读取')
        for name,value in values.items():
            self.metric_bars[name].set(max(0,min(1,value/100)) if value is not None else 0)
            self.metric_bars[name].configure(progress_color=ACCENT if value is not None else BORDER)
        if self.active_device_type=='hybrid':
            state='当前协同工作 · CPU + GPU';name=self.active_device_name
        elif self.active_device_type=='gpu':
            state='当前正在使用';name=self.active_device_name
        elif self.active_device_type=='cpu':
            state='当前正在使用';name='CPU · 处理器'
        elif self.active_device_type=='loading':
            state='正在加载模型';name=self.active_device_name
        elif self.active_device_type=='unknown':
            state='等待后台确认设备';name='正在准备'
        else:
            state='待机 · 下次任务使用' if not self.worker else '缓存复用 · 暂无模型计算'
            name=gpu['name'] if gpu else 'CPU · 处理器' if selected=='cpu' else '尚未检测到显卡'
        self.device_state.set(state)
        self.device_name.set(name.replace('NVIDIA GeForce ','').replace('NVIDIA ',''))

    def show_devices(self):
        lines=[f"{a['name']}\n  {a['note']}" for a in self.adapters]
        messagebox.showinfo('本机显卡与加速支持','\n\n'.join(lines) if lines else '暂未读到显卡信息。当前模型也可手动选择 CPU。')

    def update_resources(self,resources):
        def gb(value): return f'{value/1024**3:.1f}'
        cpu=resources.get('cpu_percent')
        self.cpu_text.set('CPU —' if cpu is None else f'CPU {cpu:.0f}% · 整机')
        used,total=resources.get('ram_used'),resources.get('ram_total')
        self.ram_text.set('电脑内存 —' if used is None or not total else
            f"内存 {gb(used)} / {gb(total)} GB · {resources['ram_percent']:.0f}%")
        working=resources.get('app_working_set')
        self.working_text.set('工具工作集 —' if working is None else f'工具工作集 {gb(working)} GB')

    def field(self,parent,index,title,variable,action,button_text,clear=False,placeholder=''):
        block=ctk.CTkFrame(parent,fg_color='transparent')
        block.grid(row=index,column=0,sticky='ew',padx=16,pady=(0,10))
        block.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(block,text=title,height=18,font=self.font(11),text_color=MUTED).grid(row=0,column=0,sticky='w',pady=(0,5))
        row=ctk.CTkFrame(block,fg_color='transparent')
        row.grid(row=1,column=0,sticky='ew')
        row.grid_columnconfigure(0,weight=1)
        entry=ctk.CTkEntry(row,textvariable=variable,height=40,corner_radius=6,border_width=1,
            border_color=BORDER,fg_color=FIELD,text_color=TEXT,font=self.font(11),placeholder_text=placeholder)
        entry.grid(row=0,column=0,sticky='ew')
        button=ctk.CTkButton(row,text=button_text,command=action,width=82,height=40,corner_radius=6,
            fg_color=BORDER,hover_color='#514A40',text_color=TEXT,text_color_disabled='#97A8AE',font=self.font(11))
        button.grid(row=0,column=1,padx=(8,0))
        self.controls.extend([entry, button])
        if clear:
            erase=ctk.CTkButton(row,text='×',command=lambda:variable.set(''),width=28,height=40,
                corner_radius=6,fg_color=FIELD,hover_color='#514A40',text_color=MUTED,font=self.font(17))
            erase.grid(row=0,column=2,padx=(5,0))
            self.controls.append(erase)

    def update_progress(self,event):
        if 'active_device_type' in event:
            self.active_device_type=event['active_device_type']
            self.active_device_name=event.get('active_device_name','待核验')
            self.active_gpu_uuid=event.get('active_gpu_uuid')
            self.render_hardware()
        if 'message' in event:
            self.status.set(event['message'])
            text=event['message']
            self.display_status.set(text[:48]+('…' if len(text)>48 else ''))
        if event.get('error'):
            self.phase_name.set('处理未完成')
            self.badge.set('●  处理未完成')
            self.badge_label.configure(fg_color='#382238',text_color='#F0ADB4')
            self.remaining.set('请查看错误信息')
        if event.get('cancelled'):
            self.phase_name.set('已停止')
            self.badge.set('●  已停止')
            self.remaining.set('可复用已完成的部分')
        if event.get('progress') is not None:
            value,detail,eta=progress_text(event)
            self.percent.set(value)
            self.number.set(f'{value:.0f}%')
            self.bar.set(value/100)
            self.detail.set(detail)
            self.remaining.set(eta)
        phase=event.get('stage')
        names={'extract':'提取原声','recognize':'识别日语','screen':'识别画面日语','translate':'翻译中文','encode':'压制视频','complete':'处理完成'}
        if phase in names:
            if phase=='screen':
                if event.get('ocr_device'):
                    self.ocr_device='NVIDIA 显卡加速' if event.get('ocr_gpu') else 'CPU（较慢）'
                    if event.get('ocr_gpu_name'):
                        self.ocr_device=event['ocr_gpu_name']+' 显卡加速'
                self.phase_name.set(names[phase])
            elif phase=='encode':
                if event.get('encoder'):
                    self.encoding_device='原音画复制 · CPU' if event['encoder']=='stream_copy' else 'NVIDIA 显卡加速' if event['encoder']=='hevc_nvenc' else 'CPU（较慢）'
                self.phase_name.set('封装字幕视频' if self.encoding_device=='原音画复制 · CPU' else names[phase])
            else:
                self.phase_name.set(names[phase])
            active={'extract':0,'recognize':0,'screen':1,'translate':2,'encode':3,'complete':4}[phase]
            for i,(circle,label) in enumerate(self.steps):
                skipped=(i==1 and not self.screen.get()) or (i==3 and event.get('subtitles_only'))
                circle.configure(text='—' if skipped else '✓' if i<active else f'0{i+1}',
                    fg_color=BORDER if skipped else THEME['tint'] if i<active else ACCENT if i==active else BORDER,
                    text_color=MUTED if skipped else TEAL if i<active else FIELD if i==active else MUTED)
                label.configure(text='仅生成字幕' if skipped and i==3 else ('对白识别','画面文字','中文翻译','封装成片' if self.output_mode.get()=='可切换字幕 · 极速' else '压制成片')[i],
                    text_color=MUTED if skipped or i>active else TEXT)
            if phase=='complete':
                self.remaining.set('已完成')
                self.badge.set('●  处理完成')

    def choose_video(self):
        path = filedialog.askopenfilename(filetypes=[('视频文件','*.mp4 *.mkv *.mov *.avi *.webm *.m4v'),('所有文件','*.*')])
        if path:
            self.video.set(path)

    def choose_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output.set(path)

    def choose_srt(self):
        path = filedialog.askopenfilename(filetypes=[('中文字幕','*.srt')])
        if path:
            self.srt.set(path)

    def select_workspace(self,name):
        self.workspace_choice.set(name)
        for title,page in self.pages.items():
            if title==name: page.grid()
            else: page.grid_remove()

    def show_style_options(self):
        self.select_workspace('字幕设计')
        self.root.update_idletasks()
        self.style_form._parent_canvas.yview_moveto(0)

    def update_style_preview(self,*_):
        self.size_text.set(str(self.font_size.get()))
        self.x_text.set(f'{self.subtitle_x.get():.0f}%');self.y_text.set(f'{self.subtitle_y.get():.0f}%')
        from subtitle_layout import normalize_style
        try:size,color=normalize_style(self.font_size.get(),self.subtitle_color.get())
        except ValueError:return
        width=self.preview_frame.winfo_width()/self.preview_frame._get_widget_scaling()
        pixels=max(10,round(size*max(240,width)/1920))
        self.preview_label.configure(text_color=color,font=self.font(pixels),width=1,height=max(20,pixels+6))
        anchors={1:'sw',2:'s',3:'se',4:'w',5:'center',6:'e',7:'nw',8:'n',9:'ne'}
        self.preview_label.place(relx=self.subtitle_x.get()/100,rely=self.subtitle_y.get()/100,
            anchor=anchors[self.subtitle_anchor])

    def choose_subtitle_position(self,*_):
        self.subtitle_anchor,x,y=PLACEMENTS[self.subtitle_position.get()]
        self.subtitle_x.set(x);self.subtitle_y.set(y);self.update_style_preview()

    def choose_subtitle_color(self):
        chosen=colorchooser.askcolor(color=self.preview_label.cget('text_color'),parent=self.root,title='字幕字体颜色')[1]
        if chosen:self.subtitle_color.set(chosen.upper())

    def start(self):
        if self.worker and self.worker.poll() is None:
            return
        if not Path(self.video.get()).is_file():
            messagebox.showinfo('选择视频', '请先选择实际的视频文件。')
            return
        if self.srt.get() and not Path(self.srt.get()).is_file():
            messagebox.showinfo('选择字幕', '已有字幕路径无效，请重新选择或清空。')
            return
        if self.srt.get() and self.retranslate.get():
            messagebox.showinfo('重译对白','请先清空导入的中文字幕。重译对白使用本工具保存的日语识别记录。')
            return
        from subtitle_layout import normalize_style
        try:size,color=normalize_style(self.font_size.get(),self.subtitle_color.get())
        except ValueError as exc:
            messagebox.showinfo('字幕外观',str(exc));return
        (ROOT/'jobs').mkdir(exist_ok=True)
        self.stop_file = ROOT/'jobs'/('stop-'+uuid.uuid4().hex)
        exe = Path(sys.executable).with_name('python.exe')
        args = [str(exe), '-u', str(ROOT/'engine.py'), self.video.get(), '--output',self.output.get(), '--stop-file',str(self.stop_file)]
        selection='cpu' if self.cpu.get() else self.requested_device
        if selection is None:
            messagebox.showinfo('选择设备','所选显卡已不可用，请重新选择。')
            return
        args+=['--gpu',selection,'--speed',PROFILE_LABELS[self.speed.get()]]
        if self.cpu.get(): args.append('--cpu')
        if self.only.get(): args.append('--subtitles-only')
        if self.output_mode.get()=='可切换字幕 · 极速':args.append('--soft-subtitles')
        if self.srt.get(): args += ['--srt',self.srt.get()]
        if not self.soft.get(): args.append('--no-soft-voice')
        if self.screen.get(): args.append('--screen-text')
        args+=['--translation-style',{'自然口语':'natural','忠实直译':'literal'}[self.translation_style.get()]]
        if self.retranslate.get(): args.append('--retranslate')
        if not self.hybrid.get(): args.append('--no-hybrid')
        args += ['--screen-position',{'跟随字幕位置':'bottom','画面顶部':'top','原文附近':'near'}[self.screen_position.get()],
            '--font-size',str(size),'--subtitle-color',color,'--subtitle-anchor',str(self.subtitle_anchor),
            '--subtitle-x',str(self.subtitle_x.get()),'--subtitle-y',str(self.subtitle_y.get())]
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        try:
            self.worker = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          encoding='utf-8', errors='replace', env=env,
                                          creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        except OSError as exc:
            messagebox.showerror('启动失败', str(exc))
            return
        self.encoding_device = None
        self.ocr_device = None
        self.update_progress({'message':'正在准备本地模型……','progress':0,'stage':'extract','stage_progress':0})
        self.badge.set('●  正在处理')
        self.badge_label.configure(fg_color=THEME['tint'],text_color=TEAL)
        self.result = None
        self.status.set('正在准备本地模型……')
        for widget in self.controls: widget.configure(state='disabled')
        self.start_button.configure(text='正在处理…')
        self.cancel_button.configure(state='normal')
        self.folder_button.configure(state='disabled')
        threading.Thread(target=self.read_worker, daemon=True).start()

    def read_worker(self):
        worker = self.worker
        for line in worker.stdout:
            try: self.messages.put(json.loads(line))
            except json.JSONDecodeError: self.messages.put({'diagnostic':line.strip()})
        self.messages.put({'finished':worker.wait()})

    def poll(self):
        while not self.messages.empty():
            event = self.messages.get()
            if 'resources' in event:
                self.update_resources(event['resources']);continue
            if 'adapters' in event:
                self.adapters=event['adapters']
                self.inventory_text.set(f'检测到 {len(self.adapters)} 个显示设备 · 查看全部')
                continue
            if 'hardware' in event:
                self.update_hardware(event['hardware'])
                continue
            if 'diagnostic' in event:
                # Keep full diagnostics local without replacing useful progress.
                with (ROOT/'jobs'/'app.log').open('a',encoding='utf-8') as f: f.write(event['diagnostic']+'\n')
            self.update_progress(event)
            if event.get('result'):
                self.result = event['folder']
                self.folder_button.configure(state='normal')
            if 'finished' in event:
                self.worker = None
                self.active_device_type='none'
                self.active_device_name='待机'
                self.active_gpu_uuid=None
                self.render_hardware()
                self.stop_file.unlink(missing_ok=True)
                for widget in self.controls: widget.configure(state='normal')
                self.hybrid_switch.configure(state='disabled' if self.cpu.get() else 'normal')
                self.start_button.configure(text='开始处理')
                self.cancel_button.configure(state='disabled')
                if event['finished'] not in (0,2):
                    messagebox.showerror('处理未完成',self.status.get())
                if self.closing:
                    self.destroy()
                    return
        self._poll_after_id=self.root.after(100,self.poll)

    def cancel(self):
        if self.worker and self.worker.poll() is None:
            self.stop_file.touch()
            self.status.set('正在停止当前小段，已完成的内容会保留……')
            self.display_status.set(self.status.get())
            self.phase_name.set('正在停止')
            self.cancel_button.configure(state='disabled')

    def open_folder(self):
        if self.result and os.name == 'nt': os.startfile(self.result)

    def show_completed(self,folder):
        folder=Path(folder).resolve()
        state=folder/'encoding.json'
        data=json.loads(state.read_text(encoding='utf-8')) if state.is_file() else {}
        video=folder/('Chinese-subtitled.mkv' if data.get('output_mode')=='soft' else 'Chinese-subtitled.mp4')
        if not video.is_file() or not data.get('complete'):
            raise ValueError('该任务还没有完成成片。')
        self.result=str(folder)
        self.update_progress({'message':'上次任务已完成，可打开结果文件夹查看成片',
            'stage':'complete','progress':100,'stage_progress':100,'active_device_type':'none',
            'active_device_name':'待机','active_gpu_uuid':None})
        self.detail.set('成片和字幕已保留 · 可以选择新视频或重新处理')
        self.folder_button.configure(state='normal')
        self.badge_label.configure(fg_color=THEME['tint'],text_color=TEAL)

    def close(self):
        if self.worker and self.worker.poll() is None:
            if messagebox.askyesno('正在处理', '停止当前任务并关闭窗口？已完成的部分会保留。'):
                self.closing = True
                self.cancel()
        else:
            self.destroy()

    def destroy(self):
        self._hardware_stop.set()
        if self._poll_after_id:
            self.root.after_cancel(self._poll_after_id)
            self._poll_after_id=None
        self.root.destroy()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='字幕工坊')
    parser.add_argument('--video')
    parser.add_argument('--output')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--gpu',default='auto')
    parser.add_argument('--speed',choices=['fast','balanced','quality'],default='fast')
    parser.add_argument('--completed-folder')
    args=parser.parse_args()
    root = ctk.CTk()
    app=App(root)
    from runtime_options import PROFILES
    app.speed.set(PROFILES[args.speed]['label']);app.choose_speed()
    if args.gpu=='cpu':
        app.gpu_choice.set('CPU（较慢）');app.choose_device()
    elif args.gpu!='auto':
        devices=query_gpus();app.update_hardware(devices)
        selected=choose_gpu(devices,args.gpu)
        label=next((name for name,value in app.gpu_map.items() if value==selected['uuid']),None)
        if label:
            app.gpu_choice.set(label);app.choose_device()
        else:
            parser.error('所选显卡不存在')
    if args.video:
        app.video.set(args.video)
    if args.output:
        app.output.set(args.output)
    if args.completed_folder:
        app.show_completed(args.completed_folder)
    elif args.resume and args.video:
        root.after(500,app.start)
    root.mainloop()
