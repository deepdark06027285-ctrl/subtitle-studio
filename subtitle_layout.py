"""Subtitle tracks, safe ASS text, and positions relative to detected source text."""
from pathlib import Path
import re


def ass_time(seconds):
    value=max(0,round(seconds*100))
    return f'{value//360000}:{value//6000%60:02}:{value//100%60:02}.{value%100:02}'


def safe_text(text):
    return str(text).replace('\\','＼').replace('{','｛').replace('}','｝').replace('\r','').replace('\n',r'\N')


def read_srt(path):
    text=Path(path).read_text(encoding='utf-8-sig')
    blocks=re.split(r'\r?\n\s*\r?\n',text.strip())
    rows=[]
    pattern=r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})'
    for block in blocks:
        match=re.search(pattern,block)
        if not match:
            continue
        values=[int(x) for x in match.groups()]
        def seconds(v): return v[0]*3600+v[1]*60+v[2]+v[3]/1000
        rows.append({'start':seconds(values[:4]),'end':seconds(values[4:]),
                     'zh':block[match.end():].strip()})
    return rows


def normalize_style(font_size=44,color='#FFFFFF'):
    try:
        size=int(font_size)
    except (TypeError,ValueError):
        raise ValueError('字幕大小请填写 16 到 96 之间的整数。')
    if not 16<=size<=96:
        raise ValueError('字幕大小请填写 16 到 96 之间的整数。')
    if not re.fullmatch(r'#[0-9a-fA-F]{6}',str(color)):
        raise ValueError('字幕颜色请使用六位颜色代码，例如 #FFFFFF。')
    return size,str(color).upper()


def normalize_placement(anchor=2,x=50,y=95):
    try:anchor=int(anchor);x=float(x);y=float(y)
    except (TypeError,ValueError):raise ValueError('请选择有效的字幕位置。')
    if anchor not in range(1,10) or not 0<=x<=100 or not 0<=y<=100:
        raise ValueError('字幕位置必须在画面范围内。')
    return anchor,x,y


def placement_tag(width,height,anchor=2,x=50,y=95):
    anchor,x,y=normalize_placement(anchor,x,y)
    return f'{{\\an{anchor}\\pos({round(width*x/100)},{round(height*y/100)})}}'


def bottom_events(speech,screen,tag=''):
    """One timed block per interval, so simultaneous tracks never overlap."""
    boundaries={}
    rows=[]
    for track,source in (('Screen',screen),('Speech',speech)):
        for row in source:
            start=round(row['start']*100);end=round(row['end']*100)
            if not row.get('zh') or end<=start:continue
            index=len(rows);rows.append((track,row['zh']))
            boundaries.setdefault(start,[]).append((True,index))
            boundaries.setdefault(end,[]).append((False,index))
    active=set();times=sorted(boundaries)
    for offset,time in enumerate(times[:-1]):
        for entering,index in boundaries[time]:
            if entering:active.add(index)
            else:active.discard(index)
        if not active:continue
        ordered=[rows[i] for i in sorted(active)]
        texts=list(dict.fromkeys(text for _,text in ordered))
        track='Speech' if any(t=='Speech' for t,_ in ordered) else 'Screen'
        yield f"Dialogue: 0,{ass_time(time/100)},{ass_time(times[offset+1]/100)},{track},,0,0,0,,{tag}{safe_text(chr(10).join(texts))}"


def write_ass(speech,screen,path,width,height,position='bottom',font_size=44,color='#FFFFFF',
        anchor=2,x=50,y=95):
    size,color=normalize_style(font_size,color)
    anchor,x,y=normalize_placement(anchor,x,y)
    if position not in ('bottom','near','top'):
        raise ValueError('请选择有效的画面译文位置。')
    font=max(12,round(size*height/1080))
    ass_color='&H00'+color[5:7]+color[3:5]+color[1:3]
    screen_alignment=2 if position=='bottom' else 8
    margin=max(12,round(height*.05))
    header=f'''[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Speech,Microsoft YaHei,{font},{ass_color},&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,40,40,{margin},1
Style: Screen,Microsoft YaHei,{font},{ass_color},&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,{screen_alignment},40,40,{margin if position=='bottom' else 20},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
    if position=='bottom':
        tag='' if (anchor,x,y)==(2,50,95) else placement_tag(width,height,anchor,x,y)
        Path(path).write_text(header+'\n'.join(bottom_events(speech,screen,tag))+'\n',encoding='utf-8-sig')
        return
    events=[]
    for row in speech:
        if row['end']>row['start'] and row.get('zh'):
            tag='' if (anchor,x,y)==(2,50,95) else placement_tag(width,height,anchor,x,y)
            events.append(f"Dialogue: 1,{ass_time(row['start'])},{ass_time(row['end'])},Speech,,0,0,0,,{tag}{safe_text(row['zh'])}")
    for row in screen:
        if not row.get('zh') or row['end']<=row['start']:
            continue
        box=row.get('box',[.1,.1,.9,.2])
        size=font
        x=round(max(.08,min(.92,(box[0]+box[2])/2))*width)
        y=round((box[3]+.015)*height)
        if box[3]>.78:
            y=round(max(.05,box[1]-.08)*height)
        if position=='top':
            x=width//2
            y=round(.06*height+box[1]*height*.2)
        y=min(round(height*.8),max(round(height*.03),y))
        # Limit line width to the frame, including labels near either edge.
        max_chars=max(6,min(32,round(min(x,width-x)*1.8/size)))
        text=row['zh']
        lines='\n'.join(text[i:i+max_chars] for i in range(0,len(text),max_chars))
        tag=f'{{\\an8\\pos({x},{y})\\fs{size}}}'
        events.append(f"Dialogue: 0,{ass_time(row['start'])},{ass_time(row['end'])},Screen,,0,0,0,,{tag}{safe_text(lines)}")
    Path(path).write_text(header+'\n'.join(events)+'\n',encoding='utf-8-sig')
