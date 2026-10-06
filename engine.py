"""Offline video subtitles; translation uses only a worker-owned loopback server."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
import queue
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
DLL_HANDLES = []
HIDDEN = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


class Cancelled(Exception):
    pass


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    # Windows readers/scanners may briefly deny replacement of an open file.
    for attempt in range(6):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt==5:
                raise
            time.sleep(.025*2**attempt)


def load_config():
    cfg = read_json(ROOT / 'config.json')
    if not cfg:
        raise RuntimeError('缺少运行环境，请双击“安装环境.cmd”。')
    for key in ('dependencies', 'cuda', 'whisper', 'translation', 'llama_server',
                'translation_gguf', 'ocr_dependencies', 'ocr_gpu_dependencies',
                'ocr_gpu_cuda', 'ocr_det', 'ocr_rec'):
        if key in cfg:
            cfg[key] = str((ROOT / cfg[key]).resolve())
    return cfg


def configure(cfg):
    sys.path.insert(0, cfg['dependencies'])
    import psutil
    cores=psutil.cpu_count(logical=False) or os.cpu_count() or 4
    cfg['cpu_threads']=max(1,min(12,cores)) if cfg.get('cpu_mode') and cfg.get('speed')=='fast' else min(4,cores)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('OMP_NUM_THREADS', str(cfg['cpu_threads']))
    bins = [str(p) for p in Path(cfg['cuda']).glob('nvidia/*/bin')]
    os.environ['PATH'] = os.pathsep.join(bins + [os.environ.get('PATH', '')])
    if os.name == 'nt':
        DLL_HANDLES.extend(os.add_dll_directory(p) for p in bins)


def emit(message, progress=None, **extra):
    if 'stage' in extra and 'active_device_type' not in extra:
        if extra['stage']=='complete' or '复用' in message or '已有字幕' in message:
            extra.update(active_device_type='none',active_device_name='缓存复用' if '复用' in message else '待机',active_gpu_uuid=None)
        elif extra['stage']=='extract':
            extra.update(active_device_type='cpu',active_device_name='CPU',active_gpu_uuid=None)
        elif extra['stage'] in ('recognize','screen','translate'):
            extra.update(active_device_type='unknown',active_device_name='正在准备',active_gpu_uuid=None)
    print(json.dumps({'message': message, 'progress': progress, **extra}, ensure_ascii=False), flush=True)


def check_cancel(stop):
    if stop and Path(stop).exists():
        raise Cancelled('已停止。再次开始时会复用已完成的部分。')


def run_ffmpeg(ff, args, cwd, stop, duration, begin, end, backend=None):
    log = Path(cwd) / 'ffmpeg.log'
    started=time.monotonic()
    stage='encode' if begin>=70 else 'extract'
    with log.open('w', encoding='utf-8') as errors:
        proc = subprocess.Popen([ff, '-hide_banner', '-loglevel', 'error', *args,
                                 '-progress', 'pipe:1', '-nostats'], cwd=cwd,
                                stdout=subprocess.PIPE, stderr=errors,
                                encoding='utf-8', errors='replace', creationflags=HIDDEN)
        try:
            for line in proc.stdout:
                check_cancel(stop)
                if line.startswith('out_time_us='):
                    try:
                        processed=max(0,min(duration,int(line.split('=')[1])/1e6))
                        fraction=processed/duration
                        elapsed=time.monotonic()-started
                        eta=elapsed*(1-fraction)/fraction if fraction>0 and elapsed>=2 else None
                        message=('正在封装视频与可切换字幕' if (backend or {}).get('output_mode')=='soft' else '正在压制视频')
                        emit(message if stage=='encode' else '正在提取原声',
                             round(begin + fraction * (end - begin), 1),stage=stage,
                             stage_progress=round(fraction*100,1),processed_seconds=processed,
                             total_seconds=duration,eta_seconds=eta,
                             encoding_speed=round(processed/elapsed,2) if stage=='encode' and elapsed>=1 else None,
                             **(backend or {}))
                    except ValueError:
                        pass
            if proc.wait() != 0:
                raise RuntimeError(log.read_text(encoding='utf-8')[-1800:] or '视频处理失败。')
            check_cancel(stop)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def timestamp(value):
    ms = max(0, round(value * 1000))
    return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'


def write_srt(rows, path, duration):
    blocks = []
    rows = sorted(rows, key=lambda r: r['start'])
    for i, row in enumerate(rows):
        text = row.get('zh', '').replace('▁', '').strip()
        if not text or '⁇' in text or '<unk>' in text:
            continue
        start = max(0, row['start'])
        end = min(duration, row['end'], start + max(2, len(text)/5 + 1))
        if i + 1 < len(rows):
            end = min(end, rows[i+1]['start'])
        if end - start < 0.1:
            continue
        lines = '\n'.join(text[j:j+20] for j in range(0, len(text), 20))
        blocks.append(f'{len(blocks)+1}\n{timestamp(start)} --> {timestamp(end)}\n{lines}')
    Path(path).write_text('\n\n'.join(blocks) + ('\n' if blocks else ''), encoding='utf-8-sig')
    return len(blocks)


def sensible_speech(text, score, no_speech, soft_voice=True):
    silence = no_speech > 0.6 and (score <= -1.0 or not soft_voice)
    if not text.strip() or score < -1.1 or silence:
        return False
    clean = re.sub(r'[\s、。！？,.!?ー…]', '', text)
    if not clean or len(set(clean)) < 3 and len(clean) > 12:
        return False
    # Nonspeech frequently decodes to Latin gibberish; retain Japanese dialogue.
    return bool(re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', clean))


def _legacy_job_key(source, cfg, soft_voice=True, screen_text=True, screen_position='near'):
    stat = source.stat()
    models = []
    for key, name in (('whisper','model.bin'),('translation','model.bin'),
                      ('translation_gguf',''),('ocr_det','inference.pdiparams'),
                      ('ocr_rec','inference.pdiparams')):
        path = Path(cfg.get(key, '__missing__')) / name
        if path.is_file():
            models.append((str(path),path.stat().st_size,path.stat().st_mtime_ns))
    payload = json.dumps([str(source), stat.st_size, stat.st_mtime_ns, models,
        cfg.get('translation_backend','nllb'),soft_voice,screen_text,screen_position,'pipeline-v6'])
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def job_key(source,cfg,soft_voice=True,screen_text=True,screen_position='bottom'):
    # Rendering options do not change recognition/translation identity.
    # Canonical identity exactly matches previously default "near" jobs.
    return _legacy_job_key(source,cfg,soft_voice,True,'near')


def job_directory(source,output,cfg,soft_voice=True,screen_text=True):
    canonical=output/(source.stem[:65]+'-'+job_key(source,cfg,soft_voice,screen_text))
    if canonical.exists():return canonical
    # Both old OCR choices share the same audio; prefer existing canonical jobs.
    for flag,position in ((True,'top'),(False,'near'),(False,'top')):
        legacy=output/(source.stem[:65]+'-'+_legacy_job_key(source,cfg,soft_voice,flag,position))
        if legacy.exists():return legacy
    return canonical


def translation_cache_name(screen_only,style):
    if style not in ('natural','literal'):
        raise ValueError('请选择自然口语或忠实直译。')
    prefix='screen-translation' if screen_only else 'translation'
    return prefix+('.natural-v1.json' if style=='natural' else '.json')


def replace_speech_subtitles(rows,srt,duration):
    """Commit a successful retranslation without losing the previous edited SRT."""
    partial=srt.with_name('subtitles.zh.pending.srt')
    count=write_srt(rows,partial,duration)
    if not count:
        partial.unlink(missing_ok=True)
        raise RuntimeError('重译没有生成有效对白，原字幕已保留。')
    if srt.exists():
        from uuid import uuid4
        shutil.copy2(srt,srt.with_name('subtitles.zh.backup-'+uuid4().hex[:12]+'.srt'))
    partial.replace(srt)


def enhance_quiet_audio(audio):
    import numpy as np
    result = audio.copy()
    for begin in range(0,len(result),16000):
        block=result[begin:begin+16000]
        rms=float(np.sqrt(np.mean(block*block)))
        peak=float(np.max(np.abs(block))) if len(block) else 0
        if rms >= .0002 and peak > 0:
            block *= max(1,min(12,.035/rms,.98/peak))
    return result


def recognize(audio_path, cfg, job, stop, device, soft_voice=True):
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import get_speech_timestamps, VadOptions
    state_path = job / 'recognition.json'
    state = read_json(state_path, {'next': 0, 'rows': [], 'complete': False})
    if state['complete']:
        emit('复用已完成的日语识别', 45,stage='recognize',stage_progress=100)
        return state['rows']
    emit('正在检测日语对白', 6,stage='recognize',stage_progress=0)
    audio = decode_audio(str(audio_path))
    enhanced = enhance_quiet_audio(audio) if soft_voice else audio
    chunks = get_speech_timestamps(enhanced, VadOptions(threshold=0.35 if soft_voice else 0.6,
              min_speech_duration_ms=100 if soft_voice else 250, min_silence_duration_ms=500,
              max_speech_duration_s=20, speech_pad_ms=400 if soft_voice else 300))
    check_cancel(stop)
    model = WhisperModel(cfg['whisper'], device=device,
                         device_index=0,
                         compute_type='int8_float16' if device == 'cuda' else 'int8',
                         cpu_threads=cfg.get('cpu_threads',4), local_files_only=True)
    from hardware import device_event
    from runtime_options import profile
    backend=device_event(cfg,model.model.device=='cuda')
    emit('正在识别日语对白',6,stage='recognize',stage_progress=0,**backend)
    search=profile(cfg.get('speed','fast'))['beam_size']
    started=time.monotonic()
    resumed_at=state['next']
    try:
        for index in range(state['next'], len(chunks)):
            check_cancel(stop)
            chunk = chunks[index]
            offset = chunk['start'] / 16000
            def decode(signal):
                segments,_=model.transcribe(signal[chunk['start']:chunk['end']],
                    language='ja',task='transcribe',beam_size=search,condition_on_previous_text=False,
                    vad_filter=False,word_timestamps=True,max_new_tokens=128)
                return list(segments)
            segments=decode(audio)
            boosted=False
            review=False
            def score(items):
                return sum(s.avg_logprob for s in items)/len(items) if items else -99
            # Preserve clear original dialogue; retry amplification only for weak windows.
            if soft_voice and score(segments)<-.8:
                check_cancel(stop)
                retry=decode(enhanced)
                review=''.join(s.text for s in retry).strip()!=''.join(s.text for s in segments).strip()
                if score(retry)>score(segments):
                    segments=retry
                    boosted=True
            for seg in segments:
                row={'start': offset+seg.start,'end': offset+seg.end,'ja':seg.text.strip(),
                     'confidence':round(seg.avg_logprob,4),'no_speech':round(seg.no_speech_prob,4),
                     'enhanced':boosted,'needs_review':review or seg.avg_logprob<-.85}
                if sensible_speech(seg.text, seg.avg_logprob, seg.no_speech_prob,soft_voice):
                    state['rows'].append(row)
                else:
                    state.setdefault('rejected',[]).append(row)
            state['next'] = index + 1
            save_json(state_path, state)
            elapsed=time.monotonic()-started
            eta=elapsed*(len(chunks)-index-1)/(index+1-resumed_at)
            emit(f'识别日语对白 {index+1}/{len(chunks)}', 6+39*(index+1)/max(1,len(chunks)),
                 stage='recognize',stage_progress=100*(index+1)/max(1,len(chunks)),
                 stage_current=index+1,stage_total=len(chunks),eta_seconds=eta,**backend)
        state['complete'] = True
        save_json(state_path, state)
    finally:
        del model
        gc.collect()
    return state['rows']


def translated_row(row,text):
    untranslated=bool(re.search(r'[\u3040-\u30ff]',text))
    return {**row,'zh':text,'translation_needs_review':untranslated,
        'needs_review':row.get('needs_review',False) or untranslated}


def translate_local(rows, cfg, job, stop, device, cache_name):
    from local_translation import LocalTranslator,text_key
    check_cancel(stop)
    path=job/cache_name
    done=read_json(path,[])
    valid=0
    for old,new in zip(done,rows):
        if any(old.get(key)!=new.get(key) for key in ('ja','start','end','track')):
            break
        valid+=1
    done=done[:valid]
    if len(done)==len(rows):
        save_json(job/'translation-performance.json',{'speed':cfg.get('speed','fast'),
            'active_device_type':'none','active_device_name':'译文缓存复用','active_gpu_uuid':None,
            'model_requests_this_run':0,'reused_rows_this_run':len(rows),'model_loaded':False})
        emit('复用已完成的中文翻译',70,stage='translate',stage_progress=100,
             active_device_type='none',active_device_name='缓存复用',active_gpu_uuid=None)
        return done
    emit('正在加载本地中文翻译模型',50,stage='translate',stage_progress=0)
    started=time.monotonic()
    resumed_at=len(done)
    memory={text_key(r['ja']):r['zh'] for r in done}
    if all(text_key(row['ja']) in memory for row in rows[len(done):]):
        reused=len(rows)-len(done)
        done.extend(translated_row(row,memory[text_key(row['ja'])]) for row in rows[len(done):])
        save_json(path,done)
        save_json(job/'translation-performance.json',{'speed':cfg.get('speed','fast'),
            'active_device_type':'none','active_device_name':'译文缓存复用','active_gpu_uuid':None,
            'model_requests_this_run':0,'reused_rows_this_run':reused,'model_loaded':False})
        emit('复用重复对白译文，无需重新加载模型',70,stage='translate',stage_progress=100,
             active_device_type='none',active_device_name='译文缓存复用',active_gpu_uuid=None)
        return done
    requests=0
    reused=0
    with LocalTranslator(cfg,job,lambda:check_cancel(stop),gpu=device=='cuda') as model:
        emit('正在翻译中文 · 独立句子并行处理',50,stage='translate',stage_progress=0,**model.backend)
        while len(done)<len(rows):
            check_cancel(stop)
            batch=rows[len(done):len(done)+model.parallel]
            unique={}
            for row in batch:
                key=text_key(row['ja'])
                if key not in memory:
                    unique.setdefault(key,row['ja'])
            values=model.translate_batch(list(unique.values()))
            requests+=len(unique)
            memory.update(zip(unique,values))
            reused+=len(batch)-len(unique)
            for row in batch:
                text=memory[text_key(row['ja'])]
                done.append(translated_row(row,text))
            save_json(path,done)
            eta=(time.monotonic()-started)*(len(rows)-len(done))/(len(done)-resumed_at)
            active=model.backend if unique else {'active_device_type':'none',
                'active_device_name':'译文缓存复用','active_gpu_uuid':None}
            emit(f'翻译中文字幕 {len(done)}/{len(rows)}',50+20*len(done)/max(1,len(rows)),
                 stage='translate',stage_progress=100*len(done)/max(1,len(rows)),
                 stage_current=len(done),stage_total=len(rows),eta_seconds=eta,**active)
        save_json(job/'translation-performance.json',{'speed':cfg.get('speed','fast'),
            **model.backend,'translated_rows_this_run':len(done)-resumed_at,
            'unique_texts_this_run':requests,'model_requests_this_run':getattr(model,'request_count',requests),
            'reused_rows_this_run':reused,
            'elapsed_seconds':round(time.monotonic()-started,3)})
    return done


def translate(rows, cfg, job, stop, device, cache_name='translation.json'):
    if cfg.get('translation_backend')=='hy-mt2':
        return translate_local(rows,cfg,job,stop,device,cache_name)
    import ctranslate2
    import sentencepiece
    path = job / cache_name
    done = read_json(path, [])
    if len(done) == len(rows):
        return done
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(Path(cfg['translation'])/'sentencepiece.bpe.model'))
    model = ctranslate2.Translator(cfg['translation'], device=device,
                                  compute_type='int8_float16' if device == 'cuda' else 'int8',
                                  intra_threads=4)
    started=time.monotonic()
    resumed_at=len(done)
    try:
        for index in range(len(done), len(rows), 8):
            check_cancel(stop)
            batch = rows[index:index+8]
            inputs = [['jpn_Jpan'] + tokenizer.encode(r['ja'], out_type=str) + ['</s>'] for r in batch]
            result = model.translate_batch(inputs, target_prefix=[['zho_Hans'] for _ in batch],
                                           beam_size=3, max_decoding_length=128)
            for row, translation in zip(batch, result):
                text = tokenizer.decode(translation.hypotheses[0][1:]).strip()
                done.append({**row, 'zh': text})
            save_json(path, done)
            eta=(time.monotonic()-started)*(len(rows)-len(done))/(len(done)-resumed_at)
            emit(f'翻译中文字幕 {len(done)}/{len(rows)}', 50+20*len(done)/max(1,len(rows)),
                 stage='translate',stage_progress=100*len(done)/max(1,len(rows)),
                 stage_current=len(done),stage_total=len(rows),eta_seconds=eta)
    finally:
        del model
        gc.collect()
    return done


def select_encoder(ff, gpu=True, device_index=0):
    if not gpu:
        return 'libx265', '已选择 CPU 模式'
    # NVIDIA HEVC rejects 64×64 inputs even when the GPU is fully supported.
    try:
        probe = subprocess.run([ff, '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                                '-i', 'color=s=256x256:d=0.1', '-c:v', 'hevc_nvenc',
                                '-gpu',str(device_index),
                                '-pix_fmt', 'yuv420p', '-f', 'null', '-'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               encoding='utf-8', errors='replace', timeout=15,
                               creationflags=HIDDEN)
    except subprocess.TimeoutExpired:
        return 'libx265', '显卡编码检测超时'
    if probe.returncode == 0:
        return 'hevc_nvenc', ''
    return 'libx265', probe.stderr.strip()[-1800:] or '显卡编码检测未通过'


def encoder_args(ff, duration, source_size, gpu=True, audio_rate=192000, encoder=None,speed='fast',device_index=0):
    if encoder is None:
        encoder, _ = select_encoder(ff, gpu,device_index)
    # Reserve room for the actual selected audio stream.
    target = max(64_000, min(35_000_000, int(source_size*8/duration) - audio_rate))
    from runtime_options import profile
    preset=profile(speed)['nvenc_preset']
    cpu_preset={'fast':'ultrafast','balanced':'veryfast','quality':'fast'}[speed]
    codec = ['-c:v', 'hevc_nvenc','-gpu',str(device_index),'-preset',preset,'-rc','vbr'] if encoder == 'hevc_nvenc' else ['-c:v', 'libx265','-preset',cpu_preset]
    return codec + ['-b:v', str(target), '-maxrate', str(int(target*1.5)),
                    '-bufsize', str(target*2), '-pix_fmt', 'yuv420p', '-tag:v', 'hvc1']


def screen_worker(source,cfg,job,stop,duration,cpu=False,software_decode=False):
    path=job/'screen-recognition.json'
    state=read_json(path,{})
    if state.get('complete'):
        emit('复用已完成的画面日语识别',50,stage='screen',stage_progress=100)
        return state['rows']
    emit('正在加载画面日语识别模型',45,stage='screen',stage_progress=0)
    command=[sys.executable,str(ROOT/'screen_text.py'),str(source),str(job),str(stop or ''),str(duration)]
    if cpu:
        command.append('--cpu')
    if software_decode:
        command.append('--software-decode')
    env=dict(os.environ, PYTHONIOENCODING='utf-8')
    with (job/'screen-runtime.log').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log,env=env,
            encoding='utf-8',errors='replace',creationflags=HIDDEN)
        messages=queue.Queue()
        def read_output():
            for line in proc.stdout:
                messages.put(line)
            messages.put(None)
        threading.Thread(target=read_output,daemon=True).start()
        last_error=''
        try:
            while True:
                check_cancel(stop)
                try:
                    line=messages.get(timeout=.2)
                except queue.Empty:
                    continue
                if line is None:
                    break
                try:
                    value=json.loads(line)
                except ValueError:
                    log.write(line)
                    continue
                if value.get('error'):
                    last_error=value.get('message','')
                else:
                    print(json.dumps(value,ensure_ascii=False),flush=True)
            code=proc.wait()
            check_cancel(stop)
            if code:
                raise RuntimeError('画面文字识别失败：'+(last_error or
                    (job/'screen-runtime.log').read_text(encoding='utf-8',errors='replace')[-1400:]))
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill();proc.wait()
    return read_json(path)['rows']


def run_screen_text(source,cfg,job,stop,duration,cpu=False):
    # Decode compatibility may change, but GPU mode always keeps OCR on the GPU.
    attempts=[(True,True)] if cpu else [(False,False),(False,True)]
    errors=[]
    for force_cpu,software_decode in attempts:
        check_cancel(stop)
        try:
            return screen_worker(source,cfg,job,stop,duration,force_cpu,software_decode)
        except RuntimeError as exc:
            errors.append(str(exc))
            save_json(job/'screen-fallback.json',{'attempts':errors})
            if len(errors)<len(attempts):
                emit('保留断点，尝试软件解码 + NVIDIA 画面识别',45,stage='screen')
    prefix='画面日语扫描失败：' if cpu else 'NVIDIA 画面识别未能启动，断点已保留；请检查显卡环境，或在设备菜单选择 CPU：'
    raise RuntimeError(prefix+errors[-1])


def prepare_speech(source,cfg,job,stop,device,soft_voice,ff,duration,cpu):
    if not Path(cfg['whisper'],'model.bin').exists():
        raise RuntimeError('本地模型缺失，请双击“安装环境.cmd”。')
    audio=job/'audio.wav'
    if not audio.exists():
        emit('正在提取原声',0,stage='extract',stage_progress=0)
        partial=job/'audio.partial.wav'
        run_ffmpeg(ff,['-i',str(source),'-vn','-ac','1','-ar','16000','-y',str(partial)],job,stop,duration,0,5)
        partial.replace(audio)
    if not cpu and device!='cuda':
        raise RuntimeError('所选显卡无法用于对白识别，已保留进度，请检查驱动或选择 CPU。')
    return [{**row,'track':'speech'} for row in recognize(audio,cfg,job,stop,device,soft_voice)]


def soft_subtitle_args(source,subtitle_file,partial,font=None):
    args=['-i',str(source),'-i',str(subtitle_file),'-map','0:v:0','-map','0:a:0?',
        '-map','1:0','-c:v','copy','-c:a','copy','-c:s','ass',
        '-metadata:s:s:0','language=zho','-metadata:s:s:0','title=简体中文字幕',
        '-disposition:s:0','default']
    if font and Path(font).is_file():
        args+=['-attach',str(font),'-metadata:s:t:0','mimetype=application/x-truetype-font',
            '-metadata:s:t:0','filename='+Path(font).name]
    return [*args,'-y',str(partial)]


def run(source, output, stop=None, cpu=False, subtitles_only=False, external_srt=None,
        soft_voice=True, screen_text=False, screen_position='bottom',gpu='auto',speed='fast',hybrid=True,
        font_size=44,subtitle_color='#FFFFFF',subtitle_anchor=2,subtitle_x=50,subtitle_y=95,
        translation_style='natural',retranslate=False,soft_subtitles=False):
    from subtitle_layout import normalize_style,normalize_placement
    font_size,subtitle_color=normalize_style(font_size,subtitle_color)
    subtitle_anchor,subtitle_x,subtitle_y=normalize_placement(subtitle_anchor,subtitle_x,subtitle_y)
    translation_cache_name(False,translation_style)
    if external_srt and retranslate:
        raise ValueError('重译对白使用本工具的日语识别记录，请先清空导入的中文字幕。')
    if screen_position not in ('bottom','near','top'):
        raise ValueError('请选择有效的画面译文位置。')
    cfg = load_config()
    from hardware import query_gpus,choose_gpu,route_gpu,device_event,cuda_driver_index
    from runtime_options import profile
    options=profile(speed)
    selected=choose_gpu(query_gpus(),'cpu' if cpu else gpu)
    route_gpu(selected)
    cpu=selected is None
    driver_index=cuda_driver_index(selected) if selected else None
    cfg.update(selected_gpu=selected,speed=speed,cuda_driver_device=driver_index,cpu_mode=cpu,
        translation_style=translation_style)
    os.environ['SUBTITLE_GPU_UUID']=selected['uuid'] if selected else ''
    os.environ['SUBTITLE_GPU_NAME']=selected['name'] if selected else ''
    os.environ['SUBTITLE_CUDA_DRIVER_DEVICE']=str(driver_index) if selected else ''
    configure(cfg)
    import psutil
    prefetch=4 if speed=='fast' and psutil.virtual_memory().available>512*1024**2 else 2
    os.environ['SUBTITLE_HYBRID']='1' if hybrid and not cpu else '0'
    os.environ['SUBTITLE_FRAME_PREFETCH']=str(prefetch)
    os.environ['SUBTITLE_CPU_THREADS']=str(cfg['cpu_threads'])
    from runtime_options import ocr_batch_size
    ocr_batch=ocr_batch_size(speed,selected,cfg.get('ocr_batch_size'))
    os.environ['SUBTITLE_OCR_BATCH']=str(ocr_batch)
    import imageio_ffmpeg
    import av
    source = Path(source).resolve()
    if not source.is_file():
        raise RuntimeError('请选择实际的视频文件。')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    job = job_directory(source,output,cfg,soft_voice,screen_text)
    job.mkdir(exist_ok=True)
    save_json(job/'processing.json',{'speed':speed,'settings':options,'selected_gpu':selected,
        'cuda_driver_device':driver_index,
        'hybrid_processing':hybrid and not cpu,'frame_prefetch':prefetch if hybrid and not cpu else 0,
        'cpu_threads':cfg['cpu_threads'],
        'ocr_batch_size':ocr_batch,
        'translate_screen_text':screen_text,'translation_style':translation_style,'retranslate_speech':retranslate,
        'output_mode':'subtitles' if subtitles_only else 'soft' if soft_subtitles else 'burn',
        'applies_to_remaining_work':True})
    check_cancel(stop)
    with av.open(str(source)) as container:
        video = next(iter(container.streams.video), None)
        if not video:
            raise RuntimeError('选中的文件没有视频画面。')
        width,height=video.codec_context.width,video.codec_context.height
        duration = (container.duration or 0) / av.time_base
        if duration <= 0 and video.duration:
            duration = float(video.duration * video.time_base)
        audio_stream = next(iter(container.streams.audio), None)
        audio_rate = 0
        audio_args = ['-an']
        if audio_stream:
            audio_rate = audio_stream.codec_context.bit_rate or 192000
            if audio_stream.codec_context.name == 'aac':
                audio_args = ['-c:a', 'copy']
            else:
                audio_rate = min(192000, max(48000,audio_rate))
                audio_args = ['-c:a', 'aac', '-b:a', str(audio_rate)]
    if duration <= 0:
        raise RuntimeError('无法读取视频时长。')
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    srt = job / 'subtitles.zh.srt'
    voice_rows=[]
    device='cpu'
    import ctranslate2
    if not cpu and ctranslate2.get_cuda_device_count()>0:
        device='cuda'
    if external_srt:
        external = Path(external_srt).resolve()
        if not external.is_file() or external.suffix.lower() != '.srt':
            raise RuntimeError('请选择有效的 SRT 字幕文件。')
        if external != srt.resolve():
            # Decode once so FFmpeg always sees UTF-8 and a simple safe filename.
            try:
                text = external.read_text(encoding='utf-8-sig')
            except UnicodeDecodeError:
                text = external.read_text(encoding='gb18030')
            if not re.search(r'\d{2}:\d{2}:\d{2},\d{3}\s+-->',text):
                raise RuntimeError('字幕文件没有有效的 SRT 时间轴。')
            srt.write_text(text, encoding='utf-8-sig')
        emit('使用已有中文字幕', 45,stage='recognize',stage_progress=100)
    elif not srt.exists() or retranslate:
        if audio_stream:
            cached=read_json(job/'recognition.json',{})
            if cached.get('complete'):
                voice_rows=[{**row,'track':'speech'} for row in cached['rows']]
                emit('复用日语识别，只重新生成中文译文',45,stage='recognize',stage_progress=100,
                    active_device_type='none',active_device_name='日语识别缓存',active_gpu_uuid=None)
            else:
                voice_rows=prepare_speech(source,cfg,job,stop,device,soft_voice,ff,duration,cpu)
    else:
        emit('复用已有字幕；勾选重新翻译对白可应用新翻译风格',45,stage='recognize',stage_progress=100)
    screen_rows=run_screen_text(source,cfg,job,stop,duration,cpu) if screen_text else []
    combined=sorted(voice_rows+screen_rows,key=lambda r:r['start'])
    # Preserve hand-edited SRT unless the user explicitly requests retranslation.
    write_speech=not srt.exists() or retranslate
    cache_name=translation_cache_name(not write_speech,translation_style)
    if not cpu and device!='cuda' and combined:
        raise RuntimeError('所选显卡无法用于翻译，已保留进度，请检查驱动或选择 CPU。')
    translated=translate(combined,cfg,job,stop,device,cache_name)
    def letters(text):
        import unicodedata
        return ''.join(c for c in unicodedata.normalize('NFKC',text) if c.isalnum())
    screen_zh=[r for r in translated if r.get('track')=='screen' and
               letters(r.get('zh',''))!=letters(r['ja'])]
    save_json(job/'screen-translated.json',screen_zh)
    if write_speech:
        save_json(job/translation_cache_name(True,translation_style),[r for r in translated if r.get('track')=='screen'])
    review=[f"{timestamp(r['start'])}  日语：{r['ja']}\n中文：{r.get('zh','')}"
            for r in translated if r.get('needs_review')]
    if review:
        (job/'待核对字幕.txt').write_text('以下字幕识别较弱、增强前后有差异，或译文仍含日语，请结合原声与画面核对。\n\n'+
            '\n\n'.join(review),encoding='utf-8-sig')
    if write_speech:
        speech=[r for r in translated if r.get('track')!='screen']
        if retranslate:
            check_cancel(stop)
            replace_speech_subtitles(speech,srt,duration)
        else:
            write_srt(speech,srt,duration)
        save_json(job/'translation-settings.json',{'style':translation_style,'cache':cache_name,
            'prompt_revision':'natural-v1' if translation_style=='natural' else 'original',
            'retranslated':retranslate})
    from subtitle_layout import read_srt,write_ass
    speech_zh=read_srt(srt)
    if not speech_zh and not screen_zh:
        srt.unlink(missing_ok=True)
        raise RuntimeError('没有识别到可翻译的日语对白或画面文字，已保留识别记录。')
    subtitle_file=job/'subtitles.zh.ass'
    write_ass(speech_zh,screen_zh,subtitle_file,width,height,screen_position,font_size,subtitle_color,
        subtitle_anchor,subtitle_x,subtitle_y)
    save_json(job/'subtitle-style.json',{'position':screen_position,'font_size':font_size,
        'reference_height':1080,'color':subtitle_color,'width':width,'height':height,
        'anchor':subtitle_anchor,'x_percent':subtitle_x,'y_percent':subtitle_y})
    emit(f'生成 {len(speech_zh)} 条对白字幕、{len(screen_zh)} 条画面译文',70,
         stage='translate',stage_progress=100)
    if subtitles_only:
        emit('中文字幕已完成', 100,stage='complete',stage_progress=100,subtitles_only=True,result=str(subtitle_file), folder=str(job))
        return subtitle_file
    fonts = job / 'fonts'
    fonts.mkdir(exist_ok=True)
    font = Path(os.environ.get('WINDIR', r'C:\Windows')) / 'Fonts' / 'msyh.ttc'
    if font.exists() and not (fonts / font.name).exists():
        shutil.copyfile(font, fonts / font.name)
    if soft_subtitles:
        from hardware import device_event
        backend={**device_event(cfg,False),'encoder':'stream_copy','hardware_encoding':False,
            'output_mode':'soft'}
        video_path=job/'Chinese-subtitled.mkv'
        partial=job/'Chinese-subtitled.partial.mkv'
        save_json(job/'encoding.json',{**backend,'complete':False})
        emit('极速输出 · 复制原音画，封装可切换字幕',70,stage='encode',stage_progress=0,**backend)
        started=time.monotonic()
        run_ffmpeg(ff,soft_subtitle_args(source,subtitle_file,partial,fonts/font.name),job,stop,duration,70,99,backend)
        with av.open(str(partial)) as result:
            measured=(result.duration or 0)/av.time_base
            if abs(measured-duration)>max(1,duration*.002) or not result.streams.subtitles:
                raise RuntimeError('极速成片的时长或字幕检查未通过，原成片保持不变。')
        check_cancel(stop)
        partial.replace(video_path)
        elapsed=time.monotonic()-started
        save_json(job/'encoding.json',{**backend,'complete':True,'actual_encoder':'stream_copy',
            'elapsed_seconds':round(elapsed,3),'average_speed':round(duration/elapsed,2)})
        emit('完成：原画质保留，可在播放器切换中文字幕',100,stage='complete',stage_progress=100,
            result=str(video_path),folder=str(job))
        return video_path
    video_path = job / 'Chinese-subtitled.mp4'
    partial = job / 'Chinese-subtitled.partial.mp4'
    vf = f"subtitles='{subtitle_file.name}':fontsdir='fonts'"
    encoder, reason = select_encoder(ff, not cpu,driver_index)
    backend = {'encoder': encoder, 'hardware_encoding': encoder == 'hevc_nvenc',
        **device_event(cfg,encoder=='hevc_nvenc')}
    encoding_state = {**backend, 'reason': reason, 'complete': False}
    save_json(job/'encoding.json', encoding_state)
    if encoder == 'hevc_nvenc':
        message = 'NVIDIA 显卡加速压制 H.265，目标体积接近原片'
    else:
        message = '使用 CPU 压制 H.265（较慢）' if cpu else '显卡编码暂不可用，使用 CPU 压制（较慢）'
    emit(message, 70, stage='encode', stage_progress=0, **backend)
    encoding_started = time.monotonic()
    run_ffmpeg(ff, ['-i', str(source), '-map', '0:v:0', '-map', '0:a:0?',
                   '-vf', vf, *encoder_args(ff, duration, source.stat().st_size, not cpu, audio_rate, encoder,speed,driver_index),
                   *audio_args, '-movflags', '+faststart',
                   '-y', str(partial)], job, stop, duration, 70, 99, backend)
    with av.open(str(partial)) as result:
        measured = (result.duration or 0) / av.time_base
        if abs(measured-duration) > max(1, duration*0.002):
            raise RuntimeError('成片时长检查未通过，保留临时文件供检查。')
        encoding_state['actual_encoder'] = result.streams.video[0].metadata.get('encoder', encoder)
    partial.replace(video_path)
    elapsed = time.monotonic()-encoding_started
    save_json(job/'encoding.json', {**encoding_state, 'complete': True,
        'elapsed_seconds': round(elapsed,3), 'average_speed': round(duration/elapsed,2)})
    emit('完成：原视频保留，中文字幕已压进画面', 100,stage='complete',stage_progress=100,
         processed_seconds=duration,total_seconds=duration,eta_seconds=0,result=str(video_path), folder=str(job))
    return video_path


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='本地日语视频转中文字幕')
    parser.add_argument('input')
    parser.add_argument('--output', default=str(ROOT/'jobs'))
    parser.add_argument('--stop-file')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--subtitles-only', action='store_true')
    parser.add_argument('--soft-subtitles',action='store_true',help='极速 MKV 输出，原音画直接复制，字幕由播放器显示')
    parser.add_argument('--srt')
    parser.add_argument('--no-soft-voice',action='store_true')
    screen_choice=parser.add_mutually_exclusive_group()
    screen_choice.add_argument('--screen-text',dest='screen_text',action='store_true',help='额外扫描并翻译画面日语（较慢）')
    screen_choice.add_argument('--no-screen-text',dest='screen_text',action='store_false')
    parser.set_defaults(screen_text=False)
    parser.add_argument('--translation-style',choices=['natural','literal'],default='natural')
    parser.add_argument('--retranslate',action='store_true',help='复用日语识别重新翻译对白，备份原 SRT')
    parser.add_argument('--screen-position',choices=['bottom','near','top'],default='bottom')
    parser.add_argument('--font-size',type=int,default=44,help='字幕大小 16–96，以 1080p 为参考')
    parser.add_argument('--subtitle-color',default='#FFFFFF',help='字体颜色，例如 #FFFFFF')
    parser.add_argument('--subtitle-anchor',type=int,choices=range(1,10),default=2)
    parser.add_argument('--subtitle-x',type=float,default=50,help='水平位置百分比')
    parser.add_argument('--subtitle-y',type=float,default=95,help='垂直位置百分比')
    parser.add_argument('--gpu',default='auto',help='auto、NVIDIA UUID/编号，或 cpu')
    parser.add_argument('--speed',choices=['fast','balanced','quality'],default='fast')
    parser.add_argument('--no-hybrid',action='store_true')
    args = parser.parse_args()
    try:
        run(args.input, args.output, args.stop_file, args.cpu, args.subtitles_only, args.srt,
            not args.no_soft_voice,args.screen_text,args.screen_position,args.gpu,args.speed,not args.no_hybrid,
            args.font_size,args.subtitle_color,args.subtitle_anchor,args.subtitle_x,args.subtitle_y,
            args.translation_style,args.retranslate,args.soft_subtitles)
    except Cancelled as exc:
        emit(str(exc), cancelled=True)
        sys.exit(2)
    except Exception as exc:
        emit(str(exc), error=True)
        sys.exit(1)
