"""Sample original frames, track Japanese text, and checkpoint locally."""
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata
from collections import deque
from frame_pipeline import prepared_frames,sample_batches,adjacent_reuse
from scan_pipeline import CheckpointWriter,screen_snapshot
from batch_ocr import predict_frames


def clean_text(text):
    return unicodedata.normalize('NFKC',str(text)).strip()


def japanese_text(text):
    return bool(re.search(r'[\u3040-\u30ff]',text) or len(re.findall(r'[\u4e00-\u9fff]',text))>=2)


def same_region(a,b):
    width=max(.04,a[2]-a[0],b[2]-b[0])
    height=max(.03,a[3]-a[1],b[3]-b[1])
    return abs(a[0]+a[2]-b[0]-b[2])<width and abs(a[1]+a[3]-b[1]-b[3])<height


def track_frame(state,detections,timestamp,interval,duration):
    active=state.setdefault('active',[])
    rows=state.setdefault('rows',[])
    next_active=[]
    unmatched=list(active)
    for detection in detections:
        text=clean_text(detection['ja'])
        if detection['score']<.75 or not japanese_text(text):
            continue
        match=next((row for row in unmatched if same_region(row['box'],detection['box'])
            and SequenceMatcher(None,clean_text(row['ja']),text).ratio()>=.85),None)
        if match:
            unmatched.remove(match)
            match['end']=min(duration,timestamp+interval)
            if detection['score']>match.get('score',0):
                match.update(ja=text,box=detection['box'],score=detection['score'])
            next_active.append(match)
        else:
            next_active.append({'start':timestamp,'end':min(duration,timestamp+interval),
                'ja':text,'box':detection['box'],'score':detection['score'],'track':'screen'})
    for row in unmatched:
        row['end']=min(row['end'],timestamp)
        if row['end']>row['start']:
            rows.append(row)
    state['active']=next_active


DLL_HANDLES=[]


def configure_runtime(cfg,cpu=False):
    # This worker has its own Paddle import and CUDA DLL search order.
    sys.path.insert(0,cfg['dependencies'])
    sys.path.insert(0,cfg['ocr_dependencies'])
    gpu_path=Path(cfg.get('ocr_gpu_dependencies','__missing__'))
    if not cpu and (gpu_path/'paddle').is_dir():
        sys.path.insert(0,str(gpu_path))
        bins=[str(p) for p in Path(cfg.get('ocr_gpu_cuda','__missing__')).glob('nvidia/*/bin')]
        os.environ['PATH']=os.pathsep.join(bins+[os.environ.get('PATH','')])
        if os.name=='nt':
            DLL_HANDLES.extend(os.add_dll_directory(p) for p in bins)
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['OMP_NUM_THREADS']=os.environ.get('SUBTITLE_CPU_THREADS','4') if cpu else '1'


def select_ocr_device(paddle,cpu=False):
    if cpu:
        return 'cpu'
    if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count()>0:
        return 'gpu:0'
    raise RuntimeError('画面识别需要 NVIDIA 显卡：请检查显卡运行库与驱动；如需慢速处理，请在设备菜单选择 CPU。')


def scan_progress(timestamp,resumed,duration,elapsed,frames):
    processed=min(duration,max(0,timestamp))
    advanced=max(0,processed-resumed)
    eta=elapsed*max(0,duration-processed)/advanced if advanced>=2 and frames>=3 else None
    speed=advanced/elapsed if elapsed>=1 else None
    return processed,processed/duration,eta,speed


def open_video(source,av,gpu=False):
    if gpu:
        from av.codec.hwaccel import HWAccel
        return av.open(str(source),hwaccel=HWAccel(device_type='cuda',
            device=os.environ.get('SUBTITLE_CUDA_DRIVER_DEVICE') or '0',allow_software_fallback=False))
    return av.open(str(source))


def run(source,cfg,job,stop,duration,interval=1,cpu=False,software_decode=False):
    # Keep Paddle's dependencies separate from CTranslate2's environment.
    configure_runtime(cfg,cpu)
    os.environ['PADDLE_PDX_CACHE_HOME']=str(Path(job)/'ocr-cache')
    from paddleocr import PaddleOCR
    import paddle
    import av
    import numpy as np
    from engine import emit,check_cancel,read_json,save_json
    state_path=Path(job)/'screen-recognition.json'
    state=read_json(state_path,{'next_time':0,'active':[],'rows':[],'complete':False})
    if state['complete']:
        emit('复用画面文字识别',50,stage='screen',stage_progress=100)
        return state['rows']
    check_cancel(stop)
    device=select_ocr_device(paddle,cpu)
    backend={'ocr_device':'NVIDIA' if device.startswith('gpu') else 'CPU',
             'ocr_gpu':device.startswith('gpu')}
    if backend['ocr_gpu']:
        backend['ocr_gpu_name']=paddle.device.cuda.get_device_name(0)
    backend.update(active_device_type='gpu' if backend['ocr_gpu'] else 'cpu',
        active_device_name=backend.get('ocr_gpu_name','CPU'),
        active_gpu_uuid=os.environ.get('SUBTITLE_GPU_UUID') or None)
    hybrid=backend['ocr_gpu'] and os.environ.get('SUBTITLE_HYBRID','1')=='1'
    capacity=int(os.environ.get('SUBTITLE_FRAME_PREFETCH','2'))
    cpu_threads=int(os.environ.get('SUBTITLE_CPU_THREADS','4'))
    batch_size=max(1,min(4,int(os.environ.get('SUBTITLE_OCR_BATCH','1')))) if backend['ocr_gpu'] else 1
    backend.update(hybrid_processing=hybrid,frame_prefetch=capacity if hybrid else 0,ocr_batch_size=batch_size)
    emit('正在加载本地日语文字模型',45,stage='screen',stage_progress=0,
         **{**backend,'active_device_type':'loading'})
    batch_options={}
    if batch_size>1:
        from paddlex.inference import load_pipeline_config
        pipeline_config=load_pipeline_config('OCR')
        pipeline_config['batch_size']=batch_size
        pipeline_config['SubModules']['TextDetection']['batch_size']=batch_size
        batch_options['paddlex_config']=pipeline_config
    ocr=PaddleOCR(**batch_options,text_detection_model_name='PP-OCRv6_small_det',text_detection_model_dir=cfg['ocr_det'],
        text_recognition_model_name='PP-OCRv6_small_rec',text_recognition_model_dir=cfg['ocr_rec'],
        use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,
        device=device,enable_mkldnn=False,cpu_threads=cpu_threads,text_det_limit_side_len=960,
        text_det_limit_type='max',text_rec_score_thresh=.75)
    started=time.monotonic()
    resumed=float(state['next_time'])
    next_time=resumed
    frames=0
    scans=0
    batches=0
    inference_seconds=0
    previous_image=None
    detections=[]
    preparation={'decode_seconds':0,'prepare_seconds':0,'frames_prepared':0}
    inference_ranges=deque(maxlen=8)
    overlap_seconds=0
    last_report=0
    def producer():
        sample_at=resumed
        if resumed>=duration:return
        sparse=backend['ocr_gpu'] and not software_decode and os.environ.get('SUBTITLE_SPARSE_DOWNLOAD','1')=='1'
        if sparse:
            with av.open(str(source)) as info:
                sparse=info.streams.video[0].codec_context.format.name in ('yuv420p','yuv420p10le')
        if sparse:
            import imageio_ffmpeg
            from sampled_decode import sampled_frames
            preparation_start=time.monotonic()
            previous=preparation_start
            iterator=sampled_frames(source,av,imageio_ffmpeg.get_ffmpeg_exe(),job,resumed,interval,
                os.environ.get('SUBTITLE_CUDA_DRIVER_DEVICE') or '0',lambda:check_cancel(stop))
            try:
                for frame in iterator:
                    now=time.monotonic();preparation['decode_seconds']+=now-previous
                    image=frame.to_ndarray(format='bgr24')
                    preparation_end=time.monotonic()
                    preparation['prepare_seconds']+=preparation_end-preparation_start
                    preparation['frames_prepared']+=1
                    yield {'timestamp':float(frame.time or 0),'image':image,'width':frame.width,'height':frame.height,
                        'hardware_decode':True,'decode_backend':'cuda-sampled-download',
                        'preparation_start':preparation_start,'preparation_end':preparation_end}
                    preparation_start=time.monotonic();previous=preparation_start
            finally:iterator.close()
            return
        with open_video(source,av,gpu=backend['ocr_gpu'] and not software_decode) as container:
            stream=container.streams.video[0]
            stream.thread_type='AUTO'
            stream.codec_context.thread_count=cpu_threads
            if resumed>0: container.seek(round(max(0,resumed-1)*av.time_base))
            iterator=iter(container.decode(stream))
            preparation_start=time.monotonic()
            while True:
                check_cancel(stop)
                before_decode=time.monotonic()
                try: frame=next(iterator)
                except StopIteration: return
                preparation['decode_seconds']+=time.monotonic()-before_decode
                timestamp=float(frame.time or 0)
                if timestamp+1e-4<sample_at: continue
                image=frame.to_ndarray(format='bgr24')
                preparation_end=time.monotonic()
                preparation['prepare_seconds']+=preparation_end-preparation_start
                preparation['frames_prepared']+=1
                yield {'timestamp':timestamp,'image':image,'width':frame.width,'height':frame.height,
                    'hardware_decode':stream.codec_context.is_hwaccel,'decode_backend':'pyav',
                    'preparation_start':preparation_start,'preparation_end':preparation_end}
                sample_at=timestamp+interval
                preparation_start=time.monotonic()
    with CheckpointWriter(lambda snapshot:save_json(state_path,snapshot)) as checkpoints, \
            prepared_frames(producer,lambda:check_cancel(stop),hybrid,capacity) as samples:
        for group in sample_batches(samples,batch_size,lambda:check_cancel(stop)):
            check_cancel(stop)
            checkpoints.check()
            unique,indices=adjacent_reuse(group,previous_image,np.array_equal)
            batch_detections=[]
            if unique:
                before_inference=time.monotonic()
                results=predict_frames(ocr,[sample['image'] for sample in unique],lambda:check_cancel(stop))
                if len(results)!=len(unique):
                    raise RuntimeError('画面批处理结果数量不一致，已保留此前完成的进度。')
                inference_end=time.monotonic()
                inference_seconds+=inference_end-before_inference
                inference_ranges.append((before_inference,inference_end))
                for sample,result in zip(unique,results):
                    found=[]
                    for text,score,box in zip(result['rec_texts'],result['rec_scores'],result['rec_boxes']):
                        found.append({'ja':text,'score':float(score),
                            'box':[float(box[0]/sample['width']),float(box[1]/sample['height']),
                                   float(box[2]/sample['width']),float(box[3]/sample['height'])]})
                    batch_detections.append(found)
                scans+=len(unique);batches+=1
            previous_detections=detections
            for sample,index in zip(group,indices):
                check_cancel(stop)
                checkpoints.check()
                timestamp=sample['timestamp']
                overlap_seconds+=sum(max(0,min(sample['preparation_end'],end)-max(sample['preparation_start'],start))
                    for start,end in inference_ranges)
                detections=previous_detections if index<0 else batch_detections[index]
                previous_image=sample['image']
                track_frame(state,detections,timestamp,interval,duration)
                next_time=timestamp+interval
                state['next_time']=next_time
                active={**backend,'active_device_type':'hybrid' if hybrid else backend['active_device_type']}
                state['backend']={**active,'hardware_decode':sample['hardware_decode'],
                    'decode_backend':sample['decode_backend']}
                frames+=1
                checkpoints.submit(screen_snapshot(state))
                elapsed=time.monotonic()-started
                processed,fraction,eta,speed=scan_progress(timestamp,resumed,duration,elapsed,frames)
                if frames==1 or elapsed-last_report>=.25:
                    suffix=f' · 批量 {batch_size} 帧' if batch_size>1 else ''
                    emit(f'扫描画面日语 · 已发现 {len(state["rows"])+len(state["active"])} 段文字'+suffix,
                        45+5*fraction,stage='screen',stage_progress=round(fraction*100,1),
                        processed_seconds=processed,total_seconds=duration,eta_seconds=eta,
                        scan_speed=round(speed,2) if speed is not None else None,**active)
                    last_report=elapsed
    check_cancel(stop)
    state['rows'].extend(state.pop('active',[]))
    state['rows'].sort(key=lambda row:row['start'])
    state['complete']=True
    state['scan_statistics']={'frames_this_run':frames,'ocr_calls_this_run':scans,
        'ocr_batches_this_run':batches,'ocr_batch_size':batch_size,
        'elapsed_seconds_this_run':round(time.monotonic()-started,3),
        'decode_seconds':round(preparation['decode_seconds'],3),'inference_seconds':round(inference_seconds,3),
        'frames_prepared':preparation['frames_prepared'],'prepare_seconds':round(preparation['prepare_seconds'],3),
        'preparation_inference_overlap_seconds':round(overlap_seconds,4),
        'checkpoint_updates':checkpoints.updates,'checkpoint_writes':checkpoints.writes,
        'checkpoint_io_seconds':round(checkpoints.seconds,4),
        'hybrid_processing':hybrid,'prefetch_capacity':capacity if hybrid else 0}
    save_json(state_path,state)
    emit(f'画面日语识别完成 · {len(state["rows"])} 段',50,stage='screen',stage_progress=100,**state.get('backend',backend))
    return state['rows']


if __name__=='__main__':
    from engine import load_config,Cancelled,emit
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        run(sys.argv[1],load_config(),sys.argv[2],sys.argv[3] or None,float(sys.argv[4]),
            cpu='--cpu' in sys.argv,software_decode='--software-decode' in sys.argv)
    except Cancelled as exc:
        emit(str(exc),cancelled=True)
        sys.exit(2)
    except Exception as exc:
        emit(str(exc),error=True)
        sys.exit(1)
