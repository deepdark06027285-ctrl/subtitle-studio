"""Select GPU frames before downloading; keep original pixels and timestamps."""
import os
from pathlib import Path
import subprocess
import threading


class PipeReader:
    # No seek attribute: FFmpeg's live NUT stream cannot be sought.
    def __init__(self,pipe):self.pipe=pipe
    def read(self,size):return self.pipe.read(size)


def sampling_filter(resumed,interval,pixel_format):
    if interval<=0:raise ValueError('采样间隔必须大于零。')
    if pixel_format not in ('nv12','p010le'):
        raise ValueError('当前显卡采样不支持该画面格式。')
    # Matches Python's timestamp + 1e-4 >= previous_timestamp + interval.
    threshold=max(0,interval-1e-4)
    # NUT does not describe P010 reliably; planar 10-bit preserves every sample.
    transport=',format=yuv420p10le' if pixel_format=='p010le' else ''
    return (f"select='gte(t,{resumed-1e-4:.9f})*(isnan(prev_selected_t)+"
        f"gte(t-prev_selected_t,{threshold:.9f}))',hwdownload,format={pixel_format}{transport}")


def sampled_frames(source,av,ffmpeg,job,resumed,interval,driver_index,check_cancel):
    with av.open(str(source)) as container:
        name=container.streams.video[0].codec_context.format.name
    # Other formats keep the established PyAV path in the caller.
    pixel_format='p010le' if name=='yuv420p10le' else 'nv12'
    command=[str(ffmpeg),'-hide_banner','-nostdin','-loglevel','error','-copyts',
        '-hwaccel','cuda','-hwaccel_device',str(driver_index),'-hwaccel_output_format','cuda']
    if resumed>0:command+=['-ss',f'{max(0,resumed-1):.9f}']
    command+=['-noautorotate','-i',str(source),'-map','0:v:0','-an','-sn','-vf',sampling_filter(resumed,interval,pixel_format),
        '-fps_mode','passthrough','-enc_time_base','demux','-c:v','rawvideo','-f','nut','pipe:1']
    finished=threading.Event();cancelled=[]
    log_path=Path(job)/'decode-runtime.log'
    with log_path.open('wb') as log:
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        def watch():
            while not finished.wait(.05):
                try:check_cancel()
                except BaseException as exc:
                    cancelled.append(exc)
                    if proc.poll() is None:
                        try:proc.terminate()
                        except OSError:pass
                    return
        watcher=threading.Thread(target=watch,daemon=True,name='subtitle-decode-cancel')
        watcher.start()
        try:
            with av.open(PipeReader(proc.stdout),format='nut') as container:
                for frame in container.decode(video=0):
                    check_cancel()
                    expected='yuv420p10le' if pixel_format=='p010le' else 'nv12'
                    if frame.format.name!=expected:
                        raise RuntimeError('采样管道画面格式不一致，已停止以保护识别质量。')
                    yield frame
            if cancelled:raise cancelled[0]
            code=proc.wait(timeout=10)
            if code:
                raise RuntimeError('显卡采样解码失败，请查看 decode-runtime.log。')
        except BaseException:
            if cancelled:raise cancelled[0]
            check_cancel()
            raise
        finally:
            finished.set()
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=3)
            proc.stdout.close();watcher.join(timeout=1)
