"""Local-only translation. The HTTP listener belongs to this worker on loopback."""
import json
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import threading
import urllib.request
import unicodedata


def text_key(text):
    return unicodedata.normalize('NFKC',text).strip()


def translation_prompt(text,style='natural'):
    if style not in ('natural','literal'):
        raise ValueError('请选择自然口语或忠实直译。')
    if style=='natural':
        return ('把下面的日语翻译成中国人日常说话的简体中文。忠实传达原意和语气，简短流畅，'
            f'避免生硬直译。不要增加原文未说的信息。只输出译文：\n\n{text}')
    return f'将以下文本翻译为简体中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}'


class LocalTranslator:
    def __init__(self,cfg,job,check_cancel,gpu=True):
        self.cfg,self.job,self.check_cancel,self.gpu=cfg,Path(job),check_cancel,gpu
        self.proc=None
        self.log=None
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        from runtime_options import profile
        self.options=profile(cfg.get('speed','fast'))
        self.parallel=self.options['translation_parallel'] if gpu else 1
        self.backend=None
        self.request_count=0
        self._request_lock=threading.Lock()

    def __enter__(self):
        executable=Path(self.cfg['llama_server'])
        model=Path(self.cfg['translation_gguf'])
        if not executable.is_file() or not model.is_file():
            raise RuntimeError('新翻译模型或运行库缺失，请重新安装本地环境。')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            port=sock.getsockname()[1]
        self.url=f'http://127.0.0.1:{port}'
        self.log=(self.job/'translation-runtime.log').open('w',encoding='utf-8')
        env=dict(os.environ)
        env['PATH']=str(executable.parent)+os.pathsep+env.get('PATH','')
        command=[str(executable),'--model',str(model),'--host','127.0.0.1','--port',str(port),
            '--ctx-size',str(2048*self.parallel),'--parallel',str(self.parallel),'--threads',str(self.cfg.get('cpu_threads',4)),
            '--n-gpu-layers','99' if self.gpu else '0','--device','CUDA0' if self.gpu else 'none',
            '--log-verbosity','5']
        if self.gpu:
            command+=['--flash-attn','on']
        self.proc=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=self.log,env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            deadline=time.monotonic()+90
            while time.monotonic()<deadline:
                self.check_cancel()
                if self.proc.poll() is not None:
                    raise RuntimeError((self.job/'translation-runtime.log').read_text(encoding='utf-8',errors='replace')[-1600:])
                try:
                    with self.opener.open(self.url+'/health',timeout=1) as response:
                        if response.status==200:
                            self.log.flush()
                            log=(self.job/'translation-runtime.log').read_text(encoding='utf-8',errors='replace')
                            layers=re.search(r'offloaded (\d+)/(\d+) layers',log)
                            if self.gpu and (not layers or int(layers[1])==0):
                                raise RuntimeError('翻译运行库没有实际加载到所选显卡，已保留断点；请检查显卡环境或选择 CPU。')
                            from hardware import device_event
                            self.backend=device_event(self.cfg,self.gpu)
                            actual=re.search(r'using device CUDA\d+ \(([^)]+)\)',log)
                            if self.gpu and actual:
                                self.backend['active_device_name']=actual[1]
                            self.backend.update(translation_parallel=self.parallel,
                                translation_gpu_layers=int(layers[1]) if layers else 0)
                            return self
                except (OSError,urllib.error.URLError):
                    pass
                time.sleep(.2)
            raise RuntimeError('本地翻译模型启动超时。')
        except BaseException:
            self.close()
            raise

    def _request(self,text,tokens):
        self.check_cancel()
        with self._request_lock:
            self.request_count+=1
        prompt=translation_prompt(text,self.cfg.get('translation_style','natural'))
        payload={'messages':[{'role':'user','content':prompt}],'stream':True,'max_tokens':tokens,
            'temperature':.7,'top_p':.6,'top_k':20,'repeat_penalty':1.05,'seed':42}
        request=urllib.request.Request(self.url+'/v1/chat/completions',
            data=json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
        pieces=[]
        finish_reason=None
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request,timeout=90) as response:
            for line in response:
                self.check_cancel()
                if not line.startswith(b'data: '):
                    continue
                data=line[6:].strip()
                if data==b'[DONE]':
                    break
                item=json.loads(data)
                for choice in item.get('choices',[]):
                    pieces.append(choice.get('delta',{}).get('content') or '')
                    finish_reason=choice.get('finish_reason') or finish_reason
        translated=''.join(pieces).strip()
        return translated,finish_reason

    def translate(self,text):
        tokens=min(self.options['translation_tokens'],max(64,len(text)*3+32))
        translated,reason=self._request(text,tokens)
        # Retry genuine truncation, empty output, or untranslated kana, never add dialogue background.
        if reason=='length' or not translated or re.search(r'[\u3040-\u30ff]',translated):
            self.check_cancel()
            translated,reason=self._request(text,max(384,len(text)*4+64))
        if not translated:
            raise RuntimeError('本地翻译没有返回译文，已保留原文和断点。')
        if reason=='length':
            raise RuntimeError('译文未能完整生成，已保留原文和断点，请使用精细档或检查这句原文。')
        return translated

    def translate_batch(self,texts):
        """Separate prompts share GPU batching; result order always follows the input."""
        if self.parallel==1 or len(texts)<2:
            return [self.translate(text) for text in texts]
        pool=ThreadPoolExecutor(max_workers=self.parallel)
        futures={pool.submit(self.translate,text):i for i,text in enumerate(texts)}
        results=[None]*len(texts)
        pending=set(futures)
        try:
            while pending:
                self.check_cancel()
                finished,pending=wait(pending,timeout=.1,return_when=FIRST_COMPLETED)
                for future in finished:
                    results[futures[future]]=future.result()
        except BaseException:
            # Terminating our local server also releases any blocked stream readers.
            self.close()
            pool.shutdown(wait=True,cancel_futures=True)
            raise
        pool.shutdown(wait=True)
        return results

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.log:
            self.log.close()

    def __exit__(self,*_):
        self.close()
