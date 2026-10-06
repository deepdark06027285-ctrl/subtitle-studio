"""First-time official downloads only; never called by the video processor."""
import json
import os
from pathlib import Path
import sys
import urllib.request
import zipfile

root = Path(__file__).resolve().parent
sys.path.insert(0,str(root/'vendor'))
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_OFFLINE']='0'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'


def prepare_runtime():
    folder=root/'translation-runtime'/'b11146'
    folder.mkdir(parents=True,exist_ok=True)
    for name,size in [('llama-b11146-bin-win-cuda-12.4-x64.zip',253869799),
                      ('cudart-llama-bin-win-cuda-12.4-x64.zip',391443627)]:
        archive=folder/name
        if not archive.is_file() or archive.stat().st_size!=size:
            print('下载本地翻译运行库：'+name,flush=True)
            request=urllib.request.Request('https://github.com/ggml-org/llama.cpp/releases/download/b11146/'+name,
                headers={'User-Agent':'Local-Subtitle-Setup'})
            partial=archive.with_suffix('.partial')
            with urllib.request.urlopen(request,timeout=120) as response,partial.open('wb') as output:
                while chunk:=response.read(1024*1024):
                    output.write(chunk)
            if partial.stat().st_size!=size:
                raise RuntimeError('翻译运行库下载不完整，请重新安装。')
            partial.replace(archive)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                if not (folder/member.filename).resolve().is_relative_to(folder.resolve()):
                    raise RuntimeError('翻译运行库包含无效文件路径。')
            bundle.extractall(folder)
    return str(next(folder.rglob('llama-server.exe')).relative_to(root))


def main():
    from huggingface_hub import snapshot_download,hf_hub_download
    snapshot_download('mobiuslabsgmbh/faster-whisper-large-v3-turbo',token=False,
        revision='0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf',local_dir=root/'models'/'whisper',
        allow_patterns=['model.bin','config.json','tokenizer.json','preprocessor_config.json','vocabulary.json'])
    gguf=hf_hub_download('tencent/Hy-MT2-1.8B-GGUF','Hy-MT2-1.8B-Q8_0.gguf',token=False,
        revision='a0c709d9fac510f2c807aa3af52872340dc37a4a',local_dir=root/'models'/'hy-mt2')
    for name,revision in [('PP-OCRv6_small_det','106c97591b235f607453300d9fc8c1cad1b25488'),
                          ('PP-OCRv6_small_rec','bd619643acac4b9650c040234da8d944476ee3f1')]:
        snapshot_download('PaddlePaddle/'+name,token=False,revision=revision,local_dir=root/'models'/name,
            allow_patterns=['inference.json','inference.pdiparams','inference.yml'])
    config={'python':sys.executable,'dependencies':'vendor','cuda':'cuda',
        'whisper':'models/whisper','translation_backend':'hy-mt2',
        'translation_gguf':str(Path(gguf).relative_to(root)), 'llama_server':prepare_runtime(),
        'ocr_dependencies':'ocr-vendor','ocr_gpu_dependencies':'ocr-gpu-vendor',
        'ocr_gpu_cuda':'ocr-gpu-cuda','ocr_det':'models/PP-OCRv6_small_det','ocr_rec':'models/PP-OCRv6_small_rec'}
    path=root/'config.json'
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)
    print('本地日语、中文翻译和画面文字模型准备完成。')


if __name__=='__main__':
    main()
