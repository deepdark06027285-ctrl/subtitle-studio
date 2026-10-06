"""Processing presets; device routing is independent of subtitle cache identity."""
PROFILES={
    'fast':{'label':'高速','beam_size':1,'translation_parallel':4,'translation_tokens':192,'nvenc_preset':'p3','ocr_batch_size':1},
    'balanced':{'label':'均衡','beam_size':3,'translation_parallel':2,'translation_tokens':256,'nvenc_preset':'p5','ocr_batch_size':1},
    'quality':{'label':'精细','beam_size':5,'translation_parallel':1,'translation_tokens':384,'nvenc_preset':'p6','ocr_batch_size':1},
}
PROFILE_LABELS={item['label']:key for key,item in PROFILES.items()}
PROFILE_NOTES={
    'fast':'优先完成 · 并行翻译、快速压制\n显存占用稍高，可能略降准确度与画质',
    'balanced':'兼顾速度与质量\n常规识别、并行翻译、均衡压制',
    'quality':'优先质量 · 更充分的对白搜索\n逐句翻译、精细压制，耗时较长',
}


def profile(name):
    if name not in PROFILES:
        raise ValueError('请选择有效的速度档位。')
    return dict(PROFILES[name])


def ocr_batch_size(name,gpu,requested=None):
    if gpu is None:return 1
    free=(gpu.get('memory_total') or 0)-(gpu.get('memory_used') or 0)
    limit=4 if free>=1536 else 2 if free>=768 else 1
    # Multi-frame inference remains opt-in: full-scan benchmarks were decode-bound.
    return min(max(1,min(4,int(requested or profile(name)['ocr_batch_size']))),limit)
