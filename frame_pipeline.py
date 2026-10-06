"""Bounded CPU frame preparation while the consumer performs GPU inference."""
from contextlib import contextmanager
import queue
import threading


def sample_batches(samples,size,check_cancel):
    """Same-size video frames only; keep every sampled timestamp in order."""
    size=max(1,min(4,size))
    group=[]
    shape=None
    for sample in samples:
        check_cancel()
        current=(sample['width'],sample['height'])
        if group and current!=shape:
            yield group;group=[]
        shape=current;group.append(sample)
        if len(group)==size:
            yield group;group=[]
    if group: yield group


def adjacent_reuse(group,previous_image,equal):
    """Reuse only byte-identical neighbouring frames, as in sequential OCR."""
    unique=[];indices=[];last=previous_image;index=-1
    for sample in group:
        image=sample['image']
        if last is None or not equal(image,last):
            unique.append(sample);index=len(unique)-1
        indices.append(index);last=image
    return unique,indices


@contextmanager
def prepared_frames(producer,check_cancel,parallel=False,capacity=2):
    if not parallel:
        iterator=iter(producer())
        try: yield iterator
        finally: getattr(iterator,'close',lambda:None)()
        return
    pending=queue.Queue(maxsize=max(1,min(4,capacity)))
    stopped=threading.Event()
    def put(value):
        while not stopped.is_set():
            try: pending.put(value,timeout=.1);return True
            except queue.Full: pass
        return False
    def prepare():
        try:
            for item in producer():
                if stopped.is_set() or not put(('frame',item)): break
        except BaseException as exc:
            put(('error',exc))
        finally:
            put(('done',None))
    worker=threading.Thread(target=prepare,daemon=True,name='subtitle-frame-preparation')
    worker.start()
    def consume():
        while True:
            check_cancel()
            try: kind,value=pending.get(timeout=.1)
            except queue.Empty: continue
            if kind=='done': return
            if kind=='error': raise value
            yield value
    try: yield consume()
    finally:
        stopped.set()
        worker.join(timeout=1)
