"""Keep slow checkpoint I/O off the OCR inference path, with a bounded backlog."""
import threading
import time


def screen_snapshot(state):
    # Finished rows never change; active tracks get updated by the next frame.
    snapshot={**state,'rows':list(state['rows']),
        'active':[{**row,'box':list(row['box'])} for row in state.get('active',[])]}
    if 'backend' in state: snapshot['backend']=dict(state['backend'])
    return snapshot


class CheckpointWriter:
    def __init__(self,save):
        self.save=save
        self.condition=threading.Condition()
        self.pending=None
        self.closing=False
        self.error=None
        self.updates=0
        self.writes=0
        self.seconds=0
        self.worker=threading.Thread(target=self._write,daemon=True,name='subtitle-checkpoint-writer')
        self.worker.start()

    def check(self):
        if self.error is not None: raise self.error

    def submit(self,snapshot):
        with self.condition:
            self.check()
            if self.closing: raise RuntimeError('Checkpoint writer is closed')
            self.pending=snapshot
            self.updates+=1
            self.condition.notify()

    def _write(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda:self.pending is not None or self.closing)
                    if self.pending is None: return
                    snapshot=self.pending;self.pending=None
                before=time.monotonic()
                self.save(snapshot)
                self.seconds+=time.monotonic()-before
                self.writes+=1
        except BaseException as exc:
            with self.condition:
                self.error=exc
                self.pending=None
                self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closing=True
            self.condition.notify()
        self.worker.join()
        self.check()

    def __enter__(self): return self

    def __exit__(self,*_): self.close()
