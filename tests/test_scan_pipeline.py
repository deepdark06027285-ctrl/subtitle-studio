import sys,threading,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scan_pipeline import CheckpointWriter,screen_snapshot
from frame_pipeline import sample_batches,adjacent_reuse
from runtime_options import ocr_batch_size


class CheckpointTests(unittest.TestCase):
    def test_cpu_unknown_memory_and_low_memory_bound_batch_size(self):
        gpu={'memory_total':8192,'memory_used':2048}
        self.assertEqual(ocr_batch_size('fast',gpu),1)
        self.assertEqual(ocr_batch_size('balanced',gpu),1)
        self.assertEqual(ocr_batch_size('fast',gpu,4),4)
        self.assertEqual(ocr_batch_size('quality',gpu),1)
        self.assertEqual(ocr_batch_size('fast',None),1)
        self.assertEqual(ocr_batch_size('fast',{}),1)
        self.assertEqual(ocr_batch_size('fast',{'memory_total':1024,'memory_used':100},4),2)
        self.assertEqual(ocr_batch_size('fast',{'memory_total':1024,'memory_used':900},4),1)

    def test_batches_keep_tail_and_split_video_resolution_changes(self):
        samples=[{'width':10,'height':10,'timestamp':i,'image':str(i)} for i in range(5)]
        samples[3].update(width=20)
        groups=list(sample_batches(iter(samples),4,lambda:None))
        self.assertEqual([[s['timestamp'] for s in group] for group in groups],[[0,1,2],[3],[4]])

    def test_identical_neighbours_reuse_results_across_batch_boundary(self):
        group=[{'image':image} for image in ('A','A','B','B','A')]
        unique,indices=adjacent_reuse(group,'A',lambda a,b:a==b)
        self.assertEqual([s['image'] for s in unique],['B','A'])
        self.assertEqual(indices,[-1,-1,0,0,1])

    def test_cancel_while_collecting_batch_does_not_yield_prepared_frames(self):
        samples=[{'width':1,'height':1,'image':'A'}]*4
        calls=[]
        def check():
            calls.append(1)
            if len(calls)==2:raise RuntimeError('cancel')
        with self.assertRaisesRegex(RuntimeError,'cancel'):
            list(sample_batches(iter(samples),4,check))

    def test_saving_overlaps_inference_and_keeps_only_latest_pending_state(self):
        saving=threading.Event();release=threading.Event();saved=[]
        def save(value):
            saving.set();self.assertTrue(release.wait(2));saved.append(value)
        writer=CheckpointWriter(save)
        try:
            writer.submit({'next_time':1})
            self.assertTrue(saving.wait(1))
            for i in range(2,101):writer.submit({'next_time':i})
            self.assertEqual(writer.pending,{'next_time':100})
            release.set();writer.close()
            self.assertEqual(saved,[{'next_time':1},{'next_time':100}])
            self.assertEqual(writer.updates,100);self.assertEqual(writer.writes,2)
        finally:release.set();writer.close()

    def test_cancel_flushes_completed_state_without_prepared_frames(self):
        saved=[]
        with self.assertRaisesRegex(RuntimeError,'cancel'):
            with CheckpointWriter(saved.append) as writer:
                writer.submit({'next_time':2,'rows':['completed']})
                raise RuntimeError('cancel')
        self.assertEqual(saved[-1],{'next_time':2,'rows':['completed']})
        self.assertFalse(writer.worker.is_alive())

    def test_storage_error_is_reported_and_not_treated_as_success(self):
        def save(_):raise PermissionError('disk unavailable')
        writer=CheckpointWriter(save);writer.submit({'next_time':1})
        with self.assertRaisesRegex(PermissionError,'disk unavailable'):writer.close()
        self.assertFalse(writer.worker.is_alive())
        with self.assertRaises(PermissionError):writer.submit({'next_time':2})

    def test_active_tracks_are_frozen_before_the_next_frame_updates_them(self):
        state={'next_time':1,'rows':[],'active':[{'ja':'こんにちは','end':1,'box':[0,0,1,1]}],
            'backend':{'active_device_type':'hybrid'}}
        snapshot=screen_snapshot(state)
        state['next_time']=2;state['active'][0]['end']=2;state['active'][0]['box'][0]=.2
        state['backend']['active_device_type']='cpu';state['rows'].append({'ja':'later'})
        self.assertEqual(snapshot['next_time'],1)
        self.assertEqual(snapshot['active'][0]['end'],1)
        self.assertEqual(snapshot['active'][0]['box'][0],0)
        self.assertEqual(snapshot['rows'],[])
        self.assertEqual(snapshot['backend']['active_device_type'],'hybrid')


if __name__=='__main__':unittest.main()
