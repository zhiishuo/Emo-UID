"""Bounded real-data admission only; never evaluates the test split or reports a ranking."""
import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch
from transformers import AutoModel

from misr.model import MISR, MISRConfig
from triagedmsa.model import TriagedMSA


class CompatUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module.startswith('numpy._core'):module=module.replace('numpy._core','numpy.core',1)
        return super().find_class(module,name)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--expected-data-sha256',required=True)
    p.add_argument('--bert',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261026)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--steps',type=int,default=2)
    args=p.parse_args()
    if not 1<=args.steps<=2 or not 2<=args.batch_size<=8:
        raise ValueError('Engineering admission is limited to two batches of at most eight')
    args.output.mkdir(parents=True,exist_ok=False)
    receipt={'stage':'real_data_engineering_smoke','scientific_benchmark_result':False,
             'dataset':'CMU-MOSEI','reason_for_dataset':'Available verified A100 artifact; no MOSI scientific run is implied',
             'seed':args.seed,'data':str(args.data),'data_sha256':digest(args.data),
             'bert':str(args.bert),'steps':args.steps,'batch_size':args.batch_size,
             'test_evaluated':False,'model_checkpoint_saved':False,'models':{},'status':'started'}
    save(args.output/'receipt.json',receipt)
    assert receipt['data_sha256']==args.expected_data_sha256
    with gzip.open(args.data,'rb') as f:payload=CompatUnpickler(f).load()
    payload.pop('test',None)
    splits={name:payload[name] for name in ['train','valid']}
    receipt['split_shapes']={name:{k:list(np.asarray(s[k]).shape) for k in ['text_bert','audio','vision','regression_labels']} for name,s in splits.items()}
    n=len(splits['train']['regression_labels'])
    indices=np.random.default_rng(args.seed).permutation(n)[:args.batch_size*args.steps]
    receipt['train_row_indices']=indices.tolist()
    receipt['valid_row_indices']=list(range(args.batch_size))
    receipt['split_identity']='artifact hash plus row indices; no externally verified discrepancy-bin identity'
    device=torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(.20,device)
    receipt['gpu']=torch.cuda.get_device_name(device)

    def batch(name,idx):
        s=splits[name]
        tokens=torch.as_tensor(np.asarray(s['text_bert'])[idx],dtype=torch.long,device=device)
        audio=torch.as_tensor(np.asarray(s['audio'])[idx],dtype=torch.float32,device=device)
        vision=torch.as_tensor(np.asarray(s['vision'])[idx],dtype=torch.float32,device=device)
        labels=torch.as_tensor(np.asarray(s['regression_labels'])[idx],dtype=torch.float32,device=device).reshape(-1)
        mask=tokens[:,1,:].bool()
        assert audio.shape[:2]==vision.shape[:2]==mask.shape
        assert audio.shape[-1]==74 and vision.shape[-1]==35
        assert mask.any(1).all() and not ((~mask[:,:-1]) & mask[:,1:]).any()
        assert torch.isfinite(audio[mask]).all() and torch.isfinite(vision[mask]).all()
        return tokens,audio,vision,labels,mask

    for name in ['misr','triagedmsa']:
        record={'status':'started','training_batches':[]}
        receipt['models'][name]=record
        save(args.output/'receipt.json',receipt)
        try:
            random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            bert=AutoModel.from_pretrained(str(args.bert),local_files_only=True).to(device)
            if name=='misr':
                core=MISR(MISRConfig(audio_dim=74,vision_dim=35)).to(device)
            else:
                core=TriagedMSA(audio_dim=74,vision_dim=35,heads=4,dropout=.7).to(device)
            bert.train();core.train()
            lr_bert=5e-6 if name=='misr' else 1e-4
            opt=torch.optim.Adam([{'params':core.parameters(),'lr':1e-4},
                                  {'params':bert.parameters(),'lr':lr_bert}])
            record['learning_rates']={'core':1e-4,'bert':lr_bert}
            record['source_sha256']=digest(Path(__file__).parent/name/'model.py')
            record['apd_scope']='epoch-1 zero targets only; complete-train reference/target-history lifecycle is not exercised' if name=='triagedmsa' else None
            torch.cuda.reset_peak_memory_stats(device)
            for step in range(args.steps):
                ids=indices[step*args.batch_size:(step+1)*args.batch_size]
                tokens,a,v,y,mask=batch('train',ids)
                opt.zero_grad(set_to_none=True)
                start=time.monotonic()
                language=bert(input_ids=tokens[:,0,:],attention_mask=tokens[:,1,:],token_type_ids=tokens[:,2,:]).last_hidden_state
                if name=='misr':out=core(language,a,v,masks={'language':mask,'audio':mask,'vision':mask},labels=y)
                else:out=core(language,a,v,masks=mask,labels=y,relationship_targets=torch.zeros_like(y))
                loss=out['losses']['total']
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward()
                grads=[x.grad for x in list(core.parameters())+list(bert.parameters()) if x.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads):raise RuntimeError('Nonfinite or missing gradients')
                grad_norm=float(torch.stack([g.detach().norm().square() for g in grads]).sum().sqrt())
                opt.step();torch.cuda.synchronize()
                entry={'step':step+1,'losses':{k:float(z.detach()) for k,z in out['losses'].items()},
                       'gradient_norm':grad_norm,'seconds':time.monotonic()-start}
                record['training_batches'].append(entry)
                print(json.dumps({'model':name,**entry}),flush=True)
                save(args.output/'receipt.json',receipt)
            core.eval();bert.eval()
            buffers={k:x.detach().cpu().clone() for k,x in core.named_buffers()}
            tokens,a,v,_,mask=batch('valid',np.arange(args.batch_size))
            with torch.no_grad():
                language=bert(input_ids=tokens[:,0,:],attention_mask=tokens[:,1,:],token_type_ids=tokens[:,2,:]).last_hidden_state
                out=core(language,a,v,masks={'language':mask,'audio':mask,'vision':mask} if name=='misr' else mask)
            assert torch.isfinite(out['prediction']).all()
            assert all(torch.equal(buffers[k],x.detach().cpu()) for k,x in core.named_buffers())
            record.update(status='passed',validation_forward_samples=args.batch_size,
                          validation_state_unchanged=True,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device))
            del out,opt,core,bert,language,grads
            gc.collect();torch.cuda.empty_cache()
        except Exception as exc:
            record.update(status='failed',error=repr(exc))
            receipt['status']='failed'
            save(args.output/'receipt.json',receipt)
            raise
        save(args.output/'receipt.json',receipt)
    receipt['status']='completed_engineering_smoke_not_benchmark'
    save(args.output/'receipt.json',receipt)
    print(json.dumps({'status':receipt['status'],'test_evaluated':False}),flush=True)


if __name__=='__main__':main()
