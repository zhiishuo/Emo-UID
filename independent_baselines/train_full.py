"""Full independent baseline training, validation selection, and deferred test export."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score
from transformers import AutoModel

from audit_data import load, sample_ids
from run_real_data_smoke import digest
from misr.model import MISR, MISRConfig
from triagedmsa.model import TriagedMSA


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def metrics(pred, target):
    # Float32 std can be positive for a constant array because its mean rounds.
    # Correlation's own float64 reduction then divides by zero; use one dtype.
    pred, target = np.asarray(pred,dtype=np.float64).reshape(-1), np.asarray(target,dtype=np.float64).reshape(-1)
    assert len(pred) == len(target) and np.isfinite(pred).all()
    out = {'n': len(target), 'mae': float(np.abs(pred-target).mean()),
           'acc7': float((np.rint(np.clip(pred,-3,3)) == np.rint(np.clip(target,-3,3))).mean()),
           'corr': float(np.corrcoef(pred,target)[0,1]) if np.std(pred)>0 and np.std(target)>0 else 0.0}
    for name, mask in [('has0', np.ones(len(target), dtype=bool)), ('nonzero', target != 0)]:
        if not mask.any():
            out.update({f'acc2_{name}': None, f'f1_{name}': None})
            continue
        actual = target[mask] >= 0 if name == 'has0' else target[mask] > 0
        estimate = pred[mask] >= 0 if name == 'has0' else pred[mask] > 0
        out[f'acc2_{name}'] = float((actual==estimate).mean())
        out[f'f1_{name}'] = float(f1_score(actual, estimate, average='weighted', zero_division=0))
    return out


class Split:
    def __init__(self, raw, name):
        self.name = name
        self.ids, self.id_kind = sample_ids(raw, name)
        tokens = np.asarray(raw['text_bert'])
        if (tokens.ndim != 3 or tokens.shape[1] != 3 or min(tokens.shape) < 1
                or not np.issubdtype(tokens.dtype, np.integer)):
            raise ValueError('text_bert must be nonempty integer [N,3,T]')
        if not np.isin(tokens[:,1], [0,1]).all():
            raise ValueError('Attention mask must contain only 0/1')
        self.tokens = tokens.astype(np.int64)
        self.audio = np.asarray(raw['audio'], dtype=np.float32)
        self.vision = np.asarray(raw['vision'], dtype=np.float32)
        labels = np.asarray(raw['regression_labels'], dtype=np.float32)
        n = len(tokens)
        if labels.shape not in ((n,), (n,1)) or not np.isfinite(labels).all():
            raise ValueError('Labels must be finite [N] or [N,1] matching features')
        self.labels = labels.reshape(-1)
        mask = self.tokens[:,1].astype(bool)
        if any(x.ndim != 3 or x.shape[:2] != mask.shape or x.shape[-1] < 1
               for x in (self.audio, self.vision)):
            raise ValueError('Expected benchmark-aligned features; no implicit resampling')
        if not mask.any(1).all() or ((~mask[:,:-1]) & mask[:,1:]).any():
            raise ValueError('Invalid text mask')
        if not all(np.isfinite(x[mask]).all() for x in [self.audio,self.vision]):
            raise ValueError('Nonfinite valid features')

    def batch(self, rows, device):
        return [torch.as_tensor(x[rows], device=device) for x in
                [self.tokens, self.audio, self.vision, self.labels]]

    def __len__(self):
        return len(self.labels)


class Baseline(nn.Module):
    def __init__(self, config, split):
        super().__init__()
        self.name = config['model']
        if self.name not in ('misr', 'triagedmsa'):
            raise ValueError('model must be misr or triagedmsa')
        self.bert = AutoModel.from_pretrained(config['bert'], local_files_only=True)
        if self.bert.config.hidden_size != config.get('model_kwargs', {}).get('language_dim', 768):
            raise ValueError('BERT hidden_size must match model_kwargs.language_dim (default 768)')
        kwargs = dict(audio_dim=split.audio.shape[-1], vision_dim=split.vision.shape[-1])
        if self.name == 'misr':
            self.core = MISR(MISRConfig(**kwargs, **config.get('model_kwargs', {})))
        else:
            self.core = TriagedMSA(**kwargs, max_length=split.tokens.shape[-1], **config.get('model_kwargs', {}))

    def forward(self, batch, targets=None, supervised=False):
        tokens, audio, vision, y = batch
        mask = tokens[:,1].bool()
        language = self.bert(input_ids=tokens[:,0], attention_mask=tokens[:,1],
                             token_type_ids=tokens[:,2]).last_hidden_state
        kwargs = {'labels': y} if supervised else {}
        if self.name == 'misr':
            kwargs['masks'] = {key: mask for key in ['language','audio','vision']}
        else:
            kwargs['masks'] = mask
            if supervised:
                kwargs['relationship_targets'] = targets
        return self.core(language, audio, vision, **kwargs)


@torch.no_grad()
def evaluate(model, split, batch_size, device):
    model.eval()
    values = []
    for start in range(0, len(split), batch_size):
        output = model(split.batch(np.arange(start,min(start+batch_size,len(split))),device))
        values.append(output['prediction'].detach().cpu().numpy())
    prediction = np.concatenate(values)
    return metrics(prediction, split.labels), prediction


@torch.no_grad()
def refresh_targets(model, split, targets, next_epoch, batch_size, device):
    # Reference and target snapshots use a fixed model and TRAIN examples only.
    if split.name != 'train':
        raise ValueError('APD refresh requires the train split')
    from triagedmsa.lifecycle import APDCache, refresh_apd_cache
    model.eval()
    def batches():
        for start in range(0, len(split), batch_size):
            rows = np.arange(start, min(start+batch_size,len(split)))
            yield [split.ids[i] for i in rows], split.batch(rows, device)
    cache = APDCache(tuple(split.ids),targets.cpu(),next_epoch-1)
    result = refresh_apd_cache(model.core,batches,lambda core,batch:model(batch),
                              cache=cache,train_labels=split.labels,
                              completed_epoch=next_epoch-1,source_split='train')
    return result.targets, {'reference_version':int(model.core.apd.reference_version),
                            'train_samples':len(split),'next_epoch':next_epoch,
                            'negative_target_fraction':float((result.targets<0).float().mean()),
                            'target_min':float(result.targets.min()),'target_max':float(result.targets.max())}


def export_predictions(path, split, prediction):
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label','prediction'])
        writer.writerows(zip(split.ids,split.labels.tolist(),prediction.tolist()))


def source_hashes():
    root = Path(__file__).parent
    return {str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*.py'))
            if p.name in {'train_full.py','audit_data.py','run_real_data_smoke.py','model.py','lifecycle.py'}
            and 'runs' not in p.parts and '__pycache__' not in p.parts}


def optimization_loss(losses, config, epoch):
    """Explicit optimization sensitivity; the paper objective remains logged as total."""
    schedule=config.get('consistency_schedule')
    if schedule is None:
        return losses['total']
    if config['model']!='triagedmsa' or 'consistency' not in losses:
        raise ValueError('Consistency schedule requires TriagedMSA')
    duration=schedule['epochs']
    if duration<1 or epoch<1:
        raise ValueError('Positive epoch/duration required')
    if schedule['kind']=='delayed':
        weight=0.0 if epoch<=duration else 1.0
    elif schedule['kind']=='linear':
        weight=min(1.0,(epoch-1)/duration)
    else:
        raise ValueError('Unknown schedule')
    result=losses['total']-(1-weight)*losses['consistency']
    losses['consistency_weight']=result.new_tensor(weight)
    losses['optimization_total']=result
    return result


def check_split_ids(*splits):
    for i, split in enumerate(splits):
        for other in splits[:i]:
            if set(split.ids) & set(other.ids):
                raise ValueError(f'{split.name}/{other.name} ID overlap')


def check_resume_state(saved, train, config):
    if saved['train_ids'] != train.ids:
        raise ValueError('Resume train IDs/order changed')
    if config['model'] == 'triagedmsa':
        from triagedmsa.lifecycle import APDCache
        cache = APDCache(tuple(saved['train_ids']), saved['targets'].cpu(), saved['targets_epoch'])
        cache.validate()
        if cache.epoch != saved['epoch'] + 1:
            raise ValueError('No next-epoch APD cache; cannot resume a stopped run')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--evaluate-only', action='store_true')
    mode.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get('_example_only', False):
        raise ValueError('Example only: replace placeholders, review choices, then remove _example_only')
    run = args.run_dir
    if not args.evaluate_only and not args.resume:
        run.mkdir(parents=True, exist_ok=False)
        write_json(run/'config.json',config)
    elif json.loads((run/'config.json').read_text()) != config:
        raise ValueError('Config does not match existing run')
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(.40,device)
    data_hash = digest(Path(config['data']))
    if data_hash != config['data_sha256']:
        raise ValueError('Data hash changed')
    payload = load(Path(config['data']))
    train, valid = Split(payload['train'],'train'), Split(payload['valid'],'valid')
    test = Split(payload['test'],'test') if args.evaluate_only else None
    del payload
    check_split_ids(train, valid, *([test] if test is not None else []))
    model = Baseline(config,train).to(device)
    receipt = {'status':'training','model':config['model'],'dataset':config['dataset'],
               'seed':seed,'data_sha256':data_hash,'source_sha256':source_hashes(),
               'train_n':len(train),'valid_n':len(valid),'test_used_for_selection':False,
               'train_id_sha256':hashlib.sha256('\n'.join(train.ids).encode()).hexdigest(),
               'params':sum(p.numel() for p in model.parameters()),'gpu':torch.cuda.get_device_name(0)}
    if args.evaluate_only:
        checkpoint = torch.load(run/'best.pt', map_location=device, weights_only=False)
        if checkpoint['receipt']['source_sha256'] != receipt['source_sha256']:
            raise ValueError('Source snapshot changed after training; evaluate with original source')
        model.load_state_dict(checkpoint['model'])
        report, pred = evaluate(model,test,config['batch_size'],device)
        export_predictions(run/'test_predictions.csv',test,pred)
        write_json(run/'test_metrics.json', {'status':'measured_independent_implementation',
                   'metrics':report,'best_epoch':checkpoint['epoch'],
                   'checkpoint_sha256':digest(run/'best.pt'),'data_sha256':data_hash,
                   'seed':seed,'config':config,'id_kind':test.id_kind})
        print(json.dumps({'event':'test_complete','run':str(run),'metrics':report}),flush=True)
        return
    optimizer = torch.optim.Adam([{'params':model.core.parameters(),'lr':config['lr']},
                                  {'params':model.bert.parameters(),'lr':config['bert_lr']}],
                                 weight_decay=config.get('weight_decay',0))
    targets = torch.zeros(len(train))
    start_epoch, best, best_epoch, stale = 1, float('inf'), 0, 0
    if args.resume:
        saved = torch.load(run/'latest.pt',map_location=device,weights_only=False)
        if saved['receipt']['source_sha256'] != receipt['source_sha256']:
            raise ValueError('Refusing resume with changed source')
        check_resume_state(saved, train, config)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start_epoch, best, best_epoch, stale = saved['epoch']+1,saved['best'],saved['best_epoch'],saved['stale']
        targets = saved['targets'].cpu()
        random.setstate(saved['rng_python']); np.random.set_state(saved['rng_numpy'])
        torch.set_rng_state(saved['rng_torch'].cpu()); torch.cuda.set_rng_state_all(saved['rng_cuda'])
    write_json(run/'receipt.json',receipt)
    started = time.monotonic()
    for epoch in range(start_epoch,config['epochs']+1):
        model.train()
        order = np.random.default_rng(seed+epoch).permutation(len(train))
        loss_sum, n, route_n = {}, 0, 0
        epoch_start = time.monotonic()
        for start in range(0,len(order),config['batch_size']):
            rows = order[start:start+config['batch_size']]
            batch = train.batch(rows,device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch, targets[rows].to(device), supervised=True)
            loss = optimization_loss(out['losses'],config,epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss epoch={epoch},batch={start}')
            loss.backward()
            parameters = [p for p in model.parameters() if p.grad is not None]
            if not torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all():
                raise FloatingPointError('Nonfinite gradient')
            if config.get('grad_clip_norm') is not None:
                torch.nn.utils.clip_grad_norm_(parameters,config['grad_clip_norm'])
            optimizer.step()
            for k,v in out['losses'].items():
                loss_sum[k] = loss_sum.get(k,0)+float(v.detach())*len(rows)
            n += len(rows)
            if 'disagreement' in out:
                route_n += int(out['disagreement'].sum())
            if start == 0 or start//config['batch_size'] % 100 == 0:
                print(json.dumps({'event':'batch','epoch':epoch,'examples':n,'loss':float(loss.detach())}),flush=True)
        validation,pred = evaluate(model,valid,config['batch_size'],device)
        score = validation['mae']
        improved = score < best-1e-8
        if improved:
            best,best_epoch,stale = score,epoch,0
            torch.save({'model':model.state_dict(),'epoch':epoch,'receipt':receipt,'validation':validation},run/'best.pt')
            export_predictions(run/'validation_predictions.csv',valid,pred)
        else:
            stale += 1
        row = {'epoch':epoch,'train_losses':{k:v/n for k,v in loss_sum.items()},
               'validation':validation,'best_epoch':best_epoch,'seconds':time.monotonic()-epoch_start,
               'disagreement_route_fraction':route_n/n if config['model']=='triagedmsa' else None}
        stop = stale >= config['patience'] or epoch == config['epochs']
        if config['model']=='triagedmsa' and not stop:
            refreshed = refresh_targets(model,train,targets,epoch+1,config['batch_size'],device)
            if isinstance(refreshed,tuple):
                targets,apd_receipt = refreshed
            else:
                targets,apd_receipt = refreshed,{}
            row['apd'] = apd_receipt
        with (run/'history.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps({'event':'epoch',**row}),flush=True)
        saved = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                 'best':best,'best_epoch':best_epoch,'stale':stale,'targets':targets,
                 'train_ids':train.ids,'targets_epoch':epoch+1 if not stop else epoch,
                 'receipt':receipt,'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),
                 'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()}
        torch.save(saved,run/'latest.tmp.pt'); os.replace(run/'latest.tmp.pt',run/'latest.pt')
        receipt.update(last_epoch=epoch,best_epoch=best_epoch,best_validation_mae=best,
                       seconds=time.monotonic()-started)
        write_json(run/'receipt.json',receipt)
        if stop:
            break
    receipt.update(status='training_complete_test_not_yet_evaluated')
    write_json(run/'receipt.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
