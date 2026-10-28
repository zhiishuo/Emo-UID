import unittest
from unittest.mock import patch
from pathlib import Path
import json
import numpy as np
import torch
from transformers import BertConfig, BertModel
from train_full import (Baseline, Split, metrics, evaluate, optimization_loss,
                        check_split_ids, check_resume_state, refresh_targets, main)


class ContractTests(unittest.TestCase):
    def raw(self):
        return {'id':['a','b'], 'text_bert':np.ones((2,3,3), dtype=np.int64),
                'audio':np.zeros((2,3,5)), 'vision':np.zeros((2,3,20)),
                'regression_labels':np.array([1.,-1.])}

    def test_split_rejects_malformed_inputs(self):
        for key, value in [
            ('text_bert', np.ones((2,2,3), dtype=int)),
            ('text_bert', np.ones((2,3,3), dtype=float)),
            ('text_bert', np.full((2,3,3), 2, dtype=int)),
            ('audio', np.zeros((2,3))),
            ('vision', np.zeros((2,3,0))),
            ('regression_labels', np.array([1.])),
            ('regression_labels', np.array([1.,np.nan])),
        ]:
            with self.subTest(key=key, shape=value.shape):
                raw = self.raw()
                raw[key] = value
                with self.assertRaises(ValueError): Split(raw, 'train')

    def test_split_overlap_including_deferred_test(self):
        train = Split(self.raw(), 'train')
        raw = self.raw()
        raw['id'] = ['c','d']
        valid = Split(raw, 'valid')
        check_split_ids(train, valid)
        with self.assertRaises(ValueError):
            check_split_ids(train, valid, Split(self.raw(), 'test'))

    def test_apd_refresh_rejects_nontrain_split(self):
        with self.assertRaises(ValueError):
            refresh_targets(None, Split(self.raw(), 'valid'), None, 2, 2, 'cpu')

    def test_resume_requires_matching_ids_and_next_epoch_apd_cache(self):
        train = Split(self.raw(), 'train')
        saved = {'train_ids':train.ids, 'targets':torch.zeros(2),
                 'targets_epoch':2, 'epoch':1}
        config = {'model':'triagedmsa'}
        check_resume_state(saved, train, config)
        with self.assertRaises(ValueError):
            check_resume_state(dict(saved, train_ids=list(reversed(train.ids))), train, config)
        with self.assertRaises(ValueError):
            check_resume_state(dict(saved, targets_epoch=1), train, config)

    def test_offline_bert_wrapper_backward_and_label_free_evaluation(self):
        torch.set_num_threads(1)
        for name in ('misr','triagedmsa'):
            with self.subTest(model=name):
                bert = BertModel(BertConfig(vocab_size=16, hidden_size=8,
                    num_hidden_layers=1, num_attention_heads=2, intermediate_size=16))
                kwargs = {'language_dim':8, 'heads':2, 'dropout':0.0,
                          ('hidden_dim' if name == 'misr' else 'hidden'):8}
                config = {'model':name, 'bert':'unused-offline-test', 'model_kwargs':kwargs}
                split = Split(self.raw(), 'train')
                with patch('train_full.AutoModel.from_pretrained', return_value=bert) as loader:
                    model = Baseline(config, split)
                    loader.assert_called_once_with('unused-offline-test', local_files_only=True)
                out = model(split.batch([0,1], 'cpu'), torch.zeros(2), supervised=True)
                out['losses']['total'].backward()
                grad = bert.embeddings.word_embeddings.weight.grad
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum().item(), 0)
                _, pred = evaluate(model, split, 2, 'cpu')
                split.labels[:] = 100
                _, changed = evaluate(model, split, 2, 'cpu')
                np.testing.assert_array_equal(pred, changed)

    def test_invalid_model_and_bert_width_fail_early(self):
        split = Split(self.raw(), 'train')
        with patch('train_full.AutoModel.from_pretrained') as loader:
            with self.assertRaises(ValueError):
                Baseline({'model':'typo', 'bert':'unused'}, split)
            loader.assert_not_called()
            loader.return_value.config.hidden_size = 1024
            with self.assertRaises(ValueError):
                Baseline({'model':'misr', 'bert':'unused'}, split)

    def test_example_configs_are_explicit_and_complete(self):
        paths = sorted((Path(__file__).parent/'configs').glob('*.example.json'))
        self.assertEqual(len(paths), 4)
        required = {'model','dataset','data','data_sha256','bert','seed','epochs',
                    'batch_size','lr','bert_lr','patience','model_kwargs'}
        for path in paths:
            config = json.loads(path.read_text())
            self.assertTrue(required <= config.keys())
            self.assertIs(config['_example_only'], True)
            self.assertNotIn('consistency_schedule', config)
            self.assertEqual(config['model_kwargs']['language_dim'], 768)

    def test_example_guard_precedes_output_creation_and_cuda(self):
        config = Path(__file__).parent/'configs/misr_mosi.example.json'
        with patch('sys.argv', ['train_full.py', '--config', str(config),
                               '--run-dir', 'runs/zzs-must-not-be-created']), \
             patch('train_full.Path.mkdir') as mkdir, \
             patch('train_full.torch.cuda.set_per_process_memory_fraction') as cuda:
            with self.assertRaisesRegex(ValueError, 'Example only'):
                main()
            mkdir.assert_not_called()
            cuda.assert_not_called()

    def test_schedule_is_explicit_and_restores_original_objective(self):
        for kind,duration in [('delayed',3),('linear',10)]:
            config={'model':'triagedmsa','consistency_schedule':{'kind':kind,'epochs':duration}}
            task=torch.tensor(2.,requires_grad=True)
            consistency=torch.tensor(3.,requires_grad=True)
            losses={'total':task+consistency,'consistency':consistency}
            early=optimization_loss(losses,config,1)
            self.assertEqual(early.item(),2.)
            early.backward(retain_graph=True)
            self.assertEqual(consistency.grad.item(),0.)
            late=optimization_loss(losses,config,duration+1)
            self.assertEqual(late.item(),5.)
            self.assertEqual(losses['total'].item(),5.)
        original={'total':torch.tensor(5.)}
        self.assertIs(optimization_loss(original,{},1),original['total'])

    def test_metrics_use_declared_neutral_conventions(self):
        m = metrics([0, -1, 1], [0, -1, 1])
        self.assertEqual(m['mae'],0)
        self.assertEqual(m['f1_nonzero'],1)
        self.assertEqual(m['f1_has0'],1)
        self.assertEqual(metrics([0,0], [-1,1])['acc2_nonzero'],.5)
        self.assertIsNone(metrics([1], [0])['f1_nonzero'])

    def test_raw_mae_is_not_clipped(self):
        self.assertEqual(metrics([5], [3])['mae'],2)
        self.assertEqual(metrics([5], [3])['acc7'],1)

    def test_constant_float32_prediction_has_finite_corr_convention(self):
        pred=np.full(229,0.12712538,dtype=np.float32)
        target=np.linspace(-3,3,229,dtype=np.float32)
        self.assertEqual(metrics(pred,target)['corr'],0.0)

    def test_split_preserves_identity(self):
        raw = {'id':['x','y'],'text_bert':np.zeros((2,3,4),dtype=int),
               'audio':np.zeros((2,4,5)), 'vision':np.zeros((2,4,20)),
               'regression_labels':np.array([1.,-1.])}
        raw['text_bert'][:,1,:2] = 1
        split = Split(raw,'train')
        self.assertEqual(split.ids,['x','y'])
        self.assertEqual(split.batch([1],torch.device('cpu'))[-1].item(),-1)
        raw['id']=['x','x']
        with self.assertRaises(ValueError): Split(raw,'train')

    def test_evaluation_does_not_pass_labels_into_model(self):
        class Fake(torch.nn.Module):
            def forward(self,batch,targets=None,supervised=False):
                assert not supervised and targets is None and not self.training
                return {'prediction':torch.zeros(len(batch[-1]))}
        raw = {'id':['a','b'],'text_bert':np.ones((2,3,2),dtype=int),
               'audio':np.zeros((2,2,5)), 'vision':np.zeros((2,2,20)),
               'regression_labels':np.array([1.,-1.])}
        m,p = evaluate(Fake(),Split(raw,'valid'),1,torch.device('cpu'))
        self.assertEqual(m['mae'],1)


if __name__ == '__main__': unittest.main()
