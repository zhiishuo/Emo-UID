import unittest
import numpy as np
import torch
from train_full import Split, metrics, evaluate, optimization_loss


class ContractTests(unittest.TestCase):
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
