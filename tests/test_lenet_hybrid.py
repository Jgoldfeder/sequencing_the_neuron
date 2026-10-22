"""CPU regression checks for the unified driver's gauge and resume contracts."""
from pathlib import Path
from types import SimpleNamespace
import copy
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import lenet_hybrid as H
H.load_dependencies()
torch=H.torch

class HybridStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(913)
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.model=H.new_net('cpu')
        with torch.no_grad():
            for layer in self.model.layers:layer.bias.normal_(0,.03)
        self.args=SimpleNamespace(device='cpu',seed=19,output=self.root,grid=1001)

    def tearDown(self):self.temp.cleanup()

    def record(self,model):
        masks={0:torch.ones(6,dtype=torch.bool),1:torch.ones(16,dtype=torch.bool),
               2:torch.arange(120)<103,3:torch.zeros(84,dtype=torch.bool)}
        state={k:v.clone() for k,v in model.state_dict().items()}
        path=self.root/'input.pt'
        torch.save(dict(input_shape=H.SHAPE,conv_cfgs=H.CONVS,fc_dims=(84,),out_dim=10,act='leaky_relu',
                        state_dict=state,partial_state=state,partial_mask=masks,frozen={}),path)
        return path

    def test_import_preserves_function_through_large_positive_gauges(self):
        import cnn_balance as B
        with torch.no_grad():B.scale_channels_(self.model,2,torch.logspace(-3,2,120,dtype=torch.double))
        oracle=H.OutputOracle(copy.deepcopy(self.model));solver=H.Recovery(oracle,self.args,self.record(self.model))
        x=torch.randn(8,784,dtype=torch.double)
        with torch.no_grad():error=float((solver.net(x)-self.model(x)).abs().max())
        self.assertLess(error,1e-11)
        self.assertEqual(int(solver.masks[2].sum()),103)

    def test_resume_restores_pending_unsigned_rng_and_queries(self):
        oracle=H.OutputOracle(copy.deepcopy(self.model));solver=H.Recovery(oracle,self.args,self.record(self.model))
        solver.pending[2]=[dict(row=torch.randn(401,dtype=torch.double),anchor=torch.randn(784,dtype=torch.double),attempts=2)]
        solver.unsigned[2]=[103];solver.anchors[2][103]=torch.randn(784,dtype=torch.double)
        oracle.queries=123;solver.rounds[2]=3;solver.save('test')
        expected=torch.randn(10,generator=solver.gen,dtype=torch.double)
        restored=H.Recovery(H.OutputOracle(copy.deepcopy(self.model)),self.args,self.root/'state.pt')
        self.assertEqual(restored.bb.queries,123);self.assertEqual(restored.rounds[2],3)
        self.assertEqual(restored.unsigned[2],[103]);self.assertEqual(restored.pending[2][0]['attempts'],2)
        self.assertTrue(torch.equal(expected,torch.randn(10,generator=restored.gen,dtype=torch.double)))

    def test_partial_prefix_is_rejected(self):
        path=self.record(self.model);r=torch.load(path,weights_only=False);r['partial_mask'][1][0]=False;torch.save(r,path)
        with self.assertRaisesRegex(ValueError,'fully recorded C1/C2'):
            H.Recovery(H.OutputOracle(self.model),self.args,path)

    def test_fresh_committee_uses_first_phase_recipe_without_a_record(self):
        args=SimpleNamespace(teacher=self.root/'teacher.pt',device='cpu',
            committee_epochs=0,committee_q=128,committee_outer=1,
            committee_retries=5,committee_resume=None,output=self.root,seed=0)
        with patch.object(H,'load_teacher',return_value=self.model), \
             patch('method.reconstruct_cnn') as reconstruct, \
             patch.dict(H.os.environ,{}):
            H.committee_worker(args)
        cfg=reconstruct.call_args.args[4]
        self.assertEqual(cfg.p,8)
        self.assertEqual(cfg.window,60)
        self.assertTrue(cfg.peel_refresh)
        self.assertEqual(cfg.retry,5)
        self.assertIsNone(reconstruct.call_args.kwargs['resume_state'])
        self.assertIsNone(reconstruct.call_args.kwargs['resume_record'])

    def test_adaptive_locator_restores_global_function_on_error(self):
        before=H.K.locate_light
        with self.assertRaises(RuntimeError):
            with H.adaptive_light():
                self.assertIsNot(H.K.locate_light,before)
                raise RuntimeError('stop')
        self.assertIs(H.K.locate_light,before)

if __name__=='__main__':unittest.main()
