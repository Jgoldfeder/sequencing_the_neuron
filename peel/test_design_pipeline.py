"""Bounded integration test: real reconstruction/refine/freeze, seeded good guesses.
Run from code/: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python peel/test_design_pipeline.py
"""
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nets import MLP
from align import scale_normalize_
import method
import kink_solve


def main():
    torch.set_num_threads(1); torch.manual_seed(19)
    dims=[6,6,6,3]
    teacher=MLP(dims)
    with torch.no_grad():
        for layer in teacher.layers: layer.bias.normal_(0,.1)
    scale_normalize_(teacher)
    # Only initialization is controlled: the actual teacher-query solver,
    # pipeline gate, scale matching, freeze and export code all run unchanged.
    def seeded_population(requested_dims, act='leaky_relu'):
        assert list(requested_dims)==dims
        model=teacher.clone()
        with torch.no_grad():
            for layer in model.layers:
                layer.weight.add_(1e-4*torch.randn_like(layer.weight))
                layer.bias.add_(1e-4*torch.randn_like(layer.bias))
        return model
    for consensus, incomplete in [(False, False), (True, False), (False, True)]:
        calls=[]
        recover=kink_solve.recover_layer
        def record(*args, **kwargs):
            assert kwargs['sampling']=='design'
            assert args[1].layers[0].weight.dtype==torch.float64
            result=recover(*args,**kwargs)
            if incomplete and args[2] == 0:
                result[2][-1] = False
            calls.append((args[2],result[2].clone(),result[3]))
            return result
        cfg=method.Cfg(p=2,q=128,outer=1,epochs=0,qg_steps=0,log_every=1,
                       cheat=True,cheat_solo=True,fast_peel=True,freeze_reinit=True,
                       cheat_peel_mean=0. if consensus else .01,
                       cheat_peel_max=0. if consensus else .1,peel_restart=consensus,
                       design_refine=True,peel_seal=True,pop_save_every=1)
        with tempfile.TemporaryDirectory() as td:
            cfg.pop_save_path=str(Path(td)/'population.pt')
            with patch.object(method,'MLP',side_effect=seeded_population), \
                 patch.object(kink_solve,'recover_layer',side_effect=record):
                best,log,final=method.reconstruct(teacher,dims,cfg,'cpu',torch.randn(256,6),
                                                  seed=2,save_recon=str(Path(td)/'recon.pt'))
            assert cfg.f64
            assert [c[0] for c in calls]==([0] if incomplete else [0,1]),calls
            assert incomplete or all(bool(c[1].all()) for c in calls)
            assert final['peel_refinement_queries']==sum(c[2] for c in calls)>0
            assert log[-1]['peel_refinement_queries']==final['peel_refinement_queries']
            assert next(best.parameters()).dtype==torch.float64
            ck=torch.load(Path(td)/'recon.pt',weights_only=False)
            pop=torch.load(Path(td)/'population.pt',weights_only=False)
            for l in [0,1]:
                assert ck['frozen'][l][0].dtype==torch.float64
                assert incomplete or bool(ck['frozen'][l][2].all())
                assert torch.equal(best.layers[l].weight,ck['refined_state'][f'layers.{l}.weight'])
                assert torch.equal(pop['frozen'][l][0],ck['frozen'][l][0])
            if not incomplete:
                assert max(final['final_max_eps_per_matrix'][:4])<1e-8,final
    print('PASS: actual peel refines both layers with design sampling, scales/freezes rows, '
          'counts refinement queries, exports fp64 weights, exercises consensus/restart/peel-direct, '
          'and defers deeper recovery after an incomplete prefix.')

if __name__=='__main__':
    main()
