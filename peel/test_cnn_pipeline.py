"""Real CNN refiner through production partial peel, warm refresh and cold retry."""
import copy
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import method
import kink_solve
from nets import ConvNet
from align import _push_to_next, cnn_param_errors


def main():
    torch.set_num_threads(1); torch.manual_seed(17)
    shape=(1,6,6); conv=[(1,2,3,1,1,2),(2,3,2,1,0,0)]; fc=(4,)
    teacher=ConvNet(shape,conv,fc,2,'leaky_relu')
    with torch.no_grad():
        for layer in teacher.layers: layer.bias.normal_(0,.05)
    for mode in ('partial','warm','retry','quorum','incomplete','resume'):
        made=[0]; calls=[]; original=kink_solve.recover_layer
        def factory(*args,**kwargs):
            model=copy.deepcopy(teacher); model.requires_grad_(True); made[0]+=1
            with torch.no_grad():
                for l,layer in enumerate(model.layers[:-1]):
                    layer.weight.add_(1e-5*torch.randn_like(layer.weight))
                    layer.bias.add_(1e-5*torch.randn_like(layer.bias))
                    factor=torch.exp(torch.randn(len(layer.bias)))
                    layer.weight.mul_(factor.reshape(-1,*([1]*(layer.weight.ndim-1))))
                    layer.bias.mul_(factor)
                    _push_to_next(model.layers[l+1],1/factor,layer.weight.ndim==4)
                if mode=='quorum' and made[0]==3:
                    model.layers[0].weight.neg_();model.layers[0].bias.neg_()
            return model
        def recover(*args,**kwargs):
            assert args[1].layers[0].weight.dtype==torch.float64
            result=original(*args,**kwargs)
            if mode=='retry' and args[2]==1 and sum(c[0]==1 for c in calls)<2:
                result[2][-1]=False
            if mode=='incomplete' and args[2]==0:
                result[2][-1]=False
            calls.append((args[2],result[2].clone(),result[3]))
            return result
        cfg=method.Cfg(p=3,q=64,outer=2,epochs=0,qg_steps=0,log_every=1,
                       act='leaky_relu',fast_peel_partial=mode!='incomplete',loc_refine=True,
                       cheat=mode=='incomplete',cheat_peel_mean=.01,cheat_peel_max=.1,
                       cluster_eps=.005,cluster_quorum=.67,freeze_reinit=mode in ('warm','incomplete'),
                       peel_refresh=mode=='warm',freeze_thresh=1.,freeze_precision=0.,
                       retry=1 if mode=='retry' else 0,pop_save_every=1)
        with tempfile.TemporaryDirectory() as td:
            cfg.pop_save_path=str(Path(td)/'pop.pt')
            with patch.object(method,'ConvNet',side_effect=factory),patch.object(kink_solve,'recover_layer',side_effect=recover):
                best,log,final=method.reconstruct_cnn(teacher,shape,conv,2,cfg,'cpu',fc_dims=fc,
                    resume_state=teacher.clone().double().state_dict() if mode=='resume' else None)
            if mode=='quorum':
                assert any(l==0 and bool(mask.all()) for l,mask,_ in calls), \
                    'A disagreeing member must not block the configured 2-of-3 quorum'
                assert max(cnn_param_errors(best,teacher.clone().double())['max_eps_per_matrix'][:2])<1e-9
                print('PIPELINE PASS quorum: 2 agreeing members suffice despite a third dissenter',flush=True)
                continue
            assert {c[0] for c in calls}==({0,1} if mode=='retry' else {0} if mode=='incomplete' else {0,1,2}),calls
            assert next(best.parameters()).dtype==torch.float64
            errors=cnn_param_errors(best,teacher.clone().double())['max_eps_per_matrix']
            if mode=='incomplete':
                masks=[torch.tensor([True,False]),torch.zeros(3,dtype=torch.bool),
                       torch.zeros(4,dtype=torch.bool),torch.zeros(2,dtype=torch.bool)]
                solved_error=method._cnn_consensus_eps(best,masks,teacher.clone().double())[0]['max']
                assert solved_error<1e-9
            elif mode=='retry':
                # Cold reinit deliberately discards unsolved guesses; test that
                # already solved channels persist, not that fresh rows are solved.
                masks=[torch.ones(2,dtype=torch.bool),torch.tensor([True,True,False]),
                       torch.zeros(4,dtype=torch.bool),torch.zeros(2,dtype=torch.bool)]
                stats=method._cnn_consensus_eps(best,masks,teacher.clone().double())
                solved_error=max(stats[l]['max'] for l in (0,1))
                assert solved_error<1e-9,stats
            else:
                solved_error=max(errors[:-2])
                assert solved_error<1e-9,(mode,errors)
            assert final['peel_refinement_queries']==sum(c[2] for c in calls)>0
            assert log[-1]['peel_refinement_queries']==final['peel_refinement_queries']
            ck=torch.load(cfg.pop_save_path,weights_only=True)
            assert ck['refined_state']['layers.0.weight'].dtype==torch.float64
            print('PIPELINE PASS',mode,'solved-row max error',solved_error,flush=True)
    print('PASS: actual refinement, scale-diverse committee, configured partial quorum, warm/cold restarts, fp64 export and query accounting')


if __name__=='__main__':main()
