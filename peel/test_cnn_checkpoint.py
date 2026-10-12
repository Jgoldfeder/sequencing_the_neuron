"""Check learned L2 guesses above the former hidden 4-degree cap.
Uses exact aligned L1 for diagnosis; teacher parameters only build/score fixture.
"""
import copy
import glob
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nets import ConvNet
from align import cnn_canonicalize_,cnn_align_to_
from kink_solve import recover_layer


def main():
    torch.set_num_threads(1);device='cuda:1'
    ck=torch.load(glob.glob('recon/_pop__mergedbest_cnn__1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0__s0.pt')[0],weights_only=True,map_location=device)
    G=ConvNet(ck['input_shape'],ck['conv_cfgs'],ck['fc_dims'],ck['out_dim'],ck['act']).double().to(device)
    G.load_state_dict(ck['state_dict']);cnn_canonicalize_(G)
    T=copy.deepcopy(G)
    T.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',weights_only=True,map_location=device))
    cnn_canonicalize_(T);cnn_align_to_(T,G)
    G.layers[0].load_state_dict(T.layers[0].state_dict())
    def rows(model):
        v=torch.cat((model.layers[1].weight.flatten(1),model.layers[1].bias[:,None]),1)
        return v/v.norm(dim=1,keepdim=True)
    gt=rows(T);gg=rows(G);angles=torch.rad2deg(torch.acos((gt*gg).sum(1).clamp(-1,1)))
    channels=((angles>4)&(angles<12)).nonzero().flatten().tolist()
    print('Checkpoint iteration',ck['iter'],'channels above hidden cap',channels,'angles',angles.tolist(),flush=True)
    if not channels:
        raise RuntimeError('Latest mutable checkpoint has no guesses in (4,12) degrees')
    W,b,mask,nq=recover_layer(T,G,1,device,only_channels=channels,gen=torch.Generator(device=device).manual_seed(1235),sampling='track')
    v=torch.cat((W.flatten(1),b[:,None]),1);v=v/v.norm(dim=1,keepdim=True)
    errors=(v-gt).abs().amax(1)
    print('Recovered',int(mask[channels].sum()),'/',len(channels),'queries',nq,'errors',errors[channels].tolist(),flush=True)
    assert bool(mask[channels].all())
    assert errors[channels].max()<1e-9


if __name__=='__main__':main()
