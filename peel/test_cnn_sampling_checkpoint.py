"""Reproduce sampling on an actual saved fp64 CNN record, without teacher prefixes.
Example: python peel/test_cnn_sampling_checkpoint.py --snapshot /tmp/cnn_sampling_snapshot.pt
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nets import ConvNet
from align import cnn_canonicalize_,cnn_align_to_
from method import _cnn_transfer_rows_
from kink_solve import recover_layer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',required=True)
    p.add_argument('--channels',default='0,15')
    p.add_argument('--refresh-guesses',action='store_true')
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--output',default='/tmp/cnn_sampling_result.json')
    args=p.parse_args();torch.set_num_threads(1)
    ck=torch.load(args.snapshot,weights_only=True,map_location=args.device)
    G=ConvNet(ck['input_shape'],ck['conv_cfgs'],ck['fc_dims'],ck['out_dim'],ck['act']).double().to(args.device)
    G.load_state_dict(ck['partial_state'])
    channels=[int(c) for c in args.channels.split(',')]
    if args.refresh_guesses:
        best=copy.deepcopy(G);best.load_state_dict(ck['state_dict']);cnn_canonicalize_(best)
        _cnn_transfer_rows_(G,best,1,torch.tensor(channels,device=args.device))
    T=copy.deepcopy(G)
    T.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',weights_only=True,map_location=args.device))
    before={k:v.clone() for k,v in G.state_dict().items()}
    norms=torch.cat((G.layers[1].weight.flatten(1),G.layers[1].bias[:,None]),1).norm(dim=1)
    start=time.time()
    W,b,mask,nq=recover_layer(T,G,1,args.device,only_channels=channels,sampling='track',gen=torch.Generator(device=args.device).manual_seed(1235))
    assert all(torch.equal(before[k],v) for k,v in G.state_dict().items())
    # Teacher parameters only enter scoring, after recovery is finished.
    reference=G.clone();cnn_canonicalize_(reference)
    target=T.clone();cnn_canonicalize_(target);cnn_align_to_(target,reference)
    result=G.clone()
    with torch.no_grad():result.layers[1].weight.copy_(W);result.layers[1].bias.copy_(b)
    cnn_canonicalize_(result)
    # Unsolved row labels are interchangeable: score each accepted filter against
    # the teacher set, and require distinct matches disjoint from solved rows.
    actual=torch.cat((result.layers[1].weight.flatten(1),result.layers[1].bias[:,None]),1)
    truth=torch.cat((target.layers[1].weight.flatten(1),target.layers[1].bias[:,None]),1)
    distances=(actual[:,None,:]-truth[None,:,:]).abs().amax(2)
    error,matches=distances.min(1)
    solved=ck['partial_mask'][1]
    previous=torch.cat((reference.layers[1].weight.flatten(1),reference.layers[1].bias[:,None]),1)[solved]
    previous_matches=(previous[:,None,:]-truth[None,:,:]).abs().amax(2).argmin(1).tolist()
    accepted_channels=[c for c in channels if bool(mask[c])]
    new_matches=matches[accepted_channels].tolist()
    assert len(set(new_matches))==len(new_matches)
    assert not set(new_matches).intersection(previous_matches)
    report=dict(snapshot_iter=ck['iter'],channels=channels,norms=norms[channels].tolist(),
                accepted=mask[channels].tolist(),errors=error[channels].tolist(),
                teacher_matches=matches[channels].tolist(),queries=nq,
                seconds=time.time()-start,refresh_guesses=args.refresh_guesses)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
    assert bool(mask[channels].all()),report
    assert error[channels].max()<1e-9,report


if __name__=='__main__':main()
