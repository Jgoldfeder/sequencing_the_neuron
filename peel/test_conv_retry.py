"""Trained LeNet regression: compare tracking with independent-sampling retry.

Run from code/: python peel/test_conv_retry.py [--device cuda]
Uses the cached trained teacher and latest saved best member. The fixture
replaces L1 with its aligned exact teacher row (diagnostic only), leaving
learned L2 guesses. Solver recovery uses teacher forward queries only.
"""
import argparse
import glob
import sys
import torch

sys.path.insert(0, '.')
from align import cnn_align_to_, cnn_canonicalize_
from nets import ConvNet
from kink_solve import recover_layer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--snapshot', default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    snapshot = args.snapshot or glob.glob('recon/_pop__mergedbest_cnn__1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0__s0.pt')[0]
    ck = torch.load(snapshot, map_location=args.device, weights_only=True)
    teacher = ConvNet(ck['input_shape'], ck['conv_cfgs'], ck['fc_dims'],
                      ck['out_dim'], ck['act']).double().to(args.device)
    teacher.load_state_dict(torch.load(
        'teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',
        map_location=args.device, weights_only=True))
    guess = teacher.clone()
    guess.load_state_dict(ck['state_dict'])
    cnn_canonicalize_(guess)
    cnn_canonicalize_(teacher)
    cnn_align_to_(teacher, guess)
    guess.layers[0].load_state_dict(teacher.layers[0].state_dict())
    channels = [1, 9, 10, 11, 14]
    for retry in (False, True):
        W, b, mask, nq = recover_layer(
            teacher, guess, 1, args.device, only_channels=channels,
            gen=torch.Generator(device=args.device).manual_seed(1235),
            sampling='track', cnn_retry=retry)
        v = torch.cat((W.flatten(1), b[:, None]), 1)
        target = torch.cat((teacher.layers[1].weight.flatten(1),
                            teacher.layers[1].bias[:, None]), 1)
        target = target / target.norm(dim=1, keepdim=True)
        error = (v[channels] - target[channels]).abs().max().item()
        print(f'retry={retry}: {int(mask[channels].sum())}/5 solved; '
              f'max parameter error={error:.3e}; queries={nq}', flush=True)
        if retry:
            assert bool(mask[channels].all()), 'Independent retry did not recover all target rows'
            assert error < 1e-10, f'Incorrect accepted filter: {error}'


if __name__ == '__main__':
    main()
