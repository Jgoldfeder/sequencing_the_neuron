"""CPU integration checks; run from code/: python peel/test_informative_smoke.py."""
import sys
import torch
sys.path.insert(0, '.')
from nets import MLP
from align import scale_normalize_
import kink_solve as K


def main():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    teacher = MLP([6, 8, 7, 3]).double()
    with torch.no_grad():
        for layer in teacher.layers:
            layer.bias.normal_(0, .1)
    scale_normalize_(teacher)
    guess = teacher.clone()
    with torch.no_grad():
        guess.layers[1].weight.add_(.002*torch.randn_like(guess.layers[1].weight))
        guess.layers[1].bias.add_(.002*torch.randn_like(guess.layers[1].bias))
        guess.layers[2].weight.zero_()
        guess.layers[2].bias.zero_()

    class ForwardOnly:
        def __init__(self):
            self.rows = 0

        def __call__(self, x):
            self.rows += len(x)
            return teacher(x)

    oracle = ForwardOnly()
    w, b, mask, nq = K.recover_layer(
        oracle, guess, 1, 'cpu', sampling='design', need=80, pool=128,
        batch=48, max_rounds=12, scales=(1., 4.), verbose=False)
    assert bool(mask.all()), mask
    assert nq == oracle.rows
    assert torch.allclose(w.square().sum(1)+b.square(), torch.ones_like(b),
                          rtol=0, atol=1e-14)
    norm = w.norm(dim=1)
    error = max((w/norm[:, None]-teacher.layers[1].weight).abs().max().item(),
                (b/norm-teacher.layers[1].bias).abs().max().item())
    assert error < 1e-10, error
    for options in [dict(max_rounds=0), dict(only_channels=[])]:
        before = oracle.rows
        w, b, mask, nq = K.recover_layer(
            oracle, guess, 1, 'cpu', sampling='design', verbose=False, **options)
        assert not bool(mask.any()) and nq == 0 and oracle.rows == before
        assert torch.equal(w, guess.layers[1].weight)
        assert torch.equal(b, guess.layers[1].bias)
    # Oversampling is a target, not an unconditional rejection threshold.
    # Enough independent points still must pass the unchanged held-out gates.
    from peel.informative_kinks import recover_layer
    diag = {}
    wf, bf, mf, _ = recover_layer(
        teacher, guess, 1, 'cpu', only_channels=[0], need=1000,
        pool=192, batch=96, max_rounds=2, diagnostics=diag, verbose=False)
    assert mf[0] and 56 <= diag[0]['points'] < 1000, diag
    nf = wf[0].norm()
    assert (wf[0]/nf-teacher.layers[1].weight[0]).abs().max() < 1e-10
    assert (bf[0]/nf-teacher.layers[1].bias[0]).abs() < 1e-10
    float_guess = guess.float()
    w, b, mask, nq = K.recover_layer(
        teacher.float(), float_guess, 1, 'cpu', sampling='design',
        max_rounds=0, verbose=False)
    assert w.dtype == b.dtype == torch.float32
    assert torch.equal(w, float_guess.layers[1].weight)
    print(f'PASS: forward-only rectangular-prefix recovery (error {error:.2e}), '
          'query accounting, return gauge, and exhausted/empty requests')


if __name__ == '__main__':
    main()
