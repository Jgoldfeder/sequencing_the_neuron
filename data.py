"""Teacher (blackbox) creation: train an MLP on MNIST or CIFAR-100, cached on
disk. The dataset is inferred from the architecture: 3072-in / 100-out => CIFAR-
100, otherwise MNIST (784-in / 10-out).

Data uses standard per-dataset normalization (zero mean, unit std per channel).
"""
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, CIFAR100

from nets import MLP

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
TEACHER_DIR = os.path.join(os.path.dirname(__file__), "teachers")

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
CIFAR_MEAN = torch.tensor([0.5071, 0.4865, 0.4409])
CIFAR_STD = torch.tensor([0.2673, 0.2564, 0.2762])


def load_mnist(device):
    def get(train):
        ds = MNIST(root=DATA_ROOT, train=train, download=True)
        x = ds.data.float().div_(255.0)
        x = ((x - MNIST_MEAN) / MNIST_STD).view(-1, 784)
        y = ds.targets
        return x.to(device), y.to(device)

    return get(True), get(False)


def load_cifar100(device):
    def get(train):
        ds = CIFAR100(root=DATA_ROOT, train=train, download=True)
        x = torch.tensor(ds.data).float().div_(255.0)   # (N, 32, 32, 3)
        x = ((x - CIFAR_MEAN) / CIFAR_STD).reshape(-1, 3072)
        y = torch.tensor(ds.targets)
        return x.to(device), y.to(device)

    return get(True), get(False)


def load_tinyimagenet(device):
    """TinyImageNet-200: 64x64x3 (=12288) inputs, 200 classes. Decoded from the
    ImageFolder once and cached as tensors (100k train JPEGs are slow to decode)."""
    cache = os.path.join(DATA_ROOT, "tinyimagenet_64.pt")
    if os.path.exists(cache):
        d = torch.load(cache, map_location="cpu")
        return ((d["xtr"].to(device), d["ytr"].to(device)),
                (d["xte"].to(device), d["yte"].to(device)))
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    root = os.path.join(DATA_ROOT, "tiny-imagenet-200")
    tf = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def get(split):
        ds = datasets.ImageFolder(os.path.join(root, split), transform=tf)
        loader = DataLoader(ds, batch_size=512, num_workers=8)
        xs, ys = [], []
        for xb, yb in loader:                 # xb: (B,3,64,64) normalized
            xs.append(xb.reshape(xb.size(0), -1))   # standard-normalized, 12288
            ys.append(yb)
        return torch.cat(xs), torch.cat(ys)

    xtr, ytr = get("train")
    xte, yte = get("val")
    torch.save({"xtr": xtr, "ytr": ytr, "xte": xte, "yte": yte}, cache)
    return ((xtr.to(device), ytr.to(device)), (xte.to(device), yte.to(device)))


def load_cifar100_split(device, sub=10, seed=0, iters=25):
    """CIFAR-100 relabeled into 100*sub classes: each class's train images are
    k-means-clustered (k=sub, raw pixels, seeded Lloyd) and labeled
    class*sub + cluster; test images take their class's nearest centroid. So a
    1000-way head has REAL, learnable signal on every logit -- unlike padding
    the label space, which trains the extra rows into near-constant
    'never the answer' functions. Deterministic; centroids from train only."""
    (xtr, ytr), (xte, yte) = load_cifar100(device)
    gen = torch.Generator().manual_seed(seed)
    ytr2 = torch.empty_like(ytr)
    yte2 = torch.empty_like(yte)
    for c in range(int(ytr.max()) + 1):
        m = (ytr == c).nonzero(as_tuple=True)[0]
        Xc = xtr[m]
        ctr = Xc[torch.randperm(len(Xc), generator=gen)[:sub]].clone()
        for _ in range(iters):
            d = torch.cdist(Xc, ctr)
            a = d.argmin(1)
            for j in range(sub):
                sel = Xc[a == j]
                if len(sel):
                    ctr[j] = sel.mean(0)
                else:                          # empty cluster -> farthest point
                    ctr[j] = Xc[d.min(1).values.argmax()]
        ytr2[m] = c * sub + a.to(ytr2.dtype)
        mt = (yte == c).nonzero(as_tuple=True)[0]
        if len(mt):
            yte2[mt] = c * sub + torch.cdist(xte[mt], ctr).argmin(1).to(yte2.dtype)
    return (xtr, ytr2), (xte, yte2)


def load_data(dims, device):
    """Pick the dataset from the architecture: 12288-in/200-out => TinyImageNet,
    3072-in/1000-out => CIFAR-100 split into 1000 k-means subclasses,
    3072-in/100-out => CIFAR-100, else MNIST."""
    if dims[0] == 12288 or dims[-1] == 200:
        return load_tinyimagenet(device)
    if dims[0] in (3072, 150528) and dims[-1] == 1000:
        # 150528 = 3x224x224 (AlexNet-exact geometry): data stays 32x32 here;
        # make_teacher_cnn upsamples per batch (a materialized 224 CIFAR
        # tensor would be ~36 GB).
        return load_cifar100_split(device)
    if dims[0] in (3072, 150528) or dims[-1] == 100:
        return load_cifar100(device)
    return load_mnist(device)


def teacher_path(dims, epochs, seed, act="leaky_relu", l1=0.0, l1_prox=False, l2=0.0):
    os.makedirs(TEACHER_DIR, exist_ok=True)
    tag = "x".join(map(str, dims))
    suffix = "" if act == "leaky_relu" else f"_{act}"
    if l1 > 0:
        suffix += f"_l1{'p' if l1_prox else ''}wb{l1:g}"   # weights AND biases (weights-only _l1 caches are stale)
    if l2 > 0:
        suffix += f"_l2wb{l2:g}"                       # weights AND biases (a weights-only _l2 cache is stale)
    return os.path.join(TEACHER_DIR,
                        f"teacher_{tag}_e{epochs}_s{seed}{suffix}.pt")


def make_teacher(dims, epochs=25, seed=0, lr=1e-3, batch=256, device="cpu",
                 verbose=True, act="leaky_relu", l1=0.0, l1_prox=False, l2=0.0):
    """Train (or load from cache) the blackbox network. l1 > 0 adds
    l1 * sum_l (sum|W_l| + sum|b_l|) (weights AND biases) to the training loss, i.e.
    a per-weight pull of l1 toward zero. Adam then chatters around zero instead of
    landing on it, so l1_prox=True instead applies the L1 as a proximal step after
    every Adam step: w <- sign(w) max(|w| - lr*l1, 0) (exact zeros). An Adam step is
    at most ~lr, so l1 >~ 1 keeps noise-driven weights pinned at exactly 0.
    l2 > 0 adds l2 * sum_l (sum W_l^2 + sum b_l^2) to the training loss. Biases are included in both penalties on
    purpose: a weights-only penalty shrinks |w| but not |b|, pushing every kink (at distance |b|/|w|) out of the query range
    (3072x256x100, l2 3e-3 weights-only: 52/256 units one-sided on N(0, 0.5) queries vs 0 unpenalized)."""
    path = teacher_path(dims, epochs, seed, act, l1, l1_prox, l2)
    if os.path.exists(path):
        net = MLP(dims, act=act)
        net.load_state_dict(torch.load(path, map_location=device))
        return net.to(device)

    torch.manual_seed(seed)
    (xtr, ytr), (xte, yte) = load_data(dims, device)
    net = MLP(dims, act=act).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=batch,
                        shuffle=True)
    for ep in range(epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            if l1 > 0 and not l1_prox:
                loss = loss + l1 * sum(L.weight.abs().sum() + L.bias.abs().sum() for L in net.layers)
            if l2 > 0:
                loss = loss + l2 * sum(L.weight.square().sum() + L.bias.square().sum() for L in net.layers)
            loss.backward()
            opt.step()
            if l1 > 0 and l1_prox:
                with torch.no_grad():
                    for L in net.layers:
                        for w in (L.weight, L.bias):
                            w.copy_(w.sign() * (w.abs() - lr * l1).clamp_min(0))
        if verbose:
            net.eval()
            with torch.no_grad():
                acc = (net(xte).argmax(1) == yte).float().mean().item()
            print(f"  teacher epoch {ep + 1}/{epochs}  test acc {acc:.4f}")
    if verbose:
        print_weight_sparsity(net)
    torch.save(net.state_dict(), path)
    return net


@torch.no_grad()
def print_weight_sparsity(net):
    """Per weight matrix: mean|w| and the fraction of entries below 1e-2 / 1e-3
    of that matrix's max|w| (Adam + an L1 penalty chatters near zero rather
    than landing on exact zeros, so report relative thresholds)."""
    for i, L in enumerate(net.layers):
        a = L.weight.abs()
        m = a.max()
        print(f"  W{i + 1} {tuple(L.weight.shape)}: mean|w| {a.mean().item():.3e}  "
              f"max|w| {m.item():.3e}  zero {(a == 0).float().mean().item():.4f}  "
              f"<1e-2max {(a < 1e-2 * m).float().mean().item():.3f}  "
              f"<1e-3max {(a < 1e-3 * m).float().mean().item():.3f}  "
              f"zero rows {int((a.sum(1) == 0).sum())}/{a.shape[0]}  "
              f"zero cols {int((a.sum(0) == 0).sum())}/{a.shape[1]}", flush=True)


def teacher_path_cnn(input_shape, conv_cfgs, fc_dims, out_dim, epochs, seed, act):
    os.makedirs(TEACHER_DIR, exist_ok=True)
    sh = "x".join(map(str, input_shape))
    cs = "_".join("-".join(map(str, c)) for c in conv_cfgs)
    fs = ("_fc" + "-".join(map(str, fc_dims))) if fc_dims else ""
    return os.path.join(
        TEACHER_DIR,
        f"teacher_cnn_{sh}__{cs}{fs}__o{out_dim}_e{epochs}_s{seed}_{act}.pt")


def make_teacher_cnn(input_shape, conv_cfgs, fc_dims=(), out_dim=10, epochs=25,
                     seed=0, lr=1e-3, batch=256, device="cpu", verbose=True,
                     act="relu"):
    """Train (or load) a ConvNet blackbox. Data is loaded FLAT (via load_data,
    keyed by in_dim/out_dim) and the ConvNet reshapes internally."""
    from nets import ConvNet
    path = teacher_path_cnn(input_shape, conv_cfgs, fc_dims, out_dim, epochs, seed, act)
    in_dim = int(input_shape[0] * input_shape[1] * input_shape[2])
    if os.path.exists(path):
        net = ConvNet(input_shape, conv_cfgs, fc_dims, out_dim, act)
        net.load_state_dict(torch.load(path, map_location=device))
        return net.to(device)
    torch.manual_seed(seed)
    (xtr, ytr), (xte, yte) = load_data([in_dim, out_dim], device)
    net = ConvNet(input_shape, conv_cfgs, fc_dims, out_dim, act).to(device)
    if getattr(net, "rbf", False):
        # LeNet-5 (1998): the RBF output parameters are FIXED +-1 codes (the paper used stylised 7x12 bitmaps of the
        # characters; we use a fixed random +-1 code per class, seed-determined), not trained
        _g = torch.Generator().manual_seed(1998 + seed)
        with torch.no_grad():
            net.layers[-1].weight.copy_((torch.randint(0, 2, net.layers[-1].weight.shape, generator=_g) * 2 - 1).float())
        net.layers[-1].weight.requires_grad_(False)
    opt = torch.optim.Adam([p_ for p_ in net.parameters() if p_.requires_grad], lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    # dataset smaller than the teacher's input (e.g. CIFAR 32x32 -> 224x224
    # AlexNet-exact geometry): bilinear-upsample per batch on the fly.
    if xtr.shape[1] != in_dim and input_shape[1:] == (32, 32) and xtr.shape[1] == 784:
        # LeNet-5: MNIST 28x28 centred in a 32x32 field (as in the paper), padded with the background value
        import torch.nn.functional as _F
        _bg = float(xtr.min())

        def _prep(xb):
            return _F.pad(xb.view(-1, 1, 28, 28), (2, 2, 2, 2), value=_bg).reshape(xb.shape[0], -1)
    elif xtr.shape[1] != in_dim:
        import torch.nn.functional as _F
        C, H, W = input_shape
        side = int(round((xtr.shape[1] // C) ** 0.5))

        def _prep(xb):
            return _F.interpolate(xb.view(-1, C, side, side), size=(H, W),
                                  mode="bilinear", align_corners=False
                                  ).reshape(xb.shape[0], -1)
    else:
        def _prep(xb):
            return xb
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=batch, shuffle=True)
    for ep in range(epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            lossf(net(_prep(xb)), yb).backward()
            opt.step()
        if verbose:
            net.eval()
            with torch.no_grad():
                acc = sum((net(_prep(xte[i:i + batch])).argmax(1)
                           == yte[i:i + batch]).sum().item()
                          for i in range(0, len(xte), batch)) / len(xte)
            print(f"  cnn teacher epoch {ep + 1}/{epochs}  test acc {acc:.4f}")
    torch.save(net.state_dict(), path)
    return net
