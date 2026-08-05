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

# This dir is nested one level under the serial root code, so it reaches up an
# extra level to share the SAME data/ and teachers/ the root code uses (avoids
# re-downloading datasets and re-training teachers).
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data")
TEACHER_DIR = os.path.join(os.path.dirname(__file__), "..", "teachers")

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


def _prepare_tinyimagenet_val(val_dir):
    """The val split ships as a flat val/images/*.JPEG + val_annotations.txt;
    reorganize into val/<wnid>/*.JPEG so ImageFolder can read it (idempotent)."""
    img_dir = os.path.join(val_dir, "images")
    ann = os.path.join(val_dir, "val_annotations.txt")
    if not os.path.isdir(img_dir) or not os.path.exists(ann):
        return                                    # already prepared
    import shutil
    with open(ann) as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            fn, wnid = parts[0], parts[1]
            cls = os.path.join(val_dir, wnid)
            os.makedirs(cls, exist_ok=True)
            src = os.path.join(img_dir, fn)
            if os.path.exists(src):
                shutil.move(src, os.path.join(cls, fn))
    shutil.rmtree(img_dir, ignore_errors=True)


def _ensure_tinyimagenet(root):
    """Download + prepare TinyImageNet-200 into `root` if absent. torchvision has
    no built-in downloader for it, so a fresh machine would otherwise error."""
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    if not os.path.isdir(train_dir):
        from torchvision.datasets.utils import download_and_extract_archive
        parent = os.path.dirname(root)            # = DATA_ROOT
        os.makedirs(parent, exist_ok=True)
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        print(f"[data] TinyImageNet-200 not found; downloading to {parent} "
              f"(~240MB)...", flush=True)
        download_and_extract_archive(url, download_root=parent,
                                     remove_finished=True)
    _prepare_tinyimagenet_val(val_dir)            # idempotent


def load_tinyimagenet(device):
    """TinyImageNet-200: 64x64x3 (=12288) inputs, 200 classes. Decoded from the
    ImageFolder once and cached as tensors (100k train JPEGs are slow to decode).
    Auto-downloads the dataset on first use if it isn't present."""
    cache = os.path.join(DATA_ROOT, "tinyimagenet_64.pt")
    if os.path.exists(cache):
        d = torch.load(cache, map_location="cpu")
        return ((d["xtr"].to(device), d["ytr"].to(device)),
                (d["xte"].to(device), d["yte"].to(device)))
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    root = os.path.join(DATA_ROOT, "tiny-imagenet-200")
    _ensure_tinyimagenet(root)
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


def load_data(dims, device):
    """Pick the dataset from the architecture: 12288-in/200-out => TinyImageNet,
    3072-in/100-out => CIFAR-100, else MNIST."""
    if dims[0] == 12288 or dims[-1] == 200:
        return load_tinyimagenet(device)
    if dims[0] == 3072 or dims[-1] == 100:
        return load_cifar100(device)
    return load_mnist(device)


def teacher_path(dims, epochs, seed):
    os.makedirs(TEACHER_DIR, exist_ok=True)
    tag = "x".join(map(str, dims))
    return os.path.join(TEACHER_DIR, f"teacher_{tag}_e{epochs}_s{seed}.pt")


def make_teacher(dims, epochs=25, seed=0, lr=1e-3, batch=256, device="cpu",
                 verbose=True):
    """Train (or load from cache) the blackbox network."""
    path = teacher_path(dims, epochs, seed)
    if os.path.exists(path):
        net = MLP(dims)
        net.load_state_dict(torch.load(path, map_location=device))
        return net.to(device)

    torch.manual_seed(seed)
    (xtr, ytr), (xte, yte) = load_data(dims, device)
    net = MLP(dims).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=batch,
                        shuffle=True)
    for ep in range(epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
        if verbose:
            net.eval()
            with torch.no_grad():
                acc = (net(xte).argmax(1) == yte).float().mean().item()
            print(f"  teacher epoch {ep + 1}/{epochs}  test acc {acc:.4f}")
    torch.save(net.state_dict(), path)
    return net
