"""Prove population parallelism with PROCESSES (separate GIL per GPU): persistent
workers, one per GPU, each holding its members; warm up, then time a training
round. Compare to single-process sequential over all 9 members. 256-wide net."""
import time
import torch
import torch.multiprocessing as mp
from nets import MLP


def worker(gpu, dims, n_members, N, batch, epochs, cmd_q, done_q):
    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(dev)
    X = torch.randn(N, dims[0], device=dev)
    Y = torch.randn(N, dims[2], device=dev)
    nets = [MLP(dims).to(dev) for _ in range(n_members)]
    opts = [torch.optim.Adam(n.parameters(), lr=1e-3) for n in nets]
    while True:
        if cmd_q.get() == "stop":
            break
        for net, opt in zip(nets, opts):
            for _ in range(epochs):
                for i in range(0, N, batch):
                    opt.zero_grad()
                    loss = ((net(X[i:i + batch]) - Y[i:i + batch]) ** 2).mean()
                    loss.backward()
                    opt.step()
        torch.cuda.synchronize(dev)
        done_q.put(gpu)


def main():
    mp.set_start_method("spawn", force=True)
    dims = [3072, 256, 100]; N = 16000; batch = 512; epochs = 2; p = 9; ng = 3

    # --- single-process sequential baseline (all 9 on one GPU) ---
    dev = torch.device("cuda:0")
    X = torch.randn(N, dims[0], device=dev); Y = torch.randn(N, dims[2], device=dev)
    nets = [MLP(dims).to(dev) for _ in range(p)]
    opts = [torch.optim.Adam(n.parameters(), lr=1e-3) for n in nets]

    def seq_round():
        for net, opt in zip(nets, opts):
            for _ in range(epochs):
                for i in range(0, N, batch):
                    opt.zero_grad()
                    loss = ((net(X[i:i + batch]) - Y[i:i + batch]) ** 2).mean()
                    loss.backward(); opt.step()
        torch.cuda.synchronize(dev)
    seq_round()  # warmup
    t = time.time(); seq_round(); seq = time.time() - t
    print(f"SEQUENTIAL 1-process 9 members: {seq:.2f}s")
    del nets, opts, X, Y; torch.cuda.empty_cache()

    # --- 3 persistent worker processes, 3 members each ---
    cmd_qs = [mp.Queue() for _ in range(ng)]
    done_q = mp.Queue()
    procs = [mp.Process(target=worker,
                        args=(g, dims, p // ng, N, batch, epochs, cmd_qs[g], done_q))
             for g in range(ng)]
    for pr in procs:
        pr.start()
    # warmup round
    for q in cmd_qs:
        q.put("go")
    for _ in range(ng):
        done_q.get()
    # timed round
    t = time.time()
    for q in cmd_qs:
        q.put("go")
    for _ in range(ng):
        done_q.get()
    par = time.time() - t
    print(f"MULTIPROCESS 3 GPUs x 3 members:  {par:.2f}s   speedup={seq/par:.2f}x")
    for q in cmd_qs:
        q.put("stop")
    for pr in procs:
        pr.join()


if __name__ == "__main__":
    main()
