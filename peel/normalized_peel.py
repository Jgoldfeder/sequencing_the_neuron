"""Normalized peeled training with a reachable, standardized disagreement search.

Teacher access is forward-only. Prefix and initial tail come from the student.
Calibration uses no teacher calls. Coordinates are fixed for one training stage.
"""
import copy
import torch


class NormalizedPeel:
    def __init__(self, guess, frontier, calibration=2048, seed=0, lr=1e-3,
                 box=4., residual=.5):
        if not 0 < frontier < len(guess.layers):
            raise ValueError('frontier is a zero-based layer index after a recovered prefix')
        if getattr(guess, 'act_name', 'leaky_relu') != 'leaky_relu':
            raise ValueError('This experimental method currently supports leaky_relu')
        self.frontier = frontier
        self.base = copy.deepcopy(guess).double().eval()
        self.base.requires_grad_(False)
        self.device = next(guess.parameters()).device
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.box, self.residual = box, residual
        if box <= 0 or residual < 0: raise ValueError('box must be positive; residual nonnegative')
        self.din = guess.layers[0].in_features
        self.width = guess.layers[frontier].in_features
        if calibration < 2*self.width: raise ValueError('Use at least twice the hidden width for calibration')
        self.tail = torch.nn.ModuleList(copy.deepcopy(list(guess.layers[frontier:]))).float()
        self.tail.requires_grad_(True)
        with torch.no_grad():
            raw = self.randn(calibration, self.din)
            x = box * torch.tanh(raw / box)
            h = self.prefix(x)
            self.mean = h.mean(0)
            centered = h-self.mean
            cov = centered.T @ centered / (len(h)-1)
            values, vectors = torch.linalg.eigh(cov)
            floor = values.max().clamp_min(1e-20)*1e-5
            roots = values.clamp_min(floor).sqrt()
            self.encode = vectors / roots[None, :]
            self.decode = roots[:, None] * vectors.T
            z = centered @ self.encode
            # Learn a bounded raw-input proposal from standardized hidden features.
            # This is regression, not a claimed inverse of the prefix.
            gram = z.T@z + .01*len(z)*torch.eye(self.width,device=self.device,dtype=z.dtype)
            self.raw_mean = raw.mean(0)
            self.proposal = torch.linalg.solve(gram, z.T@(raw-self.raw_mean))
            w = self.tail[0].weight.double().clone()
            self.tail[0].weight.copy_(w @ self.decode.T)
            self.tail[0].bias.add_((w@self.mean).float())
            self.calibration_stats = dict(retained_directions=int((values>floor).sum()),
                dimensions=self.width, normalized_std_min=float(z.std(0).min()),
                normalized_std_max=float(z.std(0).max()))
        self.optimizer = torch.optim.Adam(self.tail.parameters(),lr=lr)
        self.queries = 0

    def randn(self, *shape):
        return torch.randn(*shape,device=self.device,dtype=torch.float64,generator=self.gen)

    def prefix(self, x):
        for layer in self.base.layers[:self.frontier]:
            x = torch.nn.functional.leaky_relu(layer(x),.01)
        return x

    def features(self, x):
        return (self.prefix(x)-self.mean)@self.encode

    def predict_features(self, z):
        h=z.float()
        for i, layer in enumerate(self.tail):
            h=layer(h)
            if i+1<len(self.tail): h=torch.nn.functional.leaky_relu(h,.01)
        return h

    def same_features(self, other):
        """Share computation only for bitwise identical frozen feature maps."""
        return (isinstance(other, NormalizedPeel) and self.frontier == other.frontier
                and torch.equal(self.mean, other.mean)
                and torch.equal(self.encode, other.encode)
                and all(torch.equal(a, b) for la, lb in zip(
                    self.base.layers[:self.frontier], other.base.layers[:other.frontier])
                    for a, b in zip(la.parameters(), lb.parameters())))

    def predict(self, x):
        return self.predict_features(self.features(x))

    def propose(self, z, residual):
        raw=self.raw_mean+z@self.proposal+self.residual*residual
        return self.box*torch.tanh(raw/self.box)

    def sample(self, peers, count=256, steps=30, lr=.05):
        """Optimize student-only committee variance. Peers accept original inputs.

        Independent fixed raw residuals diversify each sample. A per-sample best
        iterate is retained, so final disagreement cannot fall below its seed.
        """
        if not peers: raise ValueError('Disagreement sampling needs another student')
        z=self.randn(count,self.width).requires_grad_(True)
        residual=self.randn(count,self.din)
        opt=torch.optim.Adam([z],lr=lr)
        # Bound stage predictors can share the expensive fp64 prefix/whitening.
        shared = [getattr(p, '__self__', None) for p in peers]
        shared = [s if self.same_features(s) else None for s in shared]
        def score(x):
            features = self.features(x)
            predictions=torch.stack([self.predict_features(features)]+[
                s.predict_features(features) if s is not None else p(x).float()
                for p,s in zip(peers,shared)])
            return predictions.var(0,unbiased=False).mean(1)
        start=self.propose(z,residual).detach()
        best = best_x = initial = None
        # Score each iterate once. Detach only the best-query bookkeeping;
        # retain its graph for the query gradient at the same iterate.
        for step in range(steps+1):
            with torch.set_grad_enabled(step < steps):
                x=self.propose(z,residual);s=score(x)
            with torch.no_grad():
                if initial is None:
                    initial=s.detach().clone();best=initial.clone();best_x=x.detach().clone()
                else:
                    improved=s>best
                    best[improved]=s[improved];best_x[improved]=x[improved]
            if step < steps:
                opt.zero_grad()
                z.grad=torch.autograd.grad(-s.mean(),z)[0]
                opt.step()
                with torch.no_grad(): z.clamp_(-3,3)
        with torch.no_grad():
            z0=self.features(start);z1=self.features(best_x)
            stats=dict(disagreement_before=float(initial.mean()),disagreement_after=float(best.mean()),
                normalized_hidden_movement=float((z1-z0).square().mean().sqrt()),
                normalized_hidden_std=float(z1.std(0,unbiased=False).mean()),
                max_raw=float(best_x.abs().max()))
        return best_x.detach(),stats

    def fit(self, x, y, epochs=20, batch=256):
        with torch.no_grad(): z=self.features(x).float();y=y.float()
        before=float((self.predict_features(z)-y).square().mean().detach())
        for _ in range(epochs):
            order=torch.randperm(len(z),device=self.device,generator=self.gen)
            for idx in order.split(batch):
                self.optimizer.zero_grad()
                loss=(self.predict_features(z[idx])-y[idx]).square().mean()
                loss.backward();self.optimizer.step()
        after=float((self.predict_features(z)-y).square().mean().detach())
        return dict(training_mse_before=before,training_mse_after=after)

    @torch.no_grad()
    def export(self):
        """Return a full MLP in original hidden coordinates, without changing prefix."""
        net=copy.deepcopy(self.base)
        for dst,src in zip(net.layers[self.frontier:],self.tail):
            dst.weight.copy_(src.weight.double());dst.bias.copy_(src.bias.double())
        layer=net.layers[self.frontier]
        w=layer.weight.clone()@self.encode.T
        layer.weight.copy_(w);layer.bias.sub_(w@self.mean)
        return net


class TrainingView(torch.nn.Module):
    """Adam parameter view consuming raw inputs; base model stays in original gauge."""
    def __init__(self, stage, cached=False):
        super().__init__()
        self.tail = stage.tail
        self.stage = stage
        self.cached = cached

    def forward(self, x):
        if self.cached:
            return self.stage.predict_features(x)
        with torch.no_grad():
            z = self.stage.features(x.double())
        return self.stage.predict_features(z)


class PopulationStages:
    def __init__(self):
        self.entries = {}

    def prepare(self, population, frontier, optimizers, masks, seed):
        stages = []
        for member, optimizer in zip(population, optimizers):
            key = id(member)
            entry = self.entries.get(key)
            params = list(member.parameters())
            current_masks = [masks(member,l).clone() for l in range(frontier,len(member.layers))]
            if (entry is None or entry[0].frontier != frontier or
                    any(not torch.equal(a,b) for a,b in zip(current_masks,entry[2])) or
                    any(not torch.equal(a, b) for a, b in zip(params, entry[1]))):
                stage = NormalizedPeel(member, frontier,
                    calibration=max(1024, 4*member.layers[frontier].in_features), seed=seed,
                    lr=optimizer.param_groups[0]['lr'])
                # Masks are refreshed with the stage after any refinement/reroll.
                for l, layer in enumerate(stage.tail, frontier):
                    mask = masks(member, l)
                    if mask is not None:
                        keep = (~mask).to(layer.weight.dtype)
                        layer.weight.register_hook(lambda g,k=keep:g*k[:,None])
                        layer.bias.register_hook(lambda g,k=keep:g*k)
                entry = (stage, [p.detach().clone() for p in params], current_masks)
                self.entries[key] = entry
            stage = entry[0]
            for group in stage.optimizer.param_groups:
                group['lr'] = optimizer.param_groups[0]['lr']
            stages.append(stage)
        self.entries = {id(m):self.entries[id(m)] for m in population}
        return stages

    @torch.no_grad()
    def write_back(self, population, stages, masks):
        for member, stage in zip(population, stages):
            exported = stage.export()
            for l in range(stage.frontier,len(member.layers)):
                dst,src=member.layers[l],exported.layers[l]
                mask=masks(member,l)
                update=torch.ones(len(dst.bias),dtype=torch.bool,device=dst.bias.device) if mask is None else ~mask
                dst.weight[update]=src.weight.to(dst.weight.dtype)[update]
                dst.bias[update]=src.bias.to(dst.bias.dtype)[update]
            self.entries[id(member)] = (stage,[p.detach().clone() for p in member.parameters()],
                [masks(member,l).clone() for l in range(stage.frontier,len(member.layers))])
