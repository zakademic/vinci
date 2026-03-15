import json, textwrap

def md(src):
    src = textwrap.dedent(src).lstrip('\n').rstrip() + '\n'
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}

def code(src):
    src = textwrap.dedent(src).lstrip('\n').rstrip() + '\n'
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(True)}

cells = []

# ── Title ──────────────────────────────────────────────────────────────────────
cells.append(md(r"""
# Physics-Informed Neural Networks — 2D Laplace & Heat Equation

A structured, pedagogical tour of PINNs from first principles to complex geometries.

**Roadmap**

| § | Topic | Key idea |
|---|-------|----------|
| 1 | Baseline — simple manual PINN | Minimal hardcoded PyTorch, square domain, u=1 on top |
| 2 | Hyperparameter study | Optimizers, depth, width, activations |
| 3 | Triangle-mesh sampling | Geometry-agnostic collocation |
| 4 | BC variants | Neumann, mixed, multi-mode |
| 5 | L-shaped domain | Non-convex geometry, FD reference |
| 6 | Transient heat equation | Full u_t = α∇²u with time |

All sections share a common analytical benchmark where one exists.
"""))

# ── Imports & setup ────────────────────────────────────────────────────────────
cells.append(code(
"""import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time, warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cpu')   # change to 'cuda' if available
print(f"PyTorch {torch.__version__} | device: {device}")
"""))

# ── Section 1 header ───────────────────────────────────────────────────────────
cells.append(md(r"""
---
## 1. Baseline — Simple Manual PINN

### Problem

Laplace equation on the unit square $[0,1]^2$:

$$\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0$$

**Boundary conditions:**

| Boundary | Condition |
|----------|-----------|
| $x=0$ (left)   | $u = 0$ |
| $x=1$ (right)  | $u = 0$ |
| $y=0$ (bottom) | $u = 0$ |
| $y=1$ (top)    | $u = 1$ |

### Analytical Solution (Fourier series)

By separation of variables, $u = X(x)\,Y(y)$:
- $X_n(x) = \sin(n\pi x)$, $\quad n = 1,2,3,\ldots$
- $Y_n(y) = \sinh(n\pi y)$, satisfying $Y_n(0)=0$

The top BC $u(x,1)=1$ is expanded in a sine series:

$$1 = \sum_{n=1}^{\infty} B_n \sin(n\pi x) \quad \Rightarrow \quad B_n = \frac{2}{\sinh(n\pi)} \int_0^1 \sin(n\pi x)\,dx = \frac{4}{n\pi\,\sinh(n\pi)} \quad (n \text{ odd})$$

$$\boxed{u(x,y) = \sum_{\substack{n=1\\n\text{ odd}}}^{\infty} \frac{4}{n\pi} \cdot \frac{\sin(n\pi x)\,\sinh(n\pi y)}{\sinh(n\pi)}}$$
"""))

# ── Analytical solution ────────────────────────────────────────────────────────
cells.append(code(
"""def u_exact_p1(x, y, N=40):
    '''
    Fourier-series solution for Laplace on [0,1]^2:
      u=0 on left, right, bottom;  u=1 on top.
    Converges very fast (sinh grows exponentially).
    '''
    u = np.zeros_like(x, dtype=float)
    for n in range(1, 2*N, 2):          # odd n only
        Bn = 4.0 / (n * np.pi * np.sinh(n * np.pi))
        u += Bn * np.sin(n * np.pi * x) * np.sinh(n * np.pi * y)
    return u

# Quick visualisation of the exact solution
x1d = np.linspace(0, 1, 120)
y1d = np.linspace(0, 1, 120)
XX, YY = np.meshgrid(x1d, y1d)
U_exact = u_exact_p1(XX, YY)

plt.figure(figsize=(5, 4))
plt.contourf(XX, YY, U_exact, levels=50, cmap='hot')
plt.colorbar(label='u')
plt.title('Analytical solution — §1 problem', fontsize=12)
plt.xlabel('x'); plt.ylabel('y')
plt.tight_layout()
plt.show()

print(f"u at centre (0.5, 0.5) = {u_exact_p1(np.array([0.5]), np.array([0.5]))[0]:.5f}")
"""))

# ── Section 1 PINN description ─────────────────────────────────────────────────
cells.append(md(r"""
### Simple Manual PINN

We build the smallest possible PINN — everything hardcoded, no abstractions.

**Network:** $\hat u_\theta(x,y)$ — 3 hidden layers of 32 neurons, tanh activation.

**Loss:**

$$\mathcal{L} = w_\text{pde}\underbrace{\frac{1}{N_\text{pde}}\sum_i (\hat u_{xx}+\hat u_{yy})^2}_{\mathcal{L}_\text{pde}} + w_\text{bc}\underbrace{\frac{1}{N_\text{bc}}\sum_j (\hat u - u^*)^2}_{\mathcal{L}_\text{bc}}$$

Gradients $\hat u_{xx}, \hat u_{yy}$ are computed via **automatic differentiation** through the network.
"""))

# ── Simple PINN class + losses ─────────────────────────────────────────────────
cells.append(code(
"""# ── Network ────────────────────────────────────────────────────────────────────
class SimplePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, xy):          # xy : (N, 2)
        return self.net(xy).squeeze(-1)


# ── PDE loss (Laplace residual) ────────────────────────────────────────────────
def laplace_residual(model, x, y):
    '''Returns mean squared PDE residual u_xx + u_yy at interior points.'''
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    u  = model(torch.stack([x, y], dim=1))
    u_x  = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_y  = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    return (u_xx + u_yy).pow(2).mean()


# ── BC loss for Problem 1 ──────────────────────────────────────────────────────
def bc_loss_p1(model, n=300):
    '''u=0 on left/right/bottom, u=1 on top.'''
    losses = []
    for xi, yi, target in [
        (torch.rand(n),   torch.zeros(n), 0.0),   # bottom
        (torch.zeros(n),  torch.rand(n),  0.0),   # left
        (torch.ones(n),   torch.rand(n),  0.0),   # right
        (torch.rand(n),   torch.ones(n),  1.0),   # top
    ]:
        pred = model(torch.stack([xi, yi], dim=1))
        losses.append((pred - target).pow(2).mean())
    return sum(losses) / len(losses)
"""))

# ── Simple training loop ───────────────────────────────────────────────────────
cells.append(code(
"""# ── Training ───────────────────────────────────────────────────────────────────
model_simple = SimplePINN()
optimizer    = torch.optim.Adam(model_simple.parameters(), lr=1e-3)

W_PDE, W_BC = 1.0, 20.0
N_PDE, N_BC = 2000, 400
EPOCHS       = 5000

hist_simple = {'pde': [], 'bc': [], 'total': []}
t0 = time.time()

for ep in range(1, EPOCHS + 1):
    optimizer.zero_grad()

    x_i = torch.rand(N_PDE)
    y_i = torch.rand(N_PDE)

    lp = laplace_residual(model_simple, x_i, y_i)
    lb = bc_loss_p1(model_simple, N_BC)
    loss = W_PDE * lp + W_BC * lb
    loss.backward()
    optimizer.step()

    hist_simple['pde'].append(lp.item())
    hist_simple['bc'].append(lb.item())
    hist_simple['total'].append(loss.item())

    if ep % 1000 == 0:
        print(f"Epoch {ep:5d} | total {loss.item():.3e} | pde {lp.item():.3e} | bc {lb.item():.3e}")

print(f"\\nTraining time: {time.time()-t0:.1f}s")
"""))

# ── Plot simple PINN results ───────────────────────────────────────────────────
cells.append(code(
"""# ── Evaluation ─────────────────────────────────────────────────────────────────
def eval_model(model, nx=100):
    x1d = np.linspace(0, 1, nx)
    y1d = np.linspace(0, 1, nx)
    XX, YY = np.meshgrid(x1d, y1d)
    xy = torch.tensor(np.stack([XX.ravel(), YY.ravel()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        U_pred = model(xy).numpy().reshape(nx, nx)
    return XX, YY, U_pred

XX, YY, U_pred_simple = eval_model(model_simple)
U_true = u_exact_p1(XX, YY)
err = np.abs(U_pred_simple - U_true)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
kw = dict(vmin=0, vmax=1, cmap='hot', levels=50)
c0 = axes[0].contourf(XX, YY, U_true, **kw);       axes[0].set_title('Analytical')
c1 = axes[1].contourf(XX, YY, U_pred_simple, **kw); axes[1].set_title('PINN')
c2 = axes[2].contourf(XX, YY, err, levels=50, cmap='Reds'); axes[2].set_title('|Error|')
for ax in axes: ax.set_xlabel('x'); ax.set_ylabel('y')
fig.colorbar(c0, ax=axes[1]); fig.colorbar(c2, ax=axes[2])
plt.suptitle('§1 — Simple PINN vs Analytical', fontsize=13)
plt.tight_layout()
plt.show()

l2_rel = np.sqrt(((U_pred_simple-U_true)**2).mean()) / np.sqrt((U_true**2).mean())
print(f"Relative L2 error: {l2_rel:.4%}")

# Loss curves
fig, ax = plt.subplots(figsize=(8, 3))
ax.semilogy(hist_simple['pde'],   label='PDE loss')
ax.semilogy(hist_simple['bc'],    label='BC loss')
ax.semilogy(hist_simple['total'], label='Total (weighted)', lw=2)
ax.set_xlabel('Epoch'); ax.legend(); ax.set_title('§1 — Training history')
plt.tight_layout(); plt.show()
"""))

# ── Section 2 header ───────────────────────────────────────────────────────────
cells.append(md(
"""---
## 2. Hyperparameter Study

We now use a **configurable PINN** to systematically explore:

1. **Optimizers** — Adam, Adam → L-BFGS, RMSprop
2. **Depth** — 2, 3, 4, 6 hidden layers
3. **Width** — 16, 32, 64, 128 neurons
4. **Activations** — Tanh, Sin, GELU, SiLU

For each experiment we run the same number of epochs and compare:
- Final **PDE loss** (residual)
- Final **relative L2 error** against the analytical solution
- **Convergence speed**
"""))

# ── Configurable PINN class ────────────────────────────────────────────────────
cells.append(code(
"""# ── Configurable PINN ──────────────────────────────────────────────────────────
class PINN(nn.Module):
    '''
    Fully-connected PINN for 2D steady-state problems.

    Parameters
    ----------
    n_in     : input dimension (2 for steady, 3 for transient)
    hidden   : list of hidden layer widths
    act      : activation class (nn.Tanh, nn.GELU, ...)
    '''
    def __init__(self, n_in=2, hidden=[64, 64, 64], act=nn.Tanh):
        super().__init__()
        dims = [n_in] + hidden + [1]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), act()]
        layers.pop()                     # remove last activation
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy).squeeze(-1)


# ── Generic training function ──────────────────────────────────────────────────
def train_pinn(model, pde_loss_fn, bc_loss_fn,
               n_epochs=4000, lr=1e-3,
               n_pde=2000, n_bc=400,
               w_pde=1.0, w_bc=20.0,
               lbfgs_steps=0, verbose=False):
    '''
    Train with Adam (and optionally L-BFGS fine-tuning).
    Returns history dict with keys 'pde', 'bc', 'total'.
    '''
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {'pde': [], 'bc': [], 'total': []}

    for ep in range(1, n_epochs + 1):
        opt.zero_grad()
        x_i, y_i = torch.rand(n_pde), torch.rand(n_pde)
        lp = pde_loss_fn(model, x_i, y_i)
        lb = bc_loss_fn(model, n_bc)
        loss = w_pde * lp + w_bc * lb
        loss.backward()
        opt.step()
        hist['pde'].append(lp.item())
        hist['bc'].append(lb.item())
        hist['total'].append(loss.item())
        if verbose and ep % 1000 == 0:
            print(f"  Adam ep {ep:5d} | pde {lp.item():.3e} | bc {lb.item():.3e}")

    if lbfgs_steps > 0:
        opt_lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.5,
                                       max_iter=20, history_size=50,
                                       line_search_fn='strong_wolfe')
        for step in range(lbfgs_steps):
            def closure():
                opt_lbfgs.zero_grad()
                x_i, y_i = torch.rand(n_pde), torch.rand(n_pde)
                lp = pde_loss_fn(model, x_i, y_i)
                lb = bc_loss_fn(model, n_bc)
                loss = w_pde * lp + w_bc * lb
                loss.backward()
                return loss
            loss = opt_lbfgs.step(closure)
            with torch.no_grad():
                x_i, y_i = torch.rand(n_pde), torch.rand(n_pde)
                lp = pde_loss_fn(model, x_i, y_i)
                lb = bc_loss_fn(model, n_bc)
            hist['pde'].append(lp.item())
            hist['bc'].append(lb.item())
            hist['total'].append((w_pde*lp + w_bc*lb).item())
            if verbose and (step+1) % 100 == 0:
                print(f"  L-BFGS step {step+1:4d} | pde {lp.item():.3e} | bc {lb.item():.3e}")

    return hist


def l2_error(model, exact_fn, nx=80):
    x1d = np.linspace(0, 1, nx)
    y1d = np.linspace(0, 1, nx)
    XX, YY = np.meshgrid(x1d, y1d)
    xy = torch.tensor(np.stack([XX.ravel(), YY.ravel()], 1), dtype=torch.float32)
    with torch.no_grad():
        U_pred = model(xy).numpy().reshape(nx, nx)
    U_true = exact_fn(XX, YY)
    return np.sqrt(((U_pred - U_true)**2).mean()) / np.sqrt((U_true**2).mean() + 1e-12)
"""))

# ── Optimizer comparison ───────────────────────────────────────────────────────
cells.append(code(
"""# ── Experiment 1: Optimizers ───────────────────────────────────────────────────
torch.manual_seed(0)

experiments_opt = {
    'Adam':          ('Adam only',         {}),
    'Adam->L-BFGS':  ('Adam then L-BFGS',  {'lbfgs_steps': 300}),
    'RMSprop':       ('RMSprop',           {}),
}

results_opt = {}
EPOCHS_OPT = 3000

for key, (label, extra) in experiments_opt.items():
    torch.manual_seed(0)
    m = PINN(hidden=[64, 64, 64])
    if key == 'RMSprop':
        opt_r = torch.optim.RMSprop(m.parameters(), lr=1e-3)
        h = {'pde': [], 'bc': [], 'total': []}
        for ep in range(EPOCHS_OPT):
            opt_r.zero_grad()
            x_i, y_i = torch.rand(2000), torch.rand(2000)
            lp = laplace_residual(m, x_i, y_i)
            lb = bc_loss_p1(m, 400)
            loss = 1.0*lp + 20.0*lb
            loss.backward()
            opt_r.step()
            h['pde'].append(lp.item()); h['bc'].append(lb.item()); h['total'].append(loss.item())
    else:
        h = train_pinn(m, laplace_residual, bc_loss_p1,
                       n_epochs=EPOCHS_OPT, lr=1e-3, **extra)
    err = l2_error(m, u_exact_p1)
    results_opt[key] = {'hist': h, 'l2': err}
    print(f"{key:15s}  PDE loss {h['pde'][-1]:.3e}  L2 err {err:.4%}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for key, res in results_opt.items():
    axes[0].semilogy(res['hist']['pde'], label=key)
    axes[1].semilogy(res['hist']['total'], label=key)
axes[0].set_title('PDE loss'); axes[1].set_title('Total loss (weighted)')
for ax in axes: ax.set_xlabel('Epoch'); ax.legend()
plt.suptitle('§2 — Optimizer comparison', fontsize=13)
plt.tight_layout(); plt.show()
"""))

# ── Architecture comparison ────────────────────────────────────────────────────
cells.append(code(
"""# ── Experiment 2: Depth & Width ────────────────────────────────────────────────
torch.manual_seed(0)

arch_configs = {
    '2x32':   [32, 32],
    '3x64':   [64, 64, 64],
    '4x64':   [64, 64, 64, 64],
    '4x128':  [128, 128, 128, 128],
    '6x64':   [64]*6,
}

results_arch = {}
for key, hidden in arch_configs.items():
    torch.manual_seed(0)
    m = PINN(hidden=hidden)
    h = train_pinn(m, laplace_residual, bc_loss_p1, n_epochs=3000, lr=1e-3)
    err = l2_error(m, u_exact_p1)
    n_params = sum(p.numel() for p in m.parameters())
    results_arch[key] = {'hist': h, 'l2': err, 'params': n_params}
    print(f"{key:8s}  params {n_params:6d}  PDE {h['pde'][-1]:.3e}  L2 {err:.4%}")

fig, ax = plt.subplots(figsize=(9, 4))
for key, res in results_arch.items():
    ax.semilogy(res['hist']['pde'], label=f"{key} ({res['params']} params)")
ax.set_xlabel('Epoch'); ax.set_ylabel('PDE loss')
ax.set_title('§2 — Architecture comparison'); ax.legend()
plt.tight_layout(); plt.show()
"""))

# ── Activation comparison ──────────────────────────────────────────────────────
cells.append(code(
"""# ── Experiment 3: Activations ──────────────────────────────────────────────────
torch.manual_seed(0)

class Sin(nn.Module):
    def forward(self, x): return torch.sin(x)

act_configs = {
    'Tanh': nn.Tanh,
    'Sin':  Sin,
    'GELU': nn.GELU,
    'SiLU': nn.SiLU,
}

results_act = {}
for key, act in act_configs.items():
    torch.manual_seed(0)
    m = PINN(hidden=[64, 64, 64], act=act)
    h = train_pinn(m, laplace_residual, bc_loss_p1, n_epochs=3000, lr=1e-3)
    err = l2_error(m, u_exact_p1)
    results_act[key] = {'hist': h, 'l2': err}
    print(f"{key:6s}  PDE {h['pde'][-1]:.3e}  L2 {err:.4%}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for key, res in results_act.items():
    axes[0].semilogy(res['hist']['pde'],   label=key)
    axes[1].semilogy(res['hist']['total'], label=key)
axes[0].set_title('PDE loss'); axes[1].set_title('Total loss')
for ax in axes: ax.set_xlabel('Epoch'); ax.legend()
plt.suptitle('§2 — Activation comparison', fontsize=13)
plt.tight_layout(); plt.show()
"""))

# ── Summary markdown ───────────────────────────────────────────────────────────
cells.append(md(
"""### Summary

From the experiments above we can observe:

- **Optimizer**: Adam → L-BFGS typically gives the lowest PDE residual. Adam alone is usually sufficient for BC fitting, but L-BFGS excels at minimising the smooth PDE residual to machine precision.
- **Depth vs Width**: Deeper networks (4–6 layers) reduce the PDE residual better than wide shallow ones for the same parameter budget.
- **Activation**: `sin` and `Tanh` work best for smooth PDE solutions; `GELU`/`SiLU` can be competitive but occasionally get stuck.

We'll use **4 × 64 + Tanh + Adam → L-BFGS** as our standard config going forward.
"""))

# ── Best config training ───────────────────────────────────────────────────────
cells.append(code(
"""# ── Best config: Adam → L-BFGS ─────────────────────────────────────────────────
torch.manual_seed(42)
model_best = PINN(hidden=[64, 64, 64, 64])
hist_best  = train_pinn(model_best, laplace_residual, bc_loss_p1,
                        n_epochs=5000, lr=1e-3, lbfgs_steps=300, verbose=True)

_, _, U_best = eval_model(model_best)
err_best = np.abs(U_best - u_exact_p1(XX, YY))
l2_best  = l2_error(model_best, u_exact_p1)
print(f"\\nBest config — Relative L2 error: {l2_best:.4%}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
kw = dict(levels=50, cmap='hot')
axes[0].contourf(XX, YY, u_exact_p1(XX, YY), **kw); axes[0].set_title('Analytical')
axes[1].contourf(XX, YY, U_best, **kw);             axes[1].set_title('PINN (best config)')
axes[2].contourf(XX, YY, err_best, levels=50, cmap='Reds'); axes[2].set_title('|Error|')
for ax in axes: ax.set_xlabel('x'); ax.set_ylabel('y')
plt.suptitle(f'§2 — Best config (L2 err = {l2_best:.3%})', fontsize=13)
plt.tight_layout(); plt.show()
"""))

# ── Section 3 header ───────────────────────────────────────────────────────────
cells.append(md(r"""
---
## 3. Triangle-Mesh Sampling

So far we sampled collocation points with `torch.rand()` — trivial on a box.
For **arbitrary geometries** we need a mesh-aware sampler.

### Strategy

Given a triangle mesh $(V, T)$:

1. Compute each triangle's area $A_k$.
2. Sample triangles proportional to $A_k$ (area-weighted).
3. For each sampled triangle, pick a random **barycentric** point:
   $$\mathbf{p} = (1-\sqrt{r_1})\,\mathbf{v}_0 + \sqrt{r_1}(1-r_2)\,\mathbf{v}_1 + r_2\sqrt{r_1}\,\mathbf{v}_2, \quad r_1,r_2 \sim U[0,1]$$

This gives **uniform coverage** regardless of triangle shape or size.
"""))

# ── Mesh infrastructure ────────────────────────────────────────────────────────
cells.append(code(
"""# ── Mesh sampling infrastructure ───────────────────────────────────────────────
def triangle_areas(verts, tris):
    '''Signed area of each triangle (always positive for CCW ordering).'''
    v0 = verts[tris[:, 0]]; v1 = verts[tris[:, 1]]; v2 = verts[tris[:, 2]]
    return 0.5 * np.abs((v1[:,0]-v0[:,0])*(v2[:,1]-v0[:,1])
                       -(v2[:,0]-v0[:,0])*(v1[:,1]-v0[:,1]))


def sample_mesh_interior(verts, tris, n):
    '''
    Sample n points uniformly from the interior of a triangle mesh.
    Returns (x, y) as float32 torch tensors.
    '''
    areas = triangle_areas(verts, tris)
    probs = areas / areas.sum()
    tri_idx = np.random.choice(len(tris), size=n, p=probs)
    r1 = np.random.rand(n); r2 = np.random.rand(n)
    s  = np.sqrt(r1)
    w0 = 1 - s; w1 = s * (1 - r2); w2 = s * r2
    tri_verts = verts[tris[tri_idx]]             # (n, 3, 2)
    pts = (w0[:,None]*tri_verts[:,0]
          + w1[:,None]*tri_verts[:,1]
          + w2[:,None]*tri_verts[:,2])
    return torch.tensor(pts[:,0], dtype=torch.float32), torch.tensor(pts[:,1], dtype=torch.float32)


def get_boundary_edges(tris):
    '''Return set of boundary edges (edge appears in only one triangle).'''
    from collections import Counter
    edge_count = Counter()
    for tri in tris:
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i+1)%3]]))
            edge_count[e] += 1
    return [e for e, cnt in edge_count.items() if cnt == 1]


def sample_boundary_segment(verts, v0_idx, v1_idx, n):
    '''Sample n uniform points along segment v0->v1.'''
    t  = torch.rand(n)
    p0 = verts[v0_idx]; p1 = verts[v1_idx]
    x  = p0[0] + t.numpy() * (p1[0] - p0[0])
    y  = p0[1] + t.numpy() * (p1[1] - p0[1])
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
"""))

# ── Unit square mesh ───────────────────────────────────────────────────────────
cells.append(code(
"""# ── Unit square as a 2-triangle mesh ──────────────────────────────────────────
# (Two right triangles along the diagonal)
sq_verts = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
sq_tris  = np.array([[0,1,2],[0,2,3]])

# Verify: should sample uniformly in [0,1]^2
x_s, y_s = sample_mesh_interior(sq_verts, sq_tris, n=2000)
plt.figure(figsize=(4,4))
plt.scatter(x_s.numpy(), y_s.numpy(), s=1, alpha=0.5)
plt.title('Mesh interior samples — unit square (2 triangles)'); plt.axis('equal')
plt.tight_layout(); plt.show()
"""))

# ── Mesh-based BC sampler ──────────────────────────────────────────────────────
cells.append(code(
"""# ── Mesh-based BC sampler for the §1 problem ──────────────────────────────────
def bc_loss_mesh_p1(model, verts, n_per_edge=200):
    '''
    Enforce BCs on each edge of the unit square mesh:
      bottom (v0->v1): u=0,  right (v1->v2): u=0,
      top    (v2->v3): u=1,  left  (v3->v0): u=0
    '''
    edges_targets = [(0,1,0.0),(1,2,0.0),(2,3,1.0),(3,0,0.0)]
    loss = torch.tensor(0.0)
    for i0, i1, tgt in edges_targets:
        xb, yb = sample_boundary_segment(verts, i0, i1, n_per_edge)
        pred = model(torch.stack([xb, yb], 1))
        loss = loss + (pred - tgt).pow(2).mean()
    return loss / len(edges_targets)


# ── Mesh-aware PDE loss ────────────────────────────────────────────────────────
def laplace_residual_mesh(model, verts, tris, n=2000):
    x, y = sample_mesh_interior(verts, tris, n)
    return laplace_residual(model, x, y)
"""))

# ── Train mesh PINN ────────────────────────────────────────────────────────────
cells.append(code(
"""# ── Train PINN using mesh samplers ─────────────────────────────────────────────
torch.manual_seed(42)
model_mesh = PINN(hidden=[64, 64, 64, 64])
opt_mesh   = torch.optim.Adam(model_mesh.parameters(), lr=1e-3)

hist_mesh  = {'pde': [], 'bc': [], 'total': []}
W_PDE_M, W_BC_M = 1.0, 20.0
EPOCHS_MESH = 5000

for ep in range(1, EPOCHS_MESH+1):
    opt_mesh.zero_grad()
    lp   = laplace_residual_mesh(model_mesh, sq_verts, sq_tris, n=2000)
    lb   = bc_loss_mesh_p1(model_mesh, sq_verts, n_per_edge=300)
    loss = W_PDE_M*lp + W_BC_M*lb
    loss.backward(); opt_mesh.step()
    hist_mesh['pde'].append(lp.item())
    hist_mesh['bc'].append(lb.item())
    hist_mesh['total'].append(loss.item())
    if ep % 1000 == 0:
        print(f"Epoch {ep:5d} | pde {lp.item():.3e} | bc {lb.item():.3e}")

l2_mesh = l2_error(model_mesh, u_exact_p1)
print(f"\\nMesh PINN — Relative L2 error: {l2_mesh:.4%}")
"""))

# ── Compare mesh vs grid ───────────────────────────────────────────────────────
cells.append(code(
"""# ── Compare: grid sampler vs mesh sampler ──────────────────────────────────────
_, _, U_mesh = eval_model(model_mesh)
U_true_sq   = u_exact_p1(XX, YY)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
kw = dict(levels=50, cmap='hot')
axes[0].contourf(XX, YY, U_true_sq, **kw);       axes[0].set_title('Analytical')
axes[1].contourf(XX, YY, U_best, **kw);          axes[1].set_title(f'Grid PINN  (L2={l2_best:.3%})')
axes[2].contourf(XX, YY, U_mesh, **kw);          axes[2].set_title(f'Mesh PINN  (L2={l2_mesh:.3%})')
for ax in axes: ax.set_xlabel('x'); ax.set_ylabel('y')
plt.suptitle('§3 — Grid vs Mesh sampling (same problem)', fontsize=13)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(8,3))
ax.semilogy(hist_best['pde'],  label='Grid PINN (§2 best)', lw=2)
ax.semilogy(hist_mesh['pde'],  label='Mesh PINN', lw=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('PDE loss'); ax.legend()
ax.set_title('§3 — Convergence: grid vs mesh')
plt.tight_layout(); plt.show()
"""))

# ── Section 4 header ───────────────────────────────────────────────────────────
cells.append(md(r"""
---
## 4. Boundary Condition Variants

We keep the same domain and network, but change the BCs to explore how different
physical setups lead to qualitatively different temperature fields.

| Variant | BCs | Analytical |
|---------|-----|------------|
| **A** — Heated strip | $u = \sin(\pi x)$ on $y=0$, $u=0$ elsewhere | $\sin(\pi x)\sinh(\pi(1-y))/\sinh(\pi)$ |
| **B** — Insulated top (Neumann) | $u=\sin(\pi x)$ on $y=0$, $u=0$ on sides, $\partial u/\partial y=0$ on $y=1$ | $\sin(\pi x)\cosh(\pi(1-y))/\cosh(\pi)$ |
| **C** — Symmetric Neumann sides | $\partial u/\partial x=0$ on $x=0,1$; $u=0$ on $y=0$; $u=1$ on $y=1$ | $u=y$ |

Each variant has a clean analytical solution for benchmark comparison.
"""))

# ── Variant A ──────────────────────────────────────────────────────────────────
cells.append(code(
r"""# ══════════════════════════════════════════════════════════════════════════════
# Variant A — heated strip: u = sin(πx) on bottom, u=0 elsewhere
# Analytical: u = sin(πx) · sinh(π(1-y)) / sinh(π)
# ══════════════════════════════════════════════════════════════════════════════
def u_exact_A(x, y):
    return np.sin(np.pi * x) * np.sinh(np.pi * (1 - y)) / np.sinh(np.pi)

def bc_loss_A(model, n=300):
    losses = []
    # bottom: u = sin(πx)
    x = torch.rand(n); y = torch.zeros(n)
    target = torch.sin(torch.tensor(float(np.pi)) * x)
    losses.append((model(torch.stack([x,y],1)) - target).pow(2).mean())
    # left: u=0
    x = torch.zeros(n); y = torch.rand(n)
    losses.append(model(torch.stack([x,y],1)).pow(2).mean())
    # right: u=0
    x = torch.ones(n); y = torch.rand(n)
    losses.append(model(torch.stack([x,y],1)).pow(2).mean())
    # top: u=0
    x = torch.rand(n); y = torch.ones(n)
    losses.append(model(torch.stack([x,y],1)).pow(2).mean())
    return sum(losses) / len(losses)

torch.manual_seed(42)
model_A = PINN(hidden=[64,64,64,64])
hist_A  = train_pinn(model_A, laplace_residual, bc_loss_A,
                     n_epochs=5000, lbfgs_steps=200, verbose=True)
l2_A = l2_error(model_A, u_exact_A)
print(f"\nVariant A — L2 error: {l2_A:.4%}")

_, _, U_A   = eval_model(model_A)
U_true_A    = u_exact_A(XX, YY)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
kw = dict(levels=50, cmap='hot')
axes[0].contourf(XX, YY, U_true_A, **kw); axes[0].set_title('Analytical')
axes[1].contourf(XX, YY, U_A, **kw);      axes[1].set_title(f'PINN (L2={l2_A:.3%})')
axes[2].contourf(XX, YY, np.abs(U_A-U_true_A), levels=50, cmap='Reds'); axes[2].set_title('|Error|')
for ax in axes: ax.set_xlabel('x'); ax.set_ylabel('y')
plt.suptitle('§4 Variant A — u=sin(πx) on bottom', fontsize=13)
plt.tight_layout(); plt.show()
"""))

# ── Variant B ──────────────────────────────────────────────────────────────────
cells.append(code(
r"""# ══════════════════════════════════════════════════════════════════════════════
# Variant B — insulated top: ∂u/∂y = 0 at y=1
# Analytical: u = sin(πx) · cosh(π(1-y)) / cosh(π)
# ══════════════════════════════════════════════════════════════════════════════
def u_exact_B(x, y):
    return np.sin(np.pi * x) * np.cosh(np.pi * (1 - y)) / np.cosh(np.pi)

def neumann_top_loss(model, n=300):
    '''Enforce ∂u/∂y = 0 at y=1.'''
    x = torch.rand(n)
    y = torch.ones(n).requires_grad_(True)
    u = model(torch.stack([x, y], 1))
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    return u_y.pow(2).mean()

def bc_loss_B(model, n=300):
    losses = []
    # bottom: u = sin(πx)
    x = torch.rand(n); y = torch.zeros(n)
    losses.append((model(torch.stack([x,y],1)) - torch.sin(torch.tensor(float(np.pi))*x)).pow(2).mean())
    # left: u=0
    x = torch.zeros(n); y = torch.rand(n)
    losses.append(model(torch.stack([x,y],1)).pow(2).mean())
    # right: u=0
    x = torch.ones(n); y = torch.rand(n)
    losses.append(model(torch.stack([x,y],1)).pow(2).mean())
    # top: Neumann ∂u/∂y=0
    losses.append(neumann_top_loss(model, n))
    return sum(losses) / len(losses)

torch.manual_seed(42)
model_B = PINN(hidden=[64,64,64,64])
hist_B  = train_pinn(model_B, laplace_residual, bc_loss_B,
                     n_epochs=5000, lbfgs_steps=200, verbose=True)
l2_B = l2_error(model_B, u_exact_B)
print(f"\nVariant B — L2 error: {l2_B:.4%}")

_, _, U_B = eval_model(model_B)
U_true_B  = u_exact_B(XX, YY)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
kw = dict(levels=50, cmap='hot')
axes[0].contourf(XX, YY, U_true_B, **kw); axes[0].set_title('Analytical')
axes[1].contourf(XX, YY, U_B, **kw);      axes[1].set_title(f'PINN (L2={l2_B:.3%})')
axes[2].contourf(XX, YY, np.abs(U_B-U_true_B), levels=50, cmap='Reds'); axes[2].set_title('|Error|')
for ax in axes: ax.set_xlabel('x'); ax.set_ylabel('y')
plt.suptitle('§4 Variant B — Neumann (insulated) top', fontsize=13)
plt.tight_layout(); plt.show()
"""))

# ── Variant C ──────────────────────────────────────────────────────────────────
cells.append(code(
r"""# ══════════════════════════════════════════════════════════════════════════════
# Variant C — Neumann (insulated) sides: ∂u/∂x = 0 at x=0,1
# u=0 on bottom, u=1 on top.
# Analytical: u = y  (uniform linear gradient)
# Explanation: with ∂u/∂x=0 on sides, all sine modes vanish; only the
# n=0 cosine mode survives: u = y.
# ══════════════════════════════════════════════════════════════════════════════
def u_exact_C(x, y):
    return y   # trivially linear!

def neumann_sides_loss(model, n=300):
    '''Enforce ∂u/∂x = 0 at x=0 and x=1.'''
    y = torch.rand(n)
    # x=0
    x0 = torch.zeros(n).requires_grad_(True)
    u0 = model(torch.stack([x0, y], 1))
    u0_x = torch.autograd.grad(u0.sum(), x0, create_graph=True)[0]
    l0 = u0_x.pow(2).mean()
    # x=1
    x1 = torch.ones(n).requires_grad_(True)
    u1 = model(torch.stack([x1, y], 1))
    u1_x = torch.autograd.grad(u1.sum(), x1, create_graph=True)[0]
    l1 = u1_x.pow(2).mean()
    return (l0 + l1) / 2

def bc_loss_C(model, n=300):
    losses = []
    # bottom: u=0
    x = torch.rand(n); y = torch.zeros(n)
    losses.append(model(torch.stack([x,y],1)).pow(2).mean())
    # top: u=1
    x = torch.rand(n); y = torch.ones(n)
    losses.append((model(torch.stack([x,y],1)) - 1).pow(2).mean())
    # sides: Neumann
    losses.append(neumann_sides_loss(model, n))
    return sum(losses) / len(losses)

torch.manual_seed(42)
model_C = PINN(hidden=[64,64,64,64])
hist_C  = train_pinn(model_C, laplace_residual, bc_loss_C,
                     n_epochs=5000, lbfgs_steps=200, verbose=True)
l2_C = l2_error(model_C, lambda x, y: np.ones_like(x) * y)
print(f"\nVariant C — L2 error: {l2_C:.4%}")

_, _, U_C = eval_model(model_C)
U_true_C  = YY  # u=y
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
kw = dict(levels=50, cmap='hot')
axes[0].contourf(XX, YY, U_true_C, **kw); axes[0].set_title('Analytical (u = y)')
axes[1].contourf(XX, YY, U_C, **kw);      axes[1].set_title(f'PINN (L2={l2_C:.3%})')
axes[2].contourf(XX, YY, np.abs(U_C-U_true_C), levels=50, cmap='Reds'); axes[2].set_title('|Error|')
for ax in axes: ax.set_xlabel('x'); ax.set_ylabel('y')
plt.suptitle('§4 Variant C — Neumann sides (du/dx=0), u=y exact', fontsize=13)
plt.tight_layout(); plt.show()

# Side-by-side summary
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (U, title) in zip(axes, [(U_A,'A: hot strip'),(U_B,'B: insulated top'),(U_C,'C: Neumann sides')]):
    c = ax.contourf(XX, YY, U, levels=50, cmap='hot')
    ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(c, ax=ax)
plt.suptitle('§4 — BC Variants side-by-side (PINN predictions)', fontsize=13)
plt.tight_layout(); plt.show()
"""))

# ── Section 5 header ───────────────────────────────────────────────────────────
cells.append(md(r"""
---
## 5. L-Shaped Domain

The L-shape is the canonical test case for PDE solvers on non-convex domains.
A **270° re-entrant corner** at $(0.5, 0.5)$ creates a stress concentration where
the solution gradient becomes singular — challenging for mesh-free methods.

**Problem:** Laplace equation on the L-shape, zero Dirichlet on all boundaries,
with a source-like BC: $u=\sin(\pi x)$ applied on the bottom edge.

Since there is no simple closed-form, we use a **finite-difference reference** on
a fine grid for comparison.
"""))

# ── L-shape mesh ───────────────────────────────────────────────────────────────
cells.append(code(
"""# ── L-shape mesh ───────────────────────────────────────────────────────────────
#
#  (0,1)──────(0.5,1)
#    |           |
#    |           |
#  (0,.5)──(.5,.5)
#    |     |
#    |     |
#  (0,0)──(1,0)──(1,0.5)
#
# Define two rectangles, each split into triangles:
# Bottom rectangle [0,1]x[0,0.5] and Left rectangle [0,0.5]x[0.5,1]
L_verts = np.array([
    [0.0, 0.0],  # 0
    [1.0, 0.0],  # 1
    [1.0, 0.5],  # 2
    [0.5, 0.5],  # 3
    [0.0, 0.5],  # 4
    [0.0, 1.0],  # 5
    [0.5, 1.0],  # 6
], dtype=np.float32)

L_tris = np.array([
    # Bottom rectangle (y in [0, 0.5])
    [0, 1, 2],
    [0, 2, 3],
    [0, 3, 4],
    # Left rectangle (x in [0, 0.5], y in [0.5, 1])
    [4, 3, 6],
    [4, 6, 5],
], dtype=np.int32)

# Visualise mesh
plt.figure(figsize=(4, 4))
plt.triplot(L_verts[:,0], L_verts[:,1], L_tris, color='steelblue')
plt.scatter(L_verts[:,0], L_verts[:,1], c='red', zorder=5)
for i,(x,y) in enumerate(L_verts):
    plt.annotate(str(i), (x,y), textcoords='offset points', xytext=(5,5))
plt.title('L-shape mesh'); plt.axis('equal'); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# Verify mesh area = 0.75 (L-shape = 3 quadrants)
areas = triangle_areas(L_verts, L_tris)
print(f"Mesh area: {areas.sum():.4f}  (expected 0.75)")
"""))

# ── FD reference ───────────────────────────────────────────────────────────────
cells.append(code(
"""# ── Finite-difference reference solution ───────────────────────────────────────
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def fd_laplace_L(N=100):
    '''
    Finite-difference solution of Laplace on L-shape.
    Grid: uniform NxN on [0,1]^2, interior of L-shape only.
    BCs: u=sin(pi*x) on bottom edge, u=0 elsewhere.
    '''
    h  = 1.0 / N
    xs = np.linspace(0, 1, N+1)
    ys = np.linspace(0, 1, N+1)

    def in_L(i, j):  # (i=col->x, j=row->y)
        x = xs[i]; y = ys[j]
        return (x <= 0.5 or y <= 0.5) and (0 <= x <= 1) and (0 <= y <= 1)

    def on_boundary(i, j):
        x = xs[i]; y = ys[j]
        if i == 0 or j == 0 or (i == N and y <= 0.5) or (j == N and x <= 0.5):
            return True
        if abs(x - 0.5) < 1e-12 and y >= 0.5 - 1e-12:
            return True
        if abs(y - 0.5) < 1e-12 and x >= 0.5 - 1e-12:
            return True
        if i == N or j == N:
            return True
        return False

    def bc_val(i, j):
        x = xs[i]; y = ys[j]
        if j == 0:  # bottom
            return np.sin(np.pi * x)
        return 0.0

    # Index only L-shape interior nodes
    idx = {}
    interior = []
    for j in range(N+1):
        for i in range(N+1):
            if in_L(i,j) and not on_boundary(i,j):
                idx[(i,j)] = len(interior)
                interior.append((i,j))

    n_int = len(interior)
    A = lil_matrix((n_int, n_int))
    b = np.zeros(n_int)

    for k, (i,j) in enumerate(interior):
        A[k, k] = 4.0
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            ni, nj = i+di, j+dj
            if in_L(ni, nj):
                if on_boundary(ni, nj):
                    b[k] += bc_val(ni, nj)
                else:
                    A[k, idx[(ni,nj)]] = -1.0

    u_vals = spsolve(A.tocsr(), b)

    # Build full grid (NaN outside L)
    U = np.full((N+1, N+1), np.nan)
    for k, (i,j) in enumerate(interior):
        U[j, i] = u_vals[k]
    for j in range(N+1):
        for i in range(N+1):
            if in_L(i,j) and on_boundary(i,j):
                U[j,i] = bc_val(i,j)

    return xs, ys, U

print("Computing FD reference (may take a few seconds)...")
xs_fd, ys_fd, U_fd = fd_laplace_L(N=80)
print("Done.")

plt.figure(figsize=(5, 5))
plt.contourf(xs_fd, ys_fd, U_fd, levels=50, cmap='hot')
plt.colorbar(label='u')
plt.title('FD reference — L-shape Laplace')
plt.xlabel('x'); plt.ylabel('y')
plt.tight_layout(); plt.show()
"""))

# ── Train PINN on L-shape ──────────────────────────────────────────────────────
cells.append(code(
r"""# ── BC loss for L-shape ────────────────────────────────────────────────────────
def bc_loss_L(model, n=300):
    '''
    u = sin(pi*x) on bottom (y=0, 0<=x<=1)
    u = 0 on all other boundary segments.
    '''
    losses = []

    # Bottom (y=0): u = sin(pi*x)
    x = torch.rand(n); y = torch.zeros(n)
    losses.append((model(torch.stack([x,y],1)) - torch.sin(torch.tensor(float(np.pi))*x)).pow(2).mean())

    # Left edge (x=0, 0<=y<=1)
    x = torch.zeros(n); y = torch.rand(n)
    losses.append(model(torch.stack([x,y],1)).pow(2).mean())

    # Right edge (x=1, 0<=y<=0.5)
    x = torch.ones(n); y = torch.rand(n) * 0.5
    losses.append(model(torch.stack([x,y],1)).pow(2).mean())

    # Bottom of notch (y=0.5, 0.5<=x<=1)
    x = torch.rand(n)*0.5 + 0.5; y = torch.ones(n)*0.5
    losses.append(model(torch.stack([x,y],1)).pow(2).mean())

    # Right of notch (x=0.5, 0.5<=y<=1)
    x = torch.ones(n)*0.5; y = torch.rand(n)*0.5 + 0.5
    losses.append(model(torch.stack([x,y],1)).pow(2).mean())

    # Top edge (y=1, 0<=x<=0.5)
    x = torch.rand(n)*0.5; y = torch.ones(n)
    losses.append(model(torch.stack([x,y],1)).pow(2).mean())

    return sum(losses) / len(losses)


def laplace_residual_L(model, n=2000):
    x, y = sample_mesh_interior(L_verts, L_tris, n)
    return laplace_residual(model, x, y)


# ── Train ───────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
model_L = PINN(hidden=[64, 64, 64, 64])
opt_L   = torch.optim.Adam(model_L.parameters(), lr=1e-3)
sched_L = torch.optim.lr_scheduler.CosineAnnealingLR(opt_L, T_max=6000)

hist_L  = {'pde': [], 'bc': [], 'total': []}
EPOCHS_L = 6000

for ep in range(1, EPOCHS_L+1):
    opt_L.zero_grad()
    lp   = laplace_residual_L(model_L)
    lb   = bc_loss_L(model_L)
    loss = 1.0*lp + 20.0*lb
    loss.backward(); opt_L.step(); sched_L.step()
    hist_L['pde'].append(lp.item())
    hist_L['bc'].append(lb.item())
    hist_L['total'].append(loss.item())
    if ep % 1000 == 0:
        print(f"Epoch {ep:5d} | pde {lp.item():.3e} | bc {lb.item():.3e}")

# L-BFGS fine-tune
opt_lbfgs_L = torch.optim.LBFGS(model_L.parameters(), lr=0.5, max_iter=20,
                                  history_size=50, line_search_fn='strong_wolfe')
for step in range(300):
    def closure_L():
        opt_lbfgs_L.zero_grad()
        loss = 1.0*laplace_residual_L(model_L) + 20.0*bc_loss_L(model_L)
        loss.backward(); return loss
    opt_lbfgs_L.step(closure_L)

print("L-BFGS fine-tuning done.")
"""))

# ── Evaluate L-shape ───────────────────────────────────────────────────────────
cells.append(code(
"""# ── Evaluate PINN on L-shape ────────────────────────────────────────────────────
N_ev = 120
x1d_ev = np.linspace(0, 1, N_ev)
y1d_ev = np.linspace(0, 1, N_ev)
XX_ev, YY_ev = np.meshgrid(x1d_ev, y1d_ev)

def in_L_np(x, y):
    return (x <= 0.5) | (y <= 0.5)

mask = in_L_np(XX_ev, YY_ev)

xy_ev = torch.tensor(np.stack([XX_ev.ravel(), YY_ev.ravel()], 1), dtype=torch.float32)
with torch.no_grad():
    U_pinn_L = model_L(xy_ev).numpy().reshape(N_ev, N_ev)
U_pinn_L[~mask] = np.nan

# Interpolate FD solution to same grid for comparison
from scipy.interpolate import RegularGridInterpolator
fd_interp = RegularGridInterpolator((ys_fd, xs_fd), np.nan_to_num(U_fd, nan=0.0),
                                     method='linear', bounds_error=False, fill_value=np.nan)
pts = np.stack([YY_ev.ravel(), XX_ev.ravel()], axis=1)
U_fd_ev = fd_interp(pts).reshape(N_ev, N_ev)
U_fd_ev[~mask] = np.nan

err_L = np.abs(U_pinn_L - U_fd_ev)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
kw = dict(levels=40, cmap='hot')
im0 = axes[0].contourf(XX_ev, YY_ev, U_fd_ev,   **kw); axes[0].set_title('FD Reference')
im1 = axes[1].contourf(XX_ev, YY_ev, U_pinn_L,  **kw); axes[1].set_title('PINN')
im2 = axes[2].contourf(XX_ev, YY_ev, err_L, levels=40, cmap='Reds'); axes[2].set_title('|Error|')
for ax in axes: ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
fig.colorbar(im0, ax=axes[1]); fig.colorbar(im2, ax=axes[2])
plt.suptitle('§5 — L-shaped domain: FD reference vs PINN', fontsize=13)
plt.tight_layout(); plt.show()

# Error stats (inside L only)
valid = mask & ~np.isnan(U_fd_ev) & ~np.isnan(U_pinn_L)
mse_L = np.sqrt(((U_pinn_L[valid] - U_fd_ev[valid])**2).mean())
print(f"RMSE vs FD reference: {mse_L:.5f}")
"""))

# ── Section 6 header ───────────────────────────────────────────────────────────
cells.append(md(r"""
---
## 6. Transient Heat Equation

We now add time, solving:

$$\frac{\partial u}{\partial t} = \alpha \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right), \quad (x,y)\in[0,1]^2,\; t\in[0,T]$$

**Setup:**

| | |
|---|---|
| $\alpha$ | 1.0 (thermal diffusivity) |
| IC | $u(x,y,0) = \sin(\pi x)\sin(\pi y)$ |
| BC | $u=0$ on all boundaries |
| $T$ | 0.5 |

**Analytical solution** (separation of variables):

$$u(x,y,t) = e^{-2\pi^2 \alpha t}\,\sin(\pi x)\,\sin(\pi y)$$

This decays exponentially in time — a clean benchmark.

**Network:** $\hat u_\theta(x,y,t)$ — 3 inputs now, same MLP architecture.
"""))

# ── Analytical transient solution ──────────────────────────────────────────────
cells.append(code(
"""ALPHA = 1.0
T_END = 0.5

def u_exact_transient(x, y, t, alpha=ALPHA):
    '''Exact solution: exp(-2*pi^2*alpha*t)*sin(pi*x)*sin(pi*y).'''
    return np.exp(-2 * np.pi**2 * alpha * t) * np.sin(np.pi * x) * np.sin(np.pi * y)

# Visualise at a few time slices
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
x1d = np.linspace(0, 1, 80); y1d = np.linspace(0, 1, 80)
XX2, YY2 = np.meshgrid(x1d, y1d)
for ax, t_val in zip(axes, [0.0, 0.1, 0.3, 0.5]):
    U_t = u_exact_transient(XX2, YY2, t_val)
    c = ax.contourf(XX2, YY2, U_t, levels=30, cmap='hot', vmin=0, vmax=1)
    ax.set_title(f't = {t_val}'); ax.set_xlabel('x'); ax.set_ylabel('y')
plt.suptitle('§6 — Analytical transient solution', fontsize=13)
plt.tight_layout(); plt.show()
"""))

# ── Transient PINN losses ──────────────────────────────────────────────────────
cells.append(code(
"""# ── Transient PDE loss ─────────────────────────────────────────────────────────
def heat_residual(model, x, y, t, alpha=ALPHA):
    '''Residual of u_t - alpha*(u_xx + u_yy) at interior space-time points.'''
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    xyt = torch.stack([x, y, t], dim=1)
    u = model(xyt)
    u_t  = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x  = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_y  = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    residual = u_t - alpha * (u_xx + u_yy)
    return residual.pow(2).mean()

def ic_loss(model, n=500):
    '''IC: u(x,y,0) = sin(pi*x)*sin(pi*y).'''
    x = torch.rand(n); y = torch.rand(n); t = torch.zeros(n)
    pred = model(torch.stack([x, y, t], 1))
    pi_t = torch.tensor(float(np.pi))
    target = torch.sin(pi_t*x) * torch.sin(pi_t*y)
    return (pred - target).pow(2).mean()

def bc_loss_transient(model, n=200):
    '''u=0 on all 4 walls for all t.'''
    losses = []
    t = torch.rand(n) * T_END
    for xi, yi in [(torch.zeros(n), torch.rand(n)),   # left
                   (torch.ones(n),  torch.rand(n)),   # right
                   (torch.rand(n),  torch.zeros(n)),  # bottom
                   (torch.rand(n),  torch.ones(n))]:  # top
        losses.append(model(torch.stack([xi, yi, t], 1)).pow(2).mean())
    return sum(losses) / len(losses)
"""))

# ── Train transient PINN ───────────────────────────────────────────────────────
cells.append(code(
"""# ── Train transient PINN ────────────────────────────────────────────────────────
torch.manual_seed(42)
model_T = PINN(n_in=3, hidden=[64, 64, 64, 64])   # 3 inputs: x, y, t

# Phase 1: Adam
opt_T   = torch.optim.Adam(model_T.parameters(), lr=1e-3)
sched_T = torch.optim.lr_scheduler.CosineAnnealingLR(opt_T, T_max=6000)
hist_T  = {'pde': [], 'ic': [], 'bc': [], 'total': []}
W_PDE_T, W_IC_T, W_BC_T = 1.0, 20.0, 20.0
EPOCHS_T = 6000

for ep in range(1, EPOCHS_T+1):
    opt_T.zero_grad()
    x_i = torch.rand(2000); y_i = torch.rand(2000); t_i = torch.rand(2000)*T_END
    lp = heat_residual(model_T, x_i, y_i, t_i)
    li = ic_loss(model_T, 500)
    lb = bc_loss_transient(model_T, 200)
    loss = W_PDE_T*lp + W_IC_T*li + W_BC_T*lb
    loss.backward(); opt_T.step(); sched_T.step()
    hist_T['pde'].append(lp.item()); hist_T['ic'].append(li.item())
    hist_T['bc'].append(lb.item()); hist_T['total'].append(loss.item())
    if ep % 1000 == 0:
        print(f"Epoch {ep:5d} | pde {lp.item():.3e} | ic {li.item():.3e} | bc {lb.item():.3e}")

# Phase 2: L-BFGS
opt_lbfgs_T = torch.optim.LBFGS(model_T.parameters(), lr=0.5, max_iter=20,
                                  history_size=50, line_search_fn='strong_wolfe')
for step in range(300):
    def closure_T():
        opt_lbfgs_T.zero_grad()
        x_i = torch.rand(2000); y_i = torch.rand(2000); t_i = torch.rand(2000)*T_END
        lp = heat_residual(model_T, x_i, y_i, t_i)
        li = ic_loss(model_T, 500)
        lb = bc_loss_transient(model_T, 200)
        loss = W_PDE_T*lp + W_IC_T*li + W_BC_T*lb
        loss.backward(); return loss
    opt_lbfgs_T.step(closure_T)
    if (step+1) % 100 == 0:
        with torch.no_grad():
            x_i=torch.rand(500); y_i=torch.rand(500); t_i=torch.rand(500)*T_END
            lp=heat_residual(model_T,x_i,y_i,t_i)
        print(f"  L-BFGS {step+1:4d} | pde {lp.item():.3e}")

print("\\nTraining complete.")
"""))

# ── Evaluate transient PINN ────────────────────────────────────────────────────
cells.append(code(
"""# ── Evaluate at time snapshots ─────────────────────────────────────────────────
def eval_transient(model, t_val, nx=80):
    x1d = np.linspace(0, 1, nx); y1d = np.linspace(0, 1, nx)
    XX_t, YY_t = np.meshgrid(x1d, y1d)
    t_arr = np.full(nx*nx, t_val, dtype=np.float32)
    xyt = torch.tensor(np.stack([XX_t.ravel(), YY_t.ravel(), t_arr], 1), dtype=torch.float32)
    with torch.no_grad():
        U_p = model(xyt).numpy().reshape(nx, nx)
    U_e = u_exact_transient(XX_t, YY_t, t_val)
    return XX_t, YY_t, U_p, U_e

t_snaps = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
fig, axes = plt.subplots(3, len(t_snaps), figsize=(18, 9))
errs = {}
for col, t_val in enumerate(t_snaps):
    XX_t, YY_t, U_p, U_e = eval_transient(model_T, t_val)
    err = np.abs(U_p - U_e)
    errs[t_val] = np.sqrt(((U_p-U_e)**2).mean()) / (np.sqrt((U_e**2).mean()) + 1e-12)
    kw = dict(levels=30, cmap='hot', vmin=0, vmax=1)
    axes[0,col].contourf(XX_t, YY_t, U_e, **kw); axes[0,col].set_title(f't={t_val}')
    axes[1,col].contourf(XX_t, YY_t, U_p, **kw)
    axes[2,col].contourf(XX_t, YY_t, err, levels=30, cmap='Reds')
    for ax in axes[:, col]: ax.set_xticks([]); ax.set_yticks([])

axes[0,0].set_ylabel('Analytical', fontsize=10)
axes[1,0].set_ylabel('PINN',       fontsize=10)
axes[2,0].set_ylabel('|Error|',    fontsize=10)
plt.suptitle('§6 — Transient PINN: snapshots', fontsize=13)
plt.tight_layout(); plt.show()

print("\\nRelative L2 error by time snapshot:")
for t_val, e in errs.items():
    bar = chr(0x2588) * int(e * 500)
    print(f"  t={t_val:.2f}  {e:.4%}  {bar}")

# Loss curves
fig, ax = plt.subplots(figsize=(9, 3))
ax.semilogy(hist_T['pde'],   label='PDE loss')
ax.semilogy(hist_T['ic'],    label='IC loss')
ax.semilogy(hist_T['bc'],    label='BC loss')
ax.semilogy(hist_T['total'], label='Total', lw=2)
ax.set_xlabel('Epoch'); ax.legend(ncol=4)
ax.set_title('§6 — Transient training history')
plt.tight_layout(); plt.show()
"""))

# ── Write notebook ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}
with open('/Users/zeyadzaky/Documents/Development/vinci/pinn_heat_v2.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print("Written pinn_heat_v2.ipynb")
