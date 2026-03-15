import json, textwrap

def src(s):
    s = textwrap.dedent(s).lstrip('\n').rstrip() + '\n'
    return s.splitlines(True)

with open('/Users/zeyadzaky/Documents/Development/vinci/pinn_heat_v2.ipynb') as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell 5: Add data_loss + sample_data_p1 at the bottom ─────────────────────
cells[5]['source'] = src("""
# ── Network ────────────────────────────────────────────────────────────────────
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


# ── Data loss (sparse interior observations) ───────────────────────────────────
def sample_data_p1(n=50, seed=7):
    '''
    Draw n random interior points and label them with the analytical solution.
    Simulates sparse sensor measurements inside the domain.
    '''
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.05, 0.95, n).astype(np.float32)
    y = rng.uniform(0.05, 0.95, n).astype(np.float32)
    u = u_exact_p1(x, y).astype(np.float32)
    return (torch.tensor(x), torch.tensor(y), torch.tensor(u))

def data_loss(model, x_d, y_d, u_d):
    '''MSE between network prediction and known values at interior data points.'''
    pred = model(torch.stack([x_d, y_d], dim=1))
    return (pred - u_d).pow(2).mean()


# ── Generate fixed data points (used throughout §1) ───────────────────────────
X_DATA, Y_DATA, U_DATA = sample_data_p1(n=50)
print(f"Data points: {len(X_DATA)}  |  u range [{U_DATA.min():.3f}, {U_DATA.max():.3f}]")
""")

# ── Cell 6: Training loop with data loss ──────────────────────────────────────
cells[6]['source'] = src("""
# ── Training ───────────────────────────────────────────────────────────────────
model_simple = SimplePINN()
optimizer    = torch.optim.Adam(model_simple.parameters(), lr=1e-3)

W_PDE, W_BC, W_DATA = 1.0, 20.0, 10.0
N_PDE, N_BC         = 2000, 400
EPOCHS              = 5000

hist_simple = {'pde': [], 'bc': [], 'data': [], 'total': []}
t0 = time.time()

for ep in range(1, EPOCHS + 1):
    optimizer.zero_grad()

    x_i = torch.rand(N_PDE)
    y_i = torch.rand(N_PDE)

    lp = laplace_residual(model_simple, x_i, y_i)
    lb = bc_loss_p1(model_simple, N_BC)
    ld = data_loss(model_simple, X_DATA, Y_DATA, U_DATA)

    loss = W_PDE * lp + W_BC * lb + W_DATA * ld
    loss.backward()
    optimizer.step()

    hist_simple['pde'].append(lp.item())
    hist_simple['bc'].append(lb.item())
    hist_simple['data'].append(ld.item())
    hist_simple['total'].append(loss.item())

    if ep % 1000 == 0:
        print(f"Epoch {ep:5d} | total {loss.item():.3e} | "
              f"pde {lp.item():.3e} | bc {lb.item():.3e} | data {ld.item():.3e}")

print(f"\\nTraining time: {time.time()-t0:.1f}s")
""")

# ── Cell 7: Plots — show data points on the error map ─────────────────────────
cells[7]['source'] = src("""
# ── Evaluation ─────────────────────────────────────────────────────────────────
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
err    = np.abs(U_pred_simple - U_true)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
kw = dict(vmin=0, vmax=1, cmap='hot', levels=50)
c0 = axes[0].contourf(XX, YY, U_true,        **kw); axes[0].set_title('Analytical')
c1 = axes[1].contourf(XX, YY, U_pred_simple, **kw); axes[1].set_title('PINN')
c2 = axes[2].contourf(XX, YY, err, levels=50, cmap='Reds'); axes[2].set_title('|Error|')

# Overlay data points on error map
axes[2].scatter(X_DATA.numpy(), Y_DATA.numpy(), c='blue', s=15,
                zorder=5, label=f'{len(X_DATA)} data pts')
axes[2].legend(loc='lower right', fontsize=8)

for ax in axes: ax.set_xlabel('x'); ax.set_ylabel('y')
fig.colorbar(c0, ax=axes[1]); fig.colorbar(c2, ax=axes[2])
plt.suptitle('§1 — Simple PINN vs Analytical', fontsize=13)
plt.tight_layout(); plt.show()

l2_rel = np.sqrt(((U_pred_simple-U_true)**2).mean()) / np.sqrt((U_true**2).mean())
print(f"Relative L2 error: {l2_rel:.4%}")

# Loss curves
fig, ax = plt.subplots(figsize=(9, 3))
ax.semilogy(hist_simple['pde'],   label='PDE loss')
ax.semilogy(hist_simple['bc'],    label='BC loss')
ax.semilogy(hist_simple['data'],  label='Data loss')
ax.semilogy(hist_simple['total'], label='Total (weighted)', lw=2)
ax.set_xlabel('Epoch'); ax.legend(); ax.set_title('§1 — Training history')
plt.tight_layout(); plt.show()
""")

# ── Cell 9: Update configurable PINN + train_pinn to include data loss ─────────
cells[9]['source'] = src("""
# ── Configurable PINN ──────────────────────────────────────────────────────────
class PINN(nn.Module):
    '''
    Fully-connected PINN for 2D steady-state problems.

    Parameters
    ----------
    n_in   : input dimension (2 for steady, 3 for transient)
    hidden : list of hidden layer widths
    act    : activation class (nn.Tanh, nn.GELU, ...)
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
               data_pts=None, w_data=10.0,
               lbfgs_steps=0, verbose=False):
    '''
    Train with Adam (and optionally L-BFGS fine-tuning).

    data_pts : tuple (x_d, y_d, u_d) of known interior observations, or None.
    Returns history dict with keys pde, bc, data (if used), total.
    '''
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    use_data = data_pts is not None
    hist = {'pde': [], 'bc': [], 'total': []}
    if use_data:
        hist['data'] = []

    for ep in range(1, n_epochs + 1):
        opt.zero_grad()
        x_i, y_i = torch.rand(n_pde), torch.rand(n_pde)
        lp = pde_loss_fn(model, x_i, y_i)
        lb = bc_loss_fn(model, n_bc)
        loss = w_pde * lp + w_bc * lb
        if use_data:
            ld = data_loss(model, *data_pts)
            loss = loss + w_data * ld
            hist['data'].append(ld.item())
        loss.backward()
        opt.step()
        hist['pde'].append(lp.item())
        hist['bc'].append(lb.item())
        hist['total'].append(loss.item())
        if verbose and ep % 1000 == 0:
            data_str = f" | data {ld.item():.3e}" if use_data else ""
            print(f"  Adam ep {ep:5d} | pde {lp.item():.3e} | bc {lb.item():.3e}{data_str}")

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
                if use_data:
                    loss = loss + w_data * data_loss(model, *data_pts)
                loss.backward()
                return loss
            opt_lbfgs.step(closure)
            with torch.no_grad():
                x_i, y_i = torch.rand(n_pde), torch.rand(n_pde)
                lp = pde_loss_fn(model, x_i, y_i)
                lb = bc_loss_fn(model, n_bc)
                ld = data_loss(model, *data_pts) if use_data else torch.tensor(0.0)
            hist['pde'].append(lp.item())
            hist['bc'].append(lb.item())
            hist['total'].append((w_pde*lp + w_bc*lb + (w_data*ld if use_data else 0)).item())
            if use_data: hist['data'].append(ld.item())
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


# ── Fixed data points for §2 experiments (same 50 points as §1) ───────────────
# X_DATA, Y_DATA, U_DATA are already defined in §1.
print("Configurable PINN and train_pinn ready.")
print(f"Data points available: {len(X_DATA)}")
""")

# ── Insert a new markdown cell before cell 10 explaining data in §2 ─────────
# We'll insert after cell 9 (index 9), shifting later cells
data_md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": src("""
### About the data loss in these experiments

All three experiments below use the **same 50 interior observation points** (`X_DATA`, `Y_DATA`, `U_DATA`)
defined in §1. The total loss is:

$$\\mathcal{L} = w_{\\text{pde}}\\,\\mathcal{L}_{\\text{pde}} + w_{\\text{bc}}\\,\\mathcal{L}_{\\text{bc}} + w_{\\text{data}}\\,\\mathcal{L}_{\\text{data}}$$

with $w_{\\text{pde}}=1$, $w_{\\text{bc}}=20$, $w_{\\text{data}}=10$.
""")
}
cells.insert(10, data_md_cell)
# Now cell indices shift by 1 for cells that were 10, 11, 12, 13, 14

# ── Cell 11 (was 10): Optimizer comparison — add data_pts argument ─────────────
cells[11]['source'] = src("""
# ── Experiment 1: Optimizers ───────────────────────────────────────────────────
torch.manual_seed(0)

experiments_opt = {
    'Adam':         {},
    'Adam->L-BFGS': {'lbfgs_steps': 300},
    'RMSprop':      None,   # handled separately
}

results_opt = {}
EPOCHS_OPT  = 3000
DATA_PTS    = (X_DATA, Y_DATA, U_DATA)

for key in experiments_opt:
    torch.manual_seed(0)
    m = PINN(hidden=[64, 64, 64])

    if key == 'RMSprop':
        opt_r = torch.optim.RMSprop(m.parameters(), lr=1e-3)
        h = {'pde': [], 'bc': [], 'data': [], 'total': []}
        for ep in range(EPOCHS_OPT):
            opt_r.zero_grad()
            x_i, y_i = torch.rand(2000), torch.rand(2000)
            lp = laplace_residual(m, x_i, y_i)
            lb = bc_loss_p1(m, 400)
            ld = data_loss(m, *DATA_PTS)
            loss = 1.0*lp + 20.0*lb + 10.0*ld
            loss.backward(); opt_r.step()
            h['pde'].append(lp.item()); h['bc'].append(lb.item())
            h['data'].append(ld.item()); h['total'].append(loss.item())
    else:
        extra = experiments_opt[key] or {}
        h = train_pinn(m, laplace_residual, bc_loss_p1,
                       n_epochs=EPOCHS_OPT, lr=1e-3,
                       data_pts=DATA_PTS, **extra)

    err = l2_error(m, u_exact_p1)
    results_opt[key] = {'hist': h, 'l2': err}
    print(f"{key:15s}  PDE {h['pde'][-1]:.3e}  data {h['data'][-1]:.3e}  L2 {err:.4%}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for key, res in results_opt.items():
    axes[0].semilogy(res['hist']['pde'],   label=key)
    axes[1].semilogy(res['hist']['data'],  label=key)
    axes[2].semilogy(res['hist']['total'], label=key)
axes[0].set_title('PDE loss'); axes[1].set_title('Data loss'); axes[2].set_title('Total loss')
for ax in axes: ax.set_xlabel('Epoch'); ax.legend()
plt.suptitle('§2 — Optimizer comparison', fontsize=13)
plt.tight_layout(); plt.show()
""")

# ── Cell 12 (was 11): Architecture comparison — add data_pts ──────────────────
cells[12]['source'] = src("""
# ── Experiment 2: Depth & Width ────────────────────────────────────────────────
torch.manual_seed(0)
DATA_PTS = (X_DATA, Y_DATA, U_DATA)

arch_configs = {
    '2x32':  [32, 32],
    '3x64':  [64, 64, 64],
    '4x64':  [64, 64, 64, 64],
    '4x128': [128, 128, 128, 128],
    '6x64':  [64]*6,
}

results_arch = {}
for key, hidden in arch_configs.items():
    torch.manual_seed(0)
    m = PINN(hidden=hidden)
    h = train_pinn(m, laplace_residual, bc_loss_p1,
                   n_epochs=3000, lr=1e-3, data_pts=DATA_PTS)
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
""")

# ── Cell 13 (was 12): Activation comparison — add data_pts ────────────────────
cells[13]['source'] = src("""
# ── Experiment 3: Activations ──────────────────────────────────────────────────
torch.manual_seed(0)
DATA_PTS = (X_DATA, Y_DATA, U_DATA)

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
    h = train_pinn(m, laplace_residual, bc_loss_p1,
                   n_epochs=3000, lr=1e-3, data_pts=DATA_PTS)
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
""")

# ── Cell 15 (was 14): Best config — add data_pts ──────────────────────────────
cells[15]['source'] = src("""
# ── Best config: Adam -> L-BFGS ────────────────────────────────────────────────
torch.manual_seed(42)
model_best = PINN(hidden=[64, 64, 64, 64])
hist_best  = train_pinn(model_best, laplace_residual, bc_loss_p1,
                        n_epochs=5000, lr=1e-3,
                        data_pts=(X_DATA, Y_DATA, U_DATA),
                        lbfgs_steps=300, verbose=True)

_, _, U_best = eval_model(model_best)
err_best = np.abs(U_best - u_exact_p1(XX, YY))
l2_best  = l2_error(model_best, u_exact_p1)
print(f"\\nBest config — Relative L2 error: {l2_best:.4%}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
kw = dict(levels=50, cmap='hot')
axes[0].contourf(XX, YY, u_exact_p1(XX, YY), **kw); axes[0].set_title('Analytical')
axes[1].contourf(XX, YY, U_best, **kw);             axes[1].set_title('PINN (best config)')
c2 = axes[2].contourf(XX, YY, err_best, levels=50, cmap='Reds'); axes[2].set_title('|Error|')
axes[2].scatter(X_DATA.numpy(), Y_DATA.numpy(), c='blue', s=15, zorder=5, label='data pts')
axes[2].legend(loc='lower right', fontsize=8)
for ax in axes: ax.set_xlabel('x'); ax.set_ylabel('y')
plt.suptitle(f'§2 — Best config (L2 err = {l2_best:.3%})', fontsize=13)
plt.tight_layout(); plt.show()
""")

nb['cells'] = cells

with open('/Users/zeyadzaky/Documents/Development/vinci/pinn_heat_v2.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Done. Total cells: {len(nb['cells'])}")

# Verify the patched cells look right
for i in [5, 6, 7, 9, 10, 11, 12, 13, 15]:
    src_preview = ''.join(nb['cells'][i]['source'])[:80].replace('\n', ' ')
    print(f"  Cell {i:2d} [{nb['cells'][i]['cell_type'][:4]}]: {src_preview}")
