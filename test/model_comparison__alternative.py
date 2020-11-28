import os
import sys
import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import pandas as pd
import seaborn as sns

SCRIPT_NAME = "model_comparison__alternative"


class Model:

    def __init__(self, p_obs,
                 design,
                 bounds_grid, n_grid):

        self.p_obs = p_obs
        n_param = len(bounds_grid)
        if n_param == 1:
            self.grid = \
                np.atleast_2d(np.linspace(*bounds_grid, n_grid)).T
        else:
            methods = (np.linspace for _ in range(n_param))
            self.grid = np.array(list(itertools.product(
                *(m(*b, n_grid) for m, b in zip(methods, bounds_grid)))))

        # Log-lik
        self.n_param_set, n_param = self.grid.shape
        n_design = len(design)
        p_one = np.zeros((n_design, self.n_param_set))
        for i, d in enumerate(design):
            p_one[i, :] = p_obs(d, self.grid.T).flatten()
        p_zero = 1 - p_one
        p = np.zeros((n_design, self.n_param_set, 2))
        p[:, :, 0] = p_zero
        p[:, :, 1] = p_one
        self.log_lik = np.log(p + np.finfo(np.float).eps)

        # Prior
        lp = np.ones(self.n_param_set)
        self.lp = lp - logsumexp(lp)

    def update(self, d_idx, resp):

        log_lik_r = self.log_lik[d_idx, :, int(resp)].flatten()

        self.lp += log_lik_r
        self.lp -= logsumexp(self.lp)

    def get_estimates(self):
        ep = np.dot(np.exp(self.lp), self.grid)

        delta = self.grid - ep
        post_cov = np.dot(delta.T, delta * np.exp(self.lp).reshape(-1, 1))
        sdp = np.sqrt(np.diag(post_cov))

        return ep, sdp

    def reset(self):
        lp = np.ones(self.n_param_set)
        self.lp = lp - logsumexp(lp)


class ModelComparison:

    def __init__(self, p_obs,
                 design,
                 bounds_grid, n_grid):

        self.n_model = len(p_obs)
        self.m = []
        for i in range(self.n_model):
            self.m.append(
                Model(
                    p_obs=p_obs[i],
                    design=design,
                    bounds_grid=bounds_grid[i],
                    n_grid=n_grid[i]))

        # Prior
        lp = np.ones(self.n_model)
        self.lp = lp - logsumexp(lp)

        self.init_lp = self.lp.copy()

    def update(self, d_idx, resp):
        mll = np.zeros(self.n_model)
        for i, m in enumerate(self.m):
            log_lik_r = m.log_lik[d_idx, :, int(resp)].flatten()
            mll_i = m.lp + log_lik_r
            lse = logsumexp(mll_i)
            mll[i] = lse

            # update lp of the model
            m.lp = mll_i - lse

        self.lp += mll
        self.lp -= logsumexp(self.lp)

        print("d idx", d_idx)
        print("resp", resp)
        print("post", np.exp(self.lp))

    def reset(self):
        lp = np.ones(self.n_model)
        self.lp = lp - logsumexp(lp)


class ADO(ModelComparison):

    def __init__(self, p_obs,
                 design,
                 bounds_grid, n_grid):
        super().__init__(p_obs=p_obs, design=design,
                         bounds_grid=bounds_grid, n_grid=n_grid)
        self.n_design = len(design)
        # self.ent_obs = -np.multiply(np.exp(self.log_lik), self.log_lik).sum(-1)

    def get_design(self):

        a, b, c = [], [], []
        for i, m in enumerate(self.m):
            expand_lp = np.expand_dims(np.expand_dims(m.lp, 0), -1)

            # shape (n_design, n_param_set, n_resp)
            a_m = m.log_lik + expand_lp
            b_m = np.exp(a_m)

            # shape (n_design, n_resp
            c_m = logsumexp(a_m, axis=1)

            a.append(a_m)
            b.append(b_m)
            c.append(c_m)

        u_design = np.zeros(self.n_design)
        for d_idx in range(self.n_design):
            u = 0
            for i, m in enumerate(self.m):

                b_m = b[i]
                c_m = c[i]

                u_m = 0
                for y in (0, 1):
                    for j in range(m.n_param_set):
                        u_m += \
                            b_m[d_idx, j, y] \
                            - np.exp(c_m[d_idx, y])*c_m[d_idx, y]

                    # shape (num_design, num_param_set, num_resp
                u += np.exp(self.lp[i]) * u_m

            u_design[d_idx] = u
        d_idx = np.argmax(u_design)

        return d_idx


def exp_decay(x, param):
    return param[0]*np.exp(-param[1] * x)


def power_law(x, param):
    return param[0] * (x+1)**(-param[1])


def exp_decay_one_param(x, param):
    return np.exp(-param[0] * x)


def power_law_one_param(x, param):
    return (x+1)**(-param[0])


def main():
    np.random.seed(12)

    # true_param = (0.8, 0.01, )
    # true_model = exp_decay
    true_model = power_law
    true_param = (0.9025, 0.4861)

    # bounds_grid_m0 = ((0., 1.), (2e-07, 0.025))
    bounds_grid_m0 = ((0, 1), (0, 1))
    n_grid_m0 = 100
    bounds_grid_m1 = bounds_grid_m0
    n_grid_m1 = n_grid_m0
    n_grid = (n_grid_m0, n_grid_m1)
    bounds_grid = (bounds_grid_m0, bounds_grid_m1)
    models = (exp_decay, power_law)

    #bounds_design = 1, 100
    #design = np.linspace(*bounds_design, n_design)
    design = np.array([0, 1, 2, 4, 7, 12, 21, 35, 59, 99])
    n_design = len(design)

    n_trial = 50

    m = ModelComparison(
        p_obs=models,
        design=design,
        bounds_grid=bounds_grid,
        n_grid=n_grid)

    # result container
    engine = ('random', 'ado')
    results = []

    # random ---------------------------------------------------------- #

    for t in range(n_trial):
        d_idx = np.random.randint(n_design)

        d = design[d_idx]

        p_r = true_model(x=d, param=true_param)
        resp = p_r > np.random.random()

        m.update(d_idx=d_idx, resp=resp)

        p = np.exp(m.lp)

        r = {"trial": t}

        for i, m_m in enumerate(m.m):
            r.update({f"model_{i}_p": p[i]})

            mean, std = m_m.get_estimates()
            for j in range(len(mean)):
                r.update({
                    f"model_{i}_param_{j}_mean": mean[j],
                    f"model_{i}_param_{j}_std": std[j]})

        results.append(r)

    random_df = pd.DataFrame(results)

    # ADO ---------------------------------------------------------- #

    print()
    print("ADO *****************************************************")

    results = []
    m = ADO(
        p_obs=models,
        design=design,
        bounds_grid=bounds_grid,
        n_grid=n_grid)

    choices = np.zeros(n_trial)

    for t in tqdm(range(n_trial), file=sys.stdout):

        d_idx = m.get_design()

        d = design[d_idx]
        choices[t] = d_idx

        p_r = true_model(x=d, param=true_param)
        resp = p_r > np.random.random()

        m.update(d_idx=d_idx, resp=resp)

        p = np.exp(m.lp)

        r = {"trial": t}

        for i, m_m in enumerate(m.m):
            r.update({f"model_{i}_p": p[i]})

            mean, std = m_m.get_estimates()
            for j in range(len(mean)):
                r.update({
                    f"model_{i}_param_{j}_mean": mean[j],
                    f"model_{i}_param_{j}_std": std[j]})
        results.append(r)

    ado_df = pd.DataFrame(results)
    df = {"random": random_df, "ado": ado_df}
    for k, v in df.items():
        v.to_csv(f"{k}.csv")

    # figure --------------------------------------------------------

    # n_model = len(bounds_grid)
    n_param = np.sum([len(b) for b in bounds_grid])

    n_rows = 1+n_param
    fig, axes = plt.subplots(figsize=(12, 6*n_rows), nrows=n_rows)

    colors = [f'C{i}' for i, e in enumerate(engine)]

    ax = axes[0]
    for j, e in enumerate(engine):
        y = df[e]["model_0_p"]

        c = colors[j]

        ax.plot(y, color=c, label=e)

        ax.set_title(f'$p(m_0)$')
        ax.set_xlabel("time")
        ax.set_ylabel(f"value")

    colors = [f'C{i}' for i, e in enumerate(engine)]

    ax_idx = 1

    for i, m in enumerate(models):
        for j in range(len(bounds_grid[i])):

            ax = axes[ax_idx]

            for k, e in enumerate(engine):

                _means = df[e][f"model_{i}_param_{j}_mean"]
                _stds = df[e][f"model_{i}_param_{j}_std"]

                print(_means)
                if m == true_model:
                    true_p = true_param[j]
                    ax.axhline(true_p, linestyle='--', color='black', alpha=.2)

                c = colors[k]

                ax.plot(_means, color=c, label=e)
                ax.fill_between(range(n_trial), _means - _stds,
                                _means + _stds, alpha=.2, color=c)

            ax.set_title(f'param {j}')
            ax.set_xlabel("time")
            ax.set_ylabel(f"value")

            ax_idx += 1

    plt.legend()
    plt.tight_layout()
    os.makedirs('fig', exist_ok=True)
    plt.savefig(os.path.join("fig", f"{SCRIPT_NAME}.pdf"))
    plt.show()


if __name__ == "__main__":
    main()

