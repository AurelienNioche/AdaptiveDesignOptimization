import os
import sys
import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools

SCRIPT_NAME = "model_comparison"


class Model:

    def __init__(self, p_obs,
                 n_param,
                 design,
                 bounds_grid, n_grid):

        self.p_obs = p_obs
        if n_param == 1:
            self.grid = \
                np.atleast_2d(np.linspace(*bounds_grid, n_grid)).T
        else:
            methods = (np.linspace for _ in range(n_param))
            self.grid = np.array(list(itertools.product(
                *(m(*b, n_grid) for m, b in zip(methods, bounds_grid)))))

        # Log-lik
        n_param_set, n_param = self.grid.shape
        n_design = len(design)
        p_one = np.zeros((n_design, n_param_set))
        for i, d in enumerate(design):
            p_one[i, :] = p_obs(d, self.grid.T).flatten()
        p_zero = 1 - p_one
        p = np.zeros((n_design, n_param_set, 2))
        p[:, :, 0] = p_zero
        p[:, :, 1] = p_one
        self.log_lik = np.log(p + np.finfo(np.float).eps)

        # Prior
        lp = np.ones(n_param_set)
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
        n_param_set, n_param = self.grid.shape
        lp = np.ones(n_param_set)
        self.lp = lp - logsumexp(lp)


class ModelComparison:

    def __init__(self, p_obs,
                 n_param,
                 design,
                 bounds_grid, n_grid):

        self.n_model = len(p_obs)
        self.m = []
        for i in range(self.n_model):
            self.m.append(
                Model(
                    p_obs=p_obs[i],
                    n_param=n_param[i],
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
            mll[i] = logsumexp(mll_i)

            # update lp of the model
            new_lp = mll_i - logsumexp(mll_i)
            m.lp = new_lp

        for i in range(self.n_model):
            self.lp[i] = self.init_lp[i] \
                - logsumexp(self.init_lp + mll - mll[i])
        # for i in range(self.n_model):
        #     log_bf = mll - mll[i]
        #     denom = logsumexp(self.init_lp + log_bf)
        #     self.lp[i] = self.init_lp[i] - denom

        print("d idx", d_idx)
        print("resp", resp)
        print("post", np.exp(self.lp))

    def reset(self):
        lp = np.ones(self.n_model)
        self.lp = lp - logsumexp(lp)


class ADO(ModelComparison):

    def __init__(self, p_obs,
                 n_param,
                 design,
                 bounds_grid, n_grid):
        super().__init__(p_obs=p_obs, n_param=n_param, design=design,
                         bounds_grid=bounds_grid, n_grid=n_grid)
        self.n_design = len(design)
        # self.ent_obs = -np.multiply(np.exp(self.log_lik), self.log_lik).sum(-1)

    def get_design(self):

        mll = np.zeros((self.n_model, self.n_design, 2)) # 2: n_resp
        for i, m in enumerate(self.m):
            extended_lp = np.expand_dims(np.expand_dims(m.lp, 0), -1)
            mll[i] = logsumexp(m.log_lik + extended_lp, axis=1)

        log_ratio = mll - logsumexp(mll + self.lp, axis=0)
        # shape (num_model, num_design, num_response)

        u = np.zeros(self.n_design)

        for i, m in enumerate(self.m):
            extended_lp = np.expand_dims(np.expand_dims(m.lp, 0), -1)
            ll = m.log_lik + extended_lp
            u += np.exp(self.lp[i]) \
                   * np.sum(np.sum(np.exp(ll), axis=1) * log_ratio[i, :, :], axis=-1)
            # shape (num_design, num_param_set, num_resp

        d_idx = np.argmax(u)

        # post = np.exp(self.lp)
        #
        # # Calculate the marginal log likelihood.
        # extended_lp = np.expand_dims(np.expand_dims(self.lp, 0), -1)
        # mll = logsumexp(self.log_lik + extended_lp, axis=1)
        # # shape (num_design, num_response)
        #
        # # Calculate the marginal entropy and conditional entropy.
        # ent_marg = -np.sum(np.exp(mll) * mll, -1)  # shape (num_designs,)
        # ent_cond = np.sum(post * self.ent_obs, axis=1)  # shape (num_designs,)
        #
        # # Calculate the mutual information.
        # mutual_info = ent_marg - ent_cond  # shape (num_designs,)
        # d_idx = np.argmax(mutual_info)

        return d_idx


def exp_decay(x, alpha):
    return np.exp(-alpha * x)


def power_law(x, alpha):
    return (x+1)**(-alpha)


def main():
    np.random.seed(12)

    param = [0.01, ]

    bounds_design = 1, 1000
    n_design = 200

    bounds_grid = (2e-07, 0.025)
    n_grid = 100

    n_trial = 100

    design = np.linspace(*bounds_design, n_design)

    m = ModelComparison(
        p_obs=(exp_decay, power_law),
        n_param=(1, 1),
        design=design,
        bounds_grid=(bounds_grid, bounds_grid),
        n_grid=(n_grid, n_grid))

    # result container
    engine = ('random', 'ado')
    means = {e: np.zeros((2, n_trial)) for e in engine}

    # random ---------------------------------------------------------- #

    for t in range(n_trial):
        d_idx = np.random.randint(n_design)

        d = design[d_idx]

        p_r = exp_decay(x=d, alpha=param[0])
        resp = p_r > np.random.random()

        m.update(d_idx=d_idx, resp=resp)

        p = np.exp(m.lp)
        means['random'][:, t] = p

    print()
    print("ADO *****************************************************")

    # ADO ---------------------------------------------------------- #
    m = ADO(
        p_obs=(exp_decay, power_law),
        n_param=(1, 1),
        design=design,
        bounds_grid=(bounds_grid, bounds_grid),
        n_grid=(n_grid, n_grid))

    choices = np.zeros(n_trial)

    for t in tqdm(range(n_trial), file=sys.stdout):

        d_idx = m.get_design()

        d = design[d_idx]
        choices[t] = d_idx

        p_r = exp_decay(x=d, alpha=param[0])
        resp = p_r > np.random.random()

        m.update(d_idx=d_idx, resp=resp)

        p = np.exp(m.lp)
        means['ado'][:, t] = p

    # figure --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [f'C{i}' for i, e in enumerate(engine)]
    ls = ['-', ':']

    for i in range(1):  # just show for model 1
        for j, e in enumerate(engine):
            _means = means[e][i, :]

            c = colors[j]

            ax.plot(_means, color=c, label=e, ls=ls[i])

            ax.set_title(f'alpha')
            ax.set_xlabel("time")
            ax.set_ylabel(f"value")

    plt.legend()
    plt.tight_layout()
    os.makedirs('fig', exist_ok=True)
    plt.savefig(os.path.join("fig", f"{SCRIPT_NAME}.pdf"))
    plt.show()


if __name__ == "__main__":
    main()

