import os
import sys
import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools


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


class ADO(Model):

    def __init__(self, p_obs,
                 n_param,
                 design,
                 bounds_grid, n_grid):
        super().__init__(p_obs=p_obs, n_param=n_param, design=design,
                         bounds_grid=bounds_grid, n_grid=n_grid)
        self.ent_obs = -np.multiply(np.exp(self.log_lik), self.log_lik).sum(-1)

    def get_design(self):

        post = np.exp(self.lp)

        # Calculate the marginal log likelihood.
        extended_lp = np.expand_dims(np.expand_dims(self.lp, 0), -1)
        mll = logsumexp(self.log_lik + extended_lp, axis=1)
        # shape (num_design, num_response)

        # Calculate the marginal entropy and conditional entropy.
        ent_marg = -np.sum(np.exp(mll) * mll, -1)  # shape (num_designs,)
        ent_cond = np.sum(post * self.ent_obs, axis=1)  # shape (num_designs,)

        # Calculate the mutual information.
        mutual_info = ent_marg - ent_cond  # shape (num_designs,)
        d_idx = np.argmax(mutual_info)

        return d_idx


def exp_decay(x, alpha):
    return np.exp(-alpha * x)


def main():
    np.random.seed(12)

    param = [0.01, ]

    bounds_design = 1, 1000
    n_design = 100

    bounds_grid = (2e-07, 0.025)
    n_grid = 100

    n_trial = 1000

    n_param = len(param)

    design = np.linspace(*bounds_design, n_design)

    m = Model(p_obs=exp_decay, n_param=n_param,
              design=design,
              bounds_grid=bounds_grid, n_grid=n_grid)

    # result container
    engine = ('random', 'ado')
    means = {e: np.zeros((n_param, n_trial)) for e in engine}
    stds = {e: np.zeros((n_param, n_trial)) for e in engine}

    # random ---------------------------------------------------------- #

    for t in range(n_trial):
        d_idx = np.random.randint(n_design)

        d = design[d_idx]

        p_r = exp_decay(x=d, alpha=param[0])
        resp = p_r > np.random.random()

        m.update(d_idx=d_idx, resp=resp)
        ep, std = m.get_estimates()

        means['random'][:, t] = ep
        stds['random'][:, t] = std

    # ADO ---------------------------------------------------------- #
    m = ADO(p_obs=exp_decay, n_param=n_param,
            design=design,
            bounds_grid=bounds_grid, n_grid=n_grid)

    choices = np.zeros(n_trial)

    for t in tqdm(range(n_trial), file=sys.stdout):

        d_idx = m.get_design()

        d = design[d_idx]
        choices[t] = d_idx

        p_r = exp_decay(x=d, alpha=param[0])
        resp = p_r > np.random.random()

        m.update(d_idx=d_idx, resp=resp)
        ep, std = m.get_estimates()

        means['ado'][:, t] = ep
        stds['ado'][:, t] = std

    # figure --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [f'C{i}' for i, e in enumerate(engine)]

    for j, e in enumerate(engine):
        _means = means[e][0, :]
        _stds = stds[e][0, :]

        true_p = param[0]
        ax.axhline(true_p, linestyle='--', color='black', alpha=.2)

        c = colors[j]

        ax.plot(_means, color=c, label=e)
        ax.fill_between(range(n_trial), _means - _stds,
                        _means + _stds, alpha=.2, color=colors[j])

        ax.set_title(f'alpha')
        ax.set_xlabel("time")
        ax.set_ylabel(f"value")

    plt.legend()
    plt.tight_layout()
    os.makedirs('fig', exist_ok=True)
    plt.savefig(os.path.join("fig", "ado_vs_random_exp_decay.pdf"))
    plt.show()


if __name__ == "__main__":
    main()

