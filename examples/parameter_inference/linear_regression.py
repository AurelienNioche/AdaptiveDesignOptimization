import os
import sys
import numpy as np
import itertools
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from tqdm import tqdm


def p_obs(x1, b0, b1):

    logit = b0 + x1 * b1
    _p_obs = 1. / (1 + np.exp(-logit))
    return _p_obs


def main():

    np.random.seed(123)

    n_design = 100
    bounds_design = 0, 10
    design = np.linspace(*bounds_design, n_design)

    bounds_param = (-10, 10), (-10, 10)
    grid = np.array(list(itertools.product(
        *(np.linspace(*b, 100) for b in bounds_param))))
    n_trial = 1000

    param = [2.0, 0.5]

    n_param_set, n_param = grid.shape

    # Results container
    engine = ('random', 'ado')
    means = {e: np.zeros((n_param, n_trial)) for e in engine}
    stds = {e: np.zeros((n_param, n_trial)) for e in engine}

    # Log-lik
    p_one = np.array([p_obs(d, b0=grid[:, 0], b1=grid[:, 1])
                      for d in design])  # shape: (n design, n param set)
    p_zero = 1 - p_one

    p = np.zeros((n_design, n_param_set, 2))
    p[:, :, 0] = p_zero
    p[:, :, 1] = p_one
    log_lik = np.log(p + np.finfo(np.float).eps)

    # Random --------------------------------------------------------

    lp = np.ones(n_param_set)
    lp -= logsumexp(lp)

    for t in range(n_trial):

        d_idx = np.random.randint(n_design)

        d = design[d_idx]

        p_r = p_obs(d, b0=param[0], b1=param[1])
        resp = p_r > np.random.random()

        log_lik_r = log_lik[d_idx, :, int(resp)].flatten()

        lp += log_lik_r
        lp -= logsumexp(lp)

        ep = np.dot(np.exp(lp), grid)

        delta = grid - ep
        post_cov = np.dot(delta.T, delta * np.exp(lp).reshape(-1, 1))
        sdp = np.sqrt(np.diag(post_cov))

        means['random'][:, t] = ep
        stds['random'][:, t] = sdp

    # Ado -----------------------------------------------

    lp = np.ones(n_param_set)
    lp -= logsumexp(lp)

    ent_obs = -np.multiply(np.exp(log_lik), log_lik).sum(-1)

    for t in tqdm(range(n_trial), file=sys.stdout):

        post = np.exp(lp)

        # Calculate the marginal log likelihood.
        extended_lp = np.expand_dims(np.expand_dims(lp, 0), -1)
        mll = logsumexp(log_lik + extended_lp, axis=1) # shape (num_design, num_response)

        # Calculate the marginal entropy and conditional entropy.
        ent_marg = -np.sum(np.exp(mll) * mll, -1)  # shape (num_designs,)
        ent_cond = np.sum(post * ent_obs, axis=1)  # shape (num_designs,)

        # Calculate the mutual information.
        mutual_info = ent_marg - ent_cond  # shape (num_designs,)
        d_idx = np.argmax(mutual_info)

        d = design[d_idx]

        p_r = p_obs(d, b0=param[0], b1=param[1])
        resp = p_r > np.random.random()

        log_lik_r = log_lik[d_idx, :, int(resp)].flatten()

        lp += log_lik_r
        lp -= logsumexp(lp)

        ep = np.dot(np.exp(lp), grid)

        delta = grid - ep
        post_cov = np.dot(delta.T, delta * np.exp(lp).reshape(-1, 1))
        sdp = np.sqrt(np.diag(post_cov))

        means['ado'][:, t] = ep
        stds['ado'][:, t] = sdp

    # Figure -----------------------------------------------

    fig, axes = plt.subplots(ncols=n_param, figsize=(12, 6))

    colors = [f'C{i}' for i, e in enumerate(engine)]

    for i, ax in enumerate(axes):

        for j, e in enumerate(engine):

            _means = means[e][i, :]
            _stds = stds[e][i, :]

            true_p = param[i]
            ax.axhline(true_p, linestyle='--', color='black',
                       alpha=.2)

            c = colors[j]

            ax.plot(_means, color=c, label=e)
            ax.fill_between(range(n_trial), _means - _stds,
                            _means + _stds, alpha=.2, color=colors[j])

            ax.set_title(f'b{i}')
            ax.set_xlabel("trial")
            ax.set_ylabel(f"value")

    plt.legend()
    plt.tight_layout()
    os.makedirs('fig', exist_ok=True)
    plt.savefig(os.path.join("fig", "ado_vs_random_linear_reg.pdf"))


if __name__ == "__main__":
    main()

