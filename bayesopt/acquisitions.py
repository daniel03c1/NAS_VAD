import networkx as nx
import numpy as np
import torch
from abc import ABC
from copy import deepcopy
from multiprocessing import Pool
from torch.distributions import Normal


class BaseAcquisition(ABC):
    def __init__(self, gp):
        self.gp = gp
        self.iters = 0

        # Storage for the current evaluation on the acquisition function
        self.next_location = None
        self.next_acq_value = None

    def propose_location(self, *args):
        """Propose new locations for subsequent sampling
        This method should be overriden by respective acquisition function implementations."""
        raise NotImplementedError

    def optimize(self):
        """This is the method that user should call for the Bayesian optimisation main loop."""
        raise NotImplementedError

    def eval(self, x):
        """Evaluate the acquisition function at point x2. This should be overridden by respective acquisition
        function implementations"""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)


class GraphExpectedImprovement(BaseAcquisition):
    def __init__(self, surrogate_model, augmented_ei=False, xi: float = 0.0,
                 in_fill: str = 'best'):
        """
        This is the graph BO version of the expected improvement
        key differences are:
        1. The input x2 is a networkx graph instead of a vectorial input
        2. the search space (a collection of x1_graphs) is discrete, so there is no gradient-based optimisation. Instead,
        we compute the EI at all candidate points and empirically select the best position during optimisation

        augmented_ei: Using the Augmented EI heuristic modification to the standard expected improvement algorithm
        according to Huang (2006)
        xi: float: manual exploration-exploitation trade-off parameter.
        in_fill: str: the criterion to be used for in-fill for the determination of mu_star. 'best' means the empirical
        best observation so far (but could be susceptible to noise), 'posterior' means the best *posterior GP mean*
        encountered so far, and is recommended for optimisationn of more noisy functions.
        """
        super(GraphExpectedImprovement, self).__init__(surrogate_model)
        assert in_fill in ['best', 'posterior']
        self.in_fill = in_fill
        self.augmented_ei = augmented_ei
        self.xi = xi
        self.mu_star = self._get_incumbent()

    def eval(self, x: nx.Graph, asscalar=False):
        """
        Return the negative expected improvement at the query point x2
        """
        mu, cov = self.gp.predict(x)
        std = torch.sqrt(torch.diag(cov))

        gauss = Normal(torch.zeros(1, device=mu.device),
                       torch.ones(1, device=mu.device))
        u = (mu - self.mu_star - self.xi) / std
        ucdf = gauss.cdf(u)
        updf = torch.exp(gauss.log_prob(u))
        ei = std * updf + (mu - self.mu_star - self.xi) * ucdf

        if self.augmented_ei:
            sigma_n = self.gp.likelihood
            ei *= (1. - torch.sqrt(torch.tensor(sigma_n, device=mu.device)) \
                / torch.sqrt(sigma_n + torch.diag(cov)))
        if asscalar:
            ei = ei.detach().numpy().item()
        return ei

    def _get_incumbent(self):
        """
        Get the incumbent
        """
        if self.in_fill == 'max':
            return torch.max(self.gp.y_)
        else:
            x = self.gp.x
            mu_train, _ = self.gp.predict(x)
            incumbent_idx = torch.argmax(mu_train)
            return self.gp.y_[incumbent_idx]

    def propose_location(self, candidates, top_n=5, return_distinct=True):
        """top_n: return the top n candidates wrt the acquisition function."""
        # selected_idx = [i for i in self.candidate_idx if self.evaluated[i] is False]
        # eis = torch.tensor([self.eval(self.candidates[c]) for c in selected_idx])
        # print(eis)
        self.iters += 1
        if return_distinct:
            eis = np.array([self.eval(c).detach() for c in candidates])
            eis_, unique_idx = np.unique(eis, return_index=True)
            try:
                i = np.argpartition(eis_, -top_n)[-top_n:]
                indices = np.array([unique_idx[j] for j in i])
            except ValueError:
                eis = torch.tensor([self.eval(c) for c in candidates])
                values, indices = eis.topk(top_n)
        else:
            eis = torch.tensor([self.eval(c) for c in candidates])
            values, indices = eis.topk(top_n)
        xs = tuple([candidates[int(i)] for i in indices])
        return xs, eis, indices

    def optimize(self):
        raise ValueError("The kernel invoked does not have hyperparameters to optimse over!")


class GraphUpperConfidentBound(GraphExpectedImprovement):
    """
    Graph version of the upper confidence bound acquisition function
    """

    def __init__(self, surrogate_model, beta=None):
        """Same as graphEI with the difference that a beta coefficient is asked for, as per standard GP-UCB acquisition
        """
        super(GraphUpperConfidentBound, self).__init__(surrogate_model)
        self.beta = beta

    def eval(self, x: nx.Graph, asscalar=False):
        mu, cov = self.gp.predict(x)
        std = torch.sqrt(torch.diag(cov))
        if self.beta is None:
            self.beta = 3. * torch.sqrt(0.5 * torch.log(torch.tensor(2. * self.iters + 1.)))
        acq = mu + self.beta * std
        if asscalar:
            acq = acq.detach().numpy().item()
        return acq

