from kernels.weisfilerlehman import GraphKernels
from kernels.vectorial_kernels import Stationary
import torch
import logging


class CombineKernel:
    def __init__(self, combined_by='sum', *kernels, **kwargs):
        self.has_graph_kernels = False
        self.has_vector_kernels = False
        for k in kernels:
            if isinstance(k, GraphKernels):
                self.has_graph_kernels = True
            if not isinstance(k, GraphKernels):
                self.has_vector_kernels = True
        self.kernels = kernels
        # Store the training graphs and vector features..
        self._gram = None
        self.gr, self.x = None, None
        assert combined_by in ['sum', 'multiply']
        self.combined_by = combined_by

    def fit_transform(self, weights, gr1: list, x1=None,
                      feature_lengthscale=None,
                      normalize=True,
                      rebuild_model=True, save_gram_matrix=True,
                      **kwargs):
        if self.has_vector_kernels and x1 is None:
            raise ValueError("The supplied kernels have one or more vectorial kernels, but no vector feature is"
                             " supplied.")
        N = len(gr1)
        if x1 is not None:
            assert N == len(x1), "Expected " + str(N) + " but got " + str(len(x1))
        x1 = torch.tensor(x1).reshape(N, -1) if x1 is not None else None
        i = 0
        K = torch.zeros(N, N) if self.combined_by == 'sum' else torch.ones(N, N)
        for k in self.kernels:
            if isinstance(k, GraphKernels):
                if self.combined_by == 'sum':
                    K += weights[i] * k.fit_transform(gr1, rebuild_model=rebuild_model,
                                                      save_gram_matrix=save_gram_matrix, **kwargs)
                else:
                    K *= weights[i] * k.fit_transform(gr1, rebuild_model=rebuild_model,
                                                      save_gram_matrix=save_gram_matrix, **kwargs)
            else:
                assert isinstance(k, Stationary), " For now, only the Stationary custom built kernels are supported!"
                if self.combined_by == 'sum':
                    K += (weights[i] * k.fit_transform(x1, l=feature_lengthscale, rebuild_model=rebuild_model,
                                                       save_gram_matrix=save_gram_matrix)).double()
                else:
                    K *= (weights[i] * k.fit_transform(x1, l=feature_lengthscale, rebuild_model=rebuild_model,
                                                       save_gram_matrix=save_gram_matrix)).double()
            i += 1
        if normalize:
            K_diag = torch.sqrt(torch.diag(K))
            K /= torch.ger(K_diag, K_diag)
        if save_gram_matrix:
            self._gram = K.clone()
        self.x = x1
        return K

    def transform(self, weights, gr: list, x=None, feature_lengthscale=None):
        if self._gram is None:
            raise ValueError("The kernel has not been fitted. Call fit_transform first to generate the training Gram"
                             "matrix.")
        i = 0
        # K is in shape of len(Y), len(X)
        K = torch.zeros(len(gr), self._gram.shape[0]) if self.combined_by == 'sum' \
            else torch.ones(len(gr), self._gram.shape[0])
        for k in self.kernels:
            if isinstance(k, GraphKernels):
                if self.combined_by == 'sum':
                    K += weights[i] * k.transform(gr)
                else:
                    K *= weights[i] * k.transform(gr)
            else:
                assert isinstance(k, Stationary), " For now, only the Stationary custom built kernels are supported!"
                if self.combined_by == 'sum':
                    K += (weights[i] * k.transform(x, l=feature_lengthscale)).double()
                else:
                    K *= (weights[i] * k.transform(x, l=feature_lengthscale)).double()
            i += 1
        return K.t()


class SumKernel(CombineKernel):
    def __init__(self,  *kernels, **kwargs):
        super(SumKernel, self).__init__('sum', *kernels, **kwargs)

    def forward_t(self, weights, gr2: list,
                  x2=None,
                  gr1: list = None,
                  x1=None,
                  feature_lengthscale=None):
        """
        Compute the kernel gradient w.r.t the feature vector
        Parameters
        ----------
        feature_lengthscale
        x2
        x1
        gr1
        weights
        gr2

        Returns
        -------
        grads: K list of 2-tuple.
        (K, x2) where K is the weighted Gram matrix of that matrix, x2 is the leaf variable on which Jacobian-vector
        product to be computed.

        """
        grads = []
        for i, k in enumerate(self.kernels):
            if isinstance(k, GraphKernels):
                handle = k.forward_t(gr2, gr1=gr1)
                grads.append((weights[i] * handle[0], handle[1], handle[2]))
            elif isinstance(k, Stationary):
                handle = k.forward_t(x2=x2, x1=x1, l=feature_lengthscale)
                grads.append((weights[i] * handle[0], handle[1], handle[2]))
            else:
                logging.warning("Gradient not implemented for kernel type" + str(k.__name__))
                grads.append((None, None))
        assert len(grads) == len(self.kernels)
        return grads


class ProductKernel(CombineKernel):
    def __init__(self,  *kernels, **kwargs):
        super(ProductKernel, self).__init__('multiply', *kernels, **kwargs)

    def dk_dphi(self, weights, gr: list = None, x=None, feature_lengthscale=None):
        raise NotImplementedError