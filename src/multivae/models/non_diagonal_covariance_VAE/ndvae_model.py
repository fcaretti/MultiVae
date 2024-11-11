import logging
from typing import Union

import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F
from pythae.models.base.base_utils import ModelOutput
from torch.distributions import Laplace, Normal

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.data.utils import drop_unused_modalities

from ..base import BaseMultiVAE
from .ndvae_config import NDVAEConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class NDVAE(BaseMultiVAE):
    """
    The Variational Mixture-of-Experts Autoencoder model.

    Args:
        model_config (MMVAEConfig): An instance of MMVAEConfig in which any model's
            parameters is made available.

        encoders (Dict[str, ~pythae.models.nn.base_architectures.BaseEncoder]): A dictionary containing
            the modalities names and the encoders for each modality. Each encoder is an instance of
            Pythae's BaseEncoder. Default: None.

        decoders (Dict[str, ~pythae.models.nn.base_architectures.BaseDecoder]): A dictionary containing
            the modalities names and the decoders for each modality. Each decoder is an instance of
            Pythae's BaseDecoder.
    """

    def __init__(
        self, model_config: NDVAEConfig, encoders: dict = None, decoders: dict = None
    ):
        super().__init__(model_config, encoders, decoders)

        self.K = model_config.K
        self.prior_rank = model_config.prior_rank
        
        self.modalities_specific_dim = model_config.modalities_specific_dim
        self.total_latent_dim = sum(
            sum(dim) if isinstance(dim, list) else dim for dim in self.modalities_specific_dim.values()
        )
        self.prior_mean = torch.nn.Parameter(
            torch.zeros(1, self.total_latent_dim), requires_grad=False
        )
        self.w = torch.nn.Parameter(
            torch.randn(self.total_latent_dim, self.prior_rank), requires_grad=True
        )
        '''self.diag_variance = torch.nn.Parameter(
            torch.randn(1, self.total_latent_dim), requires_grad=True
        )'''

        self.diag_variance_unconstrained = torch.nn.Parameter(
            torch.randn(1, self.total_latent_dim), requires_grad=True
        )
        #self.diag_variance = torch.nn.functional.softplus(self.diag_variance_unconstrained).to(self.w.device)
        self.reconstruction_loss = {}
        
        for modality, dist in model_config.decoders_dist.items():
            # Retrieve additional parameters for the distribution, if any
            dist_params = model_config.decoder_dist_params.get(modality, {})

            # Define the appropriate loss function based on distribution type
            if dist == 'normal':
                # Normal distribution reconstruction loss (mean squared error)
                scale = dist_params.get('scale', 1.0)  # default scale if none specified
                self.reconstruction_loss[modality] = lambda x_recon, x_true: \
                    0.5 * ((x_recon - x_true) / scale) ** 2

            elif dist == 'bernoulli':
                # Bernoulli distribution reconstruction loss (binary cross-entropy)
                self.reconstruction_loss[modality] = lambda x_recon, x_true: \
                    F.binary_cross_entropy_with_logits(x_recon, x_true, reduction='none')

            elif dist == 'laplace':
                # Laplace distribution reconstruction loss (mean absolute error)
                scale = dist_params.get('scale', 1.0)  # default scale if none specified
                self.reconstruction_loss[modality] = lambda x_recon, x_true: \
                    torch.abs(x_recon - x_true) / scale
            else:
                raise ValueError(f"Unknown decoder distribution '{dist}' for modality '{modality}'")

        # Set up rescaling factors if using likelihood rescaling
        if model_config.uses_likelihood_rescaling:
            self.rescale_factors = model_config.rescale_factors or self._compute_rescale_factors(model_config)
        else:
            self.rescale_factors = {modality: 1.0 for modality in model_config.input_dims}

    def _compute_rescale_factors(self, model_config):
        # Default rescale factors based on input dimensions for each modality
        total_dim = sum([torch.prod(torch.tensor(dim)).item() for dim in model_config.input_dims.values()])
        return {modality: total_dim / torch.prod(torch.tensor(dim)).item() for modality, dim in model_config.input_dims.items()}

        self.model_name = "NDVAE"

    def generate_covariance_prior(self):
        """
        Generate the covariance matrix of the prior distribution from the parameters
        Parameterization ensures that the covariance matrix is positive definite.
        """
        #print the devices
        device = self.w.device
        diag_variance = torch.nn.functional.softplus(self.diag_variance_unconstrained).to(device) 
        return torch.mm(self.w, self.w.t()) + torch.diag(diag_variance) + torch.eye(self.w.size(0), device=device) * 1e-3


    def forward(self, inputs, **kwargs):
        # Initialize dictionaries to store embeddings and other necessary data
        embeddings = {}
        embeddings_list = []
        latent_dims = {}
        modalities = list(inputs.data.keys())

        # Compute embeddings for each modality
        for mod in modalities:
            x = inputs.data[mod]
            # Pass data through the encoder (projects onto a single variable)
            z = self.encoders[mod](x).embedding  # Shape: (batch_size, latent_dim_mod)
            embeddings[mod] = z
            embeddings_list.append(z)
            latent_dims[mod] = z.size(1)

        # Concatenate embeddings from all modalities along the feature dimension
        # Resulting shape: (batch_size, latent_dim_total)
        z_stacked = torch.cat(embeddings_list, dim=1)
        latent_dim_total = z_stacked.size(1)

        # Prepare the output object to store embeddings and other necessary data
        output = ModelOutput()
        output.embeddings = embeddings
        output.z_stacked = z_stacked
        output.latent_dims = latent_dims
        output.modalities = modalities
        output.inputs = inputs

        # Compute loss if specified
        compute_loss = kwargs.get('compute_loss', True)
        if compute_loss:
            loss_output = self.loss_function(output)
            output.update(loss_output)

        return output

    def loss_function(self, output):
        # Retrieve necessary variables from the output of the forward pass
        z_stacked = output.z_stacked  # (batch_size, latent_dim_total)
        latent_dims = output.latent_dims
        embeddings = output.embeddings
        inputs = output.inputs
        modalities = output.modalities

        # Generate the covariance matrix for the prior distribution
        prior_covariance = self.generate_covariance_prior()  # Shape: (latent_dim_total, latent_dim_total)
        condition_number = torch.linalg.cond(prior_covariance)
        # Ensure the covariance matrix has the correct dimensions
        latent_dim_total = z_stacked.size(1)
        if prior_covariance.size() != (latent_dim_total, latent_dim_total):
            raise ValueError("The generated covariance matrix has incorrect dimensions.")

        # Create the prior distribution with mean zero and generated covariance
        prior_mean = torch.zeros(latent_dim_total, device=z_stacked.device)
        prior_dist = torch.distributions.MultivariateNormal(
            loc=prior_mean, covariance_matrix=prior_covariance
        )

        # Compute the negative log-likelihood (equivalent to KL divergence)
        # Since embeddings are deterministic, KL divergence simplifies to -log_prob
        negative_log_likelihood = -prior_dist.log_prob(z_stacked)  # Shape: (batch_size)
        total_kl_divergence = negative_log_likelihood.sum()
        #print(total_kl_divergence)

        # Initialize variables for reconstructions and total reconstruction loss
        reconstructions = {}
        reconstruction_loss = 0

        # Separate the latent space back into modality-specific components
        current_index = 0
        for mod in modalities:
            latent_dim_mod = latent_dims[mod]
            # Extract the corresponding component from the stacked latent space
            z_mod = z_stacked[:, current_index:current_index + latent_dim_mod]  # (batch_size, latent_dim_mod)
            current_index += latent_dim_mod

            # Decode the latent embeddings to reconstruct the input
            x_recon = self.decoders[mod](z_mod)['reconstruction']
            reconstructions[mod] = x_recon

            # Compute reconstruction loss for the modality
            x_true = inputs.data[mod]
            rec_loss_mod = self.reconstruction_loss[mod](x_recon, x_true)
            reconstruction_loss += rec_loss_mod.sum()

        # Sum the KL divergence and reconstruction loss to obtain the total loss
        total_loss = total_kl_divergence + reconstruction_loss

        # Return the total loss encapsulated in a ModelOutput object
        loss_output = ModelOutput(loss=total_loss, metrics=dict())
        loss_output.reconstructions = reconstructions

        return loss_output



    def encode(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        N: int = 1,
        **kwargs,
    ):
        """
        Generate encodings conditioning on all modalities or a subset of modalities.

        Args:
            inputs (MultimodalBaseDataset): The dataset to use for the conditional generation.
            cond_mod (Union[list, str]): Either 'all' or a list of str containing the modalities
                names to condition on.
            N (int) : The number of encodings to sample for each datapoint. Default to 1.

        Returns:
            ModelOutput instance with fields:
                z (torch.Tensor (n_data, N, latent_dim))
                one_latent_space (bool) = True




        """

        cond_mod = super().encode(inputs, cond_mod, N, **kwargs).cond_mod

        return_mean = kwargs.pop("return_mean", False)
        if all([s in self.encoders.keys() for s in cond_mod]):
            if return_mean:
                emb = torch.stack(
                    [self.encoders[mod](inputs.data[mod]).embedding for mod in cond_mod]
                ).mean(0)
                if N > 1:
                    z = torch.stack([emb] * N)
                else:
                    z = emb

            else:
                # Choose one of the conditioning modalities at random
                mod = np.random.choice(cond_mod)

                output = self.encoders[mod](inputs.data[mod])

                mu, log_var = output.embedding, output.log_covariance
                sigma = self.log_var_to_std(log_var)
                qz_x = self.post_dist(mu, sigma)
                sample_shape = torch.Size([]) if N == 1 else torch.Size([N])
                z = qz_x.rsample(sample_shape)

            flatten = kwargs.pop("flatten", False)
            if flatten:
                z = z.reshape(-1, self.latent_dim)

            return ModelOutput(z=z, one_latent_space=True)

    def compute_joint_nll(
        self, inputs: MultimodalBaseDataset, K: int = 1000, batch_size_K: int = 100
    ):
        """
        Return the estimated negative log-likelihood summed over the inputs.
        The negative log-likelihood is estimated using importance sampling.

        Args:
            inputs : the data to compute the joint likelihood

        """

        self.eval()

        # First compute all the parameters of the joint posterior q(z|x,y)
        post_params = []
        for cond_mod in self.encoders:
            output = self.encoders[cond_mod](inputs.data[cond_mod])
            mu, log_var = output.embedding, output.log_covariance
            sigma = self.log_var_to_std(log_var)
            post_params.append((mu, sigma))

        z_joint = self.encode(inputs, N=K).z
        z_joint = z_joint.permute(1, 0, 2)
        n_data, _, latent_dim = z_joint.shape

        # Then iter on each datapoint to compute the iwae estimate of ln(p(x))
        ll = 0
        for i in range(n_data):
            start_idx = 0
            stop_idx = min(start_idx + batch_size_K, K)
            lnpxs = []
            while start_idx < stop_idx:
                latents = z_joint[i][start_idx:stop_idx]

                # Compute p(x_m|z) for z in latents and for each modality m
                lpx_zs = 0  # ln(p(x,y|z))
                for mod in inputs.data:
                    decoder = self.decoders[mod]
                    recon = decoder(latents)[
                        "reconstruction"
                    ]  # (batch_size_K, nb_channels, w, h)
                    x_m = inputs.data[mod][i]  # (nb_channels, w, h)

                    lpx_zs += (
                        self.recon_log_probs[mod](recon, x_m)
                        .reshape(recon.size(0), -1)
                        .sum(-1)
                    )

                # Compute ln(p(z))
                prior = self.prior_dist(*self.pz_params)
                lpz = prior.log_prob(latents).sum(dim=-1)

                # Compute posteriors -ln(q(z|x,y))
                qz_xs = [self.post_dist(p[0][i], p[1][i]) for p in post_params]
                lqz_xy = torch.logsumexp(
                    torch.stack([q.log_prob(latents).sum(-1) for q in qz_xs]), dim=0
                ) - np.log(self.n_modalities)

                ln_px = torch.logsumexp(lpx_zs + lpz - lqz_xy, dim=0)
                lnpxs.append(ln_px)

                # next batch
                start_idx += batch_size_K
                stop_idx = min(stop_idx + batch_size_K, K)

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0) - np.log(K)

        return -ll

    @torch.no_grad()
    def compute_joint_nll_paper(
        self, inputs: MultimodalBaseDataset, K: int = 1000, batch_size_K: int = 10
    ):
        """Computes the joint likelihood like in the original dataset, using all Mixture of experts
        samples and modality rescaling."""

        self.eval()

        lws = []
        nb_computed_samples = 0
        while nb_computed_samples < K:
            n_samples = min(batch_size_K, K - nb_computed_samples)
            nb_computed_samples += n_samples
            # Compute a iwae likelihood estimate using n_samples
            output = self.forward(
                inputs, compute_loss=False, K=n_samples, detailed_output=True
            )
            lw = self.iwae(output.qz_xs, output.zss, output.recon, inputs).loss
            lws.append(lw + np.log(n_samples * self.n_modalities))

        ll = torch.logsumexp(torch.stack(lws), dim=0) - np.log(
            nb_computed_samples * self.n_modalities
        )  # n_batch
        return -ll

    def generate_from_prior(self, n_samples, **kwargs):
        sample_shape = [n_samples] if n_samples > 1 else []
        z = self.prior_dist(*self.pz_params).rsample(sample_shape).to(self.device)
        return ModelOutput(z=z.squeeze(), one_latent_space=True)
