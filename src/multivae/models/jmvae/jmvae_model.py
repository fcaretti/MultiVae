import logging
from typing import Union

import torch
import numpy as np
import torch.distributions as dist
from pythae.models.base.base_utils import ModelOutput

from ...data.datasets.base import MultimodalBaseDataset
from ..base.base_utils import rsample_from_gaussian, stable_poe
from ..joint_models import BaseJointModel
from ..nn.base_architectures import BaseJointEncoder
from .jmvae_config import JMVAEConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class JMVAE(BaseJointModel):
    """The Joint Multimodal Variational Autoencoder model.

    Args:
        model_config (JMVAEConfig): An instance of JMVAEConfig in which any model's
            parameters is made available.

        encoders (Dict[str, ~pythae.models.nn.base_architectures.BaseEncoder]): A dictionary containing
            the modalities names and the encoders for each modality. Each encoder is an instance of
            Pythae's BaseEncoder. Default: None.

        decoder (Dict[str, ~pythae.models.nn.base_architectures.BaseDecoder]): A dictionary containing
            the modalities names and the decoders for each modality. Each decoder is an instance of
            Pythae's BaseDecoder.

        joint_encoder (~multivae.models.nn.base_architectures.BaseJointEncoder) : Takes all the modalities
            as input. If none is provided, one is
            created from the unimodal encoders. Default : None.
    """

    def __init__(
        self,
        model_config: JMVAEConfig,
        encoders: dict = None,
        decoders: dict = None,
        joint_encoder: Union[BaseJointEncoder, None] = None,
        **kwargs,
    ):
        super().__init__(model_config, encoders, decoders, joint_encoder, **kwargs)

        self.model_name = "JMVAE"

        self.alpha = model_config.alpha
        self.warmup = model_config.warmup
        self.start_keep_best_epoch = model_config.warmup + 1
        self.beta = model_config.beta

    def encode(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        N: int = 1,
        return_mean=False,
        **kwargs,
    ):
        """Generate encodings conditioning on all modalities or a subset of modalities.

        Args:
            inputs (MultimodalBaseDataset): The data to encode.
            cond_mod (Union[list, str], optional): The modalities to use to compute the posterior
            distribution. Defaults to 'all'.
            N (int, optional): The number of samples to generate from the posterior distribution
            for each datapoint. Defaults to 1.
            return_mean (bool) : if True, returns the mean of the posterior distribution (instead of a sample).


        Raises:
            AttributeError: _description_
            AttributeError: _description_

                Generate encodings conditioning on all modalities or a subset of modalities.

        Returns:
            ModelOutput instance with fields:
                z (torch.Tensor (N, n_data, latent_dim))
                one_latent_space (bool) = True
        """
        self.eval()

        cond_mod = super().encode(inputs, cond_mod, N, **kwargs).cond_mod
        flatten = kwargs.pop("flatten", False)
        if len(cond_mod) == self.n_modalities:
            output = self.joint_encoder(inputs.data)
            z = rsample_from_gaussian(
                output.embedding, output.log_covariance, N, return_mean, flatten=flatten
            )

        elif len(cond_mod) != 1:
            z = self._sample_from_poe_subset_exact(
                cond_mod, inputs.data, N, return_mean=return_mean, flatten=flatten
            )

        elif len(cond_mod) == 1:
            cond_mod = cond_mod[0]
            output = self.encoders[cond_mod](inputs.data[cond_mod])
            z = rsample_from_gaussian(
                output.embedding, output.log_covariance, N, return_mean, flatten=flatten
            )

        else:
            raise AttributeError(
                f"Too many modalities passed to the encode function : {cond_mod}."
            )

        return ModelOutput(z=z, one_latent_space=True)

    def forward(self, inputs: MultimodalBaseDataset, **kwargs) -> ModelOutput:
        """Performs a forward pass of the JMVAE model on inputs.

        Args:
            inputs (MultimodalBaseDataset)
            warmup (int) : number of warmup epochs to do. The weigth of the regularization augments
                linearly to reach 1 at the end of the warmup. The enforces the optimization of
                the reconstruction term only at first.
            epoch (int) : the epoch number during which forward is called.

        Returns:
            ModelOutput
        """
        # check that the dataset is not incomplete
        super().forward(inputs)

        epoch = kwargs.pop("epoch", 1)

        # Compute the reconstruction term
        joint_output = self.joint_encoder(inputs.data)
        mu, log_var = joint_output.embedding, joint_output.log_covariance

        sigma = torch.exp(0.5 * log_var)
        qz_xy = dist.Normal(mu, sigma)
        z_joint = qz_xy.rsample()

        recon_loss = 0

        # Decode in each modality
        len_batch = 0
        for mod in self.decoders:
            x_mod = inputs.data[mod]
            len_batch = len(x_mod)
            recon_mod = self.decoders[mod](z_joint).reconstruction
            recon_loss += (
                -self.recon_log_probs[mod](recon_mod, x_mod) * self.rescale_factors[mod]
            ).sum()

        # Compute the KLD to the prior
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) * self.beta

        # Compute the KL between unimodal and joint encoders
        LJM = 0
        for mod in self.encoders:
            output = self.encoders[mod](inputs.data[mod])
            uni_mu, uni_log_var = output.embedding, output.log_covariance
            LJM += (
                1
                / 2
                * (
                    uni_log_var
                    - log_var
                    + (torch.exp(log_var) + (mu - uni_mu) ** 2) / torch.exp(uni_log_var)
                    - 1
                )
            )

        LJM = LJM.sum() * self.alpha

        # Compute the total loss to minimize

        reg_loss = KLD + LJM
        if epoch >= self.warmup:
            annealing_factor = 1
        else:
            annealing_factor = epoch / self.warmup
        elbo = (recon_loss + KLD) / len_batch
        loss_sum = recon_loss + annealing_factor * reg_loss
        loss = loss_sum / len_batch

        metrics = dict(
            loss_no_ponderation=reg_loss + recon_loss, beta=annealing_factor, elbo=elbo
        )

        output = ModelOutput(loss=loss, loss_sum=loss_sum, metrics=metrics)

        return output

    def _sample_from_poe_subset_exact(
        self, subset: list, data: dict, N=1, return_mean=False, flatten=False
    ):
        """Sample from the product of experts for infering from a subset of modalities."""
        # Get all the experts' means and logvars
        mus, logvars = [], []
        for mod in subset:
            vae_output = self.encoders[mod](data[mod])
            mus.append(vae_output.embedding)
            logvars.append(vae_output.log_covariance)

        # Compute the product of experts
        joint_mu, joint_logvar = stable_poe(torch.stack(mus), torch.stack(logvars))
        z = rsample_from_gaussian(joint_mu, joint_logvar, N, return_mean, flatten)
        return z


    @torch.no_grad()
    def compute_conditional_nll(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, tuple, str],
        target_mod: Union[list, tuple, str] = "rest",
        K: int = 1000,
        batch_size_K: int = 100,
    ):
        """Estimate the negative conditional log-likelihood for JMVAE.

        Args:
            inputs (MultimodalBaseDataset): a batch of complete samples (no masks).
            cond_mod (Union[list, tuple, str]): conditioning set C; list/tuple of names or 'all'.
            target_mod (Union[list, tuple, str]): target set T; 'rest' (default) = complement of C,
                'all' = all modalities, or list/tuple/single name.
            K (int): number of importance samples. Default: 1000.
            batch_size_K (int): mini-batch size along K. Default: 100.

        Returns:
            The negative conditional log-likelihood summed over the batch.
        """
        self.eval()
        if hasattr(inputs, "masks"):
            raise AttributeError(
                "The compute_conditional_nll method is not implemented for incomplete datasets."
            )

        # --- Resolve conditioning set C ---
        if isinstance(cond_mod, str):
            if cond_mod == "all":
                C = list(self.encoders.keys())
            else:
                C = [cond_mod]
        else:
            C = list(cond_mod)
        if len(C) == 0:
            raise ValueError("cond_mod must contain at least one modality.")

        # --- Resolve target set T ---
        all_mods = list(inputs.data.keys())
        if isinstance(target_mod, str):
            if target_mod == "rest":
                T = [m for m in all_mods if m not in C]
            elif target_mod == "all":
                T = all_mods
            else:
                T = [target_mod]
        else:
            T = list(target_mod)
        if len(T) == 0:
            raise ValueError("target_mod resolves to an empty set.")

        # --- Obtain q(z | x_C) parameters (mu_C, logvar_C) ---
        if len(C) == self.n_modalities:
            # Joint encoder
            joint_out = self.joint_encoder(inputs.data)
            mu_C, logvar_C = joint_out.embedding, joint_out.log_covariance
        elif len(C) == 1:
            # Single encoder
            m = C[0]
            out = self.encoders[m](inputs.data[m])
            mu_C, logvar_C = out.embedding, out.log_covariance
        else:
            # PoE over the subset C (Gaussian PoE => Gaussian)
            mus, logvars = [], []
            for m in C:
                out = self.encoders[m](inputs.data[m])
                mus.append(out.embedding)
                logvars.append(out.log_covariance)
            mu_C, logvar_C = stable_poe(torch.stack(mus), torch.stack(logvars))

        sigma_C = torch.exp(0.5 * logvar_C)
        qz_xc = dist.Normal(mu_C, sigma_C)

        # --- Sample K latents: (n_data, K, latent_dim) ---
        z_samples = qz_xc.rsample([K]).permute(1, 0, 2)
        n_data, _, _ = z_samples.shape

        # Standard Normal prior
        prior = dist.Normal(0.0, 1.0)

        ll = 0
        for i in range(n_data):
            start_idx = 0
            stop_idx = min(start_idx + batch_size_K, K)
            ln_terms = []

            while start_idx < stop_idx:
                latents = z_samples[i][start_idx:stop_idx]  # (bK, latent_dim)

                # Sum log p(x_t | z) over targets T
                lpx_t_z = 0
                for mod in T:
                    recon = self.decoders[mod](latents)["reconstruction"]  # (bK, ...)
                    x_m = inputs.data[mod][i]                               # (...)
                    lpx_t_z += (
                        self.recon_log_probs[mod](
                            recon, torch.stack([x_m] * len(recon))
                        )
                        .reshape(recon.size(0), -1)
                        .sum(-1)
                    )

                # log p(z) and log q(z | x_C)
                lpz = prior.log_prob(latents).sum(dim=-1)
                q_i = dist.Normal(mu_C[i], sigma_C[i])
                lqz_xc = q_i.log_prob(latents).sum(dim=-1)

                # IWAE log-weights over this chunk
                ln_terms.append(torch.logsumexp(lpx_t_z + lpz - lqz_xc, dim=0))

                # Next chunk
                start_idx += batch_size_K
                stop_idx = min(stop_idx + batch_size_K, K)

            # Aggregate across chunks, then account for K
            ll += torch.logsumexp(torch.stack(ln_terms), dim=0) - np.log(K)

        # Return negative conditional log-likelihood summed over the batch
        return -ll
