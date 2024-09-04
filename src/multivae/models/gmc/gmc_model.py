from typing import Dict, Union
import torch
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseEncoder
from multivae.models.nn import BaseJointEncoder
from torch.nn import Module, ModuleDict

from ...data.datasets import MultimodalBaseDataset
from .gmc_config import GMCConfig


class GMC(Module):
    def __init__(self, config: GMCConfig, processors: Dict[str, BaseEncoder], joint_encoder : BaseJointEncoder, shared_encoder : BaseEncoder) -> None:
        # """
        # Implements the Geometric Multimodal Contrastive Learning from :
        # https://arxiv.org/abs/2202.03390
        
        # This implementation is based on the original implementation : https://github.com/miguelsvasco/gmc
        
        # Args:
        #     config (GMCConfig) : Contains all the hyperparameters for the model
        #     processors (Dict[str, BaseEncoder]) : A dictionary containing an encoder for each modality. Each encoder is
        #         expected to be a pythae BaseEncoder instance.
        #     joint_encoder (BaseJointEncoder) : The joint encoder used for computing the joint representation
        #     shared_encoder (BaseEncoder) : The shared projection head. 
        # """
        
        super().__init__()

        self.config = config
        self.n_modalities = config.n_modalities
        self.latent_dim = config.embedding_dim
        self.common_dim = config.common_dim
        self.temperature = config.temperature
        self.use_all_singular_values = config.use_all_singular_values
        self.set_networks(processors)
        self.set_joint_encoder(joint_encoder)
        self.set_shared_encoder(shared_encoder)


    def set_networks(self, networks):
        self.networks = ModuleDict()
        assert (
            len(networks) == self.n_modalities
        ), "The number of provided processors doesn't match the number of modalities."

        for m in networks:
            if not isinstance(networks[m], BaseEncoder):
                raise AttributeError(
                    "The GMC processors must be instances of pythae BaseEncoder class."
                )
            if networks[m].latent_dim != self.common_dim:
                raise AttributeError(
                    f"One of the GMC processor network (modality : {m}) doesn't have the same common dim as the model"
                    f" itself. ({networks[m].latent_dim} is different {self.common_dim})"
                )
            self.networks[m] = networks[m]
            
    def set_shared_encoder(self, shared_encoder):
        if not isinstance(shared_encoder, BaseEncoder):
            raise AttributeError("The shared encoder must be an instance of pythae BaseEncoder class")
        if shared_encoder.latent_dim != self.latent_dim :
            raise AttributeError("The shared encoder must have the same latent dim as the model. "
                                 f"In the model config latent dim is {self.latent_dim} different than shared encoder latent dim {shared_encoder.latent_dim}")
        self.shared_encoder = shared_encoder
        
    def set_joint_encoder(self, joint_encoder):
        
        if not isinstance(joint_encoder, BaseJointEncoder):
            raise AttributeError("The joint encoder must be an instance of the ~multivae.models.nn.base_architectures.BaseJointEncoder class.")
        if not joint_encoder.latent_dim != self.common_dim:
            raise AttributeError("The joint encoder 'latent_dim' attribute doesn't match the model config 'common_dim' attribute. ")
        self.joint_encoder = joint_encoder
        
    def forward(self, inputs: MultimodalBaseDataset):
        """Basic training step"""
        
        # Compute all the modalities specific representations
        
        modalities_z = dict()
        
        for m in inputs:
            
            h = self.networks[m](inputs.data[m]).embedding
            z = self.shared_encoder(h).embedding
            modalities_z[m] = z
        
        # Compute the joint representation
        joint_h = self.joint_encoder(inputs.data).embedding
        joint_z = self.shared_encoder(joint_h).embedding
        
        # Compute the loss
        loss = self.infonce(modalities_z, joint_z)
        
        return loss
        
    
    def infonce(self, modalities_z, joint_z):
        
        batch_size = len(joint_z)
        joint_mod_loss_sum = 0
        
        for mod in modalities_z:
            # Negative pairs : joint and mod
            out_joint_mod = torch.cat(
                [joint_z, modalities_z[mod]], dim=0
            )
            # [2*B, 2*B]
            sim_matrix_joint_mod = torch.exp(
                torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / self.temperature
            )
            # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
            mask_joint_mod = (
                torch.ones_like(sim_matrix_joint_mod)
                - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
            ).bool()
            # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
            sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
                mask_joint_mod
            ).view(2 * batch_size, -1)

            # Positive pairs: cosine loss joint-modality
            pos_sim_joint_mod = torch.exp(
                torch.sum(
                    joint_z * modalities_z[mod], dim=-1
                )
                / self.temperature
            )
            # [2*B]
            pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod

        loss = torch.mean(joint_mod_loss_sum)
        
        return loss
        
    def encode(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: str = "all",
        **kwargs,
    ) :
        """Encode the data using the one or several modalities.

        Args:
            inputs (MultimodalBaseDataset): The data to encode.
            cond_mod (str, optional): Either a modality's name or 'all' (to compute joint encoding). Defaults to "all".

        Raises:
            AttributeError: _description_
        """
        
        
        
        # joint encoding
        if cond_mod == 'all':
            z = self.shared_encoder(self.joint_encoder(inputs.data).embedding)
        # modality encoding
        elif cond_mod in self.networks:
            z = self.shared_encoder(self.networks[cond_mod](inputs.data).embedding)
        else :
            raise AttributeError("cond_mod must be : either a modality's name or equal to 'all' (for joint representation). ")
        
        return ModelOutput(embedding = z)
            
                