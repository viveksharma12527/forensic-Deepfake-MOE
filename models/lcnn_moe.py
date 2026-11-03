import torch
import torch.nn as nn
from typing import Sequence
from models.moe_layer import MoELayer
from models.gating_networks import FCNNGatingNetwork, TFGatingNetwork
from models.audio_frontend import AudioFrontend


class LCNNMoE(nn.Module):
    def __init__(self,
                 num_class: int = 2,
                 num_experts: Sequence[int] = (8, 8, 8, 8, 8, 8, 8, 8, 8, 4),  # Iterable of number of experts for each MoE layer in the modified LCNN (the last value controls fully-connected experts).
                 redux_channels: int = 16,  # Known (reduced) number of channels for the TFGatingNetwork models.
                 gating_hidden_dim: int = 64,  # Hidden dim of to_logits
                 noisy_gating: bool = True,  # If True: add noise to the logit before top-k sampling. If False, sampling is deterministic (and it disables differentiable load computation).
                 routing_type: str = "topk_softmax",  # Routing strategy: "topk_softmax" normalizes over selected experts; "full_softmax" normalizes over all experts before selecting top-k.
                 return_emb: bool = False,  # Whether to return the audio embeddings alongside the prediction logits
                 ):
        super().__init__()

        self.return_emb = return_emb

        # Helper dictionary to initialize TFGatingNetwork models.
        tf_kwargs_factory = {
            'redux_channels': redux_channels,
            'noisy_gating': noisy_gating,
            'hidden_dim': gating_hidden_dim,
            "routing_type": routing_type,
        }

        # Instantiate the audio frontend
        self.audio_frontend = AudioFrontend()

        # ======== Feature Extraction Part (First Part) ======== #

        self.dropout1 = nn.Dropout(0.2)
        self.conv1 = MoELayer(
            # Explicit definition of TFGatingNetwork. This ensures that the gating network does not increase the number
            # of channels (the first MoE layer is the only case where in_channels < redux_channels). For convenience,
            # other MoE layers unpack tf_kwargs_factory.
            gating_net=TFGatingNetwork(num_experts=num_experts[0], in_channels=1, **tf_kwargs_factory),
            # List of experts via list comprehension: [nn.Layer(...) for _ in range(num_experts)]
            experts=[nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), padding=(2, 2), stride=(1, 1))
                     for _ in range(num_experts[0])]
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv4 = MoELayer(
            gating_net=TFGatingNetwork(num_experts=num_experts[1], in_channels=32, **tf_kwargs_factory),
            experts=[nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
                     for _ in range(num_experts[1])]
        )
        self.batchnorm6 = nn.BatchNorm2d(32)
        self.conv7 = MoELayer(
            gating_net=TFGatingNetwork(num_experts=num_experts[2], in_channels=32, **tf_kwargs_factory),
            experts=[nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
                     for _ in range(num_experts[2])]
        )
        self.maxpool9 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.batchnorm10 = nn.BatchNorm2d(48)
        self.conv11 = MoELayer(
            gating_net=TFGatingNetwork(num_experts=num_experts[3], in_channels=48, **tf_kwargs_factory),
            experts=[nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
                     for _ in range(num_experts[3])]
        )
        self.batchnorm13 = nn.BatchNorm2d(48)
        self.conv14 = MoELayer(
            gating_net=TFGatingNetwork(num_experts=num_experts[4], in_channels=48, **tf_kwargs_factory),
            experts=[nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
                     for _ in range(num_experts[4])]
        )
        self.maxpool16 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv17 = MoELayer(
            gating_net=TFGatingNetwork(num_experts=num_experts[5], in_channels=64, **tf_kwargs_factory),
            experts=[nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
                     for _ in range(num_experts[5])]
        )
        self.batchnorm19 = nn.BatchNorm2d(64)
        self.conv20 = MoELayer(
            gating_net=TFGatingNetwork(num_experts=num_experts[6], in_channels=64, **tf_kwargs_factory),
            experts=[nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
                     for _ in range(num_experts[6])]
        )
        self.batchnorm22 = nn.BatchNorm2d(32)
        self.conv23 = MoELayer(
            gating_net=TFGatingNetwork(num_experts=num_experts[7], in_channels=32, **tf_kwargs_factory),
            experts=[nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
                     for _ in range(num_experts[7])]
        )
        self.batchnorm25 = nn.BatchNorm2d(32)
        self.conv26 = MoELayer(
            gating_net=TFGatingNetwork(num_experts=num_experts[8], in_channels=32, **tf_kwargs_factory),
            experts=[nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
                     for _ in range(num_experts[8])]
        )
        self.maxpool28 = nn.AdaptiveMaxPool2d((16, 8))

        # ======== Classification Part (Second Part) ======== #

        # First FC layer (fc29) is a MoE. To use a shared layer uncomment:
        # > self.fc29 = nn.Linear(32 * 16 * 8, 128)
        self.fc29 = MoELayer(
            gating_net=FCNNGatingNetwork(
                num_experts=num_experts[9],
                noisy_gating=noisy_gating,
                input_dim=32 * 16 * 8,
                routing_type=routing_type
            ),
            experts=[nn.Linear(32 * 16 * 8, 128) for _ in range(num_experts[9])]
        )
        self.batchnorm31 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.7)

        self.fc32 = nn.Linear(64, num_class)

        # Last FC layer (fc32) is not a MoE (but it could be...)
        # > self.fc32 = MoELayer(
        # >   gating_net=FCNNGatingNetwork(num_experts=num_experts[10], noisy_gating=noisy_gating, input_dim=64),
        # >   experts=[nn.Linear(64, num_class) for _ in range(num_experts[10])]
        # > )


    def mfm2(self, x):
        out1, out2 = torch.chunk(x, 2, 1)
        return torch.max(out1, out2)

    def mfm3(self, x):
        n, c, y, z = x.shape
        out1, out2, out3 = torch.chunk(x, 3, 1)
        res1 = torch.max(torch.max(out1, out2), out3)
        tmp1 = out1.flatten()
        tmp1 = tmp1.reshape(len(tmp1), -1)
        tmp2 = out2.flatten()
        tmp2 = tmp2.reshape(len(tmp2), -1)
        tmp3 = out3.flatten()
        tmp3 = tmp3.reshape(len(tmp3), -1)
        res2 = torch.cat((tmp1, tmp2, tmp3), 1)
        res2 = torch.median(res2, 1)[0]
        res2 = res2.reshape(n, -1, y, z)
        return torch.cat((res1, res2), 1)

    def forward(self, x, k: int = 1, temperature: float = 1.0) -> torch.tensor:
        """
        Note that "k" can be passed as an argument, and it is not a class property (e.g., self.k). This allows to pass
        a different k at each epoch; you can control this at training time to implement a schedule. For example, for the
        first N epochs, use k=4. Then, for the next N epochs, use k=2. Then, finalize the training with k=1.

        The same goes for "temperature" that could be scheduled and increase over the epochs to keep experts more evenly
        activated. In short, temperature controls the entropy of a distribution:
            - T=1 (default): standard (top-k) gating softmax.
            - Low temperature (T<1): softmax probabilities become more “peaked”, the highest logit dominates even more.
            - High temperature (T>1): the probabilities become more uniform or “softened.”

        Calling the forward method with k=1 ensures a single-expert routing through the model at inference time.
        """

        with torch.no_grad():
            x = self.audio_frontend(x)

        # Auxiliary losses are initialized to zero and accumulated across all MoE/GatingNetworks
        total_importance_loss = 0.0
        total_load_loss = 0.0

        x = x.unsqueeze(1)
        x, importance_loss, load_loss = self.conv1(x, k, temperature)
        total_importance_loss += importance_loss
        total_load_loss += load_loss
        x = self.mfm2(x)
        x = self.maxpool3(x)
        x, importance_loss, load_loss = self.conv4(x, k, temperature)
        total_importance_loss += importance_loss
        total_load_loss += load_loss
        x = self.mfm2(x)
        x = self.batchnorm6(x)
        x, importance_loss, load_loss = self.conv7(x, k, temperature)
        total_importance_loss += importance_loss
        total_load_loss += load_loss
        x = self.mfm2(x)
        x = self.maxpool9(x)
        x = self.batchnorm10(x)
        x, importance_loss, load_loss = self.conv11(x, k, temperature)
        total_importance_loss += importance_loss
        total_load_loss += load_loss
        x = self.mfm2(x)
        x = self.batchnorm13(x)
        x, importance_loss, load_loss = self.conv14(x, k, temperature)
        total_importance_loss += importance_loss
        total_load_loss += load_loss
        x = self.mfm2(x)
        x = self.maxpool16(x)
        x, importance_loss, load_loss = self.conv17(x, k, temperature)
        total_importance_loss += importance_loss
        total_load_loss += load_loss
        x = self.mfm2(x)
        x = self.batchnorm19(x)
        x, importance_loss, load_loss = self.conv20(x, k, temperature)
        total_importance_loss += importance_loss
        total_load_loss += load_loss
        x = self.mfm2(x)
        x = self.batchnorm22(x)
        x, importance_loss, load_loss = self.conv23(x, k, temperature)
        total_importance_loss += importance_loss
        total_load_loss += load_loss
        x = self.mfm2(x)
        x = self.batchnorm25(x)
        x, importance_loss, load_loss = self.conv26(x, k, temperature)
        total_importance_loss += importance_loss
        total_load_loss += load_loss
        x = self.mfm2(x)
        x = self.maxpool28(x)

        x = x.view(-1, 32 * 16 * 8)
        x, importance_loss, load_loss = self.fc29(x, k, temperature)
        total_importance_loss += importance_loss
        total_load_loss += load_loss
        emb = self.mfm2(x)
        x = self.batchnorm31(emb)
        logits = self.fc32(x)

        if self.return_emb:
            # You shouldn't need this here since embeddings are not used in this model...
            return logits, emb, total_importance_loss, total_load_loss

        return logits, total_importance_loss, total_load_loss
