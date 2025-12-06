import torch
from torch import nn

from .augmentations import GaussianSmoothing


'''class GRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
    ):
        super(GRUDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # GRU layers
        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out'''

'''import torch
from torch import nn

from .augmentations import GaussianSmoothing


class GRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
    ):
        super(GRUDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional

        self.inputLayerNonlinearity = torch.nn.Softsign()
        # LayerNorm on neural feature dimension
        self.inputLayerNorm = nn.LayerNorm(neural_dim)

        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # GRU layers
        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # LayerNorm on RNN hidden dimension
        if self.bidirectional:
            self.rnnLayerNorm = nn.LayerNorm(hidden_dim * 2)
        else:
            self.rnnLayerNorm = nn.LayerNorm(hidden_dim)

        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)
        transformedNeural = self.inputLayerNorm(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # LayerNorm on RNN outputs
        hid = self.rnnLayerNorm(hid)

        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out
'''

'''class GRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
    ):
        super(GRUDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # LSTM layers (replacing GRU)
        self.lstm_decoder = nn.LSTM(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.lstm_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # apply RNN layer (now LSTM)
        num_directions = 2 if self.bidirectional else 1
        batch_size = transformedNeural.size(0)

        h0 = torch.zeros(
            self.layer_dim * num_directions,
            batch_size,
            self.hidden_dim,
            device=self.device,
        ).requires_grad_()

        c0 = torch.zeros(
            self.layer_dim * num_directions,
            batch_size,
            self.hidden_dim,
            device=self.device,
        ).requires_grad_()

        hid, _ = self.lstm_decoder(stridedInputs, (h0.detach(), c0.detach()))

        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out

'''

'''import torch
from torch import nn

from .augmentations import GaussianSmoothing


class GRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
    ):
        super(GRUDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional

        self.inputLayerNonlinearity = torch.nn.Softsign()

        # Unfold over time dimension to create temporal "patches"
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )

        # Gaussian smoothing over time
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )

        # Day-specific linear transforms
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))
        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # ───────── LayerNorms ─────────
        # Normalize per time step over feature dimension
        self.day_norm = nn.LayerNorm(neural_dim)

        # Normalize giant unfolded vectors before LSTM
        self.pre_rnn_norm = nn.LayerNorm(neural_dim * kernelLen)

        # Normalize LSTM hidden output before final FC
        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.post_rnn_norm = nn.LayerNorm(rnn_out_dim)

        # ───────── LSTM (replacing GRU) ─────────
        # Keep attribute name "gru_decoder" so existing code still works.
        self.gru_decoder = nn.LSTM(
            neural_dim * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )
        # Use default PyTorch init (no custom orthogonal/Xavier on stacked gates)

        # Input layers (kept for compatibility with original model)
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # Output layer: from hidden state(s) to class logits (+1 for CTC blank)
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(hidden_dim * 2, n_classes + 1)
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)

    def forward(self, neuralInput, dayIdx):
        # neuralInput: [B, T, D]
        # 1) Smooth over time
        neuralInput = torch.permute(neuralInput, (0, 2, 1))       # [B, D, T]
        neuralInput = self.gaussianSmoother(neuralInput)          # [B, D, T]
        neuralInput = torch.permute(neuralInput, (0, 2, 1))       # [B, T, D]

        # 2) Day-specific transform
        # dayIdx: [B]
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)  # [B, D, D]
        dayBias = torch.index_select(self.dayBias, 0, dayIdx)        # [B, 1, D]

        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + dayBias  # [B, T, D]

        transformedNeural = self.inputLayerNonlinearity(transformedNeural)  # [B, T, D]
        transformedNeural = self.day_norm(transformedNeural)                # LN over D

        # 3) Temporal unfolding (kernel/stride)
        # transformedNeural: [B, T, D] → [B, D, T, 1] for Unfold
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )  # [B, T', D * kernelLen]

        # LayerNorm before LSTM to tame huge input dim
        stridedInputs = self.pre_rnn_norm(stridedInputs)  # LN over last dim

        # 4) LSTM
        # Let PyTorch handle h0, c0 as zeros internally.
        hid, _ = self.gru_decoder(stridedInputs)  # [B, T', H] or [B, T', 2H]

        # LayerNorm on hidden states before classification
        hid = self.post_rnn_norm(hid)

        # 5) Project to logits
        seq_out = self.fc_decoder_out(hid)  # [B, T', n_classes + 1]
        return seq_out

'''

import torch
from torch import nn

from .augmentations import GaussianSmoothing


class GRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0.25,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
    ):
        super(GRUDecoder, self).__init__()

        # ----------------- original attributes -----------------
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional

        self.inputLayerNonlinearity = torch.nn.Softsign()

        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )

        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )

        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # ----------------- NEW: LayerNorms -----------------
        # LayerNorm over feature dimension after day-specific transform
        self.day_norm = nn.LayerNorm(neural_dim)

        # LayerNorm over unfolded temporal "patch" (size = neural_dim * kernelLen)
        self.pre_rnn_norm = nn.LayerNorm(neural_dim * kernelLen)

        # ----------------- GRU layers (unchanged) -----------------
        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # ----------------- NEW: post-GRU MLP stack -----------------
        rnn_out_dim = hidden_dim * (2 if self.bidirectional else 1)

        self.post_rnn_stack = nn.Sequential(
            nn.Linear(rnn_out_dim, rnn_out_dim),
            nn.LayerNorm(rnn_out_dim),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear(rnn_out_dim, rnn_out_dim),
            nn.LayerNorm(rnn_out_dim),
            nn.Dropout(self.dropout),
        )

        # ----------------- Input layers (unchanged) -----------------
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # ----------------- Output proj (slightly moved) -----------------
        # Now projects from rnn_out_dim → (n_classes + 1)
        self.fc_decoder_out = nn.Linear(rnn_out_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):
        # neuralInput: [B, T, D]
        neuralInput = torch.permute(neuralInput, (0, 2, 1))  # [B, D, T]
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))  # [B, T, D]

        # apply day layer
        # dayIdx: [B]
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)  # [B, D, D]
        dayBias = torch.index_select(self.dayBias, 0, dayIdx)        # [B, 1, D]

        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + dayBias  # [B, T, D]

        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # NEW: LayerNorm over feature dim per time step
        transformedNeural = self.day_norm(transformedNeural)  # [B, T, D]

        # stride/kernel via unfold
        # transformedNeural: [B, T, D] → [B, D, T, 1] for Unfold
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )  # [B, T', D * kernelLen]

        # NEW: normalize the huge patch vector before GRU
        stridedInputs = self.pre_rnn_norm(stridedInputs)  # [B, T', D*kernelLen]

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())  # [B, T', H] or [B, T', 2H]

        # NEW: post-GRU linear + LN + dropout stack
        hid = self.post_rnn_stack(hid)  # [B, T', rnn_out_dim]

        # get seq logits
        seq_out = self.fc_decoder_out(hid)  # [B, T', n_classes+1]
        return seq_out
