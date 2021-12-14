# https://github.com/voithru/voice-activity-detection/blob/013da89d9e6e9b0d3dff2ec8153a52352adf5ce1/vad/models/acam.py#L118
import torch
from torch import Tensor, nn


class ACAM(nn.Module):
    def __init__(self, feature_size,
                 window_size: int = 7,
                 core_hidden_size: int = 128,
                 encoder_hidden_size: int = 128,
                 encoder_output_size: int = 128,
                 dropout: float = 0.5,
                 num_steps: int = 7,
                 action_hidden_size_1: int = 256,
                 action_hidden_size_2: int = 256):
        super(ACAM, self).__init__()

        # get_glimpse == attention + Encoder
        self.attention = Attention() # sw_sensor
        self.encoder = Encoder(
            window_feature_size=feature_size * window_size,
            window_size=window_size,
            encoder_hidden_size=encoder_hidden_size,
            encoder_output_size=encoder_output_size)

        # get_next_input == decoder + get_glimpse
        self.decoder = Decoder(core_hidden_size=core_hidden_size,
                               window_size=window_size)

        self.core = Core(encoder_output_size, core_hidden_size, dropout, False)

        self.action_network = nn.Sequential(
            nn.Linear(core_hidden_size, action_hidden_size_1),
            nn.BatchNorm1d(num_features=action_hidden_size_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(action_hidden_size_1, action_hidden_size_2),
            nn.BatchNorm1d(num_features=action_hidden_size_2),
            nn.ReLU(),
            nn.Dropout(dropout))

        self.classifier = nn.Linear(action_hidden_size_2, window_size)
        self.sigmoid = nn.Sigmoid()

        self.num_steps = num_steps

    def forward(self, features: Tensor):
        if features.dim() == 4 and features.shape[1] == 1:
            features = torch.squeeze(features, 1)

        batch_size, window_size, feature_size = features.size()

        attention = torch.zeros(batch_size, window_size, dtype=torch.float32,
                                device=features.device)
        core_state = None

        for i in range(self.num_steps):
            attended_input = self.attention(selected_input=features,
                                            attention=attention)
            aggregation = self.encoder(attention=attention,
                                       attended_input=attended_input)
            core_output, core_state = self.core(aggregation=aggregation,
                                                state=core_state)

            if i != self.num_steps - 1:
                attention = self.decoder(core_output)

        output = self.action_network(core_output)
        output = self.classifier(output)
        return self.sigmoid(output)


class Decoder(nn.Module):
    def __init__(self, core_hidden_size: int, window_size: int):
        super(Decoder, self).__init__()

        self.transform = nn.Sequential(
            nn.Linear(core_hidden_size, window_size),
            nn.BatchNorm1d(num_features=window_size))
        self.smooth_softmax = SmoothSoftmax()

    def forward(self, core_output: Tensor):
        # core_output: (batch_size, core_hidden_size)
        attention = self.smooth_softmax(self.transform(core_output))
        return attention


class SmoothSoftmax(nn.Module):
    def forward(self, x: Tensor):
        logistic_value = torch.sigmoid(x)
        return logistic_value / logistic_value.sum(dim=-1, keepdim=True)


class Attention(nn.Module):
    def forward(self, selected_input: Tensor, attention: Tensor):
        # selected_input: (batch_size, window_size, feature_size)
        # attention: (batch_size, window_size)

        attended_input = selected_input * attention.unsqueeze(-1)
        return attended_input


class Encoder(nn.Module):
    def __init__(self,
                 window_feature_size: int,
                 window_size: int,
                 encoder_hidden_size: int,
                 encoder_output_size: int):
        super(Encoder, self).__init__()

        self.transform_attention = nn.Sequential(
            nn.Linear(window_size, encoder_hidden_size),
            nn.BatchNorm1d(num_features=encoder_hidden_size),
            nn.ReLU(),
            nn.Linear(encoder_hidden_size, encoder_output_size),
            nn.BatchNorm1d(num_features=encoder_output_size),
        )
        self.transform_attended_input = nn.Sequential(
            nn.Linear(window_feature_size, encoder_hidden_size),
            nn.BatchNorm1d(num_features=encoder_hidden_size),
            nn.ReLU(),
            nn.Linear(encoder_hidden_size, encoder_output_size),
            nn.BatchNorm1d(num_features=encoder_output_size),
        )
        self.aggregate = nn.ReLU()

    def forward(self, attention: Tensor, attended_input: Tensor):
        batch_size, window_size, feature_size = attended_input.size()
        attended_input = attended_input.view(batch_size,
                                             window_size * feature_size)
        return self.aggregate(
            self.transform_attention(attention)
            + self.transform_attended_input(attended_input))


class Core(nn.Module):
    def __init__(self, encoder_output_size: int, hidden_size: int,
                 dropout: float, bidirectional: bool):
        super(Core, self).__init__()

        self.lstm = nn.LSTM(
            input_size=encoder_output_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.recurrent_dropout = nn.Dropout(p=dropout)

    def forward(self, aggregation: Tensor, state=None):
        aggregation = aggregation.unsqueeze(dim=1) # Make sequence length of 1
        if state is not None:
            state = [self.recurrent_dropout(self.layer_norm(s)) for s in state]
        core_output, core_state = self.lstm(aggregation, state)
        core_output = core_output.squeeze(dim=1)
        return core_output, core_state


if __name__ == '__main__':
    import numpy as np

    model = ACAM(64)
    
    inputs = torch.zeros((256, 1, 7, 64))
    print(model(inputs).shape)
    print(sum([np.prod(p.size()) for name, p in model.named_parameters()]))

