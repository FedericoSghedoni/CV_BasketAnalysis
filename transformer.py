import torch
import torch.nn as nn
    
class Transformer(nn.Module):
    def __init__(self, tgt_size, n_feature, d_model, nhead=4, dropout_rate=0.1, num_layers=2):
        super(Transformer, self).__init__()

        self.to_embed = nn.Linear(n_feature, d_model)

        # learnable positional embedding
        # $> initialized as N(0, 0.02)
        self.pos_emb = nn.Parameter(torch.normal(
            mean=0, std=0.02,
            size=(d_model + 1, d_model)
        ))
        torch.nn.init.xavier_normal_(self.pos_emb)

        # learnable class token
        # $> initialized as N(0.5, 0.02)
        self.class_token = nn.Parameter(torch.normal(
            mean=0.5, std=0.02,
            size=(1, d_model)
        ))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048,
            activation=lambda x: nn.SiLU()(x), batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model + 1),
            nn.Linear(d_model + 1, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, d_model // 4),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 4, tgt_size),
            nn.Sigmoid()
        )

    def forward(self, src):

        tokens = self.to_embed(src)
        __class_token = self.class_token.repeat((1, 1))
        tokens = torch.cat([__class_token, tokens], 0)
        tokens = tokens + self.pos_emb

        # apply encoding to tokens
        # (T+1, E) -> (T+1, E)
        encoded_tokens = self.encoder(tokens)

        # use (encoded) class token only to predict the output class
        # $> (E) -> (N_classes)
        encoded_class_token = encoded_tokens[:, 0]
        y = self.classifier(encoded_class_token)

        return y