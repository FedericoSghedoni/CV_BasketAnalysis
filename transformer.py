import torch
import torch.nn as nn
    
class Transformer(nn.Module):
    def __init__(self, tgt_size, n_feature, d_model, dropout):
        super(Transformer, self).__init__()

        self.to_embed = nn.Linear(n_feature, d_model)

        # learnable positional embedding
        # $> initialized as N(0, 0.02)
        # SOLITAMENTE IL POSITIONAL EMBEDDING È FISSATO DA VALORI SINUSOIDALI,
        # IN QUESTA IMPLEMENTAZIONE INVECE VIENE IMPARATO.
        self.pos_emb = nn.Parameter(torch.normal(
            mean=0, std=0.02,
            size=(1, d_model + 1, d_model)
        ))
        torch.nn.init.xavier_normal_(self.pos_emb)

        # self.fc = nn.Linear(d_model, tgt_size)
        self.dropout = nn.Dropout(dropout)

        # learnable class token
        # $> initialized as N(0, 0.02)
        self.class_token = nn.Parameter(torch.normal(
            mean=0, std=0.02,
            size=(1, d_model)
        ))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=16, dim_feedforward=2048,
            activation=lambda x: nn.SiLU()(x), batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model + 1),
            nn.Linear(d_model + 1, d_model // 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, d_model // 4),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, tgt_size),
        )

    def forward(self, src):

        tokens = self.to_embed(src)
        __class_token = self.class_token.repeat((src.shape[0], 1, 1))
        tokens = torch.cat([__class_token, tokens], 1)
        tokens = tokens + self.pos_emb

        # apply encoding to tokens
        # (B, T+1, E) -> (B, T+1, E)
        encoded_tokens = self.encoder(tokens)
	
	    # QUI PRENDIAMO SOLO IL CLASS TOKEN IN USCITA E LO DIAMO ALLA TESTA DI CLASSIFICAZIONE
        # use (encoded) class token only
        # to predict the output class
        # $> (B, E) -> (B, N_classes)
        encoded_class_token = encoded_tokens[:, :, 0]
        # d = torch.cat([encoded_class_token, fs_len[:, None]], 1)
        y = self.classifier(encoded_class_token)

        # Check the output
        print(f'La prediction del modello è: {y}')
        return y[0]