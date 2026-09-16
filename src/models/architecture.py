from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SVHunterSubwindowEncoder(nn.Module):
    def __init__(
        self,
        feature_count: int = 9,
    ) -> None:
        """Initialize the convolutional subwindow encoder."""

        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(1, feature_count), padding="valid"),  # 200
            nn.MaxPool2d(kernel_size=(2, 1)),  # 100
            nn.Conv2d(128, 64, kernel_size=(3, 1), padding="valid"),  # 98
            nn.MaxPool2d(kernel_size=(2, 1)),  # 49
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(2, 1), padding="valid"),  # 48
            nn.MaxPool2d(kernel_size=(2, 1)),  # 24
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding="valid"),  # 22
            nn.MaxPool2d(kernel_size=(2, 1)),  # 11
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(2, 1), padding="valid"),  # 10
            nn.MaxPool2d(kernel_size=(2, 1)),  # 5
            nn.Conv2d(64, 64, kernel_size=(2, 1), padding="valid"),  # 4
            nn.MaxPool2d(kernel_size=(2, 1)),  # 2
            nn.Conv2d(64, 64, kernel_size=(2, 1), padding="valid"),  # 1
        )
        self.output_dimension = 64

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Encode one batch of subwindows into flat CNN embeddings."""

        encoded_inputs = self.layers(inputs)
        return torch.flatten(encoded_inputs, start_dim=1)


class SVHunterMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dimension: int = 100,
        attention_head_count: int = 32,
        key_dimension: int = 32,
        dropout: float = 0.3,
    ) -> None:
        """Initialize the attention projections and dropout rate."""

        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.attention_head_count = attention_head_count
        self.key_dimension = key_dimension
        self.attention_inner_dimension = attention_head_count * key_dimension
        self.query_projection = nn.Linear(
            embedding_dimension, self.attention_inner_dimension
        )
        self.key_projection = nn.Linear(
            embedding_dimension, self.attention_inner_dimension
        )
        self.value_projection = nn.Linear(
            embedding_dimension, self.attention_inner_dimension
        )
        self.output_projection = nn.Linear(
            self.attention_inner_dimension, embedding_dimension
        )
        self.dropout = dropout

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Apply multi-head scaled dot-product self-attention."""

        batch_size, sequence_length, _ = inputs.shape
        query = self.query_projection(inputs).view(
            batch_size,
            sequence_length,
            self.attention_head_count,
            self.key_dimension,
        )
        key = self.key_projection(inputs).view(
            batch_size,
            sequence_length,
            self.attention_head_count,
            self.key_dimension,
        )
        value = self.value_projection(inputs).view(
            batch_size,
            sequence_length,
            self.attention_head_count,
            self.key_dimension,
        )

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attention_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(
            batch_size, sequence_length, self.attention_inner_dimension
        )
        return self.output_projection(attention_output)


class SVHunterTransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dimension: int = 100,
        attention_head_count: int = 32,
        key_dimension: int = 32,
        dropout: float = 0.3,
    ) -> None:
        """Initialize the attention and feed-forward layers."""

        super().__init__()
        self.layer_normalization_1 = nn.LayerNorm(embedding_dimension)
        self.attention = SVHunterMultiHeadAttention(
            embedding_dimension=embedding_dimension,
            attention_head_count=attention_head_count,
            key_dimension=key_dimension,
            dropout=dropout,
        )
        self.layer_normalization_2 = nn.LayerNorm(embedding_dimension)
        self.feed_forward_network = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Apply one residual transformer block."""

        attended_inputs = inputs + self.attention(self.layer_normalization_1(inputs))
        return attended_inputs + self.feed_forward_network(
            self.layer_normalization_2(attended_inputs)
        )


class SVHunterModel(nn.Module):
    def __init__(
        self,
        input_length: int = 2000,
        feature_count: int = 9,
        subwindow_size: int = 200,
        subwindow_count: int = 10,
        embedding_dimension: int = 100,
        attention_head_count: int = 4,
        key_dimension: int = 32,
        transformer_block_count: int = 3,
        multilayer_perceptron_hidden_dimension: int = 128,
        attention_dropout: float = 0.3,
        head_dropout: float = 0.4,
    ) -> None:
        """Initialize the CNN-Transformer classifier."""

        super().__init__()
        if input_length != subwindow_size * subwindow_count:
            raise ValueError("input_length must equal subwindow_size * subwindow_count")

        self.input_length = input_length
        self.feature_count = feature_count
        self.subwindow_size = subwindow_size
        self.subwindow_count = subwindow_count
        self.encoder = SVHunterSubwindowEncoder(feature_count=feature_count)
        self.patch_projection = nn.Linear(
            self.encoder.output_dimension, embedding_dimension
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, subwindow_count, embedding_dimension)
        )
        self.transformer_blocks = nn.ModuleList(
            [
                SVHunterTransformerBlock(
                    embedding_dimension=embedding_dimension,
                    attention_head_count=attention_head_count,
                    key_dimension=key_dimension,
                    dropout=attention_dropout,
                )
                for _ in range(transformer_block_count)
            ]
        )
        self.sequence_normalization = nn.LayerNorm(embedding_dimension)
        self.classifier = nn.Sequential(
            nn.Linear(
                embedding_dimension,
                multilayer_perceptron_hidden_dimension,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(
                multilayer_perceptron_hidden_dimension,
                multilayer_perceptron_hidden_dimension,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(multilayer_perceptron_hidden_dimension, 1),
        )

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Predict subwindow-level structural variant logits."""

        if inputs.ndim != 3:
            raise ValueError("Expected input shape (batch, 2000, 9)")
        if (
            inputs.shape[1] != self.input_length
            or inputs.shape[2] != self.feature_count
        ):
            raise ValueError(
                f"Expected input shape (batch, {self.input_length}, {self.feature_count})"
            )

        batch_size = inputs.shape[0]
        subwindow_inputs = inputs.view(
            batch_size,
            self.subwindow_count,
            self.subwindow_size,
            self.feature_count,
        )
        subwindow_inputs = subwindow_inputs.unsqueeze(2).reshape(
            batch_size * self.subwindow_count,
            1,
            self.subwindow_size,
            self.feature_count,
        )
        subwindow_embeddings = self.encoder(subwindow_inputs)
        subwindow_embeddings = subwindow_embeddings.view(
            batch_size,
            self.subwindow_count,
            self.encoder.output_dimension,
        )
        subwindow_embeddings = self.patch_projection(subwindow_embeddings)
        subwindow_embeddings = subwindow_embeddings + self.position_embedding

        for block in self.transformer_blocks:
            subwindow_embeddings = block(subwindow_embeddings)

        subwindow_embeddings = self.sequence_normalization(subwindow_embeddings)
        return self.classifier(subwindow_embeddings).squeeze(-1)
