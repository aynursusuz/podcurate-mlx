# Vendored from aufklarer/SpeechBrain-ECAPA-VoxCeleb-20M-MLX
# Revision: e749e6e08557f4a3ceb6ce3bf6f0b79efe592a77. Apache-2.0.
# See ecapa_license.py for the full license.
# Modified: removed unused language-classification classes.
"""Pure MLX ECAPA runtime used to validate exported safetensors."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def reflect_pad_1d(values: mx.array, padding: int) -> mx.array:
    """Match SpeechBrain ``Conv1d``'s reflected temporal padding."""
    if padding == 0:
        return values
    if values.shape[1] <= padding:
        raise ValueError(
            f"reflection padding {padding} requires more than {padding} frames"
        )
    left = values[:, 1 : padding + 1, :][:, ::-1, :]
    right = values[:, -padding - 1 : -1, :][:, ::-1, :]
    return mx.concatenate([left, values, right], axis=1)


class TDNNBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=True,
        )
        self.norm = nn.BatchNorm(output_channels)

    def __call__(self, values: mx.array) -> mx.array:
        # SpeechBrain orders this block as convolution, activation, batch norm.
        padded = reflect_pad_1d(values, self.padding)
        return self.norm(nn.relu(self.conv(padded)))


class Res2NetBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        scale: int = 8,
    ) -> None:
        super().__init__()
        self.scale = scale
        width = channels // scale
        self.blocks = [
            TDNNBlock(width, width, kernel_size, dilation=dilation)
            for _ in range(scale - 1)
        ]

    def __call__(self, values: mx.array) -> mx.array:
        chunks = mx.split(values, self.scale, axis=-1)
        outputs = [chunks[0]]
        for index, block in enumerate(self.blocks):
            block_input = chunks[index + 1]
            if index > 0:
                block_input = block_input + outputs[-1]
            outputs.append(block(block_input))
        return mx.concatenate(outputs, axis=-1)


class SEBlock(nn.Module):
    def __init__(self, channels: int, se_channels: int = 128) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, se_channels, 1, bias=True)
        self.conv2 = nn.Conv1d(se_channels, channels, 1, bias=True)

    def __call__(self, values: mx.array) -> mx.array:
        pooled = mx.mean(values, axis=1, keepdims=True)
        scale = mx.sigmoid(self.conv2(nn.relu(self.conv1(pooled))))
        return values * scale


class SERes2NetBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.tdnn1 = TDNNBlock(channels, channels, 1)
        self.res2net_block = Res2NetBlock(channels, kernel_size, dilation)
        self.tdnn2 = TDNNBlock(channels, channels, 1)
        self.se_block = SEBlock(channels)

    def __call__(self, values: mx.array) -> mx.array:
        hidden = self.tdnn1(values)
        hidden = self.res2net_block(hidden)
        hidden = self.tdnn2(hidden)
        return self.se_block(hidden) + values


class ECAPABlocks(nn.Module):
    def __init__(self, n_mels: int, channels: int) -> None:
        super().__init__()
        self.block0 = TDNNBlock(n_mels, channels, 5)
        self.block1 = SERes2NetBlock(channels, 3, 2)
        self.block2 = SERes2NetBlock(channels, 3, 3)
        self.block3 = SERes2NetBlock(channels, 3, 4)


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels: int, attention_channels: int = 128) -> None:
        super().__init__()
        self.tdnn = TDNNBlock(channels * 3, attention_channels, 1)
        self.conv = nn.Conv1d(attention_channels, channels, 1, bias=True)

    def __call__(self, values: mx.array) -> mx.array:
        mean = mx.mean(values, axis=1, keepdims=True)
        centered = values - mean
        std = mx.sqrt(
            mx.maximum(mx.mean(centered * centered, axis=1, keepdims=True), 1e-12)
        )
        context = mx.concatenate(
            [
                values,
                mx.broadcast_to(mean, values.shape),
                mx.broadcast_to(std, values.shape),
            ],
            axis=-1,
        )
        attention = mx.softmax(self.conv(mx.tanh(self.tdnn(context))), axis=1)
        weighted_mean = mx.sum(attention * values, axis=1)
        weighted_centered = values - mx.expand_dims(weighted_mean, axis=1)
        weighted_std = mx.sqrt(
            mx.maximum(mx.sum(attention * weighted_centered**2, axis=1), 1e-12)
        )
        return mx.concatenate([weighted_mean, weighted_std], axis=-1)


class ECAPAEmbedding(nn.Module):
    def __init__(self, n_mels: int, embedding_dimension: int) -> None:
        super().__init__()
        channels = 1024
        self.blocks = ECAPABlocks(n_mels, channels)
        self.mfa = TDNNBlock(channels * 3, channels * 3, 1)
        self.asp = AttentiveStatisticsPooling(channels * 3)
        self.asp_bn = nn.BatchNorm(channels * 6)
        self.fc = nn.Conv1d(channels * 6, embedding_dimension, 1, bias=True)

    def __call__(self, values: mx.array) -> mx.array:
        hidden = self.blocks.block0(values)
        output1 = self.blocks.block1(hidden)
        output2 = self.blocks.block2(output1)
        output3 = self.blocks.block3(output2)
        hidden = self.mfa(mx.concatenate([output1, output2, output3], axis=-1))
        hidden = self.asp_bn(self.asp(hidden))
        return self.fc(mx.expand_dims(hidden, axis=1))


class SpeakerModel(nn.Module):
    def __init__(self, n_mels: int = 80, embedding_dimension: int = 192) -> None:
        super().__init__()
        self.embedding_model = ECAPAEmbedding(n_mels, embedding_dimension)

    def __call__(self, mel_features: mx.array) -> mx.array:
        centered = mel_features - mx.mean(mel_features, axis=1, keepdims=True)
        embedding = mx.squeeze(self.embedding_model(centered), axis=1)
        norm = mx.sqrt(mx.sum(embedding * embedding, axis=-1, keepdims=True))
        return embedding / mx.maximum(norm, 1e-12)


