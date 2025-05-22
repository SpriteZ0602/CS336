from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor



def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    给定线性层的权重，计算批量输入的变换结果。

    参数:
        d_in (int): 输入维度的大小
        d_out (int): 输出维度的大小
        weights (Float[Tensor, "d_out d_in"]): 要使用的线性权重
        in_features (Float[Tensor, "... d_in"]): 要应用该函数的输入张量

    返回:
        Float[Tensor, "... d_out"]: 线性模块的变换输出。
    """

    raise NotImplementedError


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    给定嵌入层的权重，获取一批标记 ID 的嵌入向量。

    参数:
        vocab_size (int): 词汇表中的嵌入数量
        d_model (int): 嵌入维度的大小
        weights (Float[Tensor, "vocab_size d_model"]): 要从中提取的嵌入向量
        token_ids (Int[Tensor, "..."]): 要从嵌入层提取的标记 ID 集合

    返回:
        Float[Tensor, "... d_model"]: 嵌入层返回的批量嵌入向量。
    """

    raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    给定 SwiGLU 网络的权重，返回使用这些权重的实现的输出。

    参数:
        d_model (int): 前馈输入和输出的维度。
        d_ff (int): SwiGLU 内部上投影的维度。
        w1_weight (Float[Tensor, "d_ff d_model"]): W1 的存储权重
        w2_weight (Float[Tensor, "d_model d_ff"]): W2 的存储权重
        w3_weight (Float[Tensor, "d_ff d_model"]): W3 的存储权重
        in_features (Float[Tensor, "... d_model"]): 前馈层的输入嵌入。

    返回:
        Float[Tensor, "... d_model"]: 与输入嵌入形状相同的输出嵌入。
    """
    # 示例:
    # 如果你的状态字典键匹配，可以使用 `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # 你也可以手动分配权重
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    给定键 (K)、查询 (Q) 和值 (V) 张量，返回缩放点积注意力实现的输出。

    参数:
        Q (Float[Tensor, "... queries d_k"]): 查询张量
        K (Float[Tensor, "... keys d_k"]): 键张量
        V (Float[Tensor, "... values d_v"]): 值张量
        mask (Float[Tensor, "... queries keys"] | None): 掩码张量

    返回:
        Float[Tensor, "... queries d_v"]: 缩放点积注意力 (SDPA) 的输出
    """
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    给定多头注意力的朴素非批量实现的键、查询和值投影权重，返回优化的批量实现的输出。
    该实现应通过单次矩阵乘法处理所有头的键、查询和值投影。
    此函数不应使用 RoPE。
    参见 Vaswani 等人，2017 年的第 3.2.2 节。

    参数:
        d_model (int): 前馈输入和输出的维度。
        num_heads (int): 多头注意力中使用的头数。
        q_proj_weight (Float[Tensor, "d_k d_in"]): 查询投影的权重
        k_proj_weight (Float[Tensor, "d_k d_in"]): 键投影的权重
        v_proj_weight (Float[Tensor, "d_k d_in"]): 值投影的权重
        o_proj_weight (Float[Tensor, "d_model d_v"]): 输出投影的权重
        in_features (Float[Tensor, "... sequence_length d_in"]): 运行实现的输入张量。

    返回:
        Float[Tensor, "... sequence_length d_out"]: 使用给定 QKV 投影权重和输入特征运行优化的批量多头注意力实现的输出张量。
    """
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    给定多头注意力的朴素非批量实现的键、查询和值投影权重，返回优化的批量实现的输出。
    该实现应通过单次矩阵乘法处理所有头的键、查询和值投影。
    此版本的 MHA 应包含 RoPE。
    此时，RoPE 嵌入维度必须是头嵌入维度 (d_model // num_heads)。
    参见 Vaswani 等人，2017 年的第 3.2.2 节。

    参数:
        d_model (int): 前馈输入和输出的维度。
        num_heads (int): 多头注意力中使用的头数。
        max_seq_len (int): 若实现需要预缓存，指定预缓存的最大序列长度。
        theta (float): RoPE 参数。
        q_proj_weight (Float[Tensor, "d_k d_in"]): 查询投影的权重
        k_proj_weight (Float[Tensor, "d_k d_in"]): 键投影的权重
        v_proj_weight (Float[Tensor, "d_k d_in"]): 值投影的权重
        o_proj_weight (Float[Tensor, "d_model d_v"]): 输出投影的权重
        in_features (Float[Tensor, "... sequence_length d_in"]): 运行实现的输入张量。
        token_positions (Int[Tensor, "... sequence_length"] | None): 可选的标记位置张量

    返回:
        Float[Tensor, "... sequence_length d_out"]: 使用给定 QKV 投影权重和输入特征运行优化的批量多头注意力（含 RoPE）实现的输出张量。
    """
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    对给定的输入张量运行 RoPE（旋转位置编码）。

    参数:
        d_k (int): 查询或键张量的嵌入维度大小。
        theta (float): RoPE 参数。
        max_seq_len (int): 若实现需要预缓存，指定预缓存的最大序列长度。
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): 应用 RoPE 的输入张量。
        token_positions (Int[Tensor, "... sequence_length"]): 形状为 (batch_size, sequence_length) 的标记位置张量

    返回:
        Float[Tensor, "... sequence_length d_k"]: 应用 RoPE 后的输入张量。
    """
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    给定前置归一化 Transformer 块的权重和输入特征，返回在输入特征上运行 Transformer 块的输出。
    该函数应使用 RoPE。
    根据实现方式，可能需要将相关参数传递给 TransformerBlock 构造函数，或手动初始化 RoPE 类并传入。

    参数:
        d_model (int): Transformer 块输入的维度。
        num_heads (int): 多头注意力中使用的头数。`d_model` 必须能被 `num_heads` 整除。
        d_ff (int): 前馈内层的维度。
        max_seq_len (int): 若实现需要预缓存，指定预缓存的最大序列长度。
        theta (float): RoPE 参数。
        weights (dict[str, Tensor]): 参考实现的状态字典，键包括：
            - `attn.q_proj.weight`: 所有 `num_heads` 注意力头的查询投影权重，形状为 (d_model, d_model)。
            - `attn.k_proj.weight`: 键投影权重，形状同上。
            - `attn.v_proj.weight`: 值投影权重，形状同上。
            - `attn.output_proj.weight`: 多头自注意力输出投影权重，形状为 (d_model, d_model)。
            - `ln1.weight`: 第一个 RMSNorm 的仿射变换权重，形状为 (d_model,)。
            - `ffn.w1.weight`: 前馈网络第一层线性变换权重，形状为 (d_model, d_ff)。
            - `ffn.w2.weight`: 前馈网络第二层线性变换权重，形状为 (d_ff, d_model)。
            - `ffn.w3.weight`: 前馈网络第三层线性变换权重，形状为 (d_model, d_ff)。
            - `ln2.weight`: 第二个 RMSNorm 的仿射变换权重，形状为 (d_model,)。
        in_features (Float[Tensor, "batch sequence_length d_model"]): 运行实现的输入张量。

    返回:
        Float[Tensor, "batch sequence_length d_model"]: 使用 RoPE 运行 Transformer 块后的输出张量。
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """
    给定 Transformer 语言模型的权重和输入索引，返回对输入索引进行前向传播的输出。
    该函数应使用 RoPE。

    参数:
        vocab_size (int): 输出词汇表中唯一元素的数量。
        context_length (int): 一次处理的最大标记数。
        d_model (int): 模型嵌入和子层输出的维度。
        num_layers (int): 使用的 Transformer 层数。
        num_heads (int): 多头注意力中使用的头数。`d_model` 必须能被 `num_heads` 整除。
        d_ff (int): 前馈内层的维度（第 3.3 节）。
        rope_theta (float): RoPE 的 $\Theta$ 参数。
        weights (dict[str, Tensor]): 参考实现的状态字典，键包括：
            - `token_embeddings.weight`: 标记嵌入矩阵，形状为 (vocab_size, d_model)。
            - `layers.{num_layers}.attn.q_proj.weight`: 第 {num_layers} 层的查询投影权重，形状为 (num_heads * (d_model/num_heads), d_model)。
            - `layers.{num_layers}.attn.k_proj.weight`: 第 {num_layers} 层的键投影权重，形状同上。
            - `layers.{num_layers}.attn.v_proj.weight`: 第 {num_layers} 层的值投影权重，形状同上。
            - `layers.{num_layers}.attn.output_proj.weight`: 第 {num_layers} 层的多头自注意力输出投影权重，形状为 ((d_model/num_heads) * num_heads, d_model)。
            - `layers.{num_layers}.ln1.weight`: 第 {num_layers} 层第一个 RMSNorm 的仿射变换权重，形状为 (d_model,)。
            - `layers.{num_layers}.ffn.w1.weight`: 第 {num_layers} 层前馈网络第一层线性变换权重，形状为 (d_model, d_ff)。
            - `layers.{num_layers}.ffn.w2.weight`: 第 {num_layers} 层前馈网络第二层线性变换权重，形状为 (d_ff, d_model)。
            - `layers.{num_layers}.ffn.w3.weight`: 第 {num_layers} 层前馈网络第三层线性变换权重，形状为 (d_model, d_ff)。
            - `layers.{num_layers}.ln2.weight`: 第 {num_layers} 层第二个 RMSNorm 的仿射变换权重，形状为 (d_model,)。
            - `ln_final.weight`: 最后一个 Transformer 块输出的 RMSNorm 仿射变换权重，形状为 (d_model, )。
            - `lm_head.weight`: 语言模型输出嵌入的权重，形状为 (vocab_size, d_model)。
        in_indices (Int[Tensor, "batch_size sequence_length"]): 输入索引张量，形状为 (batch_size, sequence_length)，其中 `sequence_length` 不超过 `context_length`。

    返回:
        Float[Tensor, "batch_size sequence_length vocab_size"]: 每个标记的预测未归一化下一词分布张量。
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    给定 RMSNorm 仿射变换的权重，返回对输入特征运行 RMSNorm 的输出。

    参数:
        d_model (int): RMSNorm 输入的维度。
        eps (float): 为数值稳定性添加到分母的值。
        weights (Float[Tensor, "d_model"]): RMSNorm 权重。
        in_features (Float[Tensor, "... d_model"]): 应用 RMSNorm 的输入特征，可包含任意前置维度。

    返回:
        Float[Tensor, "... d_model"]: 与 `in_features` 形状相同的 RMSNorm 输出张量。
    """
    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """
    给定输入张量，返回对每个元素应用 SiLU 激活函数的输出。

    参数:
        in_features (Float[Tensor, "..."]): 应用 SiLU 的输入特征，形状任意。

    返回:
        Float[Tensor, "..."]: 与 `in_features` 形状相同的 SiLU 输出张量。
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    给定数据集（一维整数 numpy 数组）和所需的批量大小及上下文长度，从数据集中采样语言模型输入序列及其对应的标签。

    参数:
        dataset (np.array): 一维整数 numpy 数组，表示数据集中的标记 ID。
        batch_size (int): 要采样的批量大小。
        context_length (int): 每个采样示例的上下文长度。
        device (str): PyTorch 设备字符串（如 'cpu' 或 'cuda:0'），指定采样的输入序列和标签放置的设备。

    返回:
        形状为 (batch_size, context_length) 的 torch.LongTensor 元组。第一个元素是采样的输入序列，第二个元素是对应的语言模型标签。
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    给定输入张量，对输入的指定 `dim` 维度应用 softmax 函数并返回输出。

    参数:
        in_features (Float[Tensor, "..."]): 应用 softmax 的输入特征，形状任意。
        dim (int): 对 `in_features` 应用 softmax 的维度。

    返回:
        Float[Tensor, "..."]: 与 `in_features` 形状相同的张量，对指定维度进行 softmax 归一化后的结果。
    """
    raise NotImplementedError


def run_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """
    给定输入张量和目标张量，计算示例间的平均交叉熵损失。

    参数:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] 表示第 i 个示例的第 j 类的未归一化 logit。
        targets (Int[Tensor, "batch_size"]): 形状为 (batch_size,) 的张量，包含正确类别的索引，每个值必须在 0 到 `num_classes - 1` 之间。

    返回:
        Float[Tensor, ""]: 示例间的平均交叉熵损失。
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    给定一组参数，将其组合梯度裁剪为 L2 范数不超过 max_l2_norm。

    参数:
        parameters (Iterable[torch.nn.Parameter]): 可训练参数的集合。
        max_l2_norm (float): 正数值，表示最大 L2 范数。

    梯度将在原地修改（parameter.grad）。
    """
    raise NotImplementedError


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    返回实现 AdamW 优化器的 torch.optim.Optimizer 类。
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    给定余弦学习率衰减调度（含线性热身）的参数和迭代次数，返回指定迭代下的学习率。

    参数:
        it (int): 要获取学习率的迭代次数。
        max_learning_rate (float): alpha_max，余弦学习率调度（含热身）的最大学习率。
        min_learning_rate (float): alpha_min，余弦学习率调度的最小/最终学习率。
        warmup_iters (int): T_w，线性热身的迭代次数。
        cosine_cycle_iters (int): T_c，余弦退火的迭代次数。

    返回:
        指定迭代下的学习率。
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    给定模型、优化器和迭代次数，将其序列化到磁盘。

    参数:
        model (torch.nn.Module): 要序列化状态的模型。
        optimizer (torch.optim.Optimizer): 要序列化状态的优化器。
        iteration (int): 要序列化的迭代次数，表示已完成的训练迭代数。
        out (str | os.PathLike | BinaryIO | IO[bytes]): 序列化模型、优化器和迭代次数的路径或文件类对象。
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    给定序列化的检查点（路径或文件类对象），将序列化状态恢复到给定的模型和优化器。
    返回检查点中先前序列化的迭代次数。

    参数:
        src (str | os.PathLike | BinaryIO | IO[bytes]): 序列化检查点的路径或文件类对象。
        model (torch.nn.Module): 要恢复状态的模型。
        optimizer (torch.optim.Optimizer): 要恢复状态的优化器。

    返回:
        int: 检查点中先前序列化的迭代次数。
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """
    给定词汇表、合并列表和特殊标记列表，返回使用提供的词汇表、合并规则和特殊标记的 BPE 分词器。

    参数:
        vocab (dict[int, bytes]): 分词器词汇表，从整数（词汇表中的标记 ID）到字节（标记字节）的映射。
        merges (list[tuple[bytes, bytes]]): BPE 合并规则，每个列表项是字节元组 (<token1>, <token2>)，表示 <token1> 与 <token2> 合并，按创建顺序排列。
        special_tokens (list[str] | None): 分词器的特殊标记字符串列表，这些字符串永远不会被拆分为多个标记，始终作为单个标记保留。

    返回:
        使用提供的词汇表、合并规则和特殊标记的 BPE 分词器。
    """
    raise NotImplementedError


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    给定输入语料路径，训练 BPE 分词器并输出其词汇表和合并规则。

    参数:
        input_path (str | os.PathLike): BPE 分词器训练数据的路径。
        vocab_size (int): 分词器词汇表的总条目数（包括特殊标记）。
        special_tokens (list[str]): 要添加到分词器词汇表的特殊标记字符串列表，这些字符串在 `input_path` 中出现时将被视为普通字符串处理。

    返回:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab: 训练后的分词器词汇表，从整数（标记 ID）到字节（标记字节）的映射。
            merges: BPE 合并规则列表，按创建顺序排列。
    """
    raise NotImplementedError
