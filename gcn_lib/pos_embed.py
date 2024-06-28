import numpy as np

import torch

# --------------------------------------------------------
# 相対位置埋め込みを取得する関数
# References: https://arxiv.org/abs/2009.13658
# --------------------------------------------------------
def get_2d_relative_pos_embed(embed_dim, grid_size):
    """
    grid_size: グリッドの高さと幅の整数
    return:
    pos_embed: [grid_size*grid_size, grid_size*grid_size]
    """
    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size) # サインコサイン位置埋め込みを取得
    relative_pos = 2 * np.matmul(pos_embed, pos_embed.transpose()) / pos_embed.shape[1] # 相対位置埋め込みを計算
    return relative_pos


# --------------------------------------------------------
# 2次元のサイン・コサイン位置埋め込みを取得する関数
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: グリッドの高さと幅の整数
    return:
    pos_embed: [grid_size*grid_size, embed_dim] または [1+grid_size*grid_size, embed_dim] (クラストークンの有無に依存)
    """
    grid_h = np.arange(grid_size, dtype=np.float32) # グリッドの高さを生成
    grid_w = np.arange(grid_size, dtype=np.float32) # グリッドの幅を生成
    grid = np.meshgrid(grid_w, grid_h)  # 幅が先に来るメッシュグリッドを生成
    grid = np.stack(grid, axis=0) # グリッドをスタック

    grid = grid.reshape([2, 1, grid_size, grid_size]) # グリッドを再整形
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid) # グリッドからサインコサイン位置埋め込みを取得
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0) # クラストークンがある場合はゼロベクトルを追加
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0 # 埋め込み次元が偶数であることを確認

    # 次元の半分を使ってgrid_hをエンコード
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D) サインコサイン埋め込みを結合
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: 各位置の出力次元
    pos: エンコードする位置のリスト: サイズ (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0 # 埋め込み次元が偶数であることを確認
    omega = np.arange(embed_dim // 2, dtype=np.float) # 周波数ベクトルを生成
    omega /= embed_dim / 2. # 正規化
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,) 位置ベクトルを再整形
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2) 外積を計算

    emb_sin = np.sin(out) # (M, D/2) サイン成分
    emb_cos = np.cos(out) # (M, D/2) コサイン成分

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D) サインとコサイン成分を結合
    return emb