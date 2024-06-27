import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath


class MRConv2d(nn.Module):
    """
    最大関連グラフ畳み込み層 (Max-Relative Graph Convolution)
    デンスデータタイプ用
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias) # 基本的な畳み込み層

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1]) # xから特定のインデックスに対応するデータを選択
        if y is not None:
            x_j = batched_index_select(y, edge_index[0]) # yから特定のインデックスに対応するデータを選択
        else:
            x_j = batched_index_select(x, edge_index[0]) # xから特定のインデックスに対応するデータを選択
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True) # x_iとx_jの差の最大値を計算
        b, c, n, _ = x.shape # 入力xの形状を取得
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _) # xとx_jを結合し，形状を変形
        return self.nn(x) # 畳み込み層を適用


class EdgeConv2d(nn.Module):
    """
    エッジ畳み込み層
    デンスデータタイプ用
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias) # 基本的な畳み込み層

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1]) # xから特定のインデックスに対応するデータを選択
        if y is not None:
            x_j = batched_index_select(y, edge_index[0]) # yから特定のインデックスに対応するデータを選択
        else:
            x_j = batched_index_select(x, edge_index[0]) # xから特定のインデックスに対応するデータを選択
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True) # 畳み込み後の値の最大値を取得
        return max_value # 最大値を返す


class GraphSAGE(nn.Module):
    """
    GraphSAGE グラフ畳み込み層
    デンスデータタイプ用
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias) # 初期の畳み込み層
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias) # 最終的な畳み込み層

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0]) # yから特定のインデックスに対応するデータを選択
        else:
            x_j = batched_index_select(x, edge_index[0]) # xから特定のインデックスに対応するデータを選択
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True) # 畳み込み後の値の最大値を取得
        return self.nn2(torch.cat([x, x_j], dim=1)) # xとx_jを結合し、畳み込み層を適用


class GINConv2d(nn.Module):
    """
    GIN グラフ畳み込み層
    デンスデータタイプ用
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias) # 初期の畳み込み層
        eps_init = 0.0 # εの初期値
        self.eps = nn.Parameter(torch.Tensor([eps_init])) # εのパラメータ

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0]) # yから特定のインデックスに対応するデータを選択
        else:
            x_j = batched_index_select(x, edge_index[0]) # xから特定のインデックスに対応するデータを選択
        x_j = torch.sum(x_j, -1, keepdim=True) # x_jの合計を計算
        return self.nn((1 + self.eps) * x + x_j) # 畳み込み層を適用


class GraphConv2d(nn.Module):
    """
    静的グラフの畳み込み層
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias) # EdgeConv2d層を使用
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias) # MRConv2d層を使用
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias) # GraphSAGE層を使用
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias) # GINConv2d層を使用
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv)) # サポートされていない畳み込みタイプのエラー処理

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y) # グラフ畳み込み層での順伝播


class DyGraphConv2d(GraphConv2d):
    """
    動的グラフ畳み込み層
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias) 
        self.k = kernel_size # カーネルサイズ
        self.d = dilation # ダイレーション値
        self.r = r # 解像度縮小係数
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon) # ダイレーテッドknnグラフ生成器

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape # 入力のバッチサイズ，チャンネル数，高さ，幅
        y = None 
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r) # 平均プーリングを適用して解像度を下げる
            y = y.reshape(B, C, -1, 1).contiguous() # yの形状を変更
        x = x.reshape(B, C, -1, 1).contiguous() # xの形状を変更
        edge_index = self.dilated_knn_graph(x, y, relative_pos) # ダイレーテッドknnグラフを使用してエッジインデックスを生成
        x = super(DyGraphConv2d, self).forward(x, edge_index, y) # 静的グラフ畳み込み層の順伝播を呼び出す
        return x.reshape(B, -1, H, W).contiguous() # 出力の形状をもとに戻して返す


class Grapher(nn.Module):
    """
    グラフ畳み込みと全結合層をもつGrapherモジュール
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()

        self.channels = in_channels # 入力チャンネル数を保持
        self.n = n # 入力特徴量の総数
        self.r = r # 解像度の縮小係数
        self.fc1 = nn.Sequential( # 第一の全結合層
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0), # 1×1の畳み込み層
            nn.BatchNorm2d(in_channels), # バッチ正規化
        )

        # グラフ畳み込み層
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        
        # 第二の全結合層
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0), # 1×1の畳み込み層
            nn.BatchNorm2d(in_channels), # バッチ正規化
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() # ドロップパスを正規化，または恒等写像

        self.relative_pos = None # 相対位置エンコーディング
        if relative_pos: # 相対位置が有効の場合
            # 相対位置エンコーディングのテンソルを生成し，保存
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n: # 相対位置が不要またはサイズは一致する場合
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x # 入力をショートかっと接続用に保存
        x = self.fc1(x) # 第一の全結合層を適用
        B, C, H, W = x.shape # 出力の形状を取得
        relative_pos = self._get_relative_pos(self.relative_pos, H, W) # 相対位置エンコーディングを調整
        x = self.graph_conv(x, relative_pos) # グラフ畳み込みを適用
        x = self.fc2(x) # 第二の全結合層を適用
        x = self.drop_path(x) + _tmp # ドロップパスを適用し，ショートかっと接続を加算
        return x # 処理後の特徴マップを返す