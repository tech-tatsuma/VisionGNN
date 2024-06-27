import math
import torch
from torch import nn
import torch.nn.functional as F


def pairwise_distance(x):
    """
    点群のペアワイズ距離を計算する関数。
    Args:
        x: テンソル (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad(): # 勾配計算を行わないブロック
        x_inner = -2*torch.matmul(x, x.transpose(2, 1)) # 点群の内積を計算し、負の2倍を取る
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True) # 各点の座標値を二乗し、その合計を計算
        return x_square + x_inner + x_square.transpose(2, 1) # 各点の二乗値の合計、内積の2倍、そして転置した二乗値の合計を足し合わせる


def part_pairwise_distance(x, start_idx=0, end_idx=1):
    """
    点群の一部のペアワイズ距離を計算する関数。
    Args:
        x: テンソル (batch_size, num_points, num_dims)
        start_idx: 開始インデックス
        end_idx: 終了インデックス
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad(): # 勾配計算を行わないブロック
        x_part = x[:, start_idx:end_idx] # xから特定の部分集合を取り出す
        x_square_part = torch.sum(torch.mul(x_part, x_part), dim=-1, keepdim=True) # 取り出した部分の各点の座標値を二乗し、その合計を計算
        x_inner = -2*torch.matmul(x_part, x.transpose(2, 1)) # 取り出した部分と全体の内積を計算し、負の2倍を取る
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True) # 全体の各点の座標値を二乗し、その合計を計算
        return x_square_part + x_inner + x_square.transpose(2, 1) # 計算した部分の二乗値の合計、内積の2倍、そして全体の二乗値の合計を足し合わせる


def xy_pairwise_distance(x, y):
    """
    点群xと点群yのペアワイズ距離を計算する関数。
    Args:
        x: テンソル (batch_size, num_points, num_dims)
        y: テンソル (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad(): # 勾配計算を行わないブロック
        xy_inner = -2*torch.matmul(x, y.transpose(2, 1)) # xとyの内積を計算し、負の2倍を取る
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True) # xの各点の座標値を二乗し、その合計を計算
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True) # yの各点の座標値を二乗し、その合計を計算
        return x_square + xy_inner + y_square.transpose(2, 1) # xの二乗値の合計、内積の2倍、そしてyの二乗値の合計を足し合わせる


def dense_knn_matrix(x, k=16, relative_pos=None):
    """
    点群データに対するK最近傍点を計算する関数。
    Args:
        x: (batch_size, num_dims, num_points, 1) - 入力データのテンソル
        k: int - 最近傍の点の数
        relative_pos: 追加的な位置情報が含まれるテンソル（オプショナル）
    Returns:
        nearest neighbors: (batch_size, num_points, k) - 最近傍点のインデックス
    """
    with torch.no_grad(): # 勾配計算を無効化
        x = x.transpose(2, 1).squeeze(-1) # 次元の入れ替えと不要な次元の削除
        batch_size, n_points, n_dims = x.shape # バッチサイズ、点の数、次元数を抽出
        ### メモリ効率の良い実装 ###
        n_part = 10000 # 分割する点の数
        if n_points > n_part: # 全点の数がn_partより多い場合、分割して処理
            nn_idx_list = []
            groups = math.ceil(n_points / n_part) # 分割数を計算
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx) # 部分的な距離計算
                if relative_pos is not None:
                    dist += relative_pos[:, start_idx:end_idx] # 相対位置情報を距離に追加
                _, nn_idx_part = torch.topk(-dist, k=k) # 距離が最小のk個のインデックスを取得
                nn_idx_list += [nn_idx_part] # インデックスをリストに追加
            nn_idx = torch.cat(nn_idx_list, dim=1) # 分割処理したインデックスを結合
        else:
            dist = pairwise_distance(x.detach()) # 全点に対する距離行列を計算
            if relative_pos is not None:
                dist += relative_pos # 相対位置情報を距離に追加
            _, nn_idx = torch.topk(-dist, k=k) # 距離が最小のk個のインデックスを取得
        ######
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1) # 自己インデックスを生成
    return torch.stack((nn_idx, center_idx), dim=0) # 最近傍点のインデックスと自己インデックスを結合


def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    """
    二つの異なる点群に対するK最近傍点を計算する関数。
    Args:
        x, y: (batch_size, num_dims, num_points, 1) - 入力データのテンソル
        k: int - 最近傍の点の数
        relative_pos: 追加的な位置情報が含まれるテンソル（オプショナル）
    Returns:
        nearest neighbors: (batch_size, num_points, k)
    """
    with torch.no_grad(): # 勾配計算を無効化
        x = x.transpose(2, 1).squeeze(-1) # xの次元の入れ替えと不要な次元の削除
        y = y.transpose(2, 1).squeeze(-1) # yの次元の入れ替えと不要な次元の削除
        batch_size, n_points, n_dims = x.shape # バッチサイズ、点の数、次元数を抽出
        dist = xy_pairwise_distance(x.detach(), y.detach()) # xとyの間の距離を計算
        if relative_pos is not None:
            dist += relative_pos # 相対位置情報を距離に追加
        _, nn_idx = torch.topk(-dist, k=k) # 距離が最小のk個のインデックスを取得
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1) # 自己インデックスを生成
    return torch.stack((nn_idx, center_idx), dim=0) # 最近傍点のインデックスと自己インデックスを結合


class DenseDilated(nn.Module):
    """
    隣接点から拡張した隣接点を選択するクラス
    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation # 拡張の度合い
        self.stochastic = stochastic # 確率的拡張をするかどうか
        self.epsilon = epsilon # 確率的拡張をする確率
        self.k = k # 最近傍の点の数

    def forward(self, edge_index):
        if self.stochastic: # 確率的拡張を行う場合
            if torch.rand(1) < self.epsilon and self.training: # 訓練中に確率epsilonでランダム選択
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k] # ランダムにk個のインデックスを選択
                edge_index = edge_index[:, :, :, randnum] # 選択したインデックスに基づいて辺を選択
            else:
                edge_index = edge_index[:, :, :, ::self.dilation] # 拡張パラメータに従って辺を選択
        else:
            edge_index = edge_index[:, :, :, ::self.dilation] # 拡張パラメータに従って辺を選択
        return edge_index # 選択した辺を返す


class DenseDilatedKnnGraph(nn.Module):
    """
    拡張されたK最近傍グラフを構築するクラス
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation # 拡張の度合い
        self.stochastic = stochastic # 確率的拡張をするかどうか
        self.epsilon = epsilon # 確率的拡張をする確率
        self.k = k # 最近傍の点の数
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon) # DenseDilatedインスタンスの生成

    def forward(self, x, y=None, relative_pos=None):
        if y is not None: # yが提供されている場合、xとyの間のKNNを計算
            #### normalize
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)
            ####
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation, relative_pos) # xy間の拡張KNN計算
        else: # yが提供されていない場合、x内でのKNNを計算
            #### normalize
            x = F.normalize(x, p=2.0, dim=1)
            ####
            # (3, 192, 196, 1)
            edge_index = dense_knn_matrix(x, self.k * self.dilation, relative_pos) # x内の拡張KNN計算
        return self._dilated(edge_index) # 拡張された辺のインデックスを返す