import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d

# 活性か関数を生成する関数
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act) # 定義されていない活性化関数に関するエラー
    return layer


def norm_layer(norm, nc):
    # 正規化層を生成する関数
    norm = norm.lower()
    if norm == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True) # バッチ正規化層
    elif norm == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False) # インスタンス正規化層
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm) # 定義されていない正規化層の場合エラー
    return layer

# 多層パーセプトロン
class MLP(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
        super(MLP, self).__init__(*m)

# 基本的な畳み込み層のシーケンス
class BasicConv(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=bias, groups=4)) # 畳み込み層を追加
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1])) # 正規化層を追加
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act)) # 活性か層を追加
            if drop > 0:
                m.append(nn.Dropout2d(drop)) # ドロップアウト層を追加

        super(BasicConv, self).__init__(*m) # Seqを継承して、リストmの層をシーケンシャルに組み込む

        self.reset_parameters() # パラメータの初期化メソッドを呼び出し

    def reset_parameters(self):
        # 各層のパラメータを初期化するメソッド
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight) # 畳み込み層の重みをHe初期化
                if m.bias is not None:
                    nn.init.zeros_(m.bias) # バイアスをゼロ初期化
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1) # 正規化層の重みを1で初期化
                m.bias.data.zero_() # 正規化層のバイアスをゼロ初期化


def batched_index_select(x, idx):
    # バッチ処理されたインデックス選択を行う関数
    """
    Args:
        x (Tensor): 入力特徴量テンソル
        idx (Tensor): 選択するエッジのインデックス
    Returns:
        Tensor: 選択された近隣の特徴量
    """
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base # バッチごとのオフセットを加える
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous() # 形状を変更して返す
    return feature