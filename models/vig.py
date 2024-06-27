import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from gcn_lib import Grapher, act_layer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath

# モデルの設定を定義する関数
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'gnn_patch16_32': _cfg(
        crop_pct=0.9, input_size=(3, 224, 224),
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

class FFN(nn.Module): # FFN(Feed Forward Network)の定義
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()

        out_features = out_features or in_features # 出力特徴量数の設定．指定がなければ入力特徴量数と同じ．
        hidden_features = hidden_features or in_features # 隠れ層の特徴量の数を設定．指定されなければ入力特徴量と同じ．
        self.fc1 = nn.Sequential( # 第一の畳み込み層とバッチ正規化層を定義
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0), # 1×1の畳み込み層
            nn.BatchNorm2d(hidden_features), # バッチ正規化
        )
        self.act = act_layer(act) # 活性化関数を定義．act_layerはどのような活性か関数を選ぶかを決定．
        self.fc2 = nn.Sequential( # 第二の畳み込み層とバッチ正規化層を定義
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0), # 1×1の畳み込み層
            nn.BatchNorm2d(out_features), # バッチ正規化層
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() # ドロップパスを適用

    def forward(self, x): # ネットワークの順伝播
        shortcut = x # 入力をショートカット接続用に保存
        x = self.fc1(x) # 第一の畳み込み層とバッチ正規化を適用
        x = self.act(x) # 活性か関数を適用
        x = self.fc2(x) # 第二の畳み込み層とバッチ正規化を適用
        x = self.drop_path(x) + shortcut # ドロップパスを適用し，ショートかっと接続を加算
        return x # 処理後の特徴マップを返す

class Stem(nn.Module): # 入力画像から視覚的単語埋め込みへの変換を行うstemモジュール
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()

        # 畳み込み層のシーケンスを定義
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//8, 3, stride=2, padding=1), # 最初の畳み込み層、入力チャネルから出力チャネルの1/8へ
            nn.BatchNorm2d(out_dim//8), # バッチ正規化層
            act_layer(act), # 活性化関数層
            nn.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1), # 2番目の畳み込み層、出力チャネルの1/8から1/4へ
            nn.BatchNorm2d(out_dim//4), # バッチ正規化層
            act_layer(act), # 活性化関数層
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=2, padding=1), # 3番目の畳み込み層、出力チャネルの1/4から1/2へ
            nn.BatchNorm2d(out_dim//2), # バッチ正規化層
            act_layer(act), # 活性化関数層
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1), # 4番目の畳み込み層、出力チャネルの1/2から全出力チャネルへ
            nn.BatchNorm2d(out_dim), # バッチ正規化層
            act_layer(act), # 活性化関数層
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1), # 5番目の畳み込み層、出力チャネルを保持
            nn.BatchNorm2d(out_dim), # バッチ正規化層
        )

    def forward(self, x):# 順伝播
        x = self.convs(x)
        return x

# DeepGCNクラスを定義
class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        channels = opt.n_filters # チャネル数(フィルタ数)の設定
        k = opt.k # knnのk値（近傍の数）
        act = opt.act # 活性化関数
        norm = opt.norm # 正規化方式
        bias = opt.bias # バイアスの使用有無
        epsilon = opt.epsilon # グラフの畳み込みの確率
        stochastic = opt.use_stochastic # 確率的グラフ畳み込みの使用有無
        conv = opt.conv # 畳み込み層のタイプ
        self.n_blocks = opt.n_blocks # ブロックの数
        drop_path = opt.drop_path # ドロップパスレート
        
        self.stem = Stem(out_dim=channels, act=act) # 画像から特徴抽出する初期層の定義

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # 各ブロックのドロップパス確率
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # 各ブロックのknnのk値
        print('num_knn', num_knn)
        max_dilation = 196 // max(num_knn) # ダイレーションの最大値
        
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14)) # 位置エンベッディングパラメータの初期化
        # self.pos_embed = nn.Parameter(torch.zeros(1, channels, 2, 2)) # 位置エンベッディングパラメータの初期化

        # バックボーンの構築(GCNとFFN層のシーケンス)
        if opt.use_dilation:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])
        else:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])

        # 予測層の構築
        self.prediction = Seq(nn.Conv2d(channels, 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        self.model_init() # モデル初期化メソッドを呼び出し

    def model_init(self): # モデルの重みを初期化するメソッド
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight) # He初期化
                m.weight.requires_grad = True # 勾配を計算する設定
                if m.bias is not None:
                    m.bias.data.zero_() # バイアスを0で初期化
                    m.bias.requires_grad = True # バイアスの勾配を計算する設定

    # ネットワークの順伝播を定義
    def forward(self, inputs):
        # 入力をstem層と位置エンベッディングで処理
        x = self.stem(inputs) + self.pos_embed
        # 出力の形状を取得
        B, C, H, W = x.shape
        
        for i in range(self.n_blocks): # 各ブロックを順に処理
            x = self.backbone[i](x) # GCNとFFN層を通じて特徴を更新

        x = F.adaptive_avg_pool2d(x, 1) # 平均プーリング
        return self.prediction(x).squeeze(-1).squeeze(-1) # 最終予測値を出力
    
def vig_b_32_gelu(pretrained=False, **kwargs): # モデル構築関数
    class OptInit:
        def __init__(self, num_classes=10, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # 近傍数
            self.conv = 'mr' # グラフ畳み込み層
            self.act = 'gelu' # 活性化関数
            self.norm = 'batch' # 正規化層
            self.bias = True # 畳み込み層のバイアス
            self.n_blocks = 16 # バックボーンのブロック数
            self.n_filters = 640 # 特徴量のチャンネル数
            self.n_classes = num_classes # 出力クラス数
            self.dropout = drop_rate # ドロップアウト率
            self.use_dilation = True # ダイレーションを使用するかどうか
            self.epsilon = 0.2 # グラフ畳み込みの確率
            self.use_stochastic = False # 確率的畳み込みを使用するかどうか
            self.drop_path = drop_path_rate # ドロップパスレート

    opt = OptInit(**kwargs) # オプションの初期化
    model = DeepGCN(opt) # モデルの初期化
    model.default_cfg = default_cfgs['gnn_patch16_32'] # デフォルト設定の適用
    return model