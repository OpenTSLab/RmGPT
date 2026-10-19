"""
RmGPT
"""
import math
import torch
import torch.nn.functional as F
from torch import nn

from timm.layers import Mlp, DropPath
from timm.layers.helpers import to_2tuple
from torch import fft   
from torch.quasirandom import SobolEngine

def initialize_high_dimensional_space(batch_size, enc_in, num_class, dimension):
    # 创建一个 Sobol 生成器
    sobol = SobolEngine(dimension=dimension, scramble=True)
    # 生成低差异序列点
    points = sobol.draw(num_class)  # 生成 num_class 个点
    # 将 [0, 1] 范围内的点转换为更标准的范围 (-1, 1) 适合初始化
    points = points * 2 - 1
    
    # 调整形状并复制到其他维度
    # 初始化完整张量为0
    full_tensor = torch.zeros(batch_size, enc_in, num_class, dimension)
    # 将每个生成的点复制到 enc_in 维度
    for i in range(enc_in):
        full_tensor[:, i, :, :] = points.clone().detach().requires_grad_(True)
    return full_tensor

def calculate_unfold_output_length(input_length, size, step):
    # Calculate the number of windows
    num_windows = (input_length - size) // step + 1
    return num_windows


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            var_num=None,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if var_num is not None:
            self.template = nn.Parameter(
                torch.zeros(var_num, dim), requires_grad=True)
            torch.nn.init.normal_(self.template, std=.02)
        self.var_num = var_num

    def forward(self, x, query=None):
        B, N, C = x.shape
        if query is not None:
            q = self.q(query).reshape(
                B, query.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            var_num = query.shape[1]
        else:
            q = self.q(self.template).reshape(1, self.var_num,
                                              self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            q = q.repeat(B, 1, 1, 1)
            var_num = self.var_num
        kv = self.kv(x).reshape(B, N, 2, self.num_heads,
                                self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        k = self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, var_num, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class FeedFoward(nn.Module):
    def __init__(
            self,
            dim,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            prefix_token_length=None,
            group=1,
    ):
        super().__init__()
        dim = dim
        hidden_features = hidden_features or 4*dim
        out_features = out_features or dim
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(dim, hidden_features,
                              bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])

        self.norm = norm_layer(
            hidden_features) if norm_layer is not None else nn.Identity()
 
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        self.hidden_features = hidden_features
        self.prefix_token_length = prefix_token_length

    def forward(self, x):
        n, var, l, d = x.shape
        # x = x.view(-1, d) # (n*var, l, c)
        # x = x.transpose(-1, -2) # (n*var, c, l)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.norm(x)
        x = self.drop2(x)+x
        return x



class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        self.pe = nn.Parameter(torch.zeros(
            1, 1, max_len, d_model), requires_grad=True)

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(0)
        self.pe.data.copy_(pe.float())
        del pe

    def forward(self, x, offset=0):
        return self.pe[:, :, offset:offset+x.size(2)]


class SeqAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(
            q, k, v,  # attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VarAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, P, C = x.shape

        qkv = self.qkv(x).reshape(B, N, P, 3, self.num_heads,
                                  self.head_dim).permute(3, 0, 2, 4, 1, 5)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.mean(dim=1, keepdim=False)
        k = k.mean(dim=1, keepdim=False)
        v = v.permute(0, 2, 3, 4, 1).reshape(B, self.num_heads, N, -1)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.view(B, self.num_heads, N, -1, P).permute(0,
                                                        2, 4, 1, 3).reshape(B, N, P, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GateLayer(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gate = nn.Linear(dim, 1)

    def forward(self, x):
        gate_value = self.gate(x)
        return gate_value.sigmoid() * x


class SeqAttBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_seq = SeqAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask):
        x_input = x
        x = self.norm1(x)
        n_vars, n_seqs = x.shape[1], x.shape[2]
        x = torch.reshape(
            x, (-1, x.shape[-2], x.shape[-1]))
        x = self.attn_seq(x, attn_mask)
        x = torch.reshape(
            x, (-1, n_vars, n_seqs, x.shape[-1]))
        x = x_input + self.drop_path1(x)
        return x


class VarAttBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_var = VarAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        # self.ls1 = GateLayer(dim, init_values=init_values)
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        x = x + self.drop_path1(self.attn_var(self.norm1(x)))
        return x


class MLPBlock(nn.Module):

    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            proj_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=None,
            prefix_token_length=0,
    ):
        super().__init__()
        self.norm2 = norm_layer(dim)
        if mlp_layer is FeedFoward:
            self.mlp = mlp_layer(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
                prefix_token_length=prefix_token_length,
            )
        else:
            self.mlp = mlp_layer(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
            )
        self.ls2 = GateLayer(dim, init_values=init_values)
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, prefix_seq_len=None):
        if prefix_seq_len is not None:
            x = x + \
                self.drop_path2(
                    self.ls2(self.mlp(self.norm2(x), prefix_seq_len=prefix_seq_len)))
        else:
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class BasicBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=8.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            prefix_token_length=0,
    ):
        super().__init__()
        self.seq_att_block = SeqAttBlock(dim=dim, num_heads=num_heads,
                                         qkv_bias=qkv_bias, qk_norm=qk_norm,
                                         attn_drop=attn_drop, init_values=init_values, proj_drop=proj_drop,
                                         drop_path=drop_path, norm_layer=norm_layer)

        self.var_att_block = VarAttBlock(dim=dim, num_heads=num_heads,
                                         qkv_bias=qkv_bias, qk_norm=qk_norm,
                                         attn_drop=attn_drop, init_values=init_values, proj_drop=proj_drop,
                                         drop_path=drop_path, norm_layer=norm_layer)


        self.feed_forward = FeedFoward(dim=dim, hidden_features=dim*4, act_layer=act_layer, drop=proj_drop)

    def forward(self, x, prefix_seq_len, attn_mask):
        x = self.var_att_block(x)
        x = self.seq_att_block(x, attn_mask)
        x = self.feed_forward(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        assert self.patch_len == self.stride, "non-overlap"
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x)
        return self.dropout(x), n_vars


class CLSHead(nn.Module):
    def __init__(self, d_model, head_dropout=0):
        super().__init__()
        d_mid = d_model
        self.proj_in = nn.Linear(d_model, d_mid)
        self.cross_att = CrossAttention(d_mid)

        self.mlp = MLPBlock(dim=3*d_mid, mlp_ratio=8, mlp_layer=Mlp,
                            proj_drop=head_dropout, init_values=None, drop_path=0.0,
                            act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                            prefix_token_length=None)

    def forward(self, x):
        x = self.proj_in(x)
        B, V, L, C = x.shape 
        x = x.view(-1, L, C) # (B*V, L, C)
        cls_token = x[:, -3:]  # (B*V, 3, C)
        cls_token = self.cross_att(x, query=cls_token)
        cls_token = cls_token.reshape(B, V, -1, C)  # (B,V, 3, C)
        cls_token = cls_token.reshape(B,V,1,-1) # (B,V, 1, 3*C)
        cls_token = self.mlp(cls_token)  
        return cls_token




class ForecastHead(nn.Module):
    def __init__(self, d_model, patch_len, stride, pad, head_dropout=0, prefix_token_length=None):
        super().__init__()
        d_mid = d_model
        self.proj_in = nn.Linear(d_model, d_mid)
        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=int(d_model * 4),
            act_layer=nn.GELU,
            drop=head_dropout,
        )
        self.proj_out = nn.Linear(d_model, patch_len)
        self.pad = pad
        self.patch_len = patch_len
        self.stride = stride
        # self.pos_proj = DynamicLinear(
        #     in_features=128, out_features=128, fixed_in=prefix_token_length)

    def forward(self, x_full, pred_len, token_len):
        x_full = self.proj_in(x_full)
        x_pred = x_full[:, :, -token_len:]
        x = self.mlp(x_pred)+x_pred
        x = self.proj_out(x)

        bs, n_vars = x.shape[0], x.shape[1]
        x = x.reshape(-1, x.shape[-2], x.shape[-1])
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.fold(x, output_size=(
            pred_len, 1), kernel_size=(self.patch_len, 1), stride=(self.stride, 1))
        x = x.squeeze(dim=-1)
        x = x.reshape(bs, n_vars, -1)
        x = x.permute(0, 2, 1)
        return x


class Model(nn.Module):
    """
    RmGPT
    """

    def __init__(self, args, configs_list, pretrain=False):
        super().__init__()

        if pretrain:
            self.right_prob = args.right_prob
            self.min_mask_ratio = args.min_mask_ratio
            self.max_mask_ratio = args.max_mask_ratio

        # Tokens settings
        self.num_task = len(configs_list)
        self.prompt_tokens = nn.ParameterDict({})
        self.cls_tokens = nn.ParameterDict({})
        self.category_tokens = nn.ParameterDict({})

        for i in range(self.num_task):
            dataset_name = configs_list[i][1]['dataset']
            task_data_name = configs_list[i][0]
            if dataset_name not in self.prompt_tokens:
                self.prompt_tokens[dataset_name] = torch.zeros(
                    1, configs_list[i][1]['enc_in'], args.prompt_num, args.d_model)
                torch.nn.init.normal_(
                    self.prompt_tokens[dataset_name], std=.02)


            if 'classification' or 'RUL' in configs_list[i][1]['task_name'] :
                self.category_tokens[task_data_name] = initialize_high_dimensional_space(
                    1, configs_list[i][1]['enc_in'], configs_list[i][1]['num_class'], args.d_model*3)
                # self.category_tokens[task_data_name] = torch.zeros(
                #     1, configs_list[i][1]['enc_in'], configs_list[i][1]['num_class'], args.d_model*3)
                # torch.nn.init.normal_(
                #     self.category_tokens[task_data_name], std=.02)
                self.cls_tokens[task_data_name] = torch.zeros(
                    1, configs_list[i][1]['enc_in'], 1, args.d_model)
                torch.nn.init.normal_(self.cls_tokens[task_data_name], std=.02)
            if pretrain:
                self.cls_tokens[task_data_name] = torch.zeros(
                    1, configs_list[i][1]['enc_in'], 1, args.d_model)
                torch.nn.init.normal_(self.cls_tokens[task_data_name], std=.02)

        self.configs_list = configs_list
        self.d_model = args.d_model
        self.amp_head = nn.Linear(2048,args.d_model)
        self.phase_head = nn.Linear(2048,args.d_model)

        ### model settings ###
        self.prompt_num = args.prompt_num
        self.stride = args.stride
        self.pad = args.stride
        self.patch_len = args.patch_len
        self.cls_len = 3
        # input processing
        self.patch_embeddings = PatchEmbedding(
            args.d_model, args.patch_len, args.stride, args.stride, args.dropout)
        self.position_embedding = LearnablePositionalEmbedding(args.d_model)
        # basic blocks
        self.block_num = args.e_layers
        self.blocks = nn.ModuleList(
            [BasicBlock(dim=args.d_model, num_heads=args.n_heads, qkv_bias=False, qk_norm=False,
                        mlp_ratio=8., proj_drop=args.dropout, attn_drop=0., drop_path=0.,
                        init_values=None, prefix_token_length=args.prompt_num) for l in range(args.e_layers)]
        )

        # output processing
        self.cls_head = CLSHead(args.d_model, head_dropout=args.dropout)
        # self.rul_head = CLSHead(args.d_model, head_dropout=args.dropout)
        # self.RUL_head = RULHead(args.d_model, head_dropout=args.dropout)
        # self.forecast_head = ForecastHead(
        #     args.d_model, args.patch_len, args.stride, args.stride, prefix_token_length=args.prompt_num, head_dropout=args.dropout)
        self.pretrain_cls_head = nn.Linear(args.d_model*3, args.d_model)
        self.pretrain_predict =nn.Linear(args.d_model, args.patch_len)
        self.rul_token_head = nn.Linear(args.d_model*3, args.d_model*3)
        self.cls_token_head = nn.Linear(args.d_model*3, args.d_model*3)
        self.debug = args.mode_debug
    def tokenize(self, x, mask=None):
        # Normalization from Non-stationary Transformer
        means = x.mean(1, keepdim=True).detach()
        x = x -means
        if mask is not None:
            x = x.masked_fill(mask == 0, 0)
            stdev = torch.sqrt(torch.sum(x * x, dim=1) /
                               torch.sum(mask == 1, dim=1) + 1e-5)
            stdev = stdev.unsqueeze(dim=1)
        else:
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x.permute(0, 2, 1)
        remainder = x.shape[2] % self.patch_len
        if remainder != 0:
            padding = self.patch_len - remainder
            x = F.pad(x, (0, padding))
        else:
            padding = 0
        x, n_vars = self.patch_embeddings(x)
        return x, means, stdev, n_vars, padding

    def prepare_prompt(self, x,means,stdevs,n_vars, prefix_prompt, task_prompt, task_prompt_num, amp_token=None,phase_token=None,task_name=None, mask=None):
        x = torch.reshape(
            x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        # append prompt tokens
        this_prompt = prefix_prompt.repeat(x.shape[0], 1, 1, 1)
        #create stastical tokens
        mean_token = means.permute(0, 2, 1).unsqueeze(2).repeat(1, 1, 1, 512)
        stdev_token =  stdevs.permute(0, 2, 1).unsqueeze(2).repeat(1, 1, 1, 512)
        cls_tokens = torch.cat((amp_token,phase_token,task_prompt.repeat(x.shape[0], 1, 1, 1)),dim=2) #[B,D,3,d]
        x = x + self.position_embedding(x)
        x = torch.cat((mean_token,stdev_token,this_prompt, x, cls_tokens), dim=2)

        return x

    def mark2token(self, x_mark):
        x_mark = x_mark.unfold(
            dimension=-1, size=self.patch_len, step=self.stride)
        x_mark = x_mark.mean(dim=-1)
        x_mark = (x_mark > 0).float()
        return x_mark

    def backbone(self, x, prefix_len, seq_len):
        attn_mask = None
        for block in self.blocks:
            x = block(x, prefix_seq_len=prefix_len +
                      seq_len, attn_mask=attn_mask)
        return x


    def classification(self, x, x_mark, task_id):
        dataset_name = self.configs_list[task_id][1]['dataset']
        task_data_name = self.configs_list[task_id][0]
        prefix_prompt = self.prompt_tokens[dataset_name]
        task_prompt = self.cls_tokens[task_data_name]
        task_prompt_num = 1
        category_token = self.category_tokens[task_data_name]
        amp_token,phase_token = self.get_freq_tokens(x) #[B,D,d]
        amp_token,phase_token = amp_token.unsqueeze(2),phase_token.unsqueeze(2) #[B,D,1,d]
        x, means, stdev, n_vars, _ = self.tokenize(x)

        seq_len = x.shape[-2]

        x = self.prepare_prompt(
            x,means,stdev,n_vars, prefix_prompt, task_prompt, task_prompt_num,
            amp_token=amp_token,phase_token=phase_token, task_name='classification')
    
        x = self.backbone(x, prefix_prompt.shape[2], seq_len)
        B, V, L, C = x.shape 
        cls_token = self.cls_head(x)
        m = category_token.shape[2]
        cls_token = cls_token.expand(B, V, m, 3*C)
        cls_token = self.cls_token_head(cls_token)
        cls_token = F.normalize(cls_token, p=2, dim=-1)
        category_token = F.normalize(category_token, p=2, dim=-1)
        if self.debug:
            return cls_token, category_token
        else:    
            distance = torch.einsum('nvkc,nvmc->nvm', cls_token, category_token)
            distance = distance.mean(dim=1).softmax(dim=-1)
            return distance


    def RUL(self, x, x_mark, task_id):
        dataset_name = self.configs_list[task_id][1]['dataset']
        task_data_name = self.configs_list[task_id][0]
        prefix_prompt = self.prompt_tokens[dataset_name]
        task_prompt = self.cls_tokens[task_data_name]
        task_prompt_num = 1
        category_token = self.category_tokens[task_data_name]
        amp_token,phase_token = self.get_freq_tokens(x) #[B,D,d]
        amp_token,phase_token = amp_token.unsqueeze(2),phase_token.unsqueeze(2) #[B,D,1,d]
        x, means, stdev, n_vars, _ = self.tokenize(x)

        seq_len = x.shape[-2]

        x = self.prepare_prompt(
            x,means,stdev, n_vars, prefix_prompt, task_prompt, task_prompt_num,
            amp_token=amp_token,phase_token=phase_token, task_name='RUL')
    
        x = self.backbone(x, prefix_prompt.shape[2], seq_len)
        B, V, L, C = x.shape 
        cls_token = self.cls_head(x)
        m = category_token.shape[2]
        cls_token = cls_token.expand(B, V, m, 3*C)
        cls_token = self.rul_token_head(cls_token)
        cls_token = F.normalize(cls_token, p=2, dim=-1)
        category_token = F.normalize(category_token, p=2, dim=-1)
        distance = torch.einsum('nvkc,nvmc->nvm', cls_token, category_token)
        distance = distance.mean(dim=1)
        #得到最小的距离
        distance_min = F.sigmoid(distance).min(dim=1,keepdim=True).values
        return distance_min


    def random_masking(self, x, min_mask_ratio, max_mask_ratio):
        """
        Perform random masking where a specified ratio of the total V*L blocks are masked.
        """
        N, V, L, D = x.shape  # batch, var, length, dim
        total_elements = V * L

        mask_ratio = (min_mask_ratio+max_mask_ratio)/2
        # Calculate the number of elements to keep based on the mask ratio
        total_keeps = int((1 - mask_ratio) * total_elements)

        # Generate a random noise array for each sample in the batch
        noise = torch.rand(N, V, L, device=x.device)  # noise in [0, 1] for V*L blocks

        # Flatten noise for easier processing
        noise_flat = noise.view(N, V * L)

        # Get indices to sort and restore noise
        ids_shuffle = torch.argsort(noise_flat, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Create the binary mask: 0 is keep, 1 is remove
        # We create a range tensor and compare it with total_keeps to generate the mask
        range_tensor = torch.arange(V * L, device=x.device).repeat(N, 1)
        mask_flat = range_tensor >= total_keeps

        # Unshuffle to get the binary mask in original order
        mask_flat = mask_flat.gather(dim=1, index=ids_restore)

        # Reshape mask back to the original V, L dimensions
        mask = mask_flat.view(N, V, L)


        return mask.float()

    def next_token_masking(self, x):
        B, V, N, D = x.shape  # batch, var, length, dim
        # 创建一个(B, D, N-1)的全0张量
        zeros = torch.zeros(B, V, N-1)

        # 创建一个(B, D, 1)的全1张量
        ones = torch.ones(B, V, 1)

        # 沿着最后一个维度拼接这两个张量
        mask = torch.cat([zeros, ones], dim=-1)
        mask = mask.float().to(x.device)
        return mask


    def right_masking(self, x, min_mask_ratio, max_mask_ratio):
        N, V, L, D = x.shape  # batch, var, length, dim

        # Randomly choose a mask ratio for each sample within the specified range
        mask_ratios = torch.rand(N, device=x.device) * \
            (max_mask_ratio - min_mask_ratio) + min_mask_ratio
        len_keeps = (L * (1 - mask_ratios)).long()

        # Binary mask creation without a for loop
        len_keeps_matrix = len_keeps.unsqueeze(1).expand(N, L)
        indices = torch.arange(L, device=x.device).expand_as(len_keeps_matrix)
        mask = indices >= len_keeps_matrix
        mask = mask.float()

        return mask

    def choose_masking(self, x, right_prob, min_mask_ratio, max_mask_ratio):
        # Generate a random number to decide which masking function to use
        # if torch.rand(1).item() > right_prob:
        #     return self.random_masking(x, min_mask_ratio, max_mask_ratio)
        # else:
        #     return self.right_masking(x, min_mask_ratio, max_mask_ratio)
        return self.next_token_masking(x)


    def get_mask_seq(self, mask, seq_len):
        """
        Convert a mask from block space back to original sequence space with multiple dimensions.
        Args:
            mask (torch.Tensor): The mask tensor of shape [B, D, N].
            seq_len (int): The length of the original sequence.

        Returns:
            torch.Tensor: Mask of the original sequence space of shape [B, D, seq_len].
        """
        # Assume the mask has the shape [B, D, N] and needs to be applied over patches
        # First, reshape the mask to prepare for folding, combining the patch dimension with the feature/channel dimension
        mask_seq = mask.unsqueeze(dim=-1).repeat(1, 1, 1, self.patch_len)
        
        # We need to collapse the last two dimensions into one for folding.
        # Permute to move the patches dimension next to the sequence length dimension
        mask_seq = mask_seq.permute(0, 1, 3, 2).contiguous()
        
        # Flatten the last two dimensions
        mask_seq = mask_seq.view(mask_seq.size(0), -1, mask_seq.size(3))
        
        # Apply the mask fill operation for non-zero patch locations
        mask_seq = mask_seq.masked_fill(mask_seq == 0, -1e9)

        # Use the fold function to reduce the dimension
        # Ensure that the kernel size and the stride match the unfolding that occurred before the masking
        mask_seq = torch.nn.functional.fold(mask_seq, output_size=(seq_len, 1), kernel_size=(self.patch_len, 1), stride=(self.stride, 1))
        
        # Threshold to bring back to 0/1 values
        mask_seq = (mask_seq > 0).float()
        
        # Remove unnecessary dimensions
        mask_seq = mask_seq.squeeze(-1).squeeze(-1)

        return mask_seq



    def get_freq_tokens(self,x):
        ###
        #x:[B,L,D]
        ###
        x = x.permute(0, 2, 1)
        exp_len = self.amp_head.weight.size()[-1]
        if x.shape[1] < exp_len:
            x = F.pad(x,(0,exp_len-x.shape[-1]))
        x_f =torch.fft.fft(x)
        x_amp,x_phase = torch.abs(x_f),torch.abs(x_f)
        amp_token = self.amp_head(x_amp[:,:exp_len,:])
        phase_token = self.phase_head(x_phase[:,:exp_len,:])
        
        return amp_token,phase_token

    def pretraining(self, x, x_mark, task_id, enable_mask=False):
        ###
        # Pretraining task
        #input: x, x_mark, task_id
        #x: [B, L, D]
        #task_id: task_id
        ###

        dataset_name = self.configs_list[task_id][1]['dataset']
        task_data_name = self.configs_list[task_id][0]
        prefix_prompt = self.prompt_tokens[dataset_name] #[1,D,10,d]
        # mask_token = self.mask_tokens[dataset_name] #[1,D,1,d]
        cls_token = self.cls_tokens[task_data_name] #[1,D,1,d]
        
        amp_token,phase_token = self.get_freq_tokens(x) #[B,D,d]
        amp_token,phase_token = amp_token.unsqueeze(2),phase_token.unsqueeze(2) #[B,D,1,d]
        seq_len = x.shape[1]
        x, means, stdevs, n_vars, padding = self.tokenize(x) #[B*D,N,d]
        seq_token_len = x.shape[-2] #N
        mean_token = means.permute(0, 2, 1).unsqueeze(2).repeat(1, 1, 1, 512)
        stdev_token =  stdevs.permute(0, 2, 1).unsqueeze(2).repeat(1, 1, 1, 512)
        # append prompt tokens
        x = torch.reshape(
            x, (-1, n_vars, x.shape[-2], x.shape[-1])) #[B,D,N,d]
        # prepare prompts
        this_prompt = prefix_prompt.repeat(x.shape[0], 1, 1, 1) #[B,D,10,d]

        if enable_mask:
            mask = self.choose_masking(x, False,
                                       self.min_mask_ratio, self.max_mask_ratio)#[B,D,N]
            mask_repeat = mask.unsqueeze(dim=-1) #[B,D,N,1]
            mask_repeat = mask_repeat.repeat(1, 1, 1, x.shape[-1])#[B,D,N,d]
            #进行掩码
            x = x*(1-mask_repeat)
            #加入位置编码
            x = x + self.position_embedding(x)
            mask_seq = self.get_mask_seq(mask, seq_len+padding) #复原到原始序列空间，用于计算损失
            mask_seq = mask_seq[:, :seq_len].transpose(1, 2) #[B,L,D]
        cls_tokens = torch.cat((amp_token,phase_token,cls_token.repeat(x.shape[0], 1, 1, 1)),dim=2) #[B,D,3,d]
        x = torch.cat((mean_token,stdev_token,this_prompt, x, cls_tokens), dim=2) #[B,D,prompt_num+N+3,d]
        #此处是backbone表征
        x = self.backbone(x, prefix_prompt.shape[2], seq_token_len) #[B,D,prompt_num+N+1,d]
        #开始双塔输出
        if enable_mask:
            # cls_token  = self.cls_head(x)
            seq_token_o = x[:,:,:-(self.cls_len+1),:]
            cls_token_o  = x[:,:,-self.cls_len:,:]
            x = torch.cat((seq_token_o,cls_token_o),dim=2)
            cls_token = self.cls_head(x)
            cls_token = self.pretrain_cls_head(cls_token)
            seq_token = x[:,:,self.prompt_num:-self.cls_len,:]
            seq_token_all = torch.cat((seq_token,cls_token),dim=2)
            mask_dec_out = self.pretrain_predict(seq_token_all)
            mask_dec_out  = mask_dec_out.reshape(*mask_dec_out.shape[:-2], -1)[:,:,-seq_len:].transpose(1,2)
            # mask_dec_out = self.forecast_head(
            #     x, seq_len+padding, seq_token_len) #[B,(prompt_num+N+2)*P,D]
            # mask_dec_out = mask_dec_out[:, :seq_len]
            

            # # De-Normalization from Non-stationary Transformer
            # # mask_dec_out = mask_dec_out * \
            # #     (stdev[:, 0, :].unsqueeze(1).repeat(
            # #         1, mask_dec_out.shape[1], 1))
            # # mask_dec_out = mask_dec_out + \
            # #     (means[:, 0, :].unsqueeze(1).repeat(
            # #         1, mask_dec_out.shape[1], 1))
            # # #因为有均值和方差的De-Normalization，所以一开始预训练就能有很低的效果


            # cls_dec_out = self.cls_head(x, return_feature=True)
            # # detach grad of the forecasting on tokens,但是会进入交互
            # cls_dec_amplitude = self.pretrain_head_amplitude(cls_dec_out)
            # cls_dec_phase = self.pretrain_head_phase(cls_dec_out)
            # cls_dec_out = [cls_dec_amplitude, cls_dec_phase]


            return mask_dec_out, mask_dec_out, mask_seq #[B,L,D],[B,L,D],[B,L]
        else:
            return mask_dec_out

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None,
                mask=None, task_id=None, task_name=None, enable_mask=None):
        #X_enc [B,L,D]
        if 'classification' in task_name:
            if self.debug:
                cls_token, category_token = self.classification(x_enc, x_mark_enc, task_id)
                return cls_token, category_token
            else:
                dec_out = self.classification(x_enc, x_mark_enc, task_id)
                return dec_out  # [B, N]
        if 'pretrain' in task_name:
            dec_out = self.pretraining(x_enc, x_mark_enc, task_id,
                                       enable_mask=enable_mask)
            return dec_out #结构化输出
        if 'RUL' in task_name:
            dec_out = self.RUL(x_enc, x_mark_enc, task_id)
            return dec_out
        return None


def test_model():
    import yaml
    import argparse

    def read_task_data_config(config_path):
        with open(config_path, 'r') as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        task_dataset_config = config.get('task_dataset', {})
        return task_dataset_config

    def get_task_data_config_list(task_data_config, default_batch_size=32):
        task_data_config_list = []

        for task_name, task_config in task_data_config.items():
            task_config['max_batch'] = default_batch_size
            task_data_config_list.append([task_name, task_config])

        return task_data_config_list
    #读取args
    parser = argparse.ArgumentParser(description='RmGPT Pretrain')
    parser.add_argument('--fix_seed', type=int, default=None, help='seed')
    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='ALL_task',
                        help='task name')
    parser.add_argument('--is_training', type=int,
                        required=False, default=1, help='status') 
    parser.add_argument('--model_id', type=str, required=False,
                        default='test', help='model id')
    parser.add_argument('--model', type=str, required=False, default='RmGPT',
                        help='model name')

    # data loader
    parser.add_argument('--data', type=str, required=False,
                        default='All', help='dataset type')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT',
                        help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--task_data_config_path', type=str,
                        default='data_provider/data_config/baseline/CWRU.yaml', help='root path of the task and data yaml file')
    parser.add_argument('--subsample_pct', type=float,
                        default=None, help='subsample percent')

    # pretrain
    parser.add_argument('--right_prob', type=float,
                        default=1.0, help='right mask prob')
    parser.add_argument('--min_mask_ratio', type=float,
                        default=0.5, help='min right mask prob')
    parser.add_argument('--max_mask_ratio', type=float,
                        default=0.8, help='max right mask prob')
    parser.add_argument('--min_keep_ratio', type=float, default=None,
                        help='min crop ratio for various length in pretraining')

    #device
    parser.add_argument('--device', type=str, default='cuda:1', help='device')
    # ddp
    parser.add_argument('--ddp',type=bool,default=False,help='whether to use ddp')
    parser.add_argument('--local-rank', type=int, help='local rank')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--num_workers', type=int, default=0,
                        help='data loader num workers')

    # optimization
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int,
                        default=10, help='train epochs')
    parser.add_argument('--warmup_epochs', type=int,
                        default=0, help='warmup epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--acc_it', type=int, default=32,
                        help='acc iteration to enlarge batch size')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help='optimizer learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='optimizer learning rate')
    parser.add_argument('--beta2', type=float,
                        default=0.999, help='optimizer beta2')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0, help='optimizer weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--eps', type=float, default=1e-08,
                        help='eps for optimizer')
    parser.add_argument('--des', type=str, default='test',
                        help='exp description')
    parser.add_argument('--debug', type=str,
                        default='disabled', help='disabled')
    parser.add_argument('--clip_grad', type=float, default=None, help="""Maximal parameter
        gradient norm if using gradient clipping.""")
    parser.add_argument('--checkpoints', type=str,
                        default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument("--memory_check", action="store_true", default=True)
    parser.add_argument("--large_model", action="store_true", default=True)

    # model settings
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='num of encoder layers')
    parser.add_argument("--patch_len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--prompt_num", type=int, default=10)
    parser.add_argument("--input_len", type=int, default=2048)
    parser.add_argument("--mode_debug", type=bool, default=False)
    args = parser.parse_args()
    # 创建一个Model实例
    config = read_task_data_config(args.task_data_config_path)
    config_list = get_task_data_config_list(config)
    model = Model(args,config_list,pretrain=True)

    

    # 创建一些假的输入数据
    x_enc = torch.randn(10, 5000, 3)  # 假设有10个样本，每个样本有50个时间步，每个时间步有100个特征
    x_mark_enc = torch.randn(10, 5000, 3)
    x_dec = torch.randn(10, 5000, 3)
    x_mark_dec = torch.randn(10, 5000, 3)
    mask = torch.ones(10, 5000, dtype=torch.bool)  # 假设所有的数据都是有效的
    task_id = torch.tensor([0])  # 假设任务ID为0
    task_name = 'classification_PHM'  # 假设任务名称为'pretrain'
    enable_mask = True  # 假设启用掩码

    # 调用forward方法
    dec_out = model.forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask, task_id, task_name, enable_mask)

    # 打印输出
    print(dec_out.shape)

# 在脚本的最后调用测试函数
if __name__ == '__main__':
    test_model()