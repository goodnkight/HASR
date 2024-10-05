import torch
import math
from torch import nn 
from torch.nn import functional as F
from torch.nn import GELU, ReLU
from torch.nn.modules.normalization import LayerNorm 

STR2FN = {"gelu": GELU(), "relu": ReLU()}

class TransformerLayer(nn.Module):
    def __init__(self, args):
        super(TransformerLayer, self).__init__()
        self.self_attention = SelfAttention(args)
        self.ffn = FeedForwardNet(args)

    def forward(self, item_emb, mask):
        attn_out = self.self_attention(item_emb, mask)
        output = self.ffn(attn_out)
        return output

class GPAttention(nn.Module):
    def __init__(self, args):
        super(GPAttention, self).__init__()
        self.num_attention_heads = args.num_attention_heads
        self.query = nn.Linear(args.hidden_size, args.hidden_size)
        self.key = nn.Linear(args.hidden_size, args.hidden_size)
        self.value = nn.Linear(args.hidden_size, args.hidden_size) 

        self.attn_dropout = nn.Dropout(args.attn_dropout_rate)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.addnorm = AddNorm(args.hidden_size, args.hidden_dropout_rate)

    def forward(self, user_emb, item_emb, mask, index):
        batch_size, ses_len, seq_len, hidden_size = item_emb.shape
        query = self.query(user_emb) 
        key = self.key(item_emb)
        value = self.value(item_emb)  

        attn_score = torch.matmul(query.unsqueeze(1).unsqueeze(1), key.transpose(-1, -2)) / math.sqrt(hidden_size) 
        attn_score = attn_score.view(-1, ses_len*seq_len)[:, index] 
        attn_score += mask 
        attn_weight = self.attn_dropout(F.softmax(attn_score, dim=-1)).unsqueeze(2) 
        # attn_weight.shape == (batch_size, ses_len*seq_len, 1, ses_len)

        value = value.view(-1, ses_len*seq_len, hidden_size)[:, index, :]
        # value.shape == (batch_size, ses_len*seq_len, ses_len, hidden_size)

        attn_output = torch.matmul(attn_weight, value) 
        attn_output = attn_output.view(item_emb.shape)

        dense_output = self.dense(attn_output)
        output = self.addnorm(item_emb, dense_output)
        return output

class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of attention heads (%d)" % (args.hidden_size, args.num_attention_heads)
                )
        self.num_attention_heads = args.num_attention_heads 
        self.attention_head_size = args.hidden_size / args.num_attention_heads

        self.query_project = nn.Linear(args.hidden_size, args.hidden_size)
        self.key_project = nn.Linear(args.hidden_size, args.hidden_size)
        self.value_project = nn.Linear(args.hidden_size, args.hidden_size)

        self.attn_dropout = nn.Dropout(args.attn_dropout_rate)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.addnorm = AddNorm(args.hidden_size, args.hidden_dropout_rate)
    
    def transpose_qkv(self, input_tensor):
        new_shape = input_tensor.shape[:-1] + (self.num_attention_heads, ) 
        out = input_tensor.view(*new_shape, -1)
        return out.permute(0, 1, 3, 2, 4)

    def forward(self, input_tensor, mask):
        init_shape = input_tensor.shape
        project_Q = self.query_project(input_tensor)
        project_K = self.query_project(input_tensor)
        project_V = self.query_project(input_tensor)

        transposed_proj_Q = self.transpose_qkv(project_Q)
        transposed_proj_K = self.transpose_qkv(project_K)
        transposed_proj_V = self.transpose_qkv(project_V)

        attn_score = torch.matmul(transposed_proj_Q, transposed_proj_K.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attn_score += mask 
        attn_weight = F.softmax(attn_score, dim=-1) 
        attn_weight = self.attn_dropout(attn_weight)
            
        attn_output = torch.matmul(attn_weight, transposed_proj_V) 
        attn_output = attn_output.permute(0, 1, 3, 2, 4).contiguous()
        attn_output = attn_output.view(init_shape)

        dense_output = self.dense(attn_output)
        output = self.addnorm(input_tensor, dense_output)
        return output

class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout_rate):
        super(AddNorm, self).__init__()
        self.layernorm = LayerNorm(norm_shape, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X, Y):
        return self.layernorm(self.dropout(Y) + X)

class FeedForwardNet(nn.Module):
    def __init__(self, args):
        super(FeedForwardNet, self).__init__()
        self.dense1 = nn.Linear(args.hidden_size, args.hidden_size*4)
        if isinstance(args.hidden_act, str):
            self.act_fn = STR2FN[args.hidden_act]
        else:
            self.act_fn = args.hidden_act
        self.dense2 = nn.Linear(args.hidden_size*4, args.hidden_size)
        self.addnorm = AddNorm(args.hidden_size, args.hidden_dropout_rate)

    def forward(self, input_tensor):
        hidden_state = self.dense2(self.act_fn(self.dense1(input_tensor)))
        return self.addnorm(input_tensor, hidden_state)

class EarlyStopping:
    def __init__(self):
        pass 

def Recall_at_k(answers, pred_list, k):
    num_user = len(pred_list) 
    recall_sum = 0 
    actual_user = 0
    for i in range(num_user): 
        if len(answers[i] > 0):
            actual_set = set(answers[i]) 
            pred_set = set(pred_list[i][:k]) 
            recall_sum += len(actual_set & pred_set) / float(len(actual_set))
            actual_user += 1 
    return recall_sum / actual_user  

def Precision_at_k(answers, pred_list, k):
    num_user = len(pred_list)
    precision_sum = 0 
    actual_user = 0
    for i in range(num_user): 
        if len(answers[i] > 0):
            actual_set = set(answers[i]) 
            pred_set = set(pred_list[i][:k]) 
            precision_sum += len(actual_set & pred_set) / float(k)
            actual_user += 1 
    return precision_sum / actual_user       

def HR_at_k(answers, pred_list, k):
    num_user = len(pred_list) 
    hit_item = 0 
    actual_item = 0 
    for i in range(num_user): 
        actual_set = set(answers[i]) 
        pred_set = set(pred_list[i][:k])
        hit_item += len(actual_set & pred_set) 
        actual_set += len(actual_set) 
    return hit_item / float(actual_set)

def NDCG_k(answers, pred_list, k):
    ndcg = 0
    for uid in range(len(answers)):
        min_k = min(k, len(answers[uid]))
        idcg = IDCG_k(min_k) 
        dcg = sum([int(pred_list[uid][j] in set(answers[uid])) / math.log(j+2, 2) for j in range(k)]) 
        ndcg += (dcg / idcg)
    return ndcg / float(len(answers)) 

def IDCG_k(k): 
    idcg = sum([1.0 / math.log(i+2, 2) for i in range(k)]) 
    return 1 if idcg==0 else idcg