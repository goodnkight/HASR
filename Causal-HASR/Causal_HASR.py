import torch
import numpy as np

from torch import nn 
from torch.nn import LayerNorm
from modules import TransformerLayer, GPAttention

class Causal_HASR(nn.Module):
    def __init__(self, args):
        super(Causal_HASR, self).__init__()
        self.item_embedding = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.out_item_embedding = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.user_embedding = nn.Embedding(args.user_size, args.hidden_size)
        self.position_embedding = nn.Embedding(args.max_seq_len, args.hidden_size) 

        self.item_encoder = TransformerLayer(args)  
        self.item_decoder = TransformerLayer(args)
        self.GP_encoder = GPAttention(args)
        
        self.args = args
        self.GP_index = self.get_GP_index(args.max_ses_len, args.max_seq_len)
        self.GP_mask = self.get_GP_mask(args.max_ses_len, args.max_seq_len)

        self.PCG_layer1 = nn.Linear(args.hidden_size*2, args.hidden_size) 
        self.PCG_dropout = nn.Dropout(args.hidden_dropout_rate)
        self.PCG_layer2 = nn.Linear(args.hidden_size, 1)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12) 
        self.Dropout = nn.Dropout(args.hidden_dropout_rate)

        self.apply(self.init_weights)
    
    def get_GP_index(self, ses_len, seq_len):
        total_index = np.zeros([ses_len*seq_len, ses_len]) 
        row_index = np.zeros([ses_len]) 

        for i in range(ses_len):
            row_index[i] = i*seq_len 

        for i in range(ses_len): 
            for j in range(seq_len): 
                row_index[i] = i*seq_len + j 
                total_index[i*seq_len+j] = row_index 

        return total_index

    def get_GP_mask(self, ses_len, seq_len):
        GP_mask =  np.zeros([ses_len*seq_len, ses_len]) 
        row_mask = np.zeros([ses_len]) 

        for i in range(ses_len):
            row_mask[i] = 1
            for j in range(seq_len): 
                GP_mask[i*seq_len+j] = row_mask

        return GP_mask

    def get_hu(self, user_ids, input_items):
        attn_mask = (input_items > 0).long() ·
        # attn_mask.shape = [batch, ses_len, seq_len]
        extend_attn_mask = attn_mask.unsqueeze(2).unsqueeze(3)

        max_len = input_items.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.tril(torch.ones(attn_shape))
        subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(1).long().to(self.args.device)

        # print(extend_attn_mask.device, subsequent_mask.device)
        sub_extend_attn_mask = (extend_attn_mask * subsequent_mask).to(next(self.parameters()).dtype)
        sub_extend_attn_mask = (1.0 - sub_extend_attn_mask) * -10000.0
        # sub_extend_attn_mask.shape == [batch, ses_len, 1, seq_len, seq_len]
        item_emb = self.add_position_embedding(input_items)
        enc_item_emb = self.item_encoder(item_emb, sub_extend_attn_mask) 
        dec_item_emb = self.item_decoder(enc_item_emb, sub_extend_attn_mask) 

        GP_total_idx = torch.tensor(self.GP_index, dtype=torch.long).to(self.args.device)
        GP_subsession_mask = torch.tensor(self.GP_mask, dtype=torch.long).to(self.args.device)

        concat_attn_mask = attn_mask.view(-1,  self.args.max_ses_len*self.args.max_seq_len) 
        GP_extend_mask = concat_attn_mask[:, GP_total_idx]

        GP_extend_sub_mask = GP_extend_mask * GP_subsession_mask
        GP_extend_sub_mask = GP_extend_sub_mask.to(dtype=next(self.parameters()).dtype)
        GP_extend_sub_mask = (1.0 - GP_extend_sub_mask) * -10000.0
        
        user_emb = self.user_embedding(user_ids) 
        GP_emb = self.GP_encoder(user_emb, dec_item_emb, GP_extend_sub_mask, GP_total_idx)

        z = self.consistency_gate(GP_emb, enc_item_emb) 
        h_u = z*enc_item_emb + (1-z)*GP_emb

        return h_u

    def add_position_embedding(self, input_items):
        seq_len = input_items.size(-1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_items.device)
        position_ids = position_ids.unsqueeze(0).unsqueeze(1).expand_as(input_items)

        item_emb = self.item_embedding(input_items) 
        position_encoding = self.position_embedding(position_ids)

        item_emb = item_emb + position_encoding
        item_emb = self.LayerNorm(item_emb)
        item_emb = self.Dropout(item_emb) 

        return item_emb

    def consistency_gate(self, GP_emb, item_emb):
        concat_emb = torch.cat([GP_emb, item_emb], dim=-1) 
        output = self.PCG_layer2(self.PCG_dropout(self.PCG_layer1(concat_emb))) 
        return output 

    def get_session_score(self, input_items):
        seq_len = input_items.size(-1)
        out_item_emb = self.out_item_embedding(input_items) 
        sub_item_emb = out_item_emb.unsqueeze(2) 

        pad_mask = (input_items > 0).long() 
        extend_pad_mask = pad_mask.unsqueeze(3).unsqueeze(2) 
        sub_mask = torch.tril(torch.ones((seq_len, seq_len)))
        sub_mask = sub_mask.unsqueeze(2).unsqueeze(0).unsqueeze(0).to(self.args.device) 
        sub_extend_pad_mask = (sub_mask * extend_pad_mask).float()

        sub_item_emb = sub_item_emb * sub_extend_pad_mask
        sub_item_emb = sub_item_emb.sum(dim=-2)

        # sub_item_emb.shape == [batch, ses_len, seq_len, hidden_size]
        return sub_item_emb 

    def cross_entropy(self, h_u, ses_sum, pos_items, neg_items, input_items):
        batch_size, ses_len, seq_len, hidden_size = h_u.shape 
        pos_item_emb = self.out_item_embedding(pos_items) 
        neg_item_emb = self.out_item_embedding(neg_items) 

        h_u = h_u.view(-1, hidden_size) 
        ses_sum = ses_sum.view(-1, hidden_size) 
        pos_item_emb = pos_item_emb.view(-1, hidden_size)
        neg_item_emb = neg_item_emb.view(-1, hidden_size) 
        
        hu_pos = torch.sum(h_u*pos_item_emb, -1)
        hu_neg = torch.sum(h_u*neg_item_emb, -1) 
        ses_pos = torch.sum(ses_sum*pos_item_emb, -1)
        ses_neg = torch.sum(ses_sum*neg_item_emb, -1) 

        pos_logits = (hu_pos+ses_pos) / 2
        neg_logits = (hu_neg+ses_neg) / 2 

        target_items = (input_items > 0).view(batch_size*ses_len*seq_len).float() 
        loss = torch.sum(-(torch.log(torch.sigmoid(pos_logits) + 1e-24) + torch.log((1 - torch.sigmoid(neg_logits)) + 1e-24)) * target_items) / torch.sum(target_items) 

        return loss

    def predict(self, h_u, ses_sum):
        out_item_emb_weight = self.out_item_embedding.weight 
        
        last_h_u = h_u[:, -1, -1, :]
        last_ses_sum = ses_sum[:, -1, -1, :] 

        h_u_pred = torch.matmul(last_h_u, out_item_emb_weight.transpose(0, 1)) 
        ses_pred = torch.matmul(last_ses_sum, out_item_emb_weight.transpose(0, 1)) 

        out_pred = h_u_pred + ses_pred 
        return out_pred 

    def forward(self, user_ids, input_items):
        h_u = self.get_hu(user_ids, input_items) 
        session_sum = self.get_session_score(input_items)
        return h_u, session_sum

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0, std=self.args.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()