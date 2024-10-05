import json
import random
import torch
import numpy as np

from torch.utils.data import Dataset
from scipy.sparse import csr_matrix

def generate_rating_matrix(user_dict, num_users, num_items, max_ses_len, max_seq_len, dtype="valid"): 
    if dtype == "valid":
        tail = -2
    elif dtype == "test":
        tail = -1 

    row, col, data = [], [], [] 
    for uid in user_dict: 
        user_ses = user_dict[uid] 
        user_seq = user_ses[-1][-max_seq_len:]
        for item in user_seq[:tail]: 
            row.append(uid)  
            col.append(item) 
            data.append(1) 

    row = np.array(row) 
    col = np.array(col) 
    data = np.array(data)  
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))
    return rating_matrix

def get_user_sessions(filename, max_ses_len, max_seq_len):
    user_dict = json.loads(open(filename).readline())

    num_users = len(user_dict)
    data_user_sessions = np.zeros((num_users, max_ses_len, max_seq_len), dtype='int32')
    # note that user id starts from index 0 while item id starts from 1

    count = 0
    for uid in user_dict:
        count += 1
        u_ses = user_dict[uid][-max_ses_len:]
        ses_len = len(u_ses)
        ses_st_idx = 0
        if ses_len < max_ses_len:
            ses_st_idx = max_ses_len - ses_len
        for i in range(ses_len):
            u_seq = user_dict[uid][i][-max_seq_len:]
            seq_len = len(u_seq)
            seq_st_idx = 0
            if seq_len < max_seq_len:
                seq_st_idx = max_seq_len - seq_len 
            for j in range(seq_len): 
                data_user_sessions[int(uid), ses_st_idx+i, seq_st_idx+j] = u_seq[j]

    #  max_item = 10802 # LastFM   
    max_item = 224185  # Tmall16
    # max_item = 168254  # Avito

    print("max item number: ",max_item)
    print("max user number: ",num_users)
    print('Shape of user data UserSessions:', data_user_sessions.shape) 

    valid_rating_matrix = generate_rating_matrix(user_dict, num_users, max_item+2, max_ses_len, max_seq_len, dtype="valid")
    test_rating_matrix = generate_rating_matrix(user_dict, num_users, max_item+2, max_ses_len, max_seq_len, dtype="test")

    return data_user_sessions, max_item, valid_rating_matrix, test_rating_matrix

def neg_sampler(item_set, item_size):  # note that the item_size here equals max_size + 2
    neg_item = random.randint(1, item_size-1)
    while neg_item in item_set:
        neg_item = random.randint(1, item_size-1)
    return neg_item

class HASRDataset(Dataset):
    def __init__(self, args, user_data_sessions, data_type="train"):
        self.args = args
        self.user_ses = user_data_sessions
        self.data_type = data_type
        self.max_ses_len = args.max_ses_len
        self.max_seq_len = args.max_seq_len

    def __getitem__(self, index):
        user_id = index
        session = self.user_ses[user_id]

        assert self.data_type in {"train", "valid", "test"}
        if self.data_type == "train":
            input_items = session[:, :-3]
            target_pos = session[:, 1:-2]
            answer = [0]
        elif self.data_type == "valid":
            input_items = session[:, 1:-2]
            target_pos = session[:, 2:-1]
            answer = [session[-1, -2]]
        else:
            input_items = session[:, 2:-1]
            target_pos = session[:, 3:]
            answer = [session[-1, -1]]

        row_len, col_len = input_items.shape
        target_neg = np.zeros(target_pos.shape, dtype="int32")
        item_set = set()
        for i in range(row_len):
            item_set = item_set | set(input_items[i])
            for j in range(col_len):
                if target_pos[i][j] != 0:
                    target_neg[i][j] = neg_sampler(item_set, self.args.item_size)

        assert input_items.shape == (self.max_ses_len, self.max_seq_len)
        assert target_pos.shape == (self.max_ses_len, self.max_seq_len)
        assert target_neg.shape == (self.max_ses_len, self.max_seq_len)

        cur_tensors=  (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(input_items, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long), 
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long)
            )
        return cur_tensors

    def __len__(self):
        return len(self.user_ses)