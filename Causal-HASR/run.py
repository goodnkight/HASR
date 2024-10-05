import torch
import argparse
import tqdm
import random
import os
import numpy as np

from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from Causal_HASR import Causal_HASR
from dataload import get_user_sessions, HASRDataset
from modules import Recall_at_k, NDCG_k, EarlyStopping

def mkdir(path):
    path.strip().rstrip("\\") 
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + "file created") 
    else:
        print(path + "file existed")

def get_scores(epoch, answer_list, pred_list):
    recall, ndcg = [], [] 
    for k in [5, 10, 20]:
        recall.append(Recall_at_k(answer_list, pred_list, k)) 
        ndcg.append(NDCG_k(answer_list, pred_list, k)) 

    post_fix = {
        "Epoch: ": epoch, 
        "HIT@5: ": "{:.4f}".format(recall[0]), "NDCG@5: ": "{:.4f}".format(ndcg[0]),
        "HIT@10: ": "{:.4f}".format(recall[1]), "NDCG@10: ": "{:.4f}".format(ndcg[1]),
        "HIT@20: ": "{:.4f}".format(recall[2]), "NDCG@20: ": "{:.4f}".format(ndcg[2]),
        }
    print(str(post_fix))

    return [recall, ndcg]

def eval(rating_pred, user_idx_batch, valid_rating_matrix):
    rating_pred[valid_rating_matrix[user_idx_batch].toarray() > 0] = 0 
    index = np.argpartition(rating_pred, -20)[:, -20:] 
    arr_index  = rating_pred[np.arange(len(rating_pred))[:, None], index] 
    arr_index_argsort = np.argsort(arr_index)[np.arange(len(rating_pred)), ::-1] 
    batch_pred_list = index[np.arange(len(rating_pred))[:, None], arr_index_argsort]
    # obtain top 20 rated items from rating_pred
    return batch_pred_list  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--data_name", type=str, default="Tmall16")
    parser.add_argument("--output_dir", type=str, default="output/")
    parser.add_argument("--save_dir", type=str, default="save/")

    parser.add_argument("--max_ses_len", type=int, default=12, help="8 for LastFM")
    parser.add_argument("--max_seq_len", type=int, default=32, help="44 for LastFM")

    parser.add_argument("--lr", type=float, default=0.0003, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--eval_mode", type=bool, default=False)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--num_attention_heads", type=int, default=4) 
    parser.add_argument("--eval_freq", type=int, default=10) 
    parser.add_argument("--random_seed", type=int, default=77) 

    parser.add_argument("--hidden_dropout_rate", type=float, default=0.5, help="hidden dropout p") 
    parser.add_argument("--attn_dropout_rate", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    
    args = parser.parse_args() 
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    args.data_file = args.data_dir + args.data_name + ".json"
    args.save_path = args.save_dir + args.data_name
    user_data_sessions, max_item, valid_rating_matrix, test_rating_matrix = get_user_sessions(args.data_file, args.max_ses_len, args.max_seq_len+3) 
    args.item_size = max_item + 2
    args.user_size = user_data_sessions.shape[0]
    print("Loading data is done")

    # Load data
    train_dataset = HASRDataset(args, user_data_sessions, data_type="train")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    valid_dataset = HASRDataset(args, user_data_sessions, data_type="valid")
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.batch_size)

    test_dataset = HASRDataset(args, user_data_sessions, data_type="test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    mkdir(args.save_dir)
    model = Causal_HASR(args=args)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(args.device)
    betas = (args.adam_beta1, args.adam_beta2)
    optim = Adam(model.parameters(), lr=args.lr, betas=betas, weight_decay=args.weight_decay) 
    # args.eval_mode = True
    if not args.eval_mode:
        for epoch in range(args.num_epochs):
            str_code = "train"
            dataloader = train_dataloader
            rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                      desc="Recommendation Epochs_%s:%d" % (str_code, epoch),
                                      total=len(dataloader),
                                      bar_format="{l_bar}{r_bar}") 

            # Train model
            model.train() 
            rec_cur_loss = 0.0
            rec_avg_loss = 0.0 
            for i, batch in rec_data_iter:
                batch = tuple(b.to(args.device) for b in batch) 
                user_ids, input_items, target_pos, target_neg, answer = batch
                h_u, session_sum = model.forward(user_ids, input_items)
                loss = model.cross_entropy(h_u, session_sum, target_pos, target_neg, input_items)

                optim.zero_grad()
                loss.backward()
                optim.step()

                rec_cur_loss = loss.item()
                rec_avg_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
            } 
            print(str(post_fix))

            # Eval model
            if epoch%args.eval_freq == 0:
                model.eval()
                str_code = "valid"
                dataloader = valid_dataloader

                rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                        desc="Recommendation Epochs_%s:%d" % (str_code, epoch),
                                        total=len(dataloader),
                                        bar_format="{l_bar}{r_bar}")

                for i, batch in rec_data_iter:
                    batch = tuple(b.to(args.device) for b in batch) 
                    user_ids, input_items, target_pos, target_neg, answer = batch 
                    h_u, session_sum = model.forward(user_ids, input_items) 
                    rating_pred = model.predict(h_u, session_sum) 
                    rating_pred = rating_pred.cpu().data.numpy().copy()

                    user_idx_batch = user_ids.cpu().numpy()
                    pred_list_batch = eval(rating_pred, user_idx_batch, valid_rating_matrix) + 1

                    if i == 0:
                        pred_list = pred_list_batch
                        answer_list = answer.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, pred_list_batch, axis=0)
                        answer_list= np.append(answer_list, answer.cpu().data.numpy(), axis=0)  
                # print(pred_list[:5], answer[:5])
                score = get_scores(epoch, answer_list, pred_list)
                
    # Save model          
    args.save_file =  args.save_path+"_best.pt"
    torch.save(model.state_dict(), args.save_file) 

    # Load the best model
    model.load_state_dict(torch.load(args.save_file))

    model.eval()
    pred_list = None
    answer_list = None 
    str_code = "test"
    dataloader = test_dataloader
    if args.eval_mode: 
        epoch = 0
    rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                              desc="Recommendation EP_%s:%d" % (str_code, epoch),
                              total=len(dataloader), 
                              bar_format="{l_bar}{r_bar}")

    for i, batch in rec_data_iter:
        batch = tuple(t.to(args.device) for t in batch)
        user_ids, input_items, target_pos, target_neg, answers = batch 
        h_u, session_sum = model.forward(user_ids, input_items)

        rating_pred = model.predict(h_u,session_sum)
        rating_pred = rating_pred.cpu().data.numpy().copy()

        user_index_batch = user_ids.cpu().numpy()
        pred_list_batch = eval(rating_pred, user_index_batch, test_rating_matrix)  + 1

        if i == 0:
            pred_list = pred_list_batch
            answer_list = answers.cpu().data.numpy()  
        else:
            pred_list = np.append(pred_list, pred_list_batch, axis=0)
            answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0) 
    score = get_scores(epoch, answer_list, pred_list)