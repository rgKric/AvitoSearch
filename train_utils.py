from torch.utils.data import Dataset, DataLoader
import torch
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

class ItemDataset(Dataset):
    def __init__(self, model, items_df, encoder_big=None):
        super().__init__()
        self.embedder = model

        item_phone_hidden = torch.from_numpy(items_df[['item_is_phone_hidden', 'item_is_message_forbidden']].to_numpy().astype(np.float32))
        item_price = torch.from_numpy(items_df['item_price'].to_numpy().astype(np.float32)).unsqueeze(-1)
        item_rating = torch.from_numpy(items_df['item_rating'].to_numpy().astype(np.float32)).unsqueeze(-1)
        item_rating_reviews_count = torch.from_numpy(items_df['item_rating_reviews_count'].to_numpy().astype(np.float32)).unsqueeze(-1)
        self.context = torch.cat([item_phone_hidden, item_price, item_rating, item_rating_reviews_count], dim=1)

        self.item_emb = prepare_embeddings(items_df, 'item_title_raw_norm', self.embedder)
        # для гибридной системы
        if encoder_big != None:
            self.item_disc = prepare_embeddings(items_df, 'item_description_raw_norm', encoder_big)
            self.disc=True
        else:
            self.disc=False

        self.item_id_to_idx = {item_id: idx
                          for idx, item_id in enumerate(items_df['item_id'])}
        self.item_ids = items_df['item_id'].to_numpy()

        vids = items_df['item_vid_services'].fillna('UNK')
        # выставить unk на первое место, чтобы дальше не было багов
        values = ['UNK'] + [x for x in vids.unique() if x != 'UNK']
        self.str_to_id = {s: i for i, s in enumerate(values)}
        self.id_to_str = {i: s for i, s in enumerate(values)}
        self.item_category = torch.from_numpy(vids.map(self.str_to_id).to_numpy().astype(np.int32))
        self.item_location_id = torch.from_numpy(items_df['item_location_id'].to_numpy().astype(np.int32))

    def __len__(self):
        return self.item_emb.shape[0]

    def __getitem__(self, idx):
        if self.disc:
            return (self.item_emb[idx], self.context[idx], self.item_disc[idx], self.item_location_id[idx], self.item_category[idx])
        else:
            return (self.item_emb[idx], self.context[idx], self.item_location_id[idx], self.item_category[idx])


class QueriesDataset(Dataset):
    def __init__(self, queries_df, item_dataset, mode='train'):
        super().__init__()
        self.mode = mode
        self.embedder = item_dataset.embedder
        self.item_dataset = item_dataset
        self.q_emb = prepare_embeddings(queries_df, 'search_query_norm', self.embedder)

        # связываем 2 таблицы
        if self.mode == 'train':
            self.query_item_idx = torch.tensor(
                [
                    self.item_dataset.item_id_to_idx[item_id]
                    for item_id in queries_df['item_id']
                ],
                dtype=torch.long
            )
        vids = queries_df['search_vid_services'].fillna('UNK')
        self.query_category = torch.from_numpy(vids.map(self.item_dataset.str_to_id).to_numpy().astype(np.int32))
        self.query_location_id = torch.from_numpy(queries_df['search_location_id'].to_numpy().astype(np.int32))

    def __len__(self, ):
        return self.q_emb.shape[0]

    def __getitem__(self, idx):
        if self.mode == 'train':
            item_idx = self.query_item_idx[idx]
            if self.item_dataset.disc:
                item_emb, context, item_disc, _, _ = self.item_dataset[item_idx]
                return (
                    self.q_emb[idx], 
                    item_emb, 
                    context, 
                    item_disc,
                    item_idx,
                    self.query_location_id[idx],
                    self.query_category[idx],
                )
            else:
                item_emb, context, _, _ = self.item_dataset[item_idx]
                return (
                    self.q_emb[idx], 
                    item_emb, 
                    context, 
                    item_idx,
                    self.query_location_id[idx],
                    self.query_category[idx],
                )
        elif self.mode == 'test':
            return (
                    self.q_emb[idx],
                    self.query_location_id[idx],
                    self.query_category[idx],
                )


def prepare_embeddings(df, col, model):
    data = df[col].to_list()
    if isinstance(model, SentenceTransformer):
        print('Use SentenceTransformer')
        embeddings = model.encode(data, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        return embeddings
    else:
        print('Use Word2Vec')
        res_vecs = []
        for i in tqdm(range(len(data)), desc=col):
            indices_matrix = [model.get_index(word, default=0) for word in data[i]] 
            vectors = model.vectors[indices_matrix]
            res_vec = np.mean(vectors, axis=0)
            res_vecs.append(res_vec)
        return torch.from_numpy(np.array(res_vecs).astype(np.float32))


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device="cuda:0"
    return device


def evaluate(queries_dataset, items_dataset, model, device, k=50, filters=True):
    model.eval()
    item_dataloader = DataLoader(dataset=items_dataset,
                                 batch_size=1024,
                                 num_workers=0,
                                 shuffle=False)
    queries_dataloader = DataLoader(dataset=queries_dataset,
                                    batch_size=64,
                                    num_workers=0,
                                    shuffle=False)
    i_embeddings = []
    with torch.no_grad():
        for batch in tqdm(item_dataloader, desc='Items'):
            if items_dataset.disc == True:
                i_emb, context, item_disc, _, _ = batch
                i_emb = i_emb.to(device)
                context = context.to(device)
                item_disc = item_disc.to(device)
                i = model.get_i_tower_emb((i_emb, context, item_disc))
            else:
                i_emb, context, _, _ = batch
                i_emb = i_emb.to(device)
                context = context.to(device)
                i = model.get_i_tower_emb((i_emb, context))
            i_embeddings.append(i)

    item_location_id = items_dataset.item_location_id
    item_category = items_dataset.item_category
    i_embeddings = torch.cat(i_embeddings, dim=0)
    item_ids = torch.arange(0, len(items_dataset.item_ids))

    hits = 0
    # ndcg_sum = 0
    n_queries = 0
    ans = []
    with torch.no_grad():
        for batch in tqdm(queries_dataloader, desc='Queries'):
            if queries_dataset.mode == 'train':
                q_emb, _, _, real_item_idx, query_location_id, query_category = batch
            elif queries_dataset.mode == 'test':
                q_emb, query_location_id, query_category = batch
            q_emb = q_emb.to(device)
            q = model.get_q_tower_emb(q_emb)
            scores = q @ i_embeddings.T

            if filters:
                # # маска локации
                location_mask = (query_location_id[:, None] == item_location_id[None, :]).to(device)
                # маска вида услуги
                category_mask = ((query_category[:, None] == 0) | (query_category[:, None] == item_category[None, :])).to(device)
                # общая маска
                valid_mask = location_mask & category_mask
                # наложение маски
                scores = scores.masked_fill(~valid_mask, -torch.inf)

            _, top_indices = torch.topk(scores, k=k, dim=1)

            top_ids = item_ids[top_indices.cpu()]
            if queries_dataset.mode == 'train':
                matches = top_ids == real_item_idx[:, None]

                hit = matches.any(dim=1)
                hits += hit.sum().item()
                n_queries += len(real_item_idx)
            ans.append(top_ids)

    if queries_dataset.mode == 'train':       
        recall = hits / n_queries
        return {f'Recall@{k}': recall,}
    elif queries_dataset.mode == 'test':
        return ans