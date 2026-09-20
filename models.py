import torch.nn as nn
import torch
import torch.nn.functional as F


class ITower(nn.Module):
    def __init__(self):
        super(ITower, self).__init__()
        self.context_proj = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 128), 
            nn.ReLU()
        )

        self.dense = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

    def forward(self, data):
        i_emb, i_context = data
        i_context_emb = self.context_proj(i_context)
        x = torch.cat((i_emb, i_context_emb), dim=1)
        i_tower_emb = self.dense(x)
        return i_tower_emb

class QTower(nn.Module):
    def __init__(self):
        super(QTower, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, q_emb):
        q_tower_emb = self.dense(q_emb)
        return q_tower_emb

class TwoTower(nn.Module):
    def __init__(self):
        super(TwoTower, self).__init__()
        self.q_tower = QTower()
        self.i_tower = ITower()

    def get_q_tower_emb(self, q_emb):
        q_tower_emb = self.q_tower(q_emb)
        q = F.normalize(q_tower_emb, dim=-1)
        return q

    def get_i_tower_emb(self, data):
        i_emb, context = data
        i_tower_emb = self.i_tower((i_emb, context))
        i = F.normalize(i_tower_emb, dim=-1)
        return i
    
    def forward(self, data):
        q_emb, i_emb, context = data
        q = self.get_q_tower_emb(q_emb)
        i = self.get_i_tower_emb((i_emb, context))
        return q, i


class ITowerHF(nn.Module):
    def __init__(self, context_proj):
        super(ITowerHF, self).__init__()
        if context_proj:
            self.context_proj = nn.Sequential(
                nn.Linear(5, 64),
                nn.ReLU(),
                nn.Linear(64, 128), 
                nn.ReLU()
            )
            start_dim = 512
        else:
            self.context_proj=None
            start_dim = 384

        self.dense = nn.Sequential(
            nn.Linear(start_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

    def forward(self, data):
        i_emb, i_context = data
        if self.context_proj != None:
            i_context_emb = self.context_proj(i_context)
            x = torch.cat((i_emb, i_context_emb), dim=1)
            i_tower_emb = self.dense(x)
        else:
            i_tower_emb = self.dense(i_emb)
        return i_tower_emb

class QTowerHF(nn.Module):
    def __init__(self):
        super(QTowerHF, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

    def forward(self, q_emb):
        q_tower_emb = self.dense(q_emb)
        return q_tower_emb

class TwoTowerHF(nn.Module):
    def __init__(self, context_proj=True):
        super(TwoTowerHF, self).__init__()
        self.q_tower = QTowerHF()
        self.i_tower = ITowerHF(context_proj)

    def get_q_tower_emb(self, q_emb):
        q_tower_emb = self.q_tower(q_emb)
        q = F.normalize(q_tower_emb, dim=-1)
        return q

    def get_i_tower_emb(self, data):
        i_emb, context = data
        i_tower_emb = self.i_tower((i_emb, context))
        i = F.normalize(i_tower_emb, dim=-1)
        return i
    
    def forward(self, data):
        q_emb, i_emb, context = data
        q = self.get_q_tower_emb(q_emb)
        i = self.get_i_tower_emb((i_emb, context))
        return q, i


class ITowerHybride(nn.Module):
    def __init__(self, context_proj):
        super(ITowerHybride, self).__init__()
        if context_proj:
            self.context_proj = nn.Sequential(
                nn.Linear(5, 64),
                nn.ReLU(),
                nn.Linear(64, 128), 
                nn.ReLU()
            )
            start_dim = 512
        else:
            self.context_proj=None
            start_dim = 384

        self.dense = nn.Sequential(
            nn.Linear(start_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

    def forward(self, data):
        i_emb, i_context, i_disc = data
        if self.context_proj != None:
            i_context_emb = self.context_proj(i_context)
            x = torch.cat((i_emb, i_context_emb), dim=1)
            i_tower_emb = self.dense(x)
        else:
            i_tower_emb = self.dense(i_emb)
        return i_tower_emb

class QTowerHybrid(nn.Module):
    def __init__(self):
        super(QTowerHybrid, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

    def forward(self, q_emb):
        q_tower_emb = self.dense(q_emb)
        return q_tower_emb

class TwoTowerHybrid(nn.Module):
    def __init__(self, context_proj=True):
        super(TwoTowerHybrid, self).__init__()
        self.q_tower = QTowerHybrid()
        self.i_tower = ITowerHybride(context_proj)

    def get_q_tower_emb(self, q_emb):
        q_tower_emb = self.q_tower(q_emb)
        q = F.normalize(q_tower_emb, dim=-1)
        return q

    def get_i_tower_emb(self, data):
        i_emb, context = data
        i_tower_emb = self.i_tower((i_emb, context))
        i = F.normalize(i_tower_emb, dim=-1)
        return i
    
    def forward(self, data):
        q_emb, i_emb, context, i_disc = data
        q = self.get_q_tower_emb(q_emb)
        i = self.get_i_tower_emb((i_emb, context, i_disc))
        return q, i