# scan age or birth age
task = 'scan_age'

model_name = 'MLP.pt'

path = '/home/daniel/data/release/'

# hyperparameters
bs = 8
lr = 0.001
epochs = 200
hidden = 64
features = 'pos+norm+dha+x'

in_channels = 0
if 'pos' in features:
    in_channels += 3
if 'norm' in features:
    in_channels += 3
if 'dha' in features:
    in_channels += 3
if 'x' in features:
    in_channels += 4

import numpy as np
import pandas as pd
import nibabel as nib
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from MLP import MLP, actor_MLP
from GCN import GCN, actor_GCN
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

log_dir=f'runs/invase/{task}/{model_name}/features={features}/bs={bs}_lr={lr}_epoch={epochs}_hidden={hidden}'
writer = SummaryWriter(log_dir)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

train_ids = pd.read_csv(task + '_train.txt', header=None)
val_ids = pd.read_csv(task + '_val.txt', header=None)
test_ids = pd.read_csv(task + '_test.txt', header=None)

df = pd.read_csv("combined.tsv", sep='\t')

df.insert(0, "ID", "sub-" + df["participant_id"] + "_" + "ses-" + df["session_id"].apply(str))
df.drop("participant_id", axis=1, inplace=True)
df.drop("session_id", axis=1, inplace=True)

transform = T.Compose([T.NormalizeScale(), T.GenerateMeshNormals(), T.FaceToEdge()])

def get_data(path, task, ids):
    dataset = []
    for _id in ids[0]:
        try:
            surface = nib.load(os.path.join(path, 'surfaces', _id + '_left.wm.surf.gii'))
            pos, face = surface.agg_data()
            feature = nib.load(os.path.join(path, 'features', _id + '_left.shape.gii'))
            x = np.stack(feature.agg_data(), axis=1)
            y = np.array([[df.loc[df['ID'] == _id, task].item()]])
            data = Data()
            data.id = _id
            if 'x' in features:
                data.x = torch.from_numpy(x).to(torch.float32)
            data.pos = torch.from_numpy(pos).to(torch.float32)
            data.face = torch.from_numpy(face.T).to(torch.long)
            data.y = torch.from_numpy(y).to(torch.float32)
            if task == 'birth_age':
                confound = np.array([[df.loc[df['ID'] == _id, 'scan_age'].item()]])
                data.confound = torch.from_numpy(confound).to(torch.float32)
            data = transform(data)
            if 'norm' not in features:
                data.norm = None
            if 'dha' in features:
                data.dha = torch.from_numpy(np.load(os.path.join(path, 'preprocess/V_dihedral_angles', \
                                                                 _id + '_left.wm.surf_V_dihedralAngles.npy'))).to(torch.float32)
            # data.eig = torch.from_numpy(np.load(os.path.join(path, 'preprocess/aligned_eigen_vectors',
            #                                                     _id + '_left.wm.surf_eigen.npy'))).to(torch.float32)
            # data.curv = torch.from_numpy(np.load(os.path.join(path, 'preprocess/gaussian_curvatures',
            #                                                     _id + '_left.wm.surf_gaussian_curvature.npy'))).to(torch.float32).unsqueeze(1)
            # data.curv = (data.curv - data.curv.min()) / (data.curv.max() - data.curv.min())
            # data.hks = torch.from_numpy(np.load(os.path.join(path, 'preprocess/HKS',
            #                                                 _id + '_left.wm.surf_hks.npy'))).to(torch.float32)
            dataset.append(data)
        except Exception as error:
            print(error)
    return dataset

train_set = get_data(path, task, train_ids)
val_set = get_data(path, task, val_ids)
test_set = get_data(path, task, test_ids)

train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
val_loader = DataLoader(val_set, batch_size=bs)
test_loader = DataLoader(test_set, batch_size=bs)

class Invase():
    def __init__(self):
        if model_name == 'MLP.pt':
            self.critic = MLP(in_channels=in_channels, hidden_channels=hidden, out_channels=1)
            self.baseline = MLP(in_channels=in_channels, hidden_channels=hidden, out_channels=1)
            self.actor = actor_MLP(in_channels=in_channels, hidden_channels=hidden, out_channels=in_channels)
        elif model_name == 'GCN.pt':
            self.critic = GCN(in_channels=in_channels, hidden_channels=hidden, out_channels=1)
            self.baseline = GCN(in_channels=in_channels, hidden_channels=hidden, out_channels=1)
            self.actor = actor_GCN(in_channels=in_channels, hidden_channels=hidden, out_channels=in_channels)
        self.critic = self.critic.to(device)
        self.critic.optimizer = torch.optim.AdamW(self.critic.parameters(), lr=lr)
        self.critic.criterion = nn.MSELoss()
        self.baseline = self.baseline.to(device)
        self.baseline.optimizer = torch.optim.AdamW(self.baseline.parameters(), lr=lr)
        self.baseline.criterion = nn.MSELoss()
        self.actor = self.actor.to(device)
        self.actor.optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr)
        self.actor.criterion = self.actor_loss
        self.lambda_ = 10.0

    def actor_loss(self, actor_pred, actor_out, critic_out, baseline_out, y_true):
        critic_loss = F.mse_loss(critic_out, y_true)
        baseline_loss = F.mse_loss(baseline_out, y_true)
        reward = -(critic_loss - baseline_loss)
        # reward * BCE(actor_pred, actor_out) - lambda * ||actor_pred||
        custom_actor_loss = reward * torch.sum(actor_out * torch.log(actor_pred + 1e-8) + \
                                               (1.0 - actor_out)* torch.log(1.0 - actor_pred + 1e-8), dim=1) - \
                                                self.lambda_ * torch.mean(actor_pred, dim=1)
        custom_actor_loss = torch.mean(-custom_actor_loss)
        return custom_actor_loss

invase = Invase()

def train(loader, invase):
    actor_losses = []
    critic_losses = []
    critic_accs = []
    baseline_losses = []
    baseline_accs = []
    for data in loader:
        data = data.to(device)
        # baseline training
        invase.baseline.train()
        invase.baseline.optimizer.zero_grad()
        baseline_out = invase.baseline(data)
        baseline_loss = invase.baseline.criterion(baseline_out, data.y)
        baseline_losses.append(baseline_loss.item())
        baseline_acc = F.l1_loss(baseline_out, data.y)
        baseline_accs.append(baseline_acc.item())
        baseline_loss.backward()
        invase.baseline.optimizer.step()
        # critic training
        invase.actor.eval()
        with torch.no_grad():
            selection_probability = invase.actor(data)
            selection = torch.bernoulli(selection_probability)
        invase.critic.train()
        invase.critic.optimizer.zero_grad()
        critic_out = invase.critic(data, selection)
        critic_loss = invase.critic.criterion(critic_out, data.y)
        critic_losses.append(critic_loss.item())
        critic_acc = F.l1_loss(critic_out, data.y)
        critic_accs.append(critic_acc.item())
        critic_loss.backward()
        invase.critic.optimizer.step()
        # actor training
        invase.actor.train()
        invase.actor.optimizer.zero_grad()
        actor_out = invase.actor(data)
        invase.critic.eval()
        with torch.no_grad():
            critic_out = invase.critic(data, selection)
        invase.baseline.eval()
        with torch.no_grad():
            baseline_out = invase.baseline(data)
        actor_loss = invase.actor.criterion(actor_out, selection, critic_out, baseline_out, data.y)
        actor_losses.append(actor_loss.item())
        actor_loss.backward()
        invase.actor.optimizer.step()
    return sum(actor_losses) / len(actor_losses), \
           sum(critic_losses) / len(critic_losses), sum(critic_accs) / len(critic_accs), \
           sum(baseline_losses) / len(baseline_losses), sum(baseline_accs) / len(baseline_accs)

@torch.no_grad()
def test(loader, invase):
    actor_losses = []
    critic_losses = []
    critic_accs = []
    baseline_losses = []
    baseline_accs = []
    invase.baseline.eval()
    invase.critic.eval()
    invase.actor.eval()
    for data in loader:
        data = data.to(device)
        # baseline testing
        baseline_out = invase.baseline(data)
        baseline_loss = invase.baseline.criterion(baseline_out, data.y)
        baseline_losses.append(baseline_loss.item())
        baseline_acc = F.l1_loss(baseline_out, data.y)
        baseline_accs.append(baseline_acc.item())
        # critic testing
        selection_probability = invase.actor(data)
        selection = torch.bernoulli(selection_probability)
        critic_out = invase.critic(data, selection)
        critic_loss = invase.critic.criterion(critic_out, data.y)
        critic_losses.append(critic_loss.item())
        critic_acc = F.l1_loss(critic_out, data.y)
        critic_accs.append(critic_acc.item())
        # actor testing
        actor_loss = invase.actor.criterion(selection_probability, selection, critic_out, baseline_out, data.y)
        actor_losses.append(actor_loss.item())
        return sum(actor_losses) / len(actor_losses), \
               sum(critic_losses) / len(critic_losses), sum(critic_accs) / len(critic_accs), \
               sum(baseline_losses) / len(baseline_losses), sum(baseline_accs) / len(baseline_accs)

@torch.no_grad()
def plot_reg(loader, invase):
    invase.baseline.eval()
    invase.critic.eval()
    invase.actor.eval()
    baseline_outs = []
    critic_outs = []
    ys = []
    for data in loader:
        data = data.to(device)
        baseline_out = invase.baseline(data)
        baseline_outs.append(baseline_out.cpu().numpy())
        selection_probability = invase.actor(data)
        selection = torch.bernoulli(selection_probability)
        critic_out = invase.critic(data, selection)
        critic_outs.append(critic_out.cpu().numpy())
        ys.append(data.y.cpu().numpy())
    plt.scatter(np.concatenate(ys), np.concatenate(baseline_outs))
    plt.xlabel('y')
    plt.ylabel('baseline_out')
    plt.savefig(os.path.join(log_dir, 'baseline_regression.png'))
    plt.close()
    plt.scatter(np.concatenate(ys), np.concatenate(critic_outs))
    plt.xlabel('y')
    plt.ylabel('critic_out')
    plt.savefig(os.path.join(log_dir, 'critic_regression.png'))
    plt.close()


train_baseline_losses = []
train_baseline_accs = []
train_critic_losses = []
train_critic_accs = []
train_actor_losses = []
val_baseline_losses = []
val_baseline_accs = []
val_critic_losses = []
val_critic_accs = []
val_actor_losses = []
test_baseline_losses = []
test_baseline_accs = []
test_critic_losses = []
test_critic_accs = []
test_actor_losses = []
best_val_index = 0
for epoch in range(epochs):
    train_actor_loss, train_critic_loss, train_critic_acc, train_baseline_loss, train_baseline_acc = train(train_loader, invase)
    train_actor_losses.append(train_actor_loss)
    train_critic_losses.append(train_critic_loss)
    train_critic_accs.append(train_critic_acc)
    train_baseline_losses.append(train_baseline_loss)
    train_baseline_accs.append(train_baseline_acc)
    writer.add_scalar('MSE/actor_train', train_actor_loss, epoch)
    writer.add_scalar('MSE/critic_train', train_critic_loss, epoch)
    writer.add_scalar('MAE/critic_train', train_critic_acc, epoch)
    writer.add_scalar('MSE/baseline_train', train_baseline_loss, epoch)
    writer.add_scalar('MAE/baseline_train', train_baseline_acc, epoch)
    val_actor_loss, val_critic_loss, val_critic_acc, val_baseline_loss, val_baseline_acc = test(val_loader, invase)
    val_actor_losses.append(val_actor_loss)
    val_critic_losses.append(val_critic_loss)
    val_critic_accs.append(val_critic_acc)
    val_baseline_losses.append(val_baseline_loss)
    val_baseline_accs.append(val_baseline_acc)
    writer.add_scalar('MSE/actor_val', val_actor_loss, epoch)
    writer.add_scalar('MSE/critic_val', val_critic_loss, epoch)
    writer.add_scalar('MAE/critic_val', val_critic_acc, epoch)
    writer.add_scalar('MSE/baseline_val', val_baseline_loss, epoch)
    writer.add_scalar('MAE/baseline_val', val_baseline_acc, epoch)
    test_actor_loss, test_critic_loss, test_critic_acc, test_baseline_loss, test_baseline_acc = test(test_loader, invase)
    test_actor_losses.append(test_actor_loss)
    test_critic_losses.append(test_critic_loss)
    test_critic_accs.append(test_critic_acc)
    test_baseline_losses.append(test_baseline_loss)
    test_baseline_accs.append(test_baseline_acc)
    writer.add_scalar('MSE/actor_test', test_actor_loss, epoch)
    writer.add_scalar('MSE/critic_test', test_critic_loss, epoch)
    writer.add_scalar('MAE/critic_test', test_critic_acc, epoch)
    writer.add_scalar('MSE/baseline_test', test_baseline_loss, epoch)
    writer.add_scalar('MAE/baseline_test', test_baseline_acc, epoch)
    if val_baseline_loss < val_baseline_losses[best_val_index]:
        best_val_index = epoch
        torch.save(invase.baseline.state_dict(), os.path.join(log_dir, model_name + '_baseline'))
        torch.save(invase.critic.state_dict(), os.path.join(log_dir, model_name + '_critic'))
        torch.save(invase.actor.state_dict(), os.path.join(log_dir, model_name + '_actor'))
    if val_baseline_loss < 1.0:
        for param_group in invase.baseline.optimizer.param_groups:
            param_group['lr'] = 0.0001
        for param_group in invase.critic.optimizer.param_groups:
            param_group['lr'] = 0.0001
        for param_group in invase.actor.optimizer.param_groups:
            param_group['lr'] = 0.0001
    print(epoch, best_val_index,
          train_actor_loss, train_critic_loss, train_critic_acc, train_baseline_loss, train_baseline_acc,
          val_actor_loss, val_critic_loss, val_critic_acc, val_baseline_loss, val_baseline_acc,
          test_actor_loss, test_critic_loss, test_critic_acc, test_baseline_loss, test_baseline_acc)

writer.flush()
writer.close()

invase.baseline.load_state_dict(torch.load(os.path.join(log_dir, model_name + '_baseline')))
invase.critic.load_state_dict(torch.load(os.path.join(log_dir, model_name + '_critic')))
invase.actor.load_state_dict(torch.load(os.path.join(log_dir, model_name + '_actor')))
plot_reg(test_loader, invase)
