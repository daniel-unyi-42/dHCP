# scan age or birth age
task = 'scan_age'

model_name = 'MLP.pt'

path = '/home/daniel/data/release/'

# hyperparameters
bs = 8
lr = 0.001
epochs = 500
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
from MLP import MLP
from GCN import GCN
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter

log_dir=f'runs/{task}/{model_name}/features={features}/bs={bs}_lr={lr}_epoch={epochs}_hidden={hidden}'
writer = SummaryWriter(log_dir)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

train_ids = pd.read_csv(task + '_train.txt', header=None)
val_ids = pd.read_csv(task + '_val.txt', header=None)
test_ids = pd.read_csv(task + '_test.txt', header=None)

df = pd.read_csv("combined.tsv", sep='\t')

df.insert(0, "ID", "sub-" + df["participant_id"] + "_" + "ses-" + df["session_id"].apply(str))
df.drop("participant_id", axis=1, inplace=True)
df.drop("session_id", axis=1, inplace=True)

# def normalize_x(data):
#     if task == 'scan_age':
#         means = torch.Tensor([1.2060, 0.0446, 1.3145, 0.3749])
#         stds = torch.Tensor([0.3509, 0.2248, 0.4062, 4.4142])
#     elif task == 'birth_age':
#         means = torch.Tensor([1.2147, 0.0440, 1.3289, 0.3958])
#         stds = torch.Tensor([0.3474, 0.2261, 0.4049, 4.4723])
#     data.x = (data.x - means) / stds
#     return data

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
print(len(train_set), len(val_set), len(test_set))

train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
val_loader = DataLoader(val_set, batch_size=bs)
test_loader = DataLoader(test_set, batch_size=bs)

if model_name == 'MLP.pt':
    model = MLP(in_channels=in_channels, hidden_channels=hidden, out_channels=1)
elif model_name == 'GCN.pt':
    model = GCN(in_channels=in_channels, hidden_channels=hidden, out_channels=1)
else:
    raise NameError("Model doesn't exist!")

print(model)

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

def train(loader, model, optimizer):
    model.train()
    losses = []
    accs = []
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.mse_loss(out, data.y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        acc = F.l1_loss(out, data.y)
        accs.append(acc.item())
    return sum(losses) / len(losses), sum(accs) / len(accs)

@torch.no_grad()
def test(loader, model):
    model.eval()
    losses = []
    accs = []
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = F.mse_loss(out, data.y)
        losses.append(loss.item())
        acc = F.l1_loss(out, data.y)
        accs.append(acc.item())
    return sum(losses) / len(losses), sum(accs) / len(accs)

@torch.no_grad()
def plot_reg(loader, model):
    model.eval()
    outs = []
    ys = []
    for data in loader:
        data = data.to(device)
        out = model(data)
        outs.append(out.cpu().numpy())
        ys.append(data.y.cpu().numpy())
    import matplotlib.pyplot as plt
    plt.scatter(np.concatenate(ys), np.concatenate(outs))
    plt.xlabel('y')
    plt.ylabel('out')
    plt.savefig(os.path.join(log_dir, 'regression.png'))
    plt.close()


train_losses = []
val_losses = []
test_losses = []
train_accs = []
val_accs = []
test_accs = []
best_val_index = 0
for epoch in range(epochs):
    train_loss, train_acc = train(train_loader, model, optimizer)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    writer.add_scalar('MSE/train', train_loss, epoch)
    writer.add_scalar('MAE/train', train_acc, epoch)
    val_loss, val_acc = test(val_loader, model)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    writer.add_scalar('MSE/val', val_loss, epoch)
    writer.add_scalar('MAE/val', val_acc, epoch)
    test_loss, test_acc = test(test_loader, model)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    writer.add_scalar('MSE/test', test_loss, epoch)
    writer.add_scalar('MAE/test', test_acc, epoch)
    if val_loss < val_losses[best_val_index]:
        best_val_index = epoch
        torch.save(model.state_dict(), os.path.join(log_dir, model_name))
    if val_loss < 1.0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
    print(epoch, best_val_index, train_loss, train_acc,
          val_loss, val_acc, test_loss, test_acc)

writer.flush()
writer.close()

model.load_state_dict(torch.load(os.path.join(log_dir, model_name)))
plot_reg(test_loader, model)
