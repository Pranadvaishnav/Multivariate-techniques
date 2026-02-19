from datasets import load_dataset
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1024
epochs = 50
latent_dim = 16
beta = 0.1

ds = load_dataset("David-Egea/Creditcard-fraud-detection", split="train")
df =ds.to_pandas()
X = df.drop("Class", axis=1).values
y = df["Class"].values

X_tr_full, X_te, y_tr_full, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42)

X_tr = X_tr_full[y_tr_full == 0]
sc = StandardScaler().fit(X_tr)
X_tr = sc.transform(X_tr)
X_te = sc.transform(X_te)

X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32)
X_te_tensor = torch.tensor(X_te, dtype=torch.float32)
y_te_tensor = torch.tensor(y_te, dtype=torch.int64)

train_loader = DataLoader(TensorDataset(X_tr_tensor), batch_size=batch_size, shuffle=True)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder_pca = nn.Sequential(
            nn.Linear(28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.encoder_raw = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )
        self.mu = nn.Linear(64 + 16, latent_dim)
        self.logvar = nn.Linear(64 + 16, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def encoder(self, x):
        pca, raw = x[:, :28], x[:, 28:]
        h = torch.cat((self.encoder_pca(pca), self.encoder_raw(raw)), 1)
        return self.mu(h), self.logvar(h)


    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar =self.encoder(x)
        z = self.reparam(mu, logvar)
        return self.dec(z), mu, logvar
    

model = VAE(X_tr.shape[1], latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
w = torch.tensor(1/X_tr.var(0), device=device, dtype=torch.float32)    

for epoch in range(1, epochs+1):
    model.train(); tot=0
    for (x,) in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        recon_loss = (((x-recon)**2 * w).mean(1)).mean()
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss
        loss.backward();optimizer.step()
        tot += loss.item() * x.size(0)
    print(f"Epoch {epoch:2d}, Loss: {tot / len(X_tr):.6f}")

    if epoch % 5 == 0:
        print("Evaluating...")


model.eval()
with torch.no_grad():
        recon, _, _ = model(X_te_tensor.to(device))
        err = ((recon.cpu() -X_te_tensor)**2 * w.cpu()).mean(1).numpy()

        roc = roc_auc_score(y_te, err)
        pr, re, th = precision_recall_curve(y_te, err)
        f1 = 2*pr*re/(pr+re+1e-8)
        best = f1.argmax(); best_th = th[best]
        print(f"ROC-AUC: {roc:.4f}, Best F1: {f1[best]:.4f} at threshold {best_th:.6f}")

plt.plot(re, pr, label=f"Epoch {epoch}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
plt.legend(); plt.show()

def _to_tensor(x):
    # Accepts numpy array OR 1-D torch tensor
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:                           # assume ndarray / list-like
        return torch.tensor(x, dtype=torch.float32, device=device)


def is_fraud(sample, thr=best_th):
    sample = _to_tensor(sample).unsqueeze(0)   # shape [1, features]
    with torch.no_grad():
        rec, _, _ = model(sample)
    err = ((rec - sample) * w).pow(2).mean().item()
    return int(err > thr), err


def top_k_anomalies(k=10):
    # returns (index, score) sorted DESC by anomaly score
    scores = err.copy()   # 'err' was computed in evaluation cell
    idx_sorted = scores.argsort()[::-1][:k]
    return list(zip(idx_sorted, scores[idx_sorted]))
# First real fraud in the test split
y_te = torch.tensor(y_te)
first_fraud_idx = (y_te == 1).nonzero(as_tuple=True)[0][0].item()
flag, score = is_fraud(X_te[first_fraud_idx])
print(f"idx {first_fraud_idx} | pred={flag} | score={score:.6f} | true=1")

# Show top-10 highest error rows
for idx, sc in top_k_anomalies(10):
    print(f"idx {idx:6d} | score={sc:.6f} | label={y_te[idx].item()}")


plt.axvline(best_th, c="k", ls="--", label=f"thr={best_th:.3f}")
plt.yscale("log"); plt.xlabel("weighted reconstruction error"); plt.legend()
plt.title("Score distribution"); plt.show()
