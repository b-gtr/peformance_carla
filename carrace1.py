import os
import sys
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from collections import namedtuple, deque
from torch.optim.lr_scheduler import StepLR

# -------------------------
# Hyperparameters
# -------------------------
NUM_EPISODES = 100   # RL training episodes
MAX_STEPS_PER_EPISODE = 1000
BATCH_SIZE = 64
LATENT_DIM = 64
SCALAR_DIM = 0        # No scalar inputs for CarRacing
ACTION_DIM = 3
LEARNING_RATE = 1e-4
GAMMA = 0.99
POLYAK = 0.995
TARGET_ENTROPY = -ACTION_DIM
VAE_EPOCHS = 2         # For demonstration; increase for better results
VAE_BATCH_SIZE = 64
VAE_LR = 1e-4
VAE_DATASET_SIZE = 5000  # Number of frames to gather for VAE training
REPLAY_CAPACITY = 20000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition', 'img scalars a r n_img n_scalars d')
Batch = namedtuple('Batch', 'img scalars a r n_img n_scalars d')

# -------------------------
# Variational Encoder/Decoder
# -------------------------
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.latent_dims = latent_dims
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # out: (32, 48, 48)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # out: (64, 24,24)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),# out: (128,12,12)
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),# out: (256,6,6)
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256*6*6, latent_dims)
        self.fc_logvar = nn.Linear(256*6*6, latent_dims)
        self.kl = None

    def forward(self, x, deterministic=False):
        # x: (B,3,96,96)
        h = self.conv(x)
        h = h.contiguous().view(h.size(0), -1)  # Ensure contiguous memory layout
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        if deterministic:
            # Just return mean if deterministic
            self.kl = torch.zeros(1, device=x.device)
            return mu
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # KL divergence
        self.kl = 0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1) / x.size(0)
        return z
    
    def encode(self, x, deterministic=False):
        return self.forward(x, deterministic=deterministic)

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dims, 256*6*6),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), #12x12
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1),  #24x24
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1),   #48x48
            nn.ReLU(),
            nn.ConvTranspose2d(32,3,4,2,1),    #96x96
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.fc(x)
        x = x.view(-1,256,6,6)
        x = self.deconv(x)
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# -------------------------
# Replay Buffer
# -------------------------
class ReplayBuffer(object):
    def __init__(self, capacity: int, device=None):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.device = device if device is not None else torch.device('cpu')

    def push(self, transition: Transition) -> None:
        self.memory.appendleft(transition)

    def ready_for(self, batch_size: int) -> bool:
        return len(self.memory) >= batch_size

    def sample(self, batch_size: int) -> Batch:
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        batch = Batch(*zip(*batch))

        img_array = np.array(batch.img)  # (B,H,W,3)
        a_array = np.array(batch.a)
        r_array = np.array(batch.r)
        n_img_array = np.array(batch.n_img)
        d_array = np.array(batch.d)

        if batch.scalars[0] is None or len(batch.scalars[0]) == 0:
            scalars_array = np.zeros((len(batch.img),0))
            n_scalars_array = np.zeros((len(batch.img),0))
        else:
            scalars_array = np.array(batch.scalars)
            n_scalars_array = np.array(batch.n_scalars)

        img = torch.tensor(img_array, dtype=torch.float).to(self.device) / 255.0
        # Move to (B,3,H,W)
        img = img.permute(0,3,1,2)
        scalars = torch.tensor(scalars_array, dtype=torch.float).to(self.device)
        a = torch.tensor(a_array, dtype=torch.float).view(len(batch.img), -1).to(self.device)
        r = torch.tensor(r_array, dtype=torch.float).view(len(batch.img), 1).to(self.device)

        n_img = torch.tensor(n_img_array, dtype=torch.float).to(self.device)/255.0
        n_img = n_img.permute(0,3,1,2)
        n_scalars = torch.tensor(n_scalars_array, dtype=torch.float).to(self.device)
        d = torch.tensor(d_array, dtype=torch.float).view(len(batch.img),1).to(self.device)

        return Batch(img, scalars, a, r, n_img, n_scalars, d)

# -------------------------
# SAC Agent
# -------------------------
def get_net(num_in: int, num_out: int, final_activation, num_hidden_layers: int = 2, num_neurons_per_hidden_layer: int = 256) -> nn.Sequential:
    layers = []
    layers.append(nn.Linear(num_in, num_neurons_per_hidden_layer))
    layers.append(nn.ReLU())
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(num_neurons_per_hidden_layer, num_neurons_per_hidden_layer))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(num_neurons_per_hidden_layer, num_out))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)

class NormalPolicyNet(nn.Module):
    def __init__(self, feature_dim, scalar_dim, action_dim):
        super(NormalPolicyNet, self).__init__()
        self.input_dim = feature_dim + scalar_dim
        self.trunk = get_net(self.input_dim, 256, None)
        self.out = nn.Linear(256, 2*action_dim)

    def forward(self, features: torch.tensor, scalars: torch.tensor):
        if scalars.shape[1] > 0:
            x = torch.cat([features, scalars], dim=1)
        else:
            x = features
        x = self.trunk(x)
        x = self.out(x)
        return x

class QNet(nn.Module):
    def __init__(self, feature_dim, scalar_dim, action_dim):
        super(QNet, self).__init__()
        self.input_dim = feature_dim + scalar_dim + action_dim
        self.net = get_net(self.input_dim,1,None)
    def forward(self, features, scalars, actions):
        if scalars.shape[1] > 0:
            x = torch.cat([features, scalars, actions], dim=1)
        else:
            x = torch.cat([features, actions], dim=1)
        return self.net(x)

class ParamsPool:
    def __init__(self, latent_dim, scalar_dim, action_dim, device=None):
        self.device = device if device is not None else torch.device('cpu')
        self.latent_dim = latent_dim
        self.scalar_dim = scalar_dim
        self.action_dim = action_dim
        self.gamma = GAMMA
        self.polyak = POLYAK
        self.target_entropy = TARGET_ENTROPY

        self.encoder = None  # will set after VAE training

        self.Normal = NormalPolicyNet(latent_dim, scalar_dim, action_dim).to(self.device)
        self.Q1 = QNet(latent_dim, scalar_dim, action_dim).to(self.device)
        self.Q2 = QNet(latent_dim, scalar_dim, action_dim).to(self.device)
        self.Q1_targ = QNet(latent_dim, scalar_dim, action_dim).to(self.device)
        self.Q2_targ = QNet(latent_dim, scalar_dim, action_dim).to(self.device)
        self.Q1_targ.load_state_dict(self.Q1.state_dict())
        self.Q2_targ.load_state_dict(self.Q2.state_dict())

        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=self.device)

        self.Normal_optimizer = optim.Adam(self.Normal.parameters(), lr=LEARNING_RATE)
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=LEARNING_RATE)
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=LEARNING_RATE)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LEARNING_RATE)

        # LR Schedulers
        self.scheduler_policy = StepLR(self.Normal_optimizer, step_size=10000, gamma=0.5)
        self.scheduler_q1 = StepLR(self.Q1_optimizer, step_size=10000, gamma=0.5)
        self.scheduler_q2 = StepLR(self.Q2_optimizer, step_size=10000, gamma=0.5)
        self.scheduler_alpha = StepLR(self.alpha_optimizer, step_size=10000, gamma=0.5)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def polyak_update(self, old_net, new_net):
        for o_param, n_param in zip(old_net.parameters(), new_net.parameters()):
            o_param.data.copy_(o_param.data*self.polyak + n_param.data*(1-self.polyak))

    def sample_action_and_logp(self, features, scalars, reparam=True):
        out = self.Normal(features, scalars)
        mean, log_std = torch.chunk(out,2,dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        dist = Independent(Normal(mean, std),1)
        if reparam:
            u = dist.rsample()
        else:
            u = dist.sample()
        a = torch.tanh(u)
        # log_pi = dist.log_prob(u) - sum(log(1 - a^2))
        # a in [-1,1], derivative trick
        log_pi = dist.log_prob(u) - torch.sum(torch.log(1 - a.pow(2) + 1e-6), dim=1)
        return a, log_pi

    def update_networks(self, b:Batch):
        with torch.no_grad():
            features_s = self.encoder.encode(b.img, deterministic=False)
            features_ns = self.encoder.encode(b.n_img, deterministic=False)

        # Compute targets
        with torch.no_grad():
            na, log_pi_na = self.sample_action_and_logp(features_ns, b.n_scalars, reparam=False)
            q1_next = self.Q1_targ(features_ns, b.n_scalars, na)
            q2_next = self.Q2_targ(features_ns, b.n_scalars, na)
            min_q_next = torch.min(q1_next, q2_next)
            targets = b.r + self.gamma*(1-b.d)*(min_q_next - self.alpha.detach()*log_pi_na.unsqueeze(-1))

        # Critic update
        q1 = self.Q1(features_s, b.scalars, b.a)
        q2 = self.Q2(features_s, b.scalars, b.a)
        q1_loss = F.mse_loss(q1, targets)
        q2_loss = F.mse_loss(q2, targets)

        self.Q1_optimizer.zero_grad()
        q1_loss.backward()
        self.Q1_optimizer.step()

        self.Q2_optimizer.zero_grad()
        q2_loss.backward()
        self.Q2_optimizer.step()

        # Actor update
        a, log_pi_a = self.sample_action_and_logp(features_s, b.scalars, reparam=True)
        q1_pi = self.Q1(features_s, b.scalars, a)
        q2_pi = self.Q2(features_s, b.scalars, a)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha.detach()*log_pi_a - min_q_pi).mean()

        self.Normal_optimizer.zero_grad()
        policy_loss.backward()
        self.Normal_optimizer.step()

        # Alpha update
        alpha_loss = -(self.log_alpha*(log_pi_a + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Polyak update
        with torch.no_grad():
            self.polyak_update(self.Q1_targ,self.Q1)
            self.polyak_update(self.Q2_targ,self.Q2)

        # Step schedulers
        self.scheduler_policy.step()
        self.scheduler_q1.step()
        self.scheduler_q2.step()
        self.scheduler_alpha.step()

    def act(self, image):
        # image shape expected: (1,3,96,96)
        with torch.no_grad():
            feat = self.encoder.encode(image, deterministic=True)
            scalars = torch.zeros((1,SCALAR_DIM), device=self.device) # no scalars
            out = self.Normal(feat, scalars)
            mean, log_std = torch.chunk(out,2,dim=-1)
            log_std = torch.clamp(log_std,-20,2)
            std = torch.exp(log_std)
            dist = Independent(Normal(mean,std),1)
            a = torch.tanh(dist.sample())
        return a.cpu().numpy()[0]

# -------------------------
# VAE Dataset
# -------------------------
class VAEDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # data[idx] is (H,W,3) in [0,255]
        img = self.data[idx]
        img = self.transform(img) # (3,H,W) in [0,1]
        return img, 0

# -------------------------
# Main Training Loop
# -------------------------
def main():
    env = gym.make("CarRacing-v3", continuous=True, domain_randomize=False)
    env.reset(seed=0)

    # 1. Collect data for VAE
    print("Collecting data for VAE training...")
    vae_data = []
    obs, info = env.reset()
    count = 0
    while count < VAE_DATASET_SIZE:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        # obs: (96,96,3), uint8
        vae_data.append(obs)
        count += 1
        if done or truncated:
            env.reset()

    # 2. Train the VAE
    print("Training VAE...")
    dataset = VAEDataset(vae_data)
    train_size = int(0.8*len(dataset))
    val_size = len(dataset)-train_size
    train_data, val_data = random_split(dataset,[train_size,val_size])
    trainloader = DataLoader(train_data, batch_size=VAE_BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_data, batch_size=VAE_BATCH_SIZE, shuffle=False)

    vae = VariationalAutoencoder(LATENT_DIM).to(device)
    optimizer_vae = optim.Adam(vae.parameters(), lr=VAE_LR)

    for epoch in range(VAE_EPOCHS):
        vae.train()
        train_loss = 0.0
        for x,_ in trainloader:
            x = x.to(device)
            x_hat = vae(x)
            recon_loss = ((x - x_hat)**2).sum()
            kl = vae.encoder.kl if vae.encoder.kl is not None else 0
            loss = recon_loss + kl
            optimizer_vae.zero_grad()
            loss.backward()
            optimizer_vae.step()
            train_loss += loss.item()
        train_loss /= len(trainloader)

        vae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x,_ in valloader:
                x = x.to(device)
                x_hat = vae(x)
                recon_loss = ((x - x_hat)**2).sum()
                kl = vae.encoder.kl if vae.encoder.kl is not None else 0
                loss = recon_loss + kl
                val_loss += loss.item()
        val_loss /= len(valloader)
        print(f"VAE Epoch {epoch+1}/{VAE_EPOCHS} Train Loss: {train_loss:.3f} Val Loss: {val_loss:.3f}")

    # 3. Initialize agent and replay buffer
    agent = ParamsPool(LATENT_DIM, SCALAR_DIM, ACTION_DIM, device=device)
    agent.encoder = vae.encoder
    replay_buffer = ReplayBuffer(REPLAY_CAPACITY, device=device)

    # 4. RL training
    print("Start RL training...")
    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        total_reward = 0.0
        for step in range(MAX_STEPS_PER_EPISODE):
            # Preprocess obs before passing to agent
            # obs: (96,96,3)
            img_tensor = torch.tensor(obs, dtype=torch.float, device=device)/255.0
            # Shape: (H,W,3)
            img_tensor = img_tensor.permute(2,0,1).unsqueeze(0) # (1,3,96,96)

            action = agent.act(img_tensor) # returns a np.array of shape (3,)
            next_obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            replay_buffer.push(Transition(
                img=obs,
                scalars=[], # no scalars
                a=action,
                r=reward,
                n_img=next_obs,
                n_scalars=[],
                d=float(done)
            ))

            obs = next_obs
            if replay_buffer.ready_for(BATCH_SIZE):
                batch = replay_buffer.sample(BATCH_SIZE)
                agent.update_networks(batch)

            if done or truncated:
                break
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}")

    env.close()
    print("Training completed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("Terminating...")
