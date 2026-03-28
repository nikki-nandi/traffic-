import torch
import torch.optim as optim
from backend.env.hybrid_env import HybridTrafficEnv
from backend.models.dqn_model import DQN

def train():
    env = HybridTrafficEnv()
    model = DQN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for ep in range(5):  # reduce for testing
        state = torch.FloatTensor(env.reset())

        for _ in range(20):
            q = model(state)
            action = torch.argmax(q).item()

            next_state, reward = env.step(action)
            next_state = torch.FloatTensor(next_state)

            target = reward + 0.9 * torch.max(model(next_state))
            loss = (q[action] - target.detach()) ** 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        print(f"✅ Episode {ep} done")

    torch.save(model.state_dict(), "backend/models/dqn.pth")
    print("🔥 TRAINING COMPLETE")

if __name__ == "__main__":
    train()