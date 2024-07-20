import torch
import gymnasium as gym
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from a3c.a3c_model import A3CModel
from a3c.opt import SharedAdam
from env import create_atari_env

NUM_STEPS = 20
LOG_WANDB = True
MAX_FRAMES = 1e7

def loss_fn(rewards, values, log_probs, entropies):
    policy_loss = 0
    value_loss = 0
    gamma = 0.99
    beta = 0.01
    gae_lambda = 1.00
    gae = torch.zeros(1, 1)
    R = values[-1]
    for i in reversed(range(len(rewards))):
        R = max(min(rewards[i], 1), -1) + (gamma * R)
        advantage = (R - values[i])

        delta_t = rewards[i] + gamma * values[i + 1] - values[i]
        gae = gae * gamma * gae_lambda + delta_t
        policy_loss = policy_loss - log_probs[i] * gae.detach() - beta * entropies[i]
        value_loss += 0.5*(advantage**2)

    return policy_loss, value_loss

def sync_grads(local_model, shared_agent):
    for param, shared_param in zip(local_model.parameters(), shared_agent.parameters()):
        if shared_param.grad is not None: 
            return
        shared_param._grad = param.grad

def train(shared_agent: A3CModel, shared_optim: SharedAdam, T, lock, pid):
    torch.manual_seed(1 + pid)
    if LOG_WANDB:
        run = wandb.init(project="a3c-pong", group="experiment_1")

    env = create_atari_env("PongDeterministic-v4")
    env.seed(1 + pid)
    local_model = A3CModel(input_size=1, action_size=env.action_space.n)
    
    observation, info = env.reset()
    hidden = None
    current_rewards = []
    local_count = 0
    done_count = 0
    while T.value < MAX_FRAMES: # Race condition is ok - worst case we run one extra frame
        local_count += 1
        local_model.load_state_dict(shared_agent.state_dict())
        local_model.train()

        log_probs = []
        entropies = []
        values = []
        rewards = []
        if hidden is not None:
            hidden = hidden.detach()

        for s in range(NUM_STEPS):
            with lock:
                T.value += 1
                if T.value % 200000 == 0:
                    torch.save(shared_agent, f"model_{T.value}.pt")
                if T.value % (MAX_FRAMES // 100) == 0:
                    print(f"AT FRAME {T.value}")

            input = torch.from_numpy(observation)
            logits, value, hidden = local_model(input, hidden)

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            h = -(log_prob * probs).sum(1, keepdim=True)

            action = probs.multinomial(num_samples=1).detach()
            log_p_a_s = log_prob.gather(1, action)

            observation, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            log_probs.append(log_p_a_s)
            rewards.append(reward)
            current_rewards.append(reward)
            values.append(value)
            entropies.append(h)

            if done:
                observation, info = env.reset()
                if LOG_WANDB and done_count % 10 == 0:
                    wandb.log({"reward": sum(current_rewards)})
                hidden = None
                current_rewards = []
                done_count += 1
                break

        R = torch.zeros(1, 1)
        if not done:
            last_state = torch.from_numpy(observation)
            _, value, _ = local_model(last_state, hidden)
            R = value.detach()
        
        values.append(R)
        value_coef = 0.5
        policy_loss, value_loss = loss_fn(rewards, values, log_probs, entropies)
        total_loss = policy_loss + value_coef*value_loss
        
        if LOG_WANDB and local_count % 100 == 0:
            wandb.log({"avg_entropy": sum(entropies)/len(entropies)})
            wandb.log({"actor": policy_loss})
            wandb.log({"critic": value_coef*value_loss})
        
        shared_optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 50)
        sync_grads(local_model, shared_agent)
        shared_optim.step()

    if len(current_rewards) > 0:
        if LOG_WANDB:
            wandb.log({"reward": sum(current_rewards)})

    env.close()
    if LOG_WANDB:
        run.finish()

if __name__ == "__main__":
    try:
        T = mp.Value('i', 0)
        lock = mp.Lock()
        torch.manual_seed(1)
        env = gym.make("PongDeterministic-v4")
        shared_agent = A3CModel(input_size=1, action_size=env.action_space.n)
        shared_agent.share_memory()

        shared_optim = SharedAdam(shared_agent.parameters())
        env.close()

        processes = []
        for rank in range(16):
            p = mp.Process(target=train, args=(shared_agent, shared_optim, T, lock, rank))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        torch.save(shared_agent, "shared_saved.pt")
    except KeyboardInterrupt:
        torch.save(shared_agent, "shared_saved.pt")
        pass