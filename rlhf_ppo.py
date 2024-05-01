import torch
import numpy as np
from transformers import CausalLMWithValueHead, AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("your_model_name_or_path")
device = "cuda:0"
def transition(state, action):
    return state+actions

def freeze_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

def copy_model_parameters(old_model, new_model):
    for param_old, param_new in zip(old_model.parameters(), new_model.parameters()):
        param_old.data.copy_(param_new.data)

def compute_prob(logits, output_length):
    generated_logits = logits[:, -output_length:, :]
    generated_probs = F.softmax(generated_logits, dim=-1)
    sequence_prob = torch.prod(torch.diagonal(generated_probs[:, :-1], dim1=1, dim2=2))
    return sequence_prob

def nll(logits, output_length):
    generated_logits = logits[:, -output_length:, :]
    probs = F.softmax(generated_logits, dim=-1)

    # Compute negative log-likelihood
    nll = -torch.sum(torch.log(probs) * probs, dim=-1)

    # Sum the negative log-likelihood for all tokens
    total_nll = torch.sum(nll)
    return total_nll

old_policy=CausalLMWithValueHead.from_pretrained("your_model_name_or_path").to(device)
new_policy=CausalLMWithValueHead.from_pretrained("your_model_name_or_path").to(device)
freeze_model(old_policy)

c1=1.
c2=1.
eps=0.1
# policy= model.base_model # the underlying LM? dont keep shared parameters, doesnt work, freeze reward model, make copy for policy
freeze_model(reward_model)
num_generations=5
for batch in dataloader: 3 #set batch_size =4 or something
    copy_model_parameters(old_policy, new_policy)
    for mini_batch in batch:
        optimizer.zero_grad()
        state= data
        toks=tokenizer(state, return_tensors="pt").input_ids.to(device)
        _, _, values_new = new_policy(toks)
        _, _, values_old = old_policy(toks)
        action= new_policy.generate(toks, num_return_sequences=num_generations)
        outpt_length = action.shape[1]
        rewards=reward_model(toks+action)
        lm_logits_new, _, _ = new_policy(toks+action)
        lm_logits_old, _, _ = old_policy(toks+action)
        advantage = rewards - values_old

        r_t = compute_prob(lm_logits_new, output_length) / compute_prob(lm_logits_old, output_length)
        l_clip_1 = r_t*advantage
        l_clip_2 = torch.clamp(r_t, 1-eps, 1+eps, retain_grad=True)*advantage
        loss= torch.mean(torch.min(l_clip_1,l_clip_2)) # clipped loss
        loss-=c1*torch.nn.functional.mse_loss(values_new, rewards) # value loss
        loss+=c2*nll(lm_logits_new, output_length) # entropy loss
        loss.backward()
        optimizer.step()