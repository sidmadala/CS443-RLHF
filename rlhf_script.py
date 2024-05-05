import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerFast, TrainingArguments, Trainer, AutoModel
from datasets import load_dataset, load_from_disk
import torch.optim as optim
from tqdm import tqdm
import warnings
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2TokenizerFast, AutoModelForCausalLM
import os
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from sklearn.model_selection import train_test_split
import subprocess
import matplotlib.pyplot as plt

# class LiveLossPlotAndSave(PlotLosses):
#     def __init__(self, output_path='./'):
#         super().__init__()
#         self.output_path = output_path
#         self.train_batch_count = 0
#         self.val_batch_count = 1
#         self.best_val_loss = float('inf')

#     def on_train_begin(self, logs={}):
#         print("came here")
#         super().on_train_begin(logs)
#         self.fig, (self.ax_train, self.ax_val) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

#     def on_batch_end(self, batch, logs={}):
#         print("came here 2")
#         super().on_batch_end(batch, logs)
#         self.train_batch_count += 1
#         self.val_batch_count += 1
#         self.ax_train.plot(self.train_batch_count, logs.get('loss'), 'r', label='train_loss')
#         self.ax_train.set_xlabel('Batch Number')
#         self.ax_train.set_ylabel('Loss')
#         self.ax_train.legend()
#         # plt.pause(0.005)  # Required to update the plot
#         self.fig.savefig("plot.png")
#         # plt.savefig("plot.png")
        
#         if self.val_batch_count % 10 == 0:
            
#             self.ax_val.plot(self.val_batch_count, logs.get('val_loss'), 'b', label='val_loss')
#             self.ax_val.set_xlabel('Batch Number')
#             self.ax_val.set_ylabel('Loss')
#             self.ax_val.legend()
#             # plt.pause(0.005)  # Required to update the plot
#             # self.fig.savefig(f"{self.output_path}/batch_{self.val_batch_count}_plot.png")
#             self.fig.savefig("plot.png")
#             # plt.savefig("plot.png")



# Suppress the warning
# warnings.filterwarnings("ignore", category=UserWarning, message="Setting `pad_token_id` to `eos_token_id`")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Using device: {torch.cuda.get_device_name({device})}")
subprocess.run(["rm", "-rf","rlhf_expts_ckpts"])
os.mkdir("rlhf_expts_ckpts")

model_id="gpt2"
dataset_id = "Dahoas/full-hh-rlhf"
batch_size = 32
max_length=256+128 # total sequence length input+output
gamma=1.
c1=0.1
c2=0.05
eps=0.1
epochs=1
beta=1.
learning_rate = 5e-5
weight_decay = 0.01
ppo_iters_per_batch =4 # gotta do 4 updates per batch
print_every = 50  # Checkpointing after every 50 batches


## it seems policy LM cant ber similar to deberta as its encoder only, so need to use gpt-2 for policy, so tokenization mismatch is eminent

# def preprocess(example):
#     example=example["prompt"]
#     return tokenizer(example, return_tensors="pt", padding="longest", truncation=True, max_length=256).to(device)

class CustomDataset(Dataset):
    def __init__(self, examples):
        self.examples=examples
        # self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]["prompt"]
        # example=self.examples[idx]["prompt"]
        # out_ = self.tokenizer(example, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        # # return out_
        # for key in out_.keys():
        #     out_[key] = out_[key].squeeze(0)
        # return out_

def freeze_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

# def copy_model_parameters(old_model, new_model):
#     for param_old, param_new in zip(old_model.parameters(), new_model.parameters()):
#         param_old.data.copy_(param_new.data)

#TODO: reimplement ALL functions below as verbatim taken from library

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

def entropy_from_logits(logits):
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd * logits, axis=-1)
    return entropy

def masked_mean(values, mask, axis= None):
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()

def logprobs_from_logits(logits, labels, gather = True):
    logp = F.log_softmax(logits, dim=2)
    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy

def kl_penalty(logprob, ref_logprob):
        return logprob - ref_logprob


"""
we maintain 1 policy, and 2 reward models (one trained and the other reference model of the reward model, and then we generate whole trajectory for input data
using policy before entering ppo iterations for a batch. We also have a linear layer on top of output embeddings which gives value function estimate for each step
of the generation. Now for that batch, lets say we wanna run 4 PPO epochs per batch, so we compute value function estimate using policy, run grad descent on policy
for the batch for 4 ppo epochs, compute generation for the next batch using policy, and then keep on doing this. reference model of the reward model is used to compute
per step rewards, as the reward model's reward is scalar for a given sequence, so we keep that as the final reward on completion, and keep kl-divergence between
reference model and reward model for modelling per step reward
"""

#TODO: Integrate past_key_values to fasten up training
# from transformers import LlamaTokenizer, LlamaForCausalLM
# policy_model_path = 'openlm-research/open_llama_3b'
# policy_tokenizer = LlamaTokenizer.from_pretrained(policy_model_path)
# old_policy = LlamaForCausalLM.from_pretrained(policy_model_path, torch_dtype=torch.float16)
# freeze_model(old_policy)


#Policy models
policy_tokenizer = AutoTokenizer.from_pretrained(model_id, padding="max_length", direction="left", padding_side="left",max_length=256, length=256)
base_reward=AutoModelForCausalLM.from_pretrained(model_id).to(device)
freeze_model(base_reward)
policy=AutoModelForCausalLM.from_pretrained(model_id).to(device)
config = policy.config
# policy_tokenizer2 = AutoTokenizer.from_pretrained(model_id, padding="max_length", direction="left",  padding_side="left",max_length=512,length=512)
embedding_dim = config.hidden_size
linear = nn.Linear(embedding_dim, 1, device=device)

# if policy_tokenizer.pad_token is None:
#     policy_tokenizer.pad_token = policy_tokenizer.eos_token
#     old_policy.config.pad_token_id = old_policy.config.eos_token_id
#     new_policy.config.pad_token_id = new_policy.config.eos_token_id
# reward_tokenizer = AutoTokenizer.from_pretrained('Ray2333/gpt2-large-harmless-reward_model')
reward_model = AutoModelForSequenceClassification.from_pretrained(
                'Ray2333/gpt2-large-harmless-reward_model',
                num_labels=1).to(device)
freeze_model(reward_model)

# if reward_tokenizer.pad_token is None:
#     reward_tokenizer.pad_token = reward_tokenizer.eos_token
#     reward_model.config.pad_token_id = reward_model.config.eos_token_id

if policy_tokenizer.pad_token is None:
    policy_tokenizer.pad_token = policy_tokenizer.eos_token
    # policy_tokenizer2.pad_token = policy_tokenizer2.eos_token
    policy.config.pad_token_id = policy.config.eos_token_id
    
if reward_model.config.pad_token_id is None:
    reward_model.config.pad_token_id = policy.config.pad_token_id
    base_reward.config.pad_token_id = reward_model.config.eos_token_id
    
# print(policy_tokenizer1.pad_token_id)
## assuming reward and policy models here are having same tokenizers atleast T_T


# policy_tokenizer_config = policy_tokenizer.config
# reward_tokenizer_config = reward_tokenizer.config

# print("policy_tokenizer_config", policy_tokenizer_config)
# print("reward_tokenizer_config", reward_tokenizer_config)
# print(reward_model.config.pad_token_id, old_policy.config.pad_token_id)
# assert policy_tokenizer_config == reward_tokenizer.config
# print(tokenizer.pad_token_id)

dataset = load_dataset(dataset_id)
train_dataset = dataset["train"]
print("original train dataset length", len(train_dataset))

all_train=[]
for i in range(len(train_dataset)):
    all_train.append(train_dataset[i])

train_dataset, val_dataset = train_test_split(all_train, test_size=0.2, random_state=42)
del all_train
train_dataset = train_dataset[:30000]
val_dataset = val_dataset[:500]
train_dataset = CustomDataset(train_dataset)
val_dataset = CustomDataset(val_dataset)
test_dataset = CustomDataset(dataset['test'])

print("train dataset size", len(train_dataset))
print("val dataset size", len(val_dataset))
print("test dataset size", len(test_dataset))


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

optimizer = optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

count=0
liveloss = PlotLosses(outputs=[MatplotlibPlot(figpath ="figures/liveloss1.png")])
liveloss2 = PlotLosses(outputs=[MatplotlibPlot(figpath ="figures/liveloss2.png")])
liveloss.best_val_loss = float('inf')  # Initialize with a very high value to ensure that the first validation loss is always lower
for e in (range(epochs)):
    for batch_ in tqdm(train_dataloader):
        count+=1
        # print(batch.keys())
        # print("batch shape", batch["input_ids"].shape)
        batch= policy_tokenizer(batch_, return_tensors="pt", padding="max_length", truncation=True, max_length=256).to(device)
        # copy_model_parameters(old_policy, new_policy)
        # inputs=policy_tokenizer(batch, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_old=policy.generate(input_ids=batch.input_ids, attention_mask=batch.attention_mask, \
                                                   max_length=max_length, num_return_sequences=1, return_dict_in_generate=True,\
                                                  output_scores=True)
            # outputs_old_policy=old_policy(batch, max_length=max_length)
            # print(outputs_old_policy)
            # print(outputs_old_policy.sequences.shape)
            # print(len(outputs_old_policy.scores))
            # print(outputs_old_policy.scores[0].shape)
            logits_old = torch.stack(outputs_old.scores, dim=1).to(device) # logits for generated sequence
            # print(logits_old_policy.shape)
            outputs_ids_old = logits_old.argmax(-1)
            # outputs_ids_old_policy = outputs_old_policy.sequences
            # logits_old_policy = outputs_old_policy.scores
            # print(logits_old.shape)
            old_logprobs = logprobs_from_logits(logits_old, outputs_ids_old)
            # print(old_policy_logprobs.shape)
            # values_old=linear(outputs_old_policy.hidden_state[-1])
            episode_length = max_length - batch.input_ids.shape[1]
            # print(episode_length)
            # freeze_model(old_policy)

            # mini_batch_gen_output_length = episode_length .....
            # aint doing mini-batches as its not done in the paper we use as reference (mini-batch of size 1 means takimng whole batch itself as mini-batch)
            output_attn_mask=(outputs_ids_old != policy.config.pad_token_id).float().to(device)
            assert output_attn_mask.shape == old_logprobs.shape, (output_attn_mask.shape, old_logprobs.shape) 
            old_output_logprob = masked_mean(old_logprobs, output_attn_mask)
            # print(old_policy_output_logprob)
            useful_output_length = int(torch.max(torch.sum(output_attn_mask, dim=1,keepdim=True)).cpu().item())
            # print(useful_output_length)
            concatenated_input = {
                                    'input_ids': torch.cat([batch["input_ids"], outputs_ids_old], dim=1).to(device),
                                    'attention_mask': torch.cat([batch["attention_mask"], output_attn_mask], dim=1).to(device)
                                }
        batch_losses=[]
        for iter in range(ppo_iters_per_batch):
            outputs_policy=policy.generate(input_ids=batch.input_ids, attention_mask=batch.attention_mask, \
                                               max_length=max_length, num_return_sequences=1, return_dict_in_generate=True,\
                                              output_scores=True)
            logits_policy = torch.stack(outputs_policy.scores, dim=1).to(device)
            assert logits_policy.shape[1] == episode_length, (logits_policy.shape[1], episode_length)
            # print(outputs_new_policy.keys())
            
            hidden_state = policy.transformer(input_ids=batch.input_ids, attention_mask=batch.attention_mask, return_dict=True).last_hidden_state
            # print(new_outs_.keys())
            # hidden_state=new_outs_.hidden_states[-1]
            # print(hidden_state.shape)
            values=linear(hidden_state[:,-(episode_length+1):-1,:]).squeeze(-1)
            # outputs_ids_old_policy = outputs_old_policy.logits.argmax(-1)
            # logits_old_policy = outputs_old_policy.logits
            # discounted_rewards=[]
            
            # rewards=[]
            reversed_rewards=[]
            # print(concatenated_input["input_ids"].shape)
            optimizer.zero_grad()
            # rewards=[]
            # for t in range(useful_output_length):
            #     rewards.append(reward_model(input_ids=concatenated_input["input_ids"][:,:t],\
            #                                          attention_mask= concatenated_input["attention_mask"][:,:t]).logits.cpu())
            
            final_rewards = reward_model(input_ids=concatenated_input["input_ids"], attention_mask= concatenated_input["attention_mask"]).logits
            # size: batch_size
            # print(final_rewards.shape)
            policy_logprobs = logprobs_from_logits(logits_policy, outputs_ids_old)
            # print("policy logprobs shape", policy_logprobs.shape)
            rewards = -beta*kl_penalty(policy_logprobs, old_logprobs)
            rewards[:,-1]+=final_rewards.squeeze(-1)
            # print(rewards.shape)
            reversed_rewards=rewards.flip(1)
            reversed_discounted_rewards=reversed_rewards[:,0].clone().unsqueeze(1)
            # print(useful_output_length)
            # print(reversed_discounted_rewards.shape)
            # print(reversed_rewards.shape)
            # print(reversed_rewards[:,t].unsqueeze(1).shape)

#             timesteps = torch.arange(useful_output_length).unsqueeze(0).to(device)
#             discounted_rewards = gamma ** timesteps

# # Compute the cumulative sum along the time dimension
# cumulative_discounted_rewards = discounted_rewards.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])

# # Extract the first column as the reversed_discounted_rewards
# reversed_discounted_rewards = cumulative_discounted_rewards[:, 0].unsqueeze(1)
            for t in range(1,useful_output_length):
                reversed_discounted_rewards = torch.cat((reversed_discounted_rewards, reversed_rewards[:,t].unsqueeze(1) + gamma*reversed_discounted_rewards[:,-1].unsqueeze(-1)), dim=1)
                # print(reversed_discounted_rewards.shape)
                # reversed_advantages.append(reversed_rewards[-1] + gamma*reversed_advantages[-1]
            # reversed_discounted_rewards.pop(0)
            # reversed_rewards = torch.Tensor(reversed_rewards).to(device)
            # print(reversed_rewards.shape)
            # reversed_discounted_rewards=torch.tensor(reversed_discounted_rewards).to(device)
            # print("reversed_discounted_rewards shape", reversed_discounted_rewards.shape)
            
            # rewards=reversed_rewards.flip(0)
            discounted_rewards=reversed_discounted_rewards.flip(1)
            # print("values shape", values.shape)
            # print("discounted_rewards shape", discounted_rewards.shape)
            advantages=-values + discounted_rewards
            
            assert output_attn_mask.shape == policy_logprobs.shape, (output_attn_mask.shape, policy_prob.shape) 
            policy_output_logprob = masked_mean(policy_logprobs, output_attn_mask)
            ratio = torch.exp(policy_logprobs - old_logprobs)
            # print("ratio shape", ratio.shape)
            # print("advantages shape", advantages.shape)
            a=torch.mul(advantages,torch.clamp(ratio, 1-eps, 1+eps))
            # print(a.shape)
            loss = torch.mean(a) ## clip loss
            loss += c1*torch.mean(advantages**2) ## mse loss 
            loss -= c2* torch.mean(entropy_from_logits(logits_policy))
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        
        if count%print_every == 0:
            ## do validation
            print("validation!")
            logs2={}
            val_losses = []
            with torch.no_grad():
                
                for batch_2 in val_dataloader:
                    batch= policy_tokenizer(batch_2, return_tensors="pt", padding="max_length", truncation=True, max_length=256).to(device)
                    with torch.no_grad():
                        outputs_val=policy.generate(input_ids=batch.input_ids, attention_mask=batch.attention_mask, \
                                                       max_length=max_length, num_return_sequences=1, return_dict_in_generate=True,\
                                                      output_scores=True)
                        logits_val = torch.stack(outputs_val.scores, dim=1).to(device) # logits for generated sequence
                        outputs_ids_val = logits_val.argmax(-1)
                        output_attn_mask=(outputs_ids_val != policy.config.pad_token_id).float().to(device)
                        concatenated_input = {
                                        'input_ids': torch.cat([batch["input_ids"], outputs_ids_val], dim=1).to(device),
                                        'attention_mask': torch.cat([batch["attention_mask"], output_attn_mask], dim=1).to(device)
                                    }
                        reward = torch.mean(reward_model(input_ids=concatenated_input["input_ids"], attention_mask= concatenated_input["attention_mask"]).logits.cpu()).item()
                        val_losses.append(1-reward)
            
            val_loss = sum(val_losses)/len(val_losses)
            if val_loss < liveloss.best_val_loss:
                    liveloss.best_val_loss = val_loss
                    torch.save(policy.state_dict(), "rlhf_expts_ckpts/best_model.pth")
            logs2['val_loss'] = val_loss
            liveloss2.update(logs2)
            liveloss2.send()
            
        logs = {'loss': sum(batch_losses)/len(batch_losses)}
        liveloss.update(logs)
        liveloss.send()          
        # liveloss.fig.savefig("results/plot.png")
            

            
        