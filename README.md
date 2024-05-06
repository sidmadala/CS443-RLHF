# CS443-RLHF
RLHF re-implementation for UIUC's CS 443: Reinforcement Learning 

## Dataset

We utilize a preprocessed version of [Anthropic's RLHF dataset](https://huggingface.co/datasets/Dahoas/full-hh-rlhf) to train our reward model and subsequently perform fine-tuning on our LLM. This dataset consists of prompts followed by responses that are human-labaled as chosen or rejected. Chosen respones are those which are both helpful and harmless, while rejected responses are LLM output that contains explicit or offensive material that should be suppressed once fine-tuned.

## Reward Model Training

Code for reward model training can be found in `reward_model.ipynb`. We use [GPT2](https://huggingface.co/openai-community/gpt2) with a text classification head as our reward model.

## Proximal Policy Optimization

Code for PPO training is in `rlhf_script.py` while evaluation is in `eval.ipynb`. Models used are listed below.

1. [GPT2-small](https://huggingface.co/openai-community/gpt2)
2. [GPT2-PPO](https://huggingface.co/smadala2/gpt2_ppo)
3. [Third Party GPT2 RLHF](https://huggingface.co/jtatman/gpt2-open-instruct-v1-Anthropic-hh-rlhf)