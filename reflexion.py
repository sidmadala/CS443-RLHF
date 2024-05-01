import sys
import os
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join("~/", 'cache')
import dspy


dspy.settings.configure(lm=turbo, rm=None)

dataset [] #list of input conversation, best output
devset = [x.with_inputs('prompt') for x in dataset]

class BasicPromptWithMemory(dspy.Signature):
    """You are an assistant. Given a prompt, memory of your previous responses to the prompt, and corresponding scores, generate a better response."""

    prompt = dspy.InputField(desc="input to Assistant")
    memory = dspy.InputField(desc="previous responses to the prompt and corresponding scores")
    response = dspy.OutputField(desc="Assistant's response")

class Reflexion(dspy.Module):
    def __init__(self, max_hops=2):
        super().__init__()

        self.generate_response = [dspy.ChainOfThought(BasicPromptWithMemory) for _ in range(max_hops)]
        self.max_hops = max_hops
    
    def forward(self, prompt, memory=[]):
        
        for hop in range(self.max_hops):
            response = self.generate_response[hop](prompt=prompt, memory=memory).response
            memory = deduplicate(memory + [f"Response: {response[0]}, Score: {score}"])

        pred = self.generate_response(prompt=prompt, memory=memory)
        return dspy.Prediction(context=context, answer=pred.answer)
