import transformers
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepeval.models import DeepEvalBaseLLM

class CustomLlama3(DeepEvalBaseLLM):
    def __init__(self, params='8B', hftoken=None, quantization='4bit'):

        if quantization == '4bit':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == '8bit':
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
            )
        elif quantization == 'full':
            quantization_config = None

        model_4bit = AutoModelForCausalLM.from_pretrained(
            f"meta-llama/Meta-Llama-3.1-{params}-Instruct",
            device_map="auto",
            quantization_config=quantization_config,
            token=hftoken,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            f"meta-llama/Meta-Llama-3.1-{params}-Instruct",
            token=hftoken,
            trust_remote_code=True
        )

        self.model = model_4bit
        self.tokenizer = tokenizer
        self.params = params

        self.enforced_format = 'Please answer by replying with ONLY the letter of the correct option (A, B, C, or D). Do NOT include explanations. Answer: '

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=False,
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant. You answer multiple choice questions concisely and accurately. Your responses consist of a single letter corresponding to the correct answer option (A, B, C, or D)."},
            {"role": "user", "content": prompt},
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=256,
        )
        
        print(outputs[0]["generated_text"][-1])
        return outputs[0]["generated_text"][-1]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return f"Llama-3.1 {self.params}"