import os
import warnings
import pandas as pd

import wandb
import torch
import transformers
import bitsandbytes
from huggingface_hub import login
from datasets import Dataset
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from transformers import (AutoModelForCausalLM,
                         AutoTokenizer,
                         BitsAndBytesConfig,
                         TrainingArguments,
                         pipeline,
                         logging,
                         TrainerCallback)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                            confusion_matrix,
                            classification_report)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TOKENIZERS_PARALLELISM'] = "false"
warnings.filterwarnings("ignore")
     

dataset = load_dataset("Amod/mental_health_counseling_conversations")
df = pd.DataFrame(dataset['train'])

print(df.head())
df.shape
df.Context[13], df.Response[13]
def generate_prompt(datapoint):
    return f"""
            The conversation between a Human and an AI Mental Health Chatbot named ThanthweChat.
            [INST] You are a mental health expert. Your task is to generate an appropriate response based on the conversation given in square brackets.
            The tone of your answer should be warm and kind.
            [{datapoint['Context']}][/INST]

            {datapoint['Response']}""".strip()

def generate_test_prompt(datapoint):
    return f"""
            The conversation between a Human and an AI Mental Health Chatbot named ThanthweChat.
            [INST] You are a mental health expert. Your task is to generate an appropriate response based on the conversation given in square brackets.
            The tone of your answer should be warm and kind.
            [{datapoint['Context']}][/INST]""".strip()

X_train, X_eval = train_test_split(df, test_size=0.2, random_state=42)
X_train.shape, X_eval.shape  
X_train = pd.DataFrame(X_train.apply(generate_prompt, axis=1), columns=['text'])
X_eval = pd.DataFrame(X_eval.apply(generate_test_prompt, axis=1),columns=['text'])
X_train.head()
train_data = Dataset.from_pandas(X_train)
eval_data = Dataset.from_pandas(X_eval)
train_data[0]
train_data = Dataset.from_pandas(X_train)
eval_data = Dataset.from_pandas(X_eval)
     

train_data[0]
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model_hub_id = "GRMenon/mental-health-mistral-7b-instructv0.2-finetuned-V2"

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = False,
    bnb_4bit_quant_type = 'nf4',
    bnb_4bit_compute_dtype = compute_dtype
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    quantization_config=bnb_config,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          trust_remote_code=True,
                                          add_eos_token=True,
                                         )
tokenizer.pad_token = tokenizer.eos_token

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
     

import json

# Load the uploaded JSON file
with open('secrets.json') as f:
    secrets = json.load(f)

# Access the Hugging Face API key
hf_key = secrets["huggingface"]
     

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))
            callbacks = [PeftSavingCallback]
     

project = "mental-health-chatbot-finetuned-v2"
base_model_name = "mistral-7b-instruct"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name
     

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    logging_dir = "logs",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_strategy='epoch',
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio = 0.05,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="wandb",
    evaluation_strategy="epoch",
    do_eval=True,
    run_name = run_name,
    push_to_hub = True,
    hub_model_id = model_hub_id,
    hub_token=hf_key,
    hub_strategy="checkpoint",
    disable_tqdm=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    callbacks=callbacks,
    packing=False,
    max_seq_length=512)

model = get_peft_model(model, peft_config)
     

if torch.cuda.is_available():
    torch.cuda.empty_cache()
torch.cuda.empty_cache()
     

import json

# Load the uploaded JSON file
with open('secretsx.json') as f:
    secrets = json.load(f)

# Access the Hugging Face API key
wb_key = secrets["wandb"]
     

wandb.login(key=wb_key)

trainer.train()
trainer.push_to_hub()
wandb.finish()
model.config.use_cache = True
