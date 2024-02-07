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

