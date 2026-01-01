# ==========================================
# Project: Brain Tumor Classification with ConvNeXt V2
# Author: [Your Name]
# Description: Fine-tunes Facebook's ConvNeXt V2 model on a custom dataset.
#              Includes automatic data splitting, training, and visual evaluation.
# ==========================================

# --- 1. Setup & Installation ---
# !pip install -q transformers datasets evaluate scikit-learn accelerate -U

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)

# --- Configuration ---
# Path to dataset (Kaggle default)
DATASET_ROOT_DIR = "/kaggle/input" 

# Hyperparameters
MODEL_CHECKPOINT = "facebook/convnextv2-tiny-1k-224" # Tiny variant for efficiency
BATCH_SIZE = 32
EPOCHS = 5 
LEARNING_RATE = 5e-5
SEED = 42

# Ensure Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"System ready. Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# --- 2. Robust Data Loading ---

def create_dataframe(root_dir):
    """
    Recursively scans directory for images and assigns labels based on folder names.
    Target classes: 'brain_glioma', 'brain_menin', 'brain_tumor'
    """
    image_paths = []
    labels = []
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    target_classes = ['brain_glioma', 'brain_menin', 'brain_tumor']
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_exts:
                label = os.path.basename(root)
                if label in target_classes:
                    image_paths.append(os.path.join(root, file))
                    labels.append(label)
    
    if not image_paths:
        raise ValueError("No images found. Check dataset structure.")
        
    return pd.DataFrame({'image_path': image_paths, 'label': labels})

# Load Data
try:
    df = create_dataframe(DATASET_ROOT_DIR)
except ValueError:
    # Handle Kaggle nested folder structure
    subdirs = [f.path for f in os.scandir(DATASET_ROOT_DIR) if f.is_dir()]
    if subdirs:
        df = create_dataframe(subdirs[0])
    else:
        raise

print(f"Dataset Loaded: {len(df)} images.")
labels_list = sorted(df['label'].unique().tolist())
label2id = {label: i for i, label in enumerate(labels_list)}
id2label = {i: label for i, label in enumerate(labels_list)}
print(f"Classes: {labels_list}")

# Stratified Train/Val Split (80/20)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=SEED)

# Convert to HF Datasets
dataset_dict = DatasetDict({
    'train': Dataset.from_pandas(train_df).remove_columns(["__index_level_0__"]),
    'validation': Dataset.from_pandas(val_df).remove_columns(["__index_level_0__"])
})

# --- 3. Preprocessing ---

image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

def transform_images(examples):
    """
    Preprocessing: Reads image bytes, converts to RGB, and applies model transforms.
    """
    images = [Image.open(path).convert("RGB") for path in examples["image_path"]]
    batch = image_processor(images, return_tensors="pt")
    batch["labels"] = [label2id[y] for y in examples["label"]]
    return batch

processed_dataset = dataset_dict.with_transform(transform_images)

# --- 4. Model Initialization ---

model = AutoModelForImageClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(labels_list),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True # Allow replacing the 1000-class head with our 3-class head
)

# --- 5. Training Setup ---

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./convnextv2_brain_tumor_results",
    remove_unused_columns=False,
    eval_strategy="epoch",  
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
    fp16=torch.cuda.is_available(), 
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    processing_class=image_processor, # FIXED: Updated from 'tokenizer' to 'processing_class'
    compute_metrics=compute_metrics,
    data_collator=DefaultDataCollator(),
)

# --- 6. Train & Evaluate ---

print("\nStarting Training...")
trainer.train()
trainer.save_model("./final_model")
print("Training Complete. Model Saved.")

# --- 7. Evaluation Report & Visualization ---

print("\nGenerating Evaluation Report...")
predictions = trainer.predict(processed_dataset["validation"])
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

# Text Classification Report
print(classification_report(y_true, y_pred, target_names=labels_list))

# Visual Sample Report
def visualize_predictions(dataset_split, num_images=6):
    print("\nVisualizing Sample Predictions...")
    indices = random.sample(range(len(dataset_split)), num_images)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for i, idx in enumerate(indices):
        item = dataset_split[idx]
        # Load raw image for display
        image = Image.open(item['image_path']).convert("RGB")
        
        # Inference
        inputs = image_processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        pred_idx = logits.argmax(-1).item()
        pred_label = id2label[pred_idx]
        true_label = item['label']
        
        # Plot
        ax = axes[i]
        ax.imshow(image)
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_predictions(dataset_dict['validation'])
