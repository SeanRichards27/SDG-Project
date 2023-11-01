import argparse
import torch
import os
import pandas as pd
import logging
import time
import matplotlib.pyplot as plt
import json
import io
import sys




os.system('pip install transformers==4.18.0')
os.system('pip install tensorboard')

try:
    from transformers import MobileBertForSequenceClassification, MobileBertTokenizer, AdamW
    print("Transformers module imported successfully!", flush=True)
except ImportError:
    print("Failed to import transformers!", flush=True)
    print(sys.executable, flush=True)
    print(sys.path, flush=True)


#from transformers import MobileBertForSequenceClassification, MobileBertTokenizer, AdamW
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau


import subprocess
def print_installed_version(package_name):
    result = subprocess.run(['pip', 'show', package_name], stdout=subprocess.PIPE)
    lines = result.stdout.decode('utf-8').split('\n')
    for line in lines:
        if line.startswith('Version:'):
            print(f"{package_name} {line}")

print_installed_version('transformers')



def train(args):
    #Set automated log messages to just "errors". 
    logging.getLogger().setLevel(logging.ERROR)
    
    #Initialise tokeniser
    tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
    
    writer = SummaryWriter()
    
    # Load training and validation data from CSV files
    train_data = pd.read_csv(os.path.join(args.train_X, 'X_train.csv'))
    train_labels = pd.read_csv(os.path.join(args.train_y, 'y_train.csv'))
    val_data = pd.read_csv(os.path.join(args.val_X, 'X_val.csv'))
    val_labels = pd.read_csv(os.path.join(args.val_y, 'y_val.csv'))

    # Tokenize & Prepare DataLoader for Training Data
    train_encodings = tokenizer(list(train_data['Text']), truncation=True, padding=True, max_length=512, return_tensors='pt')
    train_labels = torch.tensor(train_labels.values)
    
    logging.basicConfig(level=logging.INFO)
    logging.info("Shape of train_encodings.input_ids: %s", str(train_encodings.input_ids.shape))
    logging.info("Shape of train_encodings.attention_mask: %s", str(train_encodings.attention_mask.shape))
    logging.info("Shape of train_labels: %s", str(train_labels.shape)) 
    
    assert train_encodings.input_ids.shape[0] == train_labels.shape[0], "Mismatched length between input_ids and labels"
    assert train_encodings.attention_mask.shape[0] == train_labels.shape[0], "Mismatched length between attention_mask and labels"
    train_dataset = TensorDataset(train_encodings.input_ids, train_encodings.attention_mask, train_labels)
    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size)
    
    
    # Tokenize & Prepare DataLoader for Validation Data
    val_encodings = tokenizer(list(val_data['Text']), truncation=True, padding=True, max_length=512, return_tensors='pt')
    val_labels = torch.tensor(val_labels.values)
    
    logging.info("Shape of val_encodings.input_ids: %s", str(val_encodings.input_ids.shape))
    logging.info("Shape of val_encodings.attention_mask: %s", str(val_encodings.attention_mask.shape))
    logging.info("Shape of val_labels: %s", str(val_labels.shape))
    
    assert val_encodings.input_ids.shape[0] == val_labels.shape[0], "Mismatched length between input_ids and labels"
    assert val_encodings.attention_mask.shape[0] == val_labels.shape[0], "Mismatched length between attention_mask and labels"
    val_dataset = TensorDataset(val_encodings.input_ids, val_encodings.attention_mask, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    
    # Initialize Model
    model = MobileBertForSequenceClassification.from_pretrained('google/mobilebert-uncased', num_labels=17)
    # Initialize Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    # Loss Function
    loss_fn = BCEWithLogitsLoss()
    
    
    # Initialiee Scheduler
    #scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.patience, factor=args.gamma, verbose=True)

    
    # Initialize list to store average loss per epoch
    epoch_losses = []
    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0  # Initialize epoch loss
        num_batches = 0  # Initialize counter for number of batches

        # Measure time at the beginning of the epoch
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)[0]
            loss = loss_fn(outputs, labels.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # Accumulate batch loss into epoch loss
            if i % args.batch_size == 0:  # Logging steps according to batch size. 
                print(f"Epoch {epoch + 1}, Step {i}, Loss: {loss.item()}")
                
            num_batches += 1  # Increment the counter
        
        # Measure time at the end of the epoch
        end_time = time.time()
        elapsed_time = end_time - start_time

        avg_epoch_loss = epoch_loss / num_batches  # Calculate average epoch loss
        print(f"Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_epoch_loss}, Time taken: {elapsed_time:.2f} seconds")

        # Append the average loss to the list
        epoch_losses.append(avg_epoch_loss)
        
#         # Decay Learning Rate
#         scheduler.step()
        
        # Begin validation loop
        model.eval()  # Set model to evaluation mode

        val_loss = 0.0  # Initialise val loss
        num_val_batches = 0  # Initialise counter for number of val batches

        #Iterate through val data
        for i, batch in enumerate(val_loader):
            input_ids, attention_mask, labels = batch

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)[0]
                loss = loss_fn(outputs, labels.float())

            val_loss += loss.item()  # Accumulate batch loss into validation epoch loss
            num_val_batches += 1  # Increment counter
            
        avg_val_loss = val_loss / num_val_batches  # Calculate average validation loss for the epoch
        print(f"Validation Loss after Epoch {epoch + 1}/{args.epochs}: {avg_val_loss}")
        
        #Decay Learning Rate
        scheduler.step(avg_val_loss)
        
        # Log the average loss value of this epoch to TensorBoard
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)

        #Add validation loss:
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)



    # Save the model
    model_save_path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), model_save_path)
    writer.close()

def model_fn(model_dir):
    """
    Load the model from the `model_dir` directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileBertForSequenceClassification.from_pretrained('google/mobilebert-uncased', num_labels=17)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"), map_location=device))
    return model


def input_fn(request_body, content_type):
    if content_type == "text/csv":
        # Handle CSV input
        data = pd.read_csv(io.StringIO(request_body), header=None)
        return data
    raise ValueError("Unsupported content type: {}".format(content_type))


def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Here we will tokenize the input data for prediction
    tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
    encodings = tokenizer(list(input_data[0]), truncation=True, padding=True, max_length=512, return_tensors='pt')

    with torch.no_grad():
        output = model(encodings.input_ids, attention_mask=encodings.attention_mask)[0]
        
    return output


def output_fn(prediction_output, accept):
    if accept == "application/json":
        # Convert the output to the format you want
        # Here, I'm just converting the tensor to a list
        return json.dumps(prediction_output.tolist()), accept
    raise ValueError("Unsupported accept type: {}".format(accept))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train_X', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_X'))
    parser.add_argument('--train_y', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_Y'))
    parser.add_argument('--val_X', type=str, default=os.environ.get('SM_CHANNEL_VAL_X'))
    parser.add_argument('--val_y', type=str, default=os.environ.get('SM_CHANNEL_VAL_Y'))

    args, _ = parser.parse_known_args()

    train(args)

