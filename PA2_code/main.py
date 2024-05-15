import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from classifier import FeedForwardClassifier
from torch import nn
from utilities import Utilities
from transformer import Transformer
from LM import LanguageModel

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def getAccuracy(name, tokenizer, language_model):

    inputfile = "speechesdataset/test_LM_" + name + ".txt"

    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtest_Text = f.read()
    
    test_LM_bush__dataset = LanguageModelingDataset(tokenizer, lmtest_Text, block_size)
    test_bushH_loader = DataLoader(test_LM_bush__dataset, batch_size=batch_size,shuffle=True)
    print(name + ": ", compute_perplexity(language_model, test_bushH_loader))

def main():

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    vocab_size = tokenizer.vocab_size
    print("Vocabulary size is", vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

  
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

     # for the classification  task, you will train for a fixed number of epochs like this:
    model = FeedForwardClassifier(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size, n_layer=n_layer, n_head=n_head, device=device, ff_input=n_input, ff_n_hidden=n_hidden, ff_n_output=n_output)
    encoder_model = Transformer(vocab_size, n_embd, block_size, n_layer, n_head, device)
    # utility = Utilities(tokenizer, encoder_model)
    # utility.sanity_check(sentence="This afternoon, I spoke to former President George W. Bush.", block_size=block_size)
    
    m = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
    size = len(train_CLS_loader.dataset)
    num_batches = len(train_CLS_loader)

    # for epoch in range(epochs_CLS):
        
    #     model.train()

    #     train_loss, correct = 0, 0
    #     for xb, yb in train_CLS_loader:
    #         xb, yb = xb.to(device), yb.to(device)
    #         # CLS training code here
            
    #         pred = model(xb)
    #         loss = loss_fn(pred, yb)
    #         train_loss += loss.item()
    #         correct += (pred.argmax(1) == yb).type(torch.float).sum().item()
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     average_train_loss = train_loss / num_batches
    #     accuracy = correct / size
    #     if epoch == 0 or epoch == 14:
    #         test_accuracy =compute_classifier_accuracy(model, test_CLS_loader)
    #         print(f"step {epoch + 1}: train loss {average_train_loss:.4f}, train accuracy {accuracy:.4f}, test accuracy {test_accuracy:.4f}")

    encoder_model = Transformer(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size, n_layer=n_layer, n_head=n_head, device=device, isDecoder = True)
    lm_utility = Utilities(tokenizer, encoder_model)
    lm_utility.sanity_check(sentence="It is costly and politically difficult to continue this conflict.", block_size=block_size)
    
    language_model = LanguageModel(vocab_size, n_embd, block_size,n_layer, n_head, device)
    optimizer = torch.optim.AdamW(language_model.parameters(), lr=learning_rate)

    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break

        xb, yb = xb.to(device), yb.to(device)
        # LM training code here
        loss = language_model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        epoch = i + 1
        if(epoch  % 100 == 0):
            print(print(f"step {epoch}: train perplexity {compute_perplexity(language_model, train_LM_loader):.4f}"))

    getAccuracy("hbush", tokenizer, language_model)
    getAccuracy("obama", tokenizer, language_model)
    getAccuracy("wbush", tokenizer, language_model)



    
if __name__ == "__main__":
    main()
