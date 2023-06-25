# this part was used in order to use the minGPT in google colab's notebook
# import sys
# !git clone https://github.com/karpathy/minGPT.git minGPT
# !pip install /content/minGPT
# sys.path.append('/content/minGPT/mingpt')

import torch
from torch.optim import Adam
from torch.utils.data import Dataset
# using my modified GPT model
from my_model import GPT
from mingpt.bpe import *
from mingpt.trainer import Trainer


# Define a custom PyTorch dataset for the encoded text data
class TextDataset(Dataset):
    """
    Custom PyTorch dataset for the encoded text data.

    Args:
        data (Any): Encoded text data.
        block_size (Any): Block size for splitting the data into chunks.

    """

    def __init__(self, data, block_size):
        self.blocks = []
        for i in range(0, len(data) - block_size, block_size):
            self.blocks.append(data[i:i + block_size])

    def __getitem__(self, idx):
        item = self.blocks[idx]
        x = torch.tensor(item[:-1], dtype=torch.long)
        y = torch.tensor(item[1:], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.blocks)


def init_gpt_model():
    """
    Initialize a standard GPT2 model

    Returns:
        GPT model, model config, and device.
    """
    # initialize a standard GPT2 model
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt2'
    model_config.vocab_size = 50257  # openai's model vocabulary
    model_config.block_size = 64  # openai's model block_size (i.e. input context length)
    model = GPT(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, model_config, device


def encode_text(txt_path):
    """
    Encode the text from a given file using Byte Pair Encoding (BPE).

    Args:
        txt_path (str): Path to the text file.

    Returns:
        Encoded text and BPE encoder.
    """
    # Load the Alice in Wonderland text file and encode it using the Byte Pair Encoding method
    with open(txt_path, encoding="utf-8") as f:
        text = f.read()
    e = get_encoder()
    encoded_text = e.encode(text)
    return encoded_text, e


def batch_end_callback(trainer):
    """
    Callback function called at the end of each training batch.

    Args:
        trainer (Trainer): Trainer instance.
    """
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")


def train_gpt(model, train_dataset):
    """
    Train the GPT model.

    Args:
        model (Any): GPT model instance.
        train_dataset (TextDataset): Training dataset.
    """
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 6e-4  # prev was 5e-4
    train_config.max_iters = 600
    train_config.num_workers = 2
    trainer = Trainer(train_config, model, train_dataset)
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()


def sample_sentence(model):
    """
    Sample a sentence using the GPT model.

    Args:
        model (Any): GPT model instance.

    Returns:
        Generated sentence as returned from the generate method in the GPT class, and it's decoding to str
    """
    model.eval()
    # preparing the prompt
    prompt = "I believe I can"
    prompt = e.encode(prompt)
    prompt = torch.tensor([prompt], dtype=torch.long, device=device)
    # generating 10 tokens sentence, using the prompt
    generated = model.generate(prompt, 10, do_sample=False)
    generated_lst = generated[0].tolist()
    # printing the generated sentence, including the prompt
    output = ""
    for pred in generated_lst:
        token = e.decode([pred])
        output += token
    print("The generated output is:")
    print(output)
    return generated, output


def invert_sentence(model, sentence, num_iterations=1200, learning_rate=0.1):
    """
    Invert a given sentence using the AR model.

    Args:
        model (Any): GPT model instance.
        sentence (Any): Input sentence tensor.
        num_iterations (int): Number of iterations for inversion.
        learning_rate (float): Learning rate for optimization.

    Returns:
        Inverted input vector.
    """
    # AR inversion
    # Freeze the AR model
    for param in model.parameters():
        param.requires_grad = False

    # Initialize the learned input vector
    inp_vec = torch.nn.Parameter(torch.randn(1, len(sentence), model_config.n_embd, requires_grad=True, device="cuda"))

    # Define the optimizer
    optimizer = Adam([inp_vec], lr=learning_rate)
    # Loop for k iterations
    for k in range(num_iterations):
        for i in range(len(sentence)):
            optimizer.zero_grad()
            prefixed_inp_vec = torch.cat((inp_vec, model.transformer.wte(sentence[:i]).unsqueeze(0)), dim=1)
            # Forward pass through the AR model
            outputs, loss = model.q2_forward(prefixed_inp_vec, sentence[i])

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
        if (k + 1) % 100 == 0:
            print(f"k {k + 1}/{num_iterations}, Loss: {loss.item():.4f}")

    # Detach the optimized inp_vec to make it a leaf tensor
    inp_vec = inp_vec.detach()

    return inp_vec


def inversion(model):
    """
    Perform target sentence inversion using the AR model.

    Args:
        model (Any): GPT model instance.
    """

    model.eval()
    # Define the target sentence you want to invert
    target_sentence = "I am a little squirrel holding a walnut"

    # Tokenize the target sentence using the same tokenizer used during training
    tokenized_sentence = e.encode(target_sentence)

    # Convert the tokenized sentence to a PyTorch tensor
    input_sentence = torch.tensor(tokenized_sentence, dtype=torch.long).to(device)  # Add batch dimension

    # Invert the sentence using the AR model
    inverted_input = invert_sentence(model, input_sentence)

    # Generating a sentence using the inverted sentence
    generated = model.q2_generate(inverted_input, max_new_tokens=len(tokenized_sentence), device=device)

    AR_output = ""
    # decoding the generate output
    for i in generated[0].tolist():
        token = e.decode([i])
        AR_output += token

    print(AR_output)



def transformer_block_attn_scores(block_number, AR_generated, attn_lst):
    """
    Calculate and print attention scores achieved in the given transformer block number while predicting
    the 11 token in the AR_generated variable.

    Args:
        block_number (int): Block number (-1 for the last block, 0 for the first block).
        AR_generated (Any): the generated tokens.
        attn_lst (Any): List of attention scores.
    """
    print()
    text_to_print = "last transformer block" if block_number == -1 else "first transformer block"
    print("------- Results using the " + text_to_print + " -------")
    # getting the required transformer block, as well as the relevant attention scores for the 11th word prediction.
    token_12_transformers_blocks_attn = torch.stack(
        attn_lst[1])  # now we get shape of [12, 1, 11, 11], where 12 is the blocks amount
    token_12_required_transformer_block_attn = token_12_transformers_blocks_attn[
        block_number].squeeze()  # now we get the attentions matrix 11x11 of the required transformer block
    relevant_attn_scores_for_token_11 = token_12_required_transformer_block_attn[-1,
                                        :]  # getting the last row of the matrix
    for j in range(11):
        cur_token = AR_generated[j]
        print("The token: " + e.decode([cur_token]),
              " has attention score of: " + str(relevant_attn_scores_for_token_11[j].item()))

    attn_sum = torch.sum(relevant_attn_scores_for_token_11).item()
    print()
    print("The sum of all attention scores is: " + str(attn_sum))



def attention_scores_eval(model):
    """
    Evaluate attention scores for a generated sentence.

    Args:
        model (Any): GPT model instance.
    """
    # questions 3, 4
    model.eval()
    prompt = "At this moment the door of the house opened,"
    prompt = e.encode(prompt)
    prompt = torch.tensor([prompt], dtype=torch.long, device=device)
    output = ""
    # generating 2 tokens
    gen, attn_lst, _ = model.q345_generate(prompt, 2, do_sample=False)

    # decoding the generate output
    AR_generated = gen[0].tolist()
    for pred in AR_generated:
        token = e.decode([pred])
        output += token
    print("The generated output is:")
    print(output)

    # question 3 - last transformer block
    transformer_block_attn_scores(block_number=-1, AR_generated=AR_generated, attn_lst=attn_lst)

    # question 4 - first transformer block
    transformer_block_attn_scores(block_number=0, AR_generated=AR_generated, attn_lst=attn_lst)



def sentence_log_probability(model):
    """
    Calculate the log probability of a generated sentence.

    Args:
        model (Any): GPT model instance.
    """
    # question 5
    model.eval()
    # preparing the prompt
    prompt = "chatGPT is"
    prompt = e.encode(prompt)
    prompt = torch.tensor([prompt], dtype=torch.long, device=device)
    prompt_len = prompt.size()[1]
    # generating 10 tokens sentence, using the prompt
    gen, attn_lst, probs = model.q345_generate(prompt, 10, do_sample=False)
    gen = gen[0].tolist()
    # printing the generated sentece, including the prompt
    output = ""
    for pred in gen:
        token = e.decode([pred])
        output += token
    print("The generated output is:")
    print(output)

    # arranging the probs tensor, so we can access the tokens probabilities
    # now probs is in size of [10, vocab_size], where 10 is the len of the predicted tokens
    probs = torch.stack(probs).squeeze(1)

    log_proba = 0.0
    print("The sentence tokens")
    for i, pred in enumerate(gen[prompt_len:]):
        print(e.decode([pred]))
        # calculating the log probability
        cur_token_probs = probs[i, :]  # getting the probability distribution for the current token
        cur_token_prob = cur_token_probs[pred]  # Probability of the predicted token
        log_proba += torch.log(cur_token_prob).item()

    print("The log probability of the sentence is: " + str(log_proba))
    return log_proba


if __name__ == '__main__':
    model, model_config, device = init_gpt_model()
    encoded_text, e = encode_text("alice_in_wonderland.txt")
    # Create a PyTorch DataLoader to handle batching and shuffling of the data during training
    train_dataset = TextDataset(encoded_text, block_size=model_config.block_size)
    train_gpt(model, train_dataset)
    # getting the results from the generate method in the GPT class (generated_output),
    # and it's decoding to str (generated_str)
    generated_output, generated_str = sample_sentence(model)

    # in order to activate the results for the PDF questions, remove the following comments:
    # question 2 - generating a target sentence, using inversion
    # inversion(model)

    # questions 3, 4 - evaluating attention scores using the last and the first transformer blocks
    # attention_scores_eval(model)

    # question 5 - sampling a sentence and calculating its log probability
    # log_probability = sentence_log_probability(model)
