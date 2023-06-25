# Auto-Regressive Model Exercise

This exercise focuses on training an auto-regressive model on the "Alice in Wonderland" text and performing various tasks using the minGPT repository by Andrej Karpathy.

## Getting Started

1. Clone the minGPT repository from GitHub: [minGPT Repository](https://github.com/karpathy/minGPT)
2. Download the "Alice in Wonderland" text from Kaggle: [Alice in Wonderland Dataset](https://www.kaggle.com/datasets/roblexnana/alice-wonderland-dataset)

## Splitting the Text to Blocks

To train the auto-regressive generator, we need to split the long text into small parts or blocks. Follow these steps:

1. Encode the entire text using the supplied Byte Pair Encoder. Refer to the example in the `bpe.py` file for encoding.
2. Save the tokenized version of the text as your data.
3. Each example will consist of the next `block_size` tokens up to `i`. Meaning, `tokens[i : i + block_size]`.
4. During training, the target variable will be `tokens[i + 1 : i + block_size + 1]`, training the model to predict the next token based on the given prefix.

## Inversion

In this part of the exercise, we aim to invert the auto-regressive model. Given a known sentence `s`, we want to find an input vector `inp_vec` that satisfies `AR(inp_vec) = s`. Follow these steps:

1. Freeze the auto-regressive (AR) model.
2. Initialize a learned input vector.
3. Loop `k` times with the loss `Linversion = Cross_Entropy(AR(inp_vec)[i], s[i])` for all `i` in the range `len(s)`.
4. This optimization process should result in an input vector `inp_vec` that satisfies `AR(inp_vec) = s`.
5. Note that `inp_vec` should replace the token embedding layer in the forward method of the AR model since the token embedding layer expects int type vectors.

## Requirements

1. It is recommended to run the exercise on Google Colab with a GPU to handle the resource requirements.
2. Add print statements during training to monitor the experiments.
3. Set the block size to no more than 64 tokens to avoid resource limitations.

## Task and Results

1. Train the model as a standard GPT-2 model and present the loss term over training time.
2. Perform an inversion process to the sentence "I am a little squirrel holding a walnut". Explain any special choices made during the experiment. Note that reaching the exact sentence is not mandatory, but explain why the model might not be able to do so.
3. Generate a sentence with at least 12 words, with the 11th and 12th words generated by the model. Extract the attention scores SoftMax((Q * K)/sqrt(dim)) for the last transformer block. Average the scores over different attention heads and analyze the tokens' relation to the 11th word. Explain why the model chose to predict the 11th word in its previous prediction step based on this analysis. Calculate the sum of all attention scores.
4. Repeat the previous question, but this time use the attention scores from the first transformer block. Discuss the differences observed in the results.
5. Sample any sentence from the model and compute the log-probability score of the sentence by multiplying the model's probability prediction for each word. Remember to work in log-scale to avoid unstable and extremely low values.
