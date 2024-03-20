# BEGIN - DO NOT CHANGE THESE IMPORTS OR IMPORT ADDITIONAL PACKAGES.
import torch
import torch.nn as nn
# END - DO NOT CHANGE THESE IMPORTS OR IMPORT ADDITIONAL PACKAGES.

class ClassificationModel(torch.nn.Module):
    # Instantiate layers for your model-
    #
    # Your model architecture will be an optionally bidirectional LSTM,
    # followed by a linear + sigmoid layer.
    #
    # You'll need 4 nn.Modules
    # 1. An embeddings layer (see nn.Embedding)
    # 2. A bidirectional LSTM (see nn.LSTM)
    # 3. A Linear layer (see nn.Linear)
    # 4. A sigmoid output (see nn.Sigmoid)
    #
    # HINT: In the forward step, the BATCH_SIZE is the first dimension.
    # HINT: Think about what happens to the linear layer's hidden_dim size
    #       if bidirectional is True or False.
    #
    def __init__(self, vocab_size, embedding_dim, hidden_dim, \
                output_dim, num_layers=1, bidirectional=True):
        ## YOUR CODE STARTS HERE (~4 lines of code) ##
        ## YOUR CODE ENDS HERE ##
        super(ClassificationModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim,num_layers=num_layers, bidirectional=bidirectional)
        # Multiply hidden_dim by 2 for bidirectional LSTM
        self.fc = torch.nn.Linear(hidden_dim*2, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    # Complete the forward pass of the model.
    #
    # Use the last hidden timestep of the LSTM as input
    # to the linear layer. When completing the forward pass,
    # concatenate the last hidden timestep for both the foward,
    # and backward LSTMs.
    #
    # args:
    # x - 2D LongTensor of shape (BATCH_SIZE, max len of all tokenized_word_tensor))
    #     This is the same output that comes out of the collate_fn function you completed-
    def forward(self, x):
        ## YOUR CODE STARTS HERE (~4-5 lines of code) ##
        embedded = self.embedding(x)
        # print(embedded.shape)
        lstm_out, _ = self.lstm(embedded)#x,self.init_hidden(x))
        # Take the output from the final time step
        final_output = lstm_out[:, -1, :]
        fc_output = self.fc(final_output)
        sigmoid_output = self.sigmoid(fc_output)
        return sigmoid_output
        ## YOUR CODE ENDS HERE ##
