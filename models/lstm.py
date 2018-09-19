import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleLSTM(nn.Module):
    """This subclass represents a simple implementation of an LSTM recurrent neural network."""

    def __init__(self, num_embeddings, embedding_dim, hidden_size, output_size, embedding_weights=None, bidirectional=False, dropout=0.0, num_layers=1):
        super(SimpleLSTM, self).__init__()
        
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=self.bidirectional, dropout=dropout, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size * (2 if self.bidirectional else 1), output_size)
        self.dropout = nn.Dropout(p=dropout)

        if embedding_weights is not None:
            self.copy_embedding_weights(embedding_weights)

    def forward(self, inputs):
        input_sequences = self.dropout(self.embedding(inputs))
        output, (hidden, _) = self.lstm(input_sequences)

        if self.bidirectional:
            hidden = self.linear(self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1)))

            return hidden.squeeze()
        else:
            output = self.linear(self.dropout(output[:,-1,:]))

            return output.squeeze()

    def copy_embedding_weights(self, embedding_weights):
        """Copies the embedding weights to the Embedding layer.

        Args:
            embedding_weights: The embedding weights to copy over to the embedding layer
        """
        self.embedding.weight.data.copy_(embedding_weights)
        