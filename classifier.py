from utils.dataset import *
from utils.learn import *
from utils.plot import *
from models.rnn import *
from models.lstm import *
import torch.optim as optim
import torch.nn as nn

if __name__ == "__main__":
    print("Loading dataset...")
    input_field, output_field = create_fields()
    train_dataset, valid_dataset, test_dataset = load_dataset(input_field, output_field, dataset_path="data")
    build_vocabularies(train_dataset, input_field, output_field, max_size=2500, vectors="glove.6B.100d")
    train_iter, valid_iter, test_iter = create_iterators(train_dataset, valid_dataset, test_dataset)
    print("Dataset loaded!\n")
    
    embedding_weights = input_field.vocab.vectors
    num_embeddings = embedding_weights.size(0)
    embedding_dim = embedding_weights.size(1)

    args = [num_embeddings, embedding_dim, 128, 1]
    kwargs = {"embedding_weights": embedding_weights, "bidirectional": True, "num_layers": 2, "dropout": 0.5}
    simple_lstm = SimpleLSTM(*args, **kwargs)
    simple_lstm.cuda()

    loss = nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.Adam(simple_lstm.parameters())

    print("Training SimpleLSTM...")
    kwargs = {"num_epochs": 10}
    simple_lstm, train_acc, train_loss, valid_acc, valid_loss = train_and_validate_model(simple_lstm, loss, optimizer, train_iter, valid_iter, **kwargs)
    print("Done training SimpleLSTM!\n")

    print("Evaluating SimpleLSTM against test set...")
    test_model(simple_lstm, loss, test_iter)
    print("Evaluation SimpleLSTM done!")
