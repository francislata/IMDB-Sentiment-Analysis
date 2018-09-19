import torchtext.datasets
import torchtext.data
import torch
import spacy

def create_fields():
    """Creates the fields used to split the data when loading the dataset.

    Returns:
        The input and output fields
    """
    input_field = torchtext.data.Field(preprocessing=preprocess, tokenize="spacy")
    output_field = torchtext.data.LabelField(tensor_type=torch.FloatTensor)

    return input_field, output_field

def load_dataset(input_field, output_field, dataset_path=".data"):
    """Loads the IMDB dataset using torchtext.

    Args:
        input_field: The field to use to store the reviews
        output_field: The field to use to store the labels associated with the inputs
        dataset_path: The dataset's location path

    Returns:
        The IMDB training, validation, and test datasets
    """
    train_dataset, test_dataset = torchtext.datasets.IMDB.splits(input_field, output_field, root=dataset_path)
    train_dataset, valid_dataset = train_dataset.split()

    return train_dataset, valid_dataset, test_dataset

def build_vocabularies(training_dataset, input_field, output_field, **input_field_kwargs):
    """Builds the vocabularies for the input and output fields
    
    Args:
        training_dataset: The training dataset used to build the vocabulary
        input_field: The field representing the reviews
        output_field: The field representing the labels
        kwargs: A keyword list of arguments used for building the vocabulary
    """
    input_field.build_vocab(training_dataset, **input_field_kwargs)
    output_field.build_vocab(training_dataset)

def create_iterators(train_dataset, valid_dataset, test_dataset):
    """Creates iterators from the datasets

    Args:
        training_dataset: The training dataset
        valid_dataset: The validation dataset
        test_dataset: The test dataset

    Returns:
        The iterators for all datasets
    """
    train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits((train_dataset, valid_dataset, test_dataset), sort_key=lambda batch: torchtext.datasets.IMDB.sort_key(batch), batch_size=64, repeat=False)

    return train_iter, valid_iter, test_iter

def preprocess(tokens):
    """Custom preprocessor applied to a token

    Args:
        tokens: The list of tokens

    Returns:
        The list of preprocessed tokens
    """
    preprocessed_tokens = []
    for token in tokens:
        preprocessed_token = token.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
        preprocessed_tokens.append(preprocessed_token)

    return preprocessed_tokens
