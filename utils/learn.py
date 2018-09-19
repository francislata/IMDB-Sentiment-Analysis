import torch
import copy
from tqdm import tqdm

def train_model(model, loss, optimizer, train_iter):
    """Trains the model given its loss and optimizer

    Args:
        model: The model to train
        loss: The loss function to use with the model
        optimizer: The optimizer to use when updating the parameters
        train_iter: The iterator to use to go through each batches

    Returns:
        The epoch accuracy and loss
    """
    model.train()

    epoch_accuracy = 0.0
    epoch_losses = 0.0
        
    for batch in tqdm(train_iter, desc="[Train]"):
        inputs, labels = batch.text.t(), batch.label
        model.zero_grad()
        predictions = model(inputs)
        current_loss = loss(predictions, labels)
        current_loss.backward()
        optimizer.step()
        current_accuracy = get_accuracy(predictions, labels)
        epoch_accuracy += current_accuracy.cpu().data.item()
        epoch_losses += current_loss.cpu().data.item()
        
    final_accuracy = epoch_accuracy / len(train_iter)
    final_loss = epoch_losses / len(train_iter)

    print(f"[Train] Accuracy is: {final_accuracy}")
    print(f"[Train] Loss is: {final_loss}")

    return final_accuracy, final_loss

def test_model(model, loss, iter, is_validation_mode=False):
    """Tests the model's performance given a dataset

    Args:
        model: The model to test
        loss: The loss function to optimize
        iter: The iterator containing the dataset to test the model with
        is_validation_mode: A flag indicating whether validation mode is used
    Returns:
        The epoch accuracy and loss
    """
    test_mode_tag = "Validation" if is_validation_mode else "Test"
    epoch_accuracy = 0.0
    epoch_losses = 0.0

    model.eval()

    with torch.no_grad():
        for batch in tqdm(iter, desc=f"{test_mode_tag}"):
            inputs, labels = batch.text.t(), batch.label
            predictions = model(inputs)
            epoch_accuracy += get_accuracy(predictions, labels)
            epoch_losses += loss(predictions, labels)

    final_accuracy = epoch_accuracy / len(iter)
    final_loss = epoch_losses / len(iter)

    print(f"[{test_mode_tag}] Accuracy is: {final_accuracy}")
    print(f"[{test_mode_tag}] Loss is: {final_loss}")

    return final_accuracy, final_loss

def train_and_validate_model(model, loss, optimizer, train_iter, valid_iter, num_epochs=5):
    """Trains and validates the model

    Args:
        model: The model to train and validate
        loss: The loss function to use with the model
        optimizer: The optimizer to use when training the model
        train_iter: The training set iterator
        valid_iter: The validation set iterator
        num_epochs: The number of epochs to run for

    Returns:
        A tuple containing the final model and the list of accuracies and losses for training and validation of the model
    """
    print(f"Training and validation for {num_epochs} epochs...")

    training_accuracies, validation_accuracies = [], []
    training_losses, validation_losses = [], []

    best_model_weights = copy.deepcopy(model.state_dict())
    best_validation_accuracy = 0.0
    
    for epoch in range(num_epochs):
        print(f"-----Epoch {epoch + 1} Start-----")
        train_accuracy, train_loss = train_model(model, loss, optimizer, train_iter)
        training_accuracies.append(train_accuracy)
        training_losses.append(train_loss)

        validation_accuracy, validation_loss = test_model(model, loss, valid_iter, is_validation_mode=True)
        validation_accuracies.append(validation_accuracy)
        validation_losses.append(validation_loss)

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_model_weights = copy.deepcopy(model.state_dict()) 
        print(f"-----Epoch {epoch + 1} Finished!-----\n")

    print("Training and validation completed!\n")

    model.load_state_dict(best_model_weights)

    return model, training_accuracies, training_losses, validation_accuracies, validation_losses

def get_accuracy(predictions, targets):
    """Calculates the accuracy of the model

    Args:
        predictions: A tensor containing the predictions of the model
        targets: A tensor containing the expected values

    Returns:
        The accuracy of the model
    """
    return (torch.sigmoid(predictions).round() == targets).float().mean()
