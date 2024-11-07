import numpy as np
import pandas as pd
from tqdm import tqdm

from scores_and_graphs import model_performances, plot_score_graphs


def compute_accuracy(labels, outputs):

    # Transform the one-hot vectors (labels and outputs) into integers
    labels = labels.argmax(dim=1)
    outputs = outputs.argmax(dim=1)

    # Compute the accuracy of the current mini-batch
    corrects = (outputs == labels)
    accuracy = corrects.sum().float() / float(labels.size(0))

    return accuracy.item()

def compute_model_outputs(inputs, labels, device, model, all_labels, all_outputs, loss_function):

    # Move the x data and y labels into the device chosen for the training
    inputs, labels = inputs.to(device), labels.to(device)

    # Compute the outputs of the network with the x data of the current mini-batch
    outputs = model(inputs)
    # Store the labels and outputs of the current mini-batch
    all_labels.extend(np.array(labels.cpu()))
    all_outputs.extend(np.array(outputs.detach().cpu()))

    # Compute the loss for each instance in the mini-batch
    loss = loss_function(outputs, labels)
    # Compute the accuracy of the current mini-batch
    accuracy = compute_accuracy(labels, outputs)

    return all_labels, all_outputs, loss, accuracy



def train_model(epoch_number, train_loader, validation_loader, model, optimizer, loss_function, device):

    # Initialize a DataFrame where to store metrics
    training_epoch_scores = pd.DataFrame(columns=["Loss", "Accuracy", "Balanced Accuracy", "F1-score", "Kappa",
                                                  "Top 2 Accuracy", "Top 3 Accuracy"])
    validation_epoch_scores = pd.DataFrame(columns=["Loss", "Accuracy", "Balanced Accuracy", "F1-score", "Kappa",
                                                    "Top 2 Accuracy", "Top 3 Accuracy"])

    # Tell to your model that your are training it
    model.train()

    # For each epoch
    for epoch in range(epoch_number):

        # Initialize a mini-batch counter
        mini_batch_counter = 0

        # Initialize the loss and accuracy
        running_loss = 0.0
        running_accuracy = 0.0

        # Initialize two variables to store the outputs of the neural network and the labels (for the whole epoch)
        all_outputs = []
        all_labels = []

        # Assign the tqdm iterator to the variable "progress_epoch"
        with tqdm(train_loader, unit=" mini-batch") as progress_epoch:

            # For each mini-batch defined in the train loader through the variable "progress_epoch"
            for inputs, labels in progress_epoch:

                # Set the description of the progress bar
                progress_epoch.set_description(f"Epoch {epoch + 1}/{epoch_number}")

                # Compute the outputs of the model with specific inputs
                all_labels, all_outputs, loss, accuracy = compute_model_outputs(inputs, labels, device, model,
                                                                                all_labels, all_outputs, loss_function)

                # Update the weights and biais of the network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update the running loss
                running_loss += loss.item()
                # Update the running accuracy
                running_accuracy += accuracy

                # Display the updated loss and the accuracy
                progress_epoch.set_postfix(train_loss=running_loss / (mini_batch_counter + 1),
                                           train_accuracy=100. * (running_accuracy / (mini_batch_counter + 1)))

                # Increment the mini-batch counter
                mini_batch_counter += 1

        # Compute the performances of the training on the current epoch and store the scores
        training_epoch_scores = model_performances(np.array(all_labels), np.array(all_outputs),
                                                   running_loss / mini_batch_counter, training_epoch_scores)

        # Check performance of the model on the validation set after each training epoch
        validation_epoch_scores = validate_model(validation_loader, model, loss_function, device,
                                                 validation_epoch_scores)

    # Plot metrics
    plot_score_graphs(training_epoch_scores, validation_epoch_scores)

    return model

def validate_model(validation_loader, model, loss_function, device, validation_epoch_scores):

    # Tell to your model that your are evaluating it
    model.eval()

    # Initialize a mini-batch counter
    mini_batch_counter = 0

    # Initialize the loss and accuracy
    running_loss = 0.0
    running_accuracy = 0.0

    # Initialize two variables to store the outputs of the neural network and the labels (for the whole validation set
    # at the end of the current epoch)
    all_outputs = []
    all_labels = []

    # Assign the tqdm iterator to the variable "progress_validation"
    with tqdm(validation_loader, unit=" mini-batch") as progress_validation:

        # For each mini-batch defined in the validation loader through the variable "progress_validation"
        for inputs, labels in progress_validation:

            # Set the description of the progress bar
            progress_validation.set_description("               Validation step")

            # Compute the outputs of the model with specific inputs
            all_labels, all_outputs, loss, accuracy = compute_model_outputs(inputs, labels, device, model, all_labels,
                                                                            all_outputs, loss_function)

            # Update the running loss
            running_loss += loss.item()
            # Update the running accuracy
            running_accuracy += accuracy

            # Display the updated loss and the accuracy
            progress_validation.set_postfix(validation_loss=running_loss / (mini_batch_counter + 1),
                                            validation_accuracy=100. * (running_accuracy / (mini_batch_counter + 1)))

            # Increment the mini-batch counter
            mini_batch_counter += 1

    # Compute the performances on the validation set of the current epoch and store the scores
    validation_epoch_scores = model_performances(np.array(all_labels), np.array(all_outputs),
                                                 running_loss / mini_batch_counter, validation_epoch_scores)

    return validation_epoch_scores