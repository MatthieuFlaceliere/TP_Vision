import numpy as np
import pandas as pd
from tqdm import tqdm

from scores_and_graphs import show_compute_model_performances
from train_process import compute_model_outputs


def compute_accuracy(labels, outputs):

    # Transform the one-hot vectors (labels and outputs) into integers
    labels = labels.argmax(dim=1)
    outputs = outputs.argmax(dim=1)

    # Compute the accuracy of the current mini-batch
    corrects = (outputs == labels)
    accuracy = corrects.sum().float() / float(labels.size(0))

    return accuracy.item()


def test_model(test_loader, model, loss_function, device, classes, txt_file):

    print()
    print()
    txt_file.write('\n\n')

    # Initialize a DataFrame where to store metrics
    test_scores = pd.DataFrame(columns=["Loss", "Accuracy", "Balanced Accuracy", "F1-score", "Kappa",
                                        "Top 2 Accuracy", "Top 3 Accuracy"])

    # Tell to your model that your are evaluating it
    model.eval()

    # Initialize a mini-batch counter
    mini_batch_counter = 0

    # Initialize the loss and accuracy
    running_loss = 0.0
    running_accuracy = 0.0

    # Initialize two variables to store the outputs of the neural network and the labels (for the whole test set)
    all_outputs = []
    all_labels = []

    # Assign the tqdm iterator to the variable "progress_testing"
    with tqdm(test_loader, unit=" mini-batch") as progress_testing:

        # For each mini-batch defined in the validation loader through the variable "progress_validation"
        for inputs, labels in progress_testing:

            progress_testing.set_description("Testing the training model")

            # Compute the outputs of the model with specific inputs
            all_labels, all_outputs, loss, accuracy = compute_model_outputs(inputs, labels, device, model,
                                                                            all_labels, all_outputs, loss_function)

            # Update the running loss
            running_loss += loss.item()
            # Update the running accuracy
            running_accuracy += accuracy

            # Display the updated loss and the accuracy
            progress_testing.set_postfix(testing_loss=running_loss / (mini_batch_counter + 1),
                                         testing_accuracy=100. * (running_accuracy / (mini_batch_counter + 1)))

            # Increment the mini-batch counter
            mini_batch_counter += 1

    # Compute the performances on the test set and store them
    test_scores = show_compute_model_performances(np.array(all_labels), np.array(all_outputs),
                                                  running_loss / mini_batch_counter, test_scores, classes)

    return test_scores