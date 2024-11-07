from tqdm import tqdm


def compute_accuracy(labels, outputs):

    # Transform the one-hot vectors (labels and outputs) into integers
    labels = labels.argmax(dim=1)
    outputs = outputs.argmax(dim=1)

    # Compute the accuracy of the current mini-batch
    corrects = (outputs == labels)
    accuracy = corrects.sum().float() / float(labels.size(0))

    return accuracy.item()


def test_model(test_loader, model, loss_function, device):

    print()
    print()

    # Tell to your model that your are evaluating it
    model.eval()

    # Initialize a mini-batch counter
    mini_batch_counter = 0

    # Initialize the loss and accuracy
    running_loss = 0.0
    running_accuracy = 0.0

    # Assign the tqdm iterator to the variable "progress_testing"
    with tqdm(test_loader, unit=" mini-batch") as progress_testing:

        # For each mini-batch defined in the validation loader through the variable "progress_validation"
        for inputs, labels in progress_testing:

            progress_testing.set_description("Testing the training model")

            # Move the x data and y labels into the device chosen for the training
            inputs, labels = inputs.to(device), labels.to(device)

            # Compute the outputs of the network with the x data of the current mini-batch
            outputs = model(inputs)
            # Compute the loss function for each instance in the mini-batch
            loss = loss_function(outputs, labels)
            # Compute the accuracy of the current mini-batch
            accuracy = compute_accuracy(labels, outputs)

            # Update the running loss
            running_loss += loss.item()
            # Update the running accuracy
            running_accuracy += accuracy

            # Display the loss and the accuracy of the current mini-batch
            progress_testing.set_postfix(testing_loss=running_loss / (mini_batch_counter + 1),
                                         testing_accuracy=100. * (running_accuracy / (mini_batch_counter + 1)))

            # Increment the mini-batch counter
            mini_batch_counter += 1

    return running_loss / mini_batch_counter, 100. * (running_accuracy / mini_batch_counter)
