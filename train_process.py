from tqdm import tqdm


def compute_accuracy(labels, outputs):

    # Transform the one-hot vectors (labels and outputs) into integers
    labels = labels.argmax(dim=1)
    outputs = outputs.argmax(dim=1)

    # Compute the accuracy of the current mini-batch
    corrects = (outputs == labels)
    accuracy = corrects.sum().float() / float(labels.size(0))

    return accuracy.item()


# Train the neural network
def train_model(epoch_number, train_loader, validation_loader, model, optimizer, loss_function, device):

    # Tell to your model that your are training it
    model.train()

    # For each epoch
    for epoch in range(epoch_number):

        # Initialize a mini-batch counter
        mini_batch_counter = 0

        # Initialize the loss and accuracy
        running_loss = 0.0
        running_accuracy = 0.0

        # Assign the tqdm iterator to the variable "progress_epoch"
        with tqdm(train_loader, unit=" mini-batch") as progress_epoch:

            # For each mini-batch defined in the train loader through the variable "progress_epoch"
            for inputs, labels in progress_epoch:

                progress_epoch.set_description(f"Epoch {epoch + 1}/{epoch_number}")

                # Move the x data and y labels into the device chosen for the training
                inputs, labels = inputs.to(device), labels.to(device)

                # Compute the outputs of the network with the x data of the current mini-batch
                outputs = model(inputs)
                # Compute the loss function for each instance in the mini-batch
                loss = loss_function(outputs, labels)
                # Compute the accuracy of the current mini-batch
                accuracy = compute_accuracy(labels, outputs)

                # Update the weights and biais of the network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update the running loss
                running_loss += loss.item()
                # Update the running accuracy
                running_accuracy += accuracy

                # Display the loss and the accuracy of the current mini-batch
                progress_epoch.set_postfix(train_loss=running_loss / (mini_batch_counter + 1),
                                           train_accuracy=100. * (running_accuracy / (mini_batch_counter + 1)))

                # Increment the mini-batch counter
                mini_batch_counter += 1

        # Check performance of the model on the validation set after each training epoch
        validation_loss, validation_accuracy = validate_model(validation_loader, model, loss_function, device)

    return model


def validate_model(validation_loader, model, loss_function, device):

    # Tell to your model that your are evaluating it
    model.eval()

    # Initialize a mini-batch counter
    mini_batch_counter = 0

    # Initialize the loss and accuracy
    running_loss = 0.0
    running_accuracy = 0.0

    # Assign the tqdm iterator to the variable "progress_validation"
    with tqdm(validation_loader, unit=" mini-batch") as progress_validation:

        # For each mini-batch defined in the validation loader through the variable "progress_validation"
        for inputs, labels in progress_validation:

            progress_validation.set_description("               Validation step")

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
            progress_validation.set_postfix(validation_loss=running_loss / (mini_batch_counter + 1),
                                            validation_accuracy=100. * (running_accuracy / (mini_batch_counter + 1)))

            # Increment the mini-batch counter
            mini_batch_counter += 1

    return running_loss / mini_batch_counter, 100. * (running_accuracy / mini_batch_counter)
