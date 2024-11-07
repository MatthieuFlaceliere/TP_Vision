import pandas as pd

def get_model_information(model, txt_file):

    # Get and print information about the parameter number
    print()
    print("Model Summary")
    txt_file.write("Model Summary")
    txt_file.write("\n")

    # Get all parameters of the model layer by layer (trainable or not)
    model_parameters = [layer for layer in model.parameters()]
    # Get the name of each layer
    layer_name = [child for child in model.children()]

    # Define the column name of the DataFrame
    column_name = ["Layer Name", "Number of Trainable Parameters", "Number of (non trainable) Parameters"]
    # Instantiate the table information with a DataFrame
    table_information = pd.DataFrame(columns=column_name)

    # Count the character number for each string element in the list "layer_name"
    character_counts = [len(str(string_element)) for string_element in layer_name]
    # Get the maximum character number
    max_character_number = max(character_counts)

    print("=" * (max_character_number + 2 + 30 + 2 + 36))
    txt_file.write("=" * (max_character_number + 2 + 30 + 2 + 36))
    txt_file.write("\n")

    # Initialize variables
    j = 0
    total_trainable_params = 0
    total_params = 0

    # print("\t" * 10)
    # For each layer
    for i in layer_name:

        # Initialize an empty list
        tmp_list = []

        # Set an exception if "i.biais" does not exist (there is no possible biais in the defined layer)
        try:

            # Get "i.biais" if it exists
            bias = (i.bias is not None)

            # If the defined biais in the layer is set to True
            if bias is True:

                # and if the parameters of the current layer require gradient (if trainable)
                if model_parameters[j].requires_grad is True:

                    # Then get the number of trainable parameters
                    trainable_params = model_parameters[j].numel() + model_parameters[j + 1].numel()
                    # Print information in the console
                    # print(str(i) + "\t" * 3 + str(trainable_params))
                    # Add information in "tmp_list"
                    tmp_list.append(str(i))
                    tmp_list.append(trainable_params)
                    tmp_list.append(0)

                    # Update the total number of trainable parameters
                    total_trainable_params += trainable_params

                else:

                    # Then get the number of parameters (non trainable)
                    params = model_parameters[j].numel() + model_parameters[j + 1].numel()
                    # Print information in the console
                    # print(str(i) + "\t" * 3 + str(params))
                    # Add information in "tmp_list"
                    tmp_list.append(str(i))
                    tmp_list.append(0)
                    tmp_list.append(params)

                    # Update the total number of parameters
                    total_params += params

                # Increment the counter
                j = j + 2

            else:  # if biais is false

                # and if the parameters of the current layer require gradient (if trainable)
                if model_parameters[j].requires_grad is True:

                    # Then get the number of trainable parameters
                    trainable_params = model_parameters[j].numel()
                    # Print information in the console
                    # print(str(i) + "\t" * 3 + str(trainable_params))
                    # Add information in "tmp_list"
                    tmp_list.append(str(i))
                    tmp_list.append(trainable_params)
                    tmp_list.append(0)

                    # Update the total number of trainable parameters
                    total_trainable_params += trainable_params

                else:

                    # Then get the number of parameters (non trainable)
                    params = model_parameters[j].numel()
                    # Print information in the console
                    # print(str(i) + "\t" * 3 + str(params))
                    # Add information in "tmp_list"
                    tmp_list.append(str(i))
                    tmp_list.append(0)
                    tmp_list.append(params)

                    # Update the total number of parameters
                    total_params += params

                # Increment the counter
                j = j + 1

        except:  # If there is no biais

            # Just print the name of the layer
            # print(str(i))
            # Add information in "tmp_list"
            tmp_list.append(str(i))
            tmp_list.append(0)
            tmp_list.append(0)

        # Update the DataFrame
        table_information.loc[len(table_information)] = tmp_list

    # Print the table of information
    print(table_information.to_string(index=False, justify="center"))
    txt_file.write(table_information.to_string(index=False, justify="center"))
    txt_file.write("\n")

    # Print the total number of trainable and non trainable parameters
    print("=" * (max_character_number + 2 + 30 + 2 + 36))
    txt_file.write("=" * (max_character_number + 2 + 30 + 2 + 36))
    txt_file.write("\n")
    print(f"Total")
    txt_file.write(f"Total")
    txt_file.write("\n")
    print(f"    Trainable Parameters: {total_trainable_params}")
    txt_file.write(f"    Trainable Parameters: {total_trainable_params}")
    txt_file.write("\n")
    print(f"    Non Trainable Parameters: {total_params}")
    txt_file.write(f"    Non Trainable Parameters: {total_params}")
    txt_file.write("\n")
    print("=" * (max_character_number + 2 + 30 + 2 + 36))
    txt_file.write("=" * (max_character_number + 2 + 30 + 2 + 36))
    print()
    txt_file.write("\n")