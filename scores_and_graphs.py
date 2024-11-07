import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, top_k_accuracy_score, \
    confusion_matrix, classification_report

import plotly.express as px


def vec_to_int(y_true, y_predicted):

    y_true = np.argmax(y_true, axis=1)
    y_predicted = np.argmax(y_predicted, axis=1)

    return y_true, y_predicted

def model_performances(y_true, y_predicted, loss, my_score_df):

    # Initialize a list
    scores = []

    # Get the index of the maximum value in the vectors "y_true" and "y_predicted"
    y_int_true, y_int_predicted = vec_to_int(y_true, y_predicted)

    # Compute scores and add them to the list scores
    scores.extend([loss])
    scores.extend([accuracy_score(y_int_true, y_int_predicted)])
    scores.extend([balanced_accuracy_score(y_int_true, y_int_predicted)])
    scores.extend([f1_score(y_int_true, y_int_predicted, average="micro")])
    scores.extend([cohen_kappa_score(y_int_true, y_int_predicted)])
    scores.extend([top_k_accuracy_score(y_int_true, y_predicted, k=2)])
    scores.extend([top_k_accuracy_score(y_int_true, y_predicted, k=3)])

    my_score_df.loc[len(my_score_df)] = scores

    return my_score_df

def show_compute_model_performances(y_true, y_predicted, loss, my_score_df, classes):

    # Initialize a list
    scores = []

    # Get the index of the maximum value in the vectors "y_true" and "y_predicted"
    y_int_true, y_int_predicted = vec_to_int(y_true, y_predicted)

    # Compute scores, add them to the list scores and show them
    scores.extend([loss])
    # Accuracy
    accuracy = accuracy_score(y_int_true, y_int_predicted)
    print("Accuracy: " + str(accuracy))
    scores.extend([accuracy])
    # Balanced accuracy
    balanced_accuracy = balanced_accuracy_score(y_int_true, y_int_predicted)
    print("Balanced Accuracy: " + str(balanced_accuracy))
    scores.extend([balanced_accuracy])
    # F1-score
    f1 = f1_score(y_int_true, y_int_predicted, average="micro")
    print("F1-score: " + str(f1))
    scores.extend([f1])
    # Cohen Kappa
    kappa = cohen_kappa_score(y_int_true, y_int_predicted)
    print("Kappa: " + str(kappa))
    scores.extend([kappa])
    # Confusion matrix
    if y_true.shape[1] <= 10:
        print(confusion_matrix(y_int_true, y_int_predicted))
    # Classification report
    print(classification_report(y_int_true, y_int_predicted, target_names=classes))
    # Top 2 accuracy
    top_2 = top_k_accuracy_score(y_int_true, y_predicted, k=2)
    print("Top 2 Accuracy: " + str(top_2))
    scores.extend([top_2])
    # Top 3 accuracy
    top_3 = top_k_accuracy_score(y_int_true, y_predicted, k=3)
    print("Top 3 Accuracy: " + str(top_3))
    scores.extend([top_3])

    my_score_df.loc[len(my_score_df)] = scores

    return my_score_df

def create_score_df(training_epoch_scores, validation_epoch_scores, score_type):

    # Create a DataFrame for plotting the train "score_type"
    train_df = pd.DataFrame(columns=["Epochs", "Stage", score_type])
    # Create the vectors of values for the epochs and the stage of the training process
    epochs = np.arange(1, training_epoch_scores.shape[0] + 1, 1)
    stage = ["Train"] * training_epoch_scores.shape[0]
    # Fill the DataFrame for the train "score_type"
    train_df["Epochs"] = epochs
    train_df["Stage"] = stage
    train_df[score_type] = training_epoch_scores[score_type]

    # Create a DataFrame for plotting the validation "score_type"
    validation_df = pd.DataFrame(columns=["Epochs", "Stage", score_type])
    # Create the vector of values for the stage of the training process
    stage = ["Validation"] * training_epoch_scores.shape[0]
    # Fill the DataFrame for the validation "score_type"
    validation_df["Epochs"] = epochs
    validation_df["Stage"] = stage
    validation_df[score_type] = validation_epoch_scores[score_type]

    # Merge the two DataFrame
    score_df = pd.concat([train_df, validation_df])

    return score_df

def plot_score_graphs(training_epoch_scores, validation_epoch_scores):

    # List of scores to plot
    scores_to_plot = ["Loss", "Accuracy", "Balanced Accuracy", "F1-score", "Kappa", "Top 2 Accuracy", "Top 3 Accuracy"]

    # For each score to plot in the defined list
    for score_type in scores_to_plot:

        # Create the DataFrame to be used with Plotly
        the_df = create_score_df(training_epoch_scores, validation_epoch_scores, score_type)

        # Plot the score evolution for the training and validation stages
        fig = px.line(the_df, x="Epochs", y=score_type, color="Stage")
        # Show the graph
        fig.show()