import numpy as np

from lib.ds.bird_classes import NUM_CLASSES

def _pretty_str_weights(weights: np.ndarray):
    return f'{[round(w, 2) for w in weights]}'


def perform_weighted_voting(
        species_predictions: np.ndarray,
        species_classifier_voting_weights: np.ndarray,
        bird_no_bird_predictions: np.ndarray,
        bird_no_bird_classifier_voting_weights: np.ndarray
):
    n_sequences, sequence_length, n_species_models = species_predictions.shape
    _, _, n_bird_no_bird_models = bird_no_bird_predictions.shape

    voting_results = np.zeros((n_sequences, sequence_length)).astype(int)
    for sequence_nr in range(n_sequences):

        for fragment_nr in range(sequence_length):

            votes = [0.0] * NUM_CLASSES

            for species_model_nr in range(n_species_models):
                votes[species_predictions[sequence_nr, fragment_nr, species_model_nr]] += \
                    species_classifier_voting_weights[species_model_nr]

            for bird_no_bird_model_nr in range(n_bird_no_bird_models):
                model_prediction = \
                    bird_no_bird_predictions[sequence_nr, fragment_nr, bird_no_bird_model_nr]
                model_weight = bird_no_bird_classifier_voting_weights[bird_no_bird_model_nr]
                if model_prediction == 0:
                    votes[0] += model_weight
                else:
                    for i in range(1, len(votes)):
                        votes[i] += model_weight

            voting_results[sequence_nr, fragment_nr] = np.argmax(votes)

    return voting_results
