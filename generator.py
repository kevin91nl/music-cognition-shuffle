import numpy as np
from midiutil.MidiFile import MIDIFile
import random


def generate_sequence(notes, path):
    """
    Create a MIDI file (specified by path) given a list of notes (list of integers, C0=60).

    :param notes: A list of notes (integers, C0=60)
    :param path:  The output path (e.g. output.mid)
    """
    track = 0
    channel = 0
    time = 0        # In beats
    duration = 1    # In beats
    tempo = 200     # In BPM
    volume = 100    # 0-127, as per the MIDI standard

    midi_file = MIDIFile(1)
    midi_file.addTempo(track, time, tempo)

    for pitch in notes:
        midi_file.addNote(track, channel, pitch, time, duration, volume)
        time += 1

    with open(path, "wb") as output_file:
        midi_file.writeFile(output_file)


def make_groupings(sequence, group_size):
    """
    Given a list, make groups of size group_size.

    :param sequence:    A list
    :param group_size:  The length of each group
    :return:            A list of list, such that the concatenation of all lists is the original sequence
    """
    if len(sequence) % group_size > 0:
        raise ValueError(
            'The number of notes (%d) should be divisible by the group size (%d)' % (len(sequence), group_size))
    groups = []
    num_groups = int(len(sequence) / group_size)
    for group_index in range(num_groups):
        start_index = group_index * group_size
        end_index = start_index + group_size
        groups.append(sequence[start_index:end_index])
    return groups


def flatten(groups):
    """
    Flatten a list of lists.

    :param groups:    A list of lists
    :return:          The flattened version of groupings
    """
    output = []
    for group in groups:
        output.extend(group)
    return output


def get_ngrams(sequence, n):
    """
    Given a sequence, get a list of all n-grams.

    :param sequence: The sequence
    :param n:        1 for unigrams, 2 for bigrams, etc.
    :return:         A list of n-grams
    """
    ngrams = []
    for index in range(len(sequence) - n + 1):
        ngrams.append(sequence[index:index + n])
    return ngrams


def get_counts(sequence):
    """
    Count the number of occurrences in a sequence

    :param sequence:  The sequence
    :return:          A mapping from index -> counts and a mapping from index -> element
    """
    counts = {}
    indices = []
    index = 0
    for item in sequence:
        if item not in indices:
            indices.append(item)
            counts[index] = 1
            index += 1
        else:
            counts[indices.index(item)] += 1
    return counts, indices


def generate_melody(alphabet, length, unigram_counts, unigram_indices, bigram_counts, bigram_indices, trigram_counts, trigram_indices, alpha=1):
    """
    Generate a melody given unigram and bigram counts.

    :param alphabet:         A list of all possible notes
    :param length:           The length of the new melody
    :param unigram_counts:   The count of the unigrams
    :param unigram_indices:  The indices of the unigrams
    :param bigram_counts:    The count of the bigrams
    :param bigram_indices:   The indices of the bigrams
    :param trigram_counts:   The count of the trigrams
    :param trigram_indices:  The indices of the trigrams
    :param alpha:            The alpha value for the Dirichlet prior
    :return:                 A melody of the specified length
    """
    new_melody = [-1, -1]

    for _ in range(length):
        prev_element_1 = new_melody[-1]
        prev_element_2 = new_melody[-2]

        probs = []
        for element in alphabet:
            new_state = [prev_element_2, prev_element_1, element]
            score = 0

            # Calculate score based on trigrams
            numerator = 1
            if new_state in trigram_indices:
                numerator = trigram_counts[trigram_indices.index(new_state)] + alpha
            denominator = alpha * len(trigram_indices) + sum(trigram_counts.values())
            score += np.log(numerator / denominator)

            # Calculate score based on bigrams
            numerator = 1
            if new_state in bigram_indices:
                numerator = bigram_counts[bigram_indices.index(new_state)] + alpha
            denominator = alpha * len(bigram_indices) + sum(bigram_counts.values())
            score += np.log(numerator / denominator)

            # Calculate score based on unigrams
            numerator = 1
            if element in unigram_indices:
                numerator += unigram_counts[unigram_indices.index(element)] + alpha
            denominator = alpha * len(unigram_indices) + sum(unigram_counts.values())
            score += np.log(numerator / denominator)

            # Add the score to the list of scores
            probs.append(np.exp(score))

        # Choose an element according to the probability distribution
        probs = probs / sum(probs)
        element_index = np.where(np.random.multinomial(1, probs))[0][0]
        element = alphabet[element_index]
        new_melody.append(element)

    # Remove the first dummy note
    return new_melody[2:]


def shuffle_elements(groups):
    """
    Shuffle the elements in each group given a list of groups.

    :param groups: A list of groups
    :return:       The same list of groups in which the elements of each group are shuffled
    """
    new_groups = []
    for group in groups:
        # Copy & Shuffle!
        new_group = [element for element in group]
        random.shuffle(new_group)
        new_groups.append(new_group)
    return new_groups


if __name__ == '__main__':
    for seed in range(10):
        # Fix the seeds
        random.seed(seed)
        np.random.seed(seed)

        # Two well-known melodies
        jacques = [60, 62, 64, 60, 60, 62, 64, 60, 64, 65, 67, 64, 65, 67, 67, 69, 67, 65, 64, 60, 67, 69, 67, 65]
        farm = [65, 65, 65, 60, 62, 62, 60, 69, 69, 67, 67, 65, 60, 65, 65, 65, 60, 62, 62, 60, 69, 69, 67, 67]

        # Generate a new melody based on the unigram counts and bigram counts
        unigram_counts, unigram_indices = get_counts(get_ngrams(jacques + farm, 1))
        bigram_counts, bigram_indices = get_counts(get_ngrams( [-1] + jacques + [-1] + farm, 2))
        trigram_counts, trigram_indices = get_counts(get_ngrams([-1, -1] + jacques + [-1, -1] + farm, 3))
        random_melody = generate_melody(list(set(jacques + farm)), 24, unigram_counts, unigram_indices, bigram_counts,
                                        bigram_indices, trigram_counts, trigram_indices)

        # Store the original melodies
        generate_sequence(jacques, 'data/jacques.mid')
        generate_sequence(farm, 'data/farm.mid')
        generate_sequence(random_melody, 'data/random.mid')

        # Loop through all melodies and apply the shuffling operator
        melodies = [('jacques', jacques), ('farm', farm), ('random', random_melody)]
        for melody_name, melody in melodies:
            for group_size in [1, 2, 3, 4, 6, 8, 12, 24]:
                groups = make_groupings(melody, group_size)
                groups = shuffle_elements(groups)
                sequence = flatten(groups)
                generate_sequence(sequence, 'data/shuffle_elements_%s_%d_%d.mid' % (melody_name, group_size, seed))
