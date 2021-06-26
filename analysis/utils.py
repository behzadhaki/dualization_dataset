import numpy as np
import pretty_midi
import note_seq
from hvo_sequence.io_helpers import note_sequence_to_hvo_sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING, DUALIZATION_ROLAND_HAND_DRUM
import math

def midi_to_LeftRightHVO(midi_filename, n_steps = 32):
    midi_data = pretty_midi.PrettyMIDI(midi_filename)
    ns = note_seq.midi_io.midi_to_note_sequence(midi_data)
    hvo_seq = note_sequence_to_hvo_sequence(ns, drum_mapping=DUALIZATION_ROLAND_HAND_DRUM)
    if len(hvo_seq.time_signatures) > 1:
        del (hvo_seq.time_signatures[1:])

    if hvo_seq.hvo.shape[0]<n_steps:
        hvo_seq.hvo = np.pad(hvo_seq.hvo, ((0, n_steps-hvo_seq.hvo.shape[0]), (0, 0)))
    else:
        hvo_seq.hvo = hvo_seq.hvo[:n_steps, :]

    return hvo_seq


def midi_to_123format(midi_filename):
    """
    similar to convert_hvo_sequence_to_123format except that it is applied directly to a midi file
    """
    hvo_seq = midi_to_LeftRightHVO(midi_filename)
    return convert_hvo_sequence_to_123format(hvo_seq)


def convert_hvo_sequence_to_123format(hvo_seq):
    """
        Converts a left/hand 2d hvo sequence into a single array where
            0 denotes No hit
            1 denotes left hand hit
            2 denotes right hand hit
            3 denotes simultaneous left and right hand hits

        :param hits_array: binary array of shape (n_steps, 2).
            hits_array[:, 0] --> left hand hits
            hits_array[:, 1] --> right hand hits

        :return:
            squeezed_hits: an array of shape (n_steps, 1)
            at each step the returned value will be 0, 1, 2, or 3 as denoted above

        """
    return convert_hits_to_123format(hvo_seq.get("h"))


def convert_hits_to_123format(hits_array):
    """
    Converts a left/hand 2d array of hits (32x2) into a single array where
        0 denotes No hit
        1 denotes left hand hit
        2 denotes right hand hit
        3 denotes simultaneous left and right hand hits

    :param hits_array: binary array of shape (n_steps, 2).
        hits_array[:, 0] --> left hand hits
        hits_array[:, 1] --> right hand hits

    :return:
        squeezed_hits: an array of shape (n_steps, 1)
        at each step the returned value will be 0, 1, 2, or 3 as denoted above

    """
    squeezed_hits = np.array([])
    for step_hit in hits_array:
        if all(step_hit == np.array([0, 0])):
            squeezed_hits = np.append(squeezed_hits, 0)
        if all(step_hit == [1, 0]):
            squeezed_hits = np.append(squeezed_hits, 1)
        if all(step_hit == [0, 1]):
            squeezed_hits = np.append(squeezed_hits, 2)
        if all(step_hit == [1, 1]):
            squeezed_hits = np.append(squeezed_hits, 3)

    return squeezed_hits


def cosine_similarity(a, b):

    a_ = a.flatten()
    b_ = b.flatten()

    return np.dot(a_, b_)/(np.linalg.norm(a_)*np.linalg.norm(b_))


def cosine_distance(a, b):
    return 1-cosine_similarity(a, b)


def hamming_distance(vel_groove_a, vel_groove_b):

    x = (vel_groove_a.flatten() - vel_groove_b.flatten())
    return math.sqrt(np.dot(x, x.T))
