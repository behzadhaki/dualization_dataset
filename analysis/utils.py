import numpy as np
import pretty_midi
import note_seq
from hvo_sequence.io_helpers import note_sequence_to_hvo_sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING, DUALIZATION_ROLAND_HAND_DRUM, DUALIZATION_ROLAND_HAND_DRUM_MIXED
import math
import os, glob

from GrooveEvaluator.feature_extractor import Feature_Extractor_From_HVO_SubSets

from hvo_sequence.hvo_seq import HVO_Sequence
import matplotlib.pyplot as plt
from GrooveEvaluator.plotting_utils import velocity_timing_heatmaps_scatter_plotter, global_features_plotter, separate_figues_by_tabs

from bokeh.io import output_file, show, save

def midi_to_HVO(midi_filename, n_steps = 32):
    midi_data = pretty_midi.PrettyMIDI(midi_filename)
    ns = note_seq.midi_io.midi_to_note_sequence(midi_data)
    hvo_seq = note_sequence_to_hvo_sequence(ns, drum_mapping=ROLAND_REDUCED_MAPPING)
    if len(hvo_seq.time_signatures) > 1:
        del (hvo_seq.time_signatures[1:])

    if hvo_seq.hvo.shape[0] < n_steps:
        hvo_seq.hvo = np.pad(hvo_seq.hvo, ((0, n_steps - hvo_seq.hvo.shape[0]), (0, 0)), mode="constant")
    else:
        hvo_seq.hvo = hvo_seq.hvo[:n_steps, :]

    return hvo_seq


def midi_to_LeftRightHVO(midi_filename, n_steps = 32, mix_hands=False):
    "if mix_hands the left and right hand will be mixed together"
    midi_data = pretty_midi.PrettyMIDI(midi_filename)
    ns = note_seq.midi_io.midi_to_note_sequence(midi_data)
    if mix_hands is False:
        hvo_seq = note_sequence_to_hvo_sequence(ns, drum_mapping=DUALIZATION_ROLAND_HAND_DRUM)
    else:
        hvo_seq = note_sequence_to_hvo_sequence(ns, drum_mapping=DUALIZATION_ROLAND_HAND_DRUM_MIXED)

    if len(hvo_seq.time_signatures) > 1:
        del (hvo_seq.time_signatures[1:])

    if hvo_seq.hvo.shape[0]<n_steps:
        hvo_seq.hvo = np.pad(hvo_seq.hvo, ((0, n_steps-hvo_seq.hvo.shape[0]), (0, 0)), mode="constant")
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

def extract_style_from(fname):
    info_list = fname.split("_")
    if info_list[0].split("-")[-1] == "eval":
        style = info_list[2].split("-")[0]
    else:
        style = info_list[1].split("-")[0]
    return style


def get_dualization_heatmap_from_midis(data_folder, save_path="temp", separate_by_style=True, regroup_by_drum_voice=False):
    original_patterns_per_style, original_heatmaps_dict, original_scatters_dict, dualized_heatmaps_dict, dualized_scatters_dict =  get_drummer_heatmap_dicts(
        data_folder, separate_by_style=separate_by_style,regroup_by_drum_voice=regroup_by_drum_voice
    )

    mixed_heatmaps = {}
    mixed_scatters_dict = {}
    mixed_heatmaps.update(original_heatmaps_dict)
    mixed_scatters_dict.update(original_scatters_dict)
    for style in mixed_heatmaps.keys():
        for dualized_voice in dualized_heatmaps_dict[style].keys():
            mixed_heatmaps[style][dualized_voice] = dualized_heatmaps_dict[style][dualized_voice]
            mixed_scatters_dict[style][dualized_voice] = dualized_scatters_dict[style][dualized_voice]

    # feature_dicts_grouped = feature_extractors_for_subsets.get_global_features_dicts()

    output_file("{}.html".format(save_path))

    number_of_loops_per_subset_dict = {tag: len(original_patterns_per_style[tag]) for tag in
                                       list(set(original_patterns_per_style.keys()))}

    p = velocity_timing_heatmaps_scatter_plotter(
        mixed_heatmaps,
        mixed_scatters_dict,
        number_of_loops_per_subset_dict=number_of_loops_per_subset_dict,
        organized_by_drum_voice=regroup_by_drum_voice,
        # denotes that the first key in heatmap and dict corresponds to drum voices
        title_prefix=save_path.split("/")[-1],
        plot_width=1200, plot_height_per_set=100, legend_fnt_size="8px",
        synchronize_plots=True,
        downsample_heat_maps_by=1
    )

    # Assign the panels to Tabs
    tabs = separate_figues_by_tabs(p, tab_titles=list(mixed_heatmaps.keys()))

    show(tabs)

    return tabs

# users = ["genis", ... , "luis"]
def get_inter_drummer_heatmaps(root_folder, users, save_path="temp",
                               separate_by_style=True, regroup_by_drum_voice=False, mix_hands=False):

    mixed_heatmaps = {}
    mixed_scatters_dict = {}

    for user_ix, user in enumerate(users):
        data_folder = os.path.join(root_folder, user)
        if user_ix == 0:
            original_patterns_per_style, original_heatmaps_dict, original_scatters_dict, dualized_heatmaps_dict, dualized_scatters_dict = get_drummer_heatmap_dicts(
                data_folder, separate_by_style=separate_by_style, regroup_by_drum_voice=regroup_by_drum_voice,
                mix_hands = mix_hands
            )
            mixed_heatmaps.update(original_heatmaps_dict)
            mixed_scatters_dict.update(original_scatters_dict)
        else:
            _, original_heatmaps_dict, original_scatters_dict, dualized_heatmaps_dict, dualized_scatters_dict = get_drummer_heatmap_dicts(
                data_folder, separate_by_style=separate_by_style, regroup_by_drum_voice=regroup_by_drum_voice,
                mix_hands=mix_hands
            )

        for style in mixed_heatmaps.keys():
            for dualized_voice in dualized_heatmaps_dict[style].keys():
                tag = "{}_{}".format(dualized_voice, user)
                mixed_heatmaps[style][tag] = dualized_heatmaps_dict[style][dualized_voice]
                mixed_scatters_dict[style][tag] = dualized_scatters_dict[style][dualized_voice]

    output_file("{}.html".format(save_path))

    number_of_loops_per_subset_dict = {tag: len(original_patterns_per_style[tag]) for tag in
                                       list(set(original_patterns_per_style.keys()))}

    p = velocity_timing_heatmaps_scatter_plotter(
        mixed_heatmaps,
        mixed_scatters_dict,
        number_of_loops_per_subset_dict=number_of_loops_per_subset_dict,
        organized_by_drum_voice=regroup_by_drum_voice,
        # denotes that the first key in heatmap and dict corresponds to drum voices
        title_prefix=save_path.split("/")[-1],
        plot_width=1200, plot_height_per_set=100, legend_fnt_size="8px",
        synchronize_plots=True,
        downsample_heat_maps_by=1
    )

    # Assign the panels to Tabs
    tabs = separate_figues_by_tabs(p, tab_titles=list(mixed_heatmaps.keys()))

    show(tabs)

    return tabs

def get_drummer_heatmap_dicts(data_folder, separate_by_style=True, regroup_by_drum_voice=False, mix_hands=False):

    # Master Folder for each of the 72 files
    full_paths = glob.glob(os.path.join(data_folder, "*"))
    unique_files_tested_from_gmd = [x.split("/")[-1] for x in full_paths]

    # Dicts to load/organize data
    original_patterns_per_style = {}
    dualized_patterns_per_style = {}

    # Load original and dualized midis and convert to HVO_Sequence
    for ix, folder_name in enumerate(unique_files_tested_from_gmd):
        style = extract_style_from(folder_name) if separate_by_style else "ALL_STYLES"

        if style not in original_patterns_per_style.keys():
            original_patterns_per_style[style] = []
            dualized_patterns_per_style[style] = []

        original_patterns_per_style[style].append(midi_to_HVO(os.path.join(full_paths[ix], "original.mid")))
        dualized_patterns_per_style[style].append(
            midi_to_LeftRightHVO(os.path.join(full_paths[ix], "repetition_0.mid"), mix_hands=mix_hands))
        dualized_patterns_per_style[style].append(
            midi_to_LeftRightHVO(os.path.join(full_paths[ix], "repetition_1.mid"), mix_hands=mix_hands))
        dualized_patterns_per_style[style].append(
            midi_to_LeftRightHVO(os.path.join(full_paths[ix], "repetition_2.mid"), mix_hands=mix_hands))

    #
    feature_extractors_for_originals = Feature_Extractor_From_HVO_SubSets(
        hvo_subsets=list(original_patterns_per_style.values()),
        tags=list(original_patterns_per_style.keys()),
        auto_extract=False,
        subsets_from_gmd=False
    )
    feature_extractors_for_dualizeds = Feature_Extractor_From_HVO_SubSets(
        hvo_subsets=list(dualized_patterns_per_style.values()),
        tags=list(dualized_patterns_per_style.keys()),
        auto_extract=False,
        subsets_from_gmd=False
    )

    original_heatmaps_dict, original_scatters_dict = feature_extractors_for_originals.get_velocity_timing_heatmap_dicts(
        s=(4, 10),
        bins=[32 * 8, 127],
        regroup_by_drum_voice=regroup_by_drum_voice)

    dualized_heatmaps_dict, dualized_scatters_dict = feature_extractors_for_dualizeds.get_velocity_timing_heatmap_dicts(
        s=(4, 10),
        bins=[32 * 8, 127],
        regroup_by_drum_voice=regroup_by_drum_voice)

    return original_patterns_per_style, original_heatmaps_dict, original_scatters_dict, dualized_heatmaps_dict, dualized_scatters_dict