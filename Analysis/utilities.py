from hvo_sequence import HVO_Sequence, midi_to_hvo_sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING, DUALIZATION_ROLAND_HAND_DRUM, ROLAND_REDUCED_MAPPING_HEATMAPS
import os, glob
from eval.GrooveEvaluator import Evaluator
from bokeh.models import Tabs
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import Panel, Range1d, HoverTool


# search for all midi files in the root path
def get_repetition_files_as_hvo_seqs(root_path, print_missed_files=False, extra_filter=None,  mapper_fn=None):
    mapping = DUALIZATION_ROLAND_HAND_DRUM.copy()

    midi_files = glob.glob(os.path.join(root_path, "**/*.mid"), recursive=True)
    hvo_sequences = []
    for midi_file in midi_files:
        if extra_filter is not None:
            if extra_filter not in midi_file:
                continue
        hvo_seq = None
        try:
            hvo_seq = midi_to_hvo_sequence(midi_file, drum_mapping=mapping, beat_division_factors=[4])
        except:
            if print_missed_files:
                print(f"Error in {midi_file} - Perhaps not a dualization file?")

        if hvo_seq is not None:
            hvo_seq.adjust_length(32)
            try:
                repetition = int(midi_file.split('Participant_')[-1].split("_")[-1].replace(".mid", "")) + 1
            except:
                repetition = midi_file.split('Participant_')[-1].split("_")[-1].replace(".mid", "")

            hvo_seq.metadata.update({
                "master_id": midi_file.split('/')[-2].split(".")[0],
                "style": midi_file.split('/')[-2].split('.')[0].split('_')[1].split('-')[0],
                "performer": int(midi_file.split('Participant_')[-1].split("_")[0]),
                "repetition": repetition,
            })

            if mapper_fn is not None:
                for i, j in zip(range(hvo_seq.number_of_steps), range(hvo_seq.number_of_voices)):
                    hvo_seq.hvo[i, j + hvo_seq.number_of_voices] = mapper_fn(
                        hvo_seq.hvo[i, j + hvo_seq.number_of_voices])
                    print("Done")

            hvo_sequences.append(hvo_seq)


    return hvo_sequences


def get_three_repetitions_sample(test_number, participant_number, repetition_number):
    paths = glob.glob("midi_files/Repetitions/tested_with_four_participants/*", recursive=True)

    if paths is None:
        raise FileNotFoundError("Could not find any files in the folder midi_files/Repetitions/tested_with_four_participants/")

    assert 0 < test_number <= len(paths), f"Test number {test_number} can't be larger than {len(paths)}"
    assert 0 < participant_number <= 4, f"Participant number must be 1, 2, 3 or 4"
    assert 0 <= repetition_number <= 2, f"Repetition number must be 0, 1 or 2"

    master_id = paths[test_number-1]

    mapping_orig = ROLAND_REDUCED_MAPPING_HEATMAPS.copy()
    mapping_rep = DUALIZATION_ROLAND_HAND_DRUM.copy()
    folder_path = paths[test_number-1]

    try:
        midi_file = os.path.join(folder_path, f"Participant_{participant_number}_repetition_{repetition_number}.mid")
        hvo_seq_rep = midi_to_hvo_sequence(
            midi_file,
            drum_mapping=mapping_rep, beat_division_factors=[4])
        hvo_seq_rep.adjust_length(32)

        hvo_seq_rep.metadata.update({
            "master_id": midi_file.split('/')[-2].split(".")[0],
            "performer": int(midi_file.split('Participant_')[-1].split("_")[0]),
            "repetition": repetition_number,
        })

        midi_file = os.path.join(folder_path, "original.mid")
        hvo_seq_orig = midi_to_hvo_sequence(midi_file, drum_mapping=mapping_orig, beat_division_factors=[4])
        hvo_seq_orig.adjust_length(32)
        hvo_seq_orig.metadata.update({
            "master_id": midi_file.split('/')[-2].split(".")[0],
            "repetition": "NA",
        })
        return hvo_seq_rep, hvo_seq_orig
    except:
        raise FileNotFoundError(f"Could not find file {os.path.join(folder_path, f'Participant_{participant_number}_repetition_{repetition_number-1}.mid')}")


def get_all_three_repetitions_for_participant(test_number, participant_number, repetition_number, ):
    paths = glob.glob("midi_files/Repetitions/*/*.mid", recursive=True)
    print(paths)
    if paths is None:
        raise FileNotFoundError("Could not find any files in the folder midi_files/Repetitions/tested_with_four_participants/")

    assert 0 < test_number <= len(paths), f"Test number {test_number} can't be larger than {len(paths)}"
    assert 0 < participant_number <= 4, f"Participant number must be 1, 2, 3 or 4"
    assert 0 <= repetition_number <= 2, f"Repetition number must be 0, 1 or 2"

    master_id = paths[test_number-1]

    mapping_orig = ROLAND_REDUCED_MAPPING_HEATMAPS.copy()
    mapping_rep = DUALIZATION_ROLAND_HAND_DRUM.copy()
    folder_path = paths[test_number-1]

    try:
        midi_file = os.path.join(folder_path, f"Participant_{participant_number}_repetition_{repetition_number}.mid")
        hvo_seq_rep = midi_to_hvo_sequence(
            midi_file,
            drum_mapping=mapping_rep, beat_division_factors=[4])
        hvo_seq_rep.adjust_length(32)

        hvo_seq_rep.metadata.update({
            "master_id": midi_file.split('/')[-2].split(".")[0],
            "performer": int(midi_file.split('Participant_')[-1].split("_")[0]),
            "repetition": repetition_number,
        })

        midi_file = os.path.join(folder_path, "original.mid")
        hvo_seq_orig = midi_to_hvo_sequence(midi_file, drum_mapping=mapping_orig, beat_division_factors=[4])
        hvo_seq_orig.adjust_length(32)
        hvo_seq_orig.metadata.update({
            "master_id": midi_file.split('/')[-2].split(".")[0],
            "repetition": "NA",
        })
        return hvo_seq_rep, hvo_seq_orig
    except:
        raise FileNotFoundError(f"Could not find file {os.path.join(folder_path, f'Participant_{participant_number}_repetition_{repetition_number-1}.mid')}")





def get_original_files_as_hvo_seqs(root_path, print_missed_files=False, extra_filter=None,  mapper_fn=None):
    mapping = ROLAND_REDUCED_MAPPING_HEATMAPS.copy()

    midi_files = glob.glob(os.path.join(root_path, "**/original.mid"), recursive=True)
    hvo_sequences = []
    for midi_file in midi_files:
        if extra_filter is not None:
            if extra_filter not in midi_file:
                continue
        hvo_seq = None
        try:
            hvo_seq = midi_to_hvo_sequence(midi_file, drum_mapping=mapping, beat_division_factors=[4])
        except:
            if print_missed_files:
                print(f"Error in {midi_file} - Perhaps not a dualization file?")

        if hvo_seq is not None:
            hvo_seq.adjust_length(32)
            try:
                repetition = int(midi_file.split('Participant_')[-1].split("_")[-1].replace(".mid", "")) + 1
            except:
                repetition = midi_file.split('Participant_')[-1].split("_")[-1].replace(".mid", "")

            hvo_seq.metadata.update({
                "master_id": midi_file.split('/')[-2].split(".")[0],
                "style": midi_file.split('/')[-2].split('.')[0].split('_')[1].split('-')[0],
                "repetition": "NA",
            })

            if mapper_fn is not None:
                for i, j in zip(range(hvo_seq.number_of_steps), range(hvo_seq.number_of_voices)):
                    hvo_seq.hvo[i, j + hvo_seq.number_of_voices] = mapper_fn(
                        hvo_seq.hvo[i, j + hvo_seq.number_of_voices])

            hvo_sequences.append(hvo_seq)

        for participant_id in [1, 2, 3, 4]:
            hvo_seq.metadata.update({"performer": participant_id})
            hvo_sequences.append(hvo_seq.copy())

    return hvo_sequences


def get_original_drum_heatmaps(root_path, divs_per_16_note_grid=12,
                          separate_by_participant_ids=None, redo_y_labels=False,
                          resize_witdth_ratio=1, resize_height_ratio=1):
    if separate_by_participant_ids is None:
        list_of_filter_dicts_for_subsets = None
    else:
        list_of_filter_dicts_for_subsets = [
            {"performer": [participant_id]} for participant_id in separate_by_participant_ids]

    hvo_sequences = get_original_files_as_hvo_seqs(root_path)

    evaluator_test_set = Evaluator(
        hvo_sequences,
        list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
        _identifier="reps_tested_with_four_participants",
        n_samples_to_use=-1,
        max_hvo_shape=(32, 21),
        need_hit_scores=False,
        need_velocity_distributions=False,
        need_offset_distributions=False,
        need_rhythmic_distances=False,
        need_heatmap=True,
        need_global_features=False,
        need_audio=False,
        need_piano_roll=False,
        n_samples_to_synthesize_and_draw=5,  # "all",
        disable_tqdm=False
    )

    # create dummy predictions
    evaluator_test_set.predictions = evaluator_test_set.get_ground_truth_hvos_array()
    final_tabs = evaluator_test_set.get_velocity_heatmaps(bins=[32 * divs_per_16_note_grid, 64])

    if redo_y_labels:
        for tab in final_tabs.tabs:
            # change y axis tick labels
            if separate_by_participant_ids is None:
                tab._property_values['child'].yaxis[0].major_label_overrides = {loc: f"All Participants" for loc in
                                                           tab._property_values['child'].yaxis[0].major_label_overrides.keys()}
            else:
                tab._property_values['child'].yaxis[0].major_label_overrides = {loc: f"Participant {x}" for x, loc in
                                                           zip(separate_by_participant_ids, tab._property_values['child'].yaxis[0].major_label_overrides.keys())}

    if resize_witdth_ratio != 1 or resize_height_ratio != 1:
        for tab in final_tabs.tabs:
            tab._property_values['child'].width = int(tab._property_values['child'].width * resize_witdth_ratio)
            tab._property_values['child'].height = int(tab._property_values['child'].height * resize_height_ratio)

    return final_tabs
def get_combined_hands(hvo_sequences):
    combined_hands = []
    for hvo_seq in hvo_sequences:
        hvo_seq_left = hvo_seq.copy()
        hvo_seq_right = hvo_seq.copy()
        hvo_seq_left.metadata.update({"hand": "combined"})
        hvo_seq_right.metadata.update({"hand": "combined"})
        hvo_seq_left.hvo = hvo_seq_left.hvo * 0;
        hvo_seq_right.hvo = hvo_seq_right.hvo * 0;
        hvo_seq_left.hvo[:,0] = hvo_seq.hvo[:, 0]
        hvo_seq_left.hvo[:,2] = hvo_seq.hvo[:, 2]
        hvo_seq_left.hvo[:,4] = hvo_seq.hvo[:, 4]
        hvo_seq_right.hvo[:,0] = hvo_seq.hvo[:, 1]
        hvo_seq_right.hvo[:,2] = hvo_seq.hvo[:, 3]
        hvo_seq_right.hvo[:,4] = hvo_seq.hvo[:, 5]
        combined_hands.append(hvo_seq_left)
        combined_hands.append(hvo_seq_right)
    return combined_hands


def getRepetitionHeatmaps(root_path, divs_per_16_note_grid=12,
                          separate_by_participant_ids=None, redo_y_labels=False,
                          resize_witdth_ratio=1, resize_height_ratio=1,  mapper_fn=None):
    """
    :param root_path: path to the root folder of the dataset (where midi files are located)
    :param participant_ids: list of participant ids to be used for the evaluation (if None, all participants are used)
    :return: a bokeh figure with the heatmaps

    """
    fig_originals = get_original_drum_heatmaps(
        root_path=root_path,
        separate_by_participant_ids=separate_by_participant_ids,
        redo_y_labels=redo_y_labels,
        divs_per_16_note_grid=divs_per_16_note_grid,
        resize_witdth_ratio=resize_witdth_ratio,
        resize_height_ratio=resize_height_ratio)

    if separate_by_participant_ids is None:
        list_of_filter_dicts_for_subsets = None
    else:
        list_of_filter_dicts_for_subsets = [
            {"performer": [participant_id]} for participant_id in separate_by_participant_ids]

    hvo_sequences = get_repetition_files_as_hvo_seqs(root_path, mapper_fn=mapper_fn)

    evaluator_test_set = Evaluator(
        hvo_sequences,
        list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
        _identifier="reps_tested_with_four_participants",
        n_samples_to_use=-1,
        max_hvo_shape=(32, 6),
        need_hit_scores=False,
        need_velocity_distributions=False,
        need_offset_distributions=False,
        need_rhythmic_distances=False,
        need_heatmap=True,
        need_global_features=False,
        need_audio=False,
        need_piano_roll=False,
        n_samples_to_synthesize_and_draw=5,  # "all",
        disable_tqdm=False
    )

    # create dummy predictions
    evaluator_test_set.predictions = evaluator_test_set.get_ground_truth_hvos_array()
    fig_left_right = evaluator_test_set.get_velocity_heatmaps(bins=[32 * divs_per_16_note_grid, 64])

    evaluator_test_set = Evaluator(
        get_combined_hands(hvo_sequences),
        list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
        _identifier="reps_tested_with_four_participants",
        n_samples_to_use=-1,
        max_hvo_shape=(32, 6),
        need_hit_scores=False,
        need_velocity_distributions=False,
        need_offset_distributions=False,
        need_rhythmic_distances=False,
        need_heatmap=True,
        need_global_features=False,
        need_audio=False,
        need_piano_roll=False,
        n_samples_to_synthesize_and_draw=5,  # "all",
        disable_tqdm=False
    )

    # create dummy predictions
    evaluator_test_set.predictions = evaluator_test_set.get_ground_truth_hvos_array()

    fig_combined = evaluator_test_set.get_velocity_heatmaps(bins=[32 * 12, 64])
    fig_combined.tabs[0].title = "Hands Overlayed"
    fig_combined.tabs[0]._property_values['child'].title.text = "Hands Overlayed"
    fig_left_right.tabs.append(fig_combined.tabs[0])


    final_tabs = Tabs(tabs=[fig_combined.tabs[0], fig_left_right.tabs[0], fig_left_right.tabs[1]])
    if redo_y_labels:
        for tab in final_tabs.tabs:
            # change y axis tick labels
            if separate_by_participant_ids is None:
                tab._property_values['child'].yaxis[0].major_label_overrides = {loc: f"All Participants" for loc in
                                                           tab._property_values['child'].yaxis[0].major_label_overrides.keys()}
            else:
                tab._property_values['child'].yaxis[0].major_label_overrides = {loc: f"Participant {x}" for x, loc in
                                                           zip(separate_by_participant_ids, tab._property_values['child'].yaxis[0].major_label_overrides.keys())}

    if resize_witdth_ratio != 1 or resize_height_ratio != 1:
        for tab in final_tabs.tabs:
            tab._property_values['child'].width = int(tab._property_values['child'].width * resize_witdth_ratio)
            tab._property_values['child'].height = int(tab._property_values['child'].height * resize_height_ratio)

    show_tabs = [tab for tab in final_tabs.tabs]
    show_tabs.extend([tab for tab in fig_originals.tabs])
    return Tabs(tabs=show_tabs)

def getSimpleORComplexHeatmaps(root_path, need_simple, divs_per_16_note_grid=12,
                          separate_by_participant_ids=None, redo_y_labels=False,
                          resize_witdth_ratio=1, resize_height_ratio=1, mapper_fn=None):
    """
    :param root_path: path to the root folder of the dataset (where midi files are located)
    :param participant_ids: list of participant ids to be used for the evaluation (if None, all participants are used)
    :return: a bokeh figure with the heatmaps

    """
    fig_originals = get_original_drum_heatmaps(
        root_path=root_path,
        separate_by_participant_ids=separate_by_participant_ids,
        redo_y_labels=redo_y_labels,
        divs_per_16_note_grid=divs_per_16_note_grid,
        resize_witdth_ratio=resize_witdth_ratio,
        resize_height_ratio=resize_height_ratio)

    if separate_by_participant_ids is None:
        list_of_filter_dicts_for_subsets = None
    else:
        list_of_filter_dicts_for_subsets = [
            {"performer": [participant_id], "repetition": [rep]} for participant_id in separate_by_participant_ids for rep in ["simple", "complex"]]

    if need_simple:
        extra_filter = "simple"
    else:
        extra_filter = "complex"

    hvo_sequences = get_repetition_files_as_hvo_seqs(root_path, extra_filter=extra_filter, mapper_fn=mapper_fn)

    evaluator_test_set = Evaluator(
        hvo_sequences,
        list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
        _identifier="reps_tested_with_four_participants",
        n_samples_to_use=-1,
        max_hvo_shape=(32, 6),
        need_hit_scores=False,
        need_velocity_distributions=False,
        need_offset_distributions=False,
        need_rhythmic_distances=False,
        need_heatmap=True,
        need_global_features=False,
        need_audio=False,
        need_piano_roll=False,
        n_samples_to_synthesize_and_draw=5,  # "all",
        disable_tqdm=False
    )

    # create dummy predictions
    evaluator_test_set.predictions = evaluator_test_set.get_ground_truth_hvos_array()
    fig_left_right = evaluator_test_set.get_velocity_heatmaps(bins=[32 * divs_per_16_note_grid, 64])

    evaluator_test_set = Evaluator(
        get_combined_hands(hvo_sequences),
        list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
        _identifier="reps_tested_with_four_participants",
        n_samples_to_use=-1,
        max_hvo_shape=(32, 6),
        need_hit_scores=False,
        need_velocity_distributions=False,
        need_offset_distributions=False,
        need_rhythmic_distances=False,
        need_heatmap=True,
        need_global_features=False,
        need_audio=False,
        need_piano_roll=False,
        n_samples_to_synthesize_and_draw=5,  # "all",
        disable_tqdm=False
    )

    # create dummy predictions
    evaluator_test_set.predictions = evaluator_test_set.get_ground_truth_hvos_array()

    fig_combined = evaluator_test_set.get_velocity_heatmaps(bins=[32 * 12, 64])
    fig_combined.tabs[0].title = "Hands Overlayed"
    fig_combined.tabs[0]._property_values['child'].title.text = "Hands Overlayed"
    fig_left_right.tabs.append(fig_combined.tabs[0])


    final_tabs = Tabs(tabs=[fig_combined.tabs[0], fig_left_right.tabs[0], fig_left_right.tabs[1]])
    if redo_y_labels:
        for tab in final_tabs.tabs:
            # change y axis tick labels
            if separate_by_participant_ids is None:
                tab._property_values['child'].yaxis[0].major_label_overrides = {loc: f"All Participants" for loc in
                                                           tab._property_values['child'].yaxis[0].major_label_overrides.keys()}
            else:
                tab._property_values['child'].yaxis[0].major_label_overrides = {loc: f"Participant {x}" for x, loc in
                                                           zip(separate_by_participant_ids, tab._property_values['child'].yaxis[0].major_label_overrides.keys())}

    if resize_witdth_ratio != 1 or resize_height_ratio != 1:
        for tab in final_tabs.tabs:
            tab._property_values['child'].width = int(tab._property_values['child'].width * resize_witdth_ratio)
            tab._property_values['child'].height = int(tab._property_values['child'].height * resize_height_ratio)

    show_tabs = [tab for tab in final_tabs.tabs]
    show_tabs.extend([tab for tab in fig_originals.tabs])
    return Tabs(tabs=show_tabs)


def get_velocities_from_original_midis(root_path, voice_idx=None):
    # Velocity Matching
    original_samples = get_original_files_as_hvo_seqs(
        root_path= root_path)
    original_samples = sorted(original_samples, key=lambda x: x.metadata["master_id"])
    original_samples = original_samples[::5] # original files are repeated 5 times
    vels = np.array([sample.get("v") for sample in original_samples])
    if voice_idx is not None:
        vels = vels[:, :, voice_idx]
    non_zero_vels_gt = np.nonzero(np.array(vels))
    return vels[non_zero_vels_gt]

def get_velocities_from_repetition_midis(root_path, mapper_fn=None):
    # Velocity Matching
    original_samples = get_repetition_files_as_hvo_seqs(
        root_path= root_path, mapper_fn=mapper_fn)
    original_samples = sorted(original_samples, key=lambda x: x.metadata["master_id"])
    original_samples = original_samples[::5]
    vels = np.array([sample.get("v") for sample in original_samples])
    non_zero_vels_gt = np.nonzero(np.array(vels))
    return vels[non_zero_vels_gt]


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid")

def violinplot_with_boxplot(data, title, ax=None):
    if ax is None:
        ax = plt.gca()
    sns.violinplot(data=data, ax=ax)
    #sns.boxplot(data=data, ax=ax, color="white")
    ax.set_title(title)
    ax.set_xlabel("Velocity")
    ax.set_ylabel("Count")
    return ax


def violinplot_grid(data, ncols=2, figsize=(9, 4), fontsize=6):
    nrows = int(np.ceil(len(data) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()
    for i, (title, d) in enumerate(data.items()):
        violinplot_with_boxplot(d, title, ax=axes[i])
        axes[i].tick_params(labelsize=fontsize)
    for i in range(len(data), len(axes)):
        axes[i].axis("off")
    fig.tight_layout()
    return fig, axes


def regroup_repetitions_by_master_id(repetitions_dict_per_participant):
    """
    Regroups the repetitions by the master id of the piece.
    :param repetitions_dict_per_participant: A dictionary of repetitions per participant. dict(list(hvo_seq))

    :return: A dictionary of repetitions per participant, regrouped by the master id of the piece. dict(dict(list(hvo_seq)))
    """
    regrouped_repetitions = dict()
    for participant_id in range(1, len(repetitions_dict_per_participant.keys())+1):
        if f"Participant_{participant_id}" not in regrouped_repetitions:
            regrouped_repetitions[f"Participant_{participant_id}"] = dict()
        for repetition in repetitions_dict_per_participant[f"Participant_{participant_id}"]:
            if repetition.metadata["master_id"] not in regrouped_repetitions[f"Participant_{participant_id}"]:
                regrouped_repetitions[f"Participant_{participant_id}"][repetition.metadata["master_id"]] = list()
            regrouped_repetitions[f"Participant_{participant_id}"][repetition.metadata["master_id"]].append(repetition)
    return regrouped_repetitions


def get_repetitions_dict_separated_by_participant(root_path, regroup_by_master_id=False, participants_id=None):
    repetitions_per_participant = dict()

    for participant_id in participants_id:
        print(f"Participant_{participant_id}")
        repetitions_per_participant[f"Participant_{participant_id}"] = get_repetition_files_as_hvo_seqs(
            root_path=root_path,
            extra_filter="Participant_{}".format(participant_id)
        )

    if regroup_by_master_id:
        return regroup_repetitions_by_master_id(repetitions_per_participant)
    else:
        return repetitions_per_participant


def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+1)
    return dp[m][n]

def jaccard_similarity(a, b):
    set_a = set([i for i, x in enumerate(a) if x == 1])
    set_b = set([i for i, x in enumerate(b) if x == 1])
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union
def extract_intra_participant_distances_and_similarities(regrouped_repetitions, return_separated_by_master_id=True):
    Edit_Distances = {}
    Jacard_Similarities = {}
    for participant_id, repetitions in regrouped_repetitions.items():
        Edit_Distances[participant_id] = {}
        Jacard_Similarities[participant_id] = {}
        for master_id in repetitions.keys():
            if master_id not in Edit_Distances[participant_id]:
                Edit_Distances[participant_id][master_id] = []
                Jacard_Similarities[participant_id][master_id] = []
            samples = [s.flatten_voices(voice_idx=0, reduce_dim=0)[:, 0] for s in repetitions[master_id]]
            for i in range(len(samples)):
                for j in range(i+1, len(samples)):
                    Edit_Distances[participant_id][master_id].append(edit_distance(samples[i], samples[j]))
                    Jacard_Similarities[participant_id][master_id].append(jaccard_similarity(samples[i], samples[j]))
    if return_separated_by_master_id:
        return Edit_Distances, Jacard_Similarities
    else:
        Edit_Distances = {k: [v for master_id in Edit_Distances[k].keys() for v in Edit_Distances[k][master_id]] for k in Edit_Distances.keys()}
        Jacard_Similarities = {k: [v for master_id in Jacard_Similarities[k].keys() for v in Jacard_Similarities[k][master_id]] for k in Jacard_Similarities.keys()}
        return Edit_Distances, Jacard_Similarities

def extract_inter_participant_distances_and_similarities(regrouped_repetitions):
    Edit_Distances = {}
    Jacard_Similarities = {}
    participant_ids = list(regrouped_repetitions.keys())
    master_ids = list(regrouped_repetitions[participant_ids[0]].keys())

    for master_id in master_ids:
        for i in range(len(participant_ids)):
            for j in range(i+1, len(participant_ids)):
                new_key = f"{participant_ids[i]} - {participant_ids[j]}"
                new_key = "P" + new_key.replace("Participant_", " ")
                if new_key not in Edit_Distances:
                    Edit_Distances[new_key] = []
                    Jacard_Similarities[new_key] = []
                samples_i = [s.flatten_voices(voice_idx=0, reduce_dim=0)[:, 0] for s in regrouped_repetitions[participant_ids[i]][master_id]]
                samples_j = [s.flatten_voices(voice_idx=0, reduce_dim=0)[:, 0] for s in regrouped_repetitions[participant_ids[j]][master_id]]
                for sample_i in samples_i:
                    for sample_j in samples_j:
                        Edit_Distances[new_key].append(edit_distance(sample_i, sample_j))
                        Jacard_Similarities[new_key].append(jaccard_similarity(sample_i, sample_j))

    return Edit_Distances, Jacard_Similarities




# side-by-side violin plots for Edit Distances
# imports
import holoviews as hv
hv.extension('bokeh')
from holoviews import opts

def get_violin_plots(data_dict, violin_width=200, violin_height= 60, font_size=6, min_=None, max_=None, single_axis=True):
    # returns a single holoviews plot with violin plots for each participant
    # data_dict is a dictionary of lists {"participant_id": [list of edit distances]}

    # get the violin plots
    if not single_axis:
        violin_plots = []
        for participant_id, distances in data_dict.items():
            violin_plots.append(hv.Violin(distances, label="P"+participant_id.replace("Participant_", " ")))


         # get the violin plots side-by-side
        violin_plots = hv.Layout(violin_plots).cols(len(data_dict.keys()))
    else:
        # use a single plot rather than multiple plots layed out side-by-side
        # PANDAS DATAFRAME
        data = []
        for participant_id, distances in data_dict.items():
            data.extend([[participant_id, d] for d in distances])
        data = pd.DataFrame(data, columns=["Participant", "data"])
        violin_plots = hv.Violin(data, kdims=["Participant"], vdims=["data"])


    # set the plot options
    violin_plots.opts(
        opts.Violin(
            width=violin_width,
            height=violin_height,
            show_legend=False,
            show_grid=True,
            xlabel="",
            ylabel="",
            xrotation=45,
            yrotation=0,
        )
    )

    # set lowest y value to 0
    violin_plots.opts(
        opts.Violin(
            ylim=(min_, max_),
        )
    )

    # change font size for all the labels
    violin_plots.opts(
        opts.Labels(
            text_font_size=f"{font_size}px",
            text_color="black",
            text_align="center",
            text_baseline="middle",
        )
    )
    return violin_plots

def get_heatmap_for_hvo_sequences(hvo_sequences, redo_y_labels, participant_ids):
    figs = []
    list_of_filter_dicts_for_subsets = [
                {"performer": [participant_id]} for participant_id in participant_ids]

    evaluator_test_set = Evaluator(
            get_combined_hands(hvo_sequences),
            list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
            _identifier="reps_tested_with_four_participants",
            n_samples_to_use=-1,
            max_hvo_shape=(32, 6),
            need_hit_scores=False,
            need_velocity_distributions=False,
            need_offset_distributions=False,
            need_rhythmic_distances=False,
            need_heatmap=True,
            need_global_features=False,
            need_audio=False,
            need_piano_roll=False,
            n_samples_to_synthesize_and_draw=5,  # "all",
            disable_tqdm=False
        )

    # create dummy predictions
    evaluator_test_set.predictions = evaluator_test_set.get_ground_truth_hvos_array()# get agreement heatmaps
    fig_combined = evaluator_test_set.get_velocity_heatmaps(bins=[32 * 12, 64])

    fig_combined.tabs[0].title = "Hands Overlayed"
    fig_combined.tabs[0]._property_values['child'].title.text = "Hands Overlayed"
    figs.append(fig_combined.tabs[0])

    evaluator_test_set = Evaluator(
            hvo_sequences,
            list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
            _identifier="reps_tested_with_four_participants",
            n_samples_to_use=-1,
            max_hvo_shape=(32, 6),
            need_hit_scores=False,
            need_velocity_distributions=False,
            need_offset_distributions=False,
            need_rhythmic_distances=False,
            need_heatmap=True,
            need_global_features=False,
            need_audio=False,
            need_piano_roll=False,
            n_samples_to_synthesize_and_draw=5,  # "all",
            disable_tqdm=False
        )

    # create dummy predictions
    evaluator_test_set.predictions = evaluator_test_set.get_ground_truth_hvos_array()# get agreement heatmaps
    fig_combined = evaluator_test_set.get_velocity_heatmaps(bins=[32 * 12, 64])

    figs.append(fig_combined.tabs[0])
    figs.append(fig_combined.tabs[1])

    figs = Tabs(tabs=figs)
    if redo_y_labels:
        for tab in figs.tabs:
            tab._property_values['child'].yaxis[0].major_label_overrides = {loc: f"Participant {x}" for x, loc in
                                                               zip(participant_ids, tab._property_values['child'].yaxis[0].major_label_overrides.keys())}
    return figs


def plot_dice(dice_scores_per_step, title, y_axis_label, width=700, height=200, need_x_labels=True, need_y_labels=True):
    p = figure(width=width, height=height, toolbar_location=None,
           title=title)

    # use colors:
    # '#084594', '#4292c6', ,

    # Histogram with four bars
    edges = np.linspace(0, 32, 33)
    p.quad(top=dice_scores_per_step, bottom=0, left=edges, right=edges+1,
             fill_color='#9ecae1', line_color="white")

    # change x tick labels to 0, 4, 8, 12, 16, 20, 24, 28, 32 at 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5
    p.xaxis.ticker = [0., 4., 8., 12., 16., 20., 24., 28., 32.]
    #p.xaxis.major_label_overrides = {0.24: '0', 4.5: '4', 8.5: '8', 12.5: '12', 16.5: '16', 20.5: '20', 24.5: '24', 28.5: '28', 32.5: '32'}
    # place legend at top left (in a single row)
    # place legend outside of plot
    p.legend.location = "top_center"
    p.legend.orientation = "horizontal"
    p.legend.click_policy="hide"
    p.legend.label_text_font_size = "8pt"
    p.legend.spacing = 8
    p.legend.glyph_height = 15
    p.legend.glyph_width = 5
    p.legend.label_height = 10
    p.legend.label_width = 10
    p.legend.margin = 2
    p.legend.padding =5

    # adjust y range to -0.2, 1
    p.y_range = Range1d(0, 1.2)
    p.yaxis.ticker = [0., 0.2, 0.4, 0.6, 0.8, 1.0]
    p.yaxis.major_label_overrides = {0.0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1.0'}
    p.xaxis.ticker = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5]
    p.xaxis.major_label_overrides = {0.5: '1', 1.5: '', 2.5: '', 3.5: '', 4.5: '5', 5.5: '', 6.5: '', 7.5: '', 8.5: '9', 9.5: '', 10.5: '', 11.5: '', 12.5: '13', 13.5: '', 14.5: '', 15.5: '', 16.5: '17', 17.5: '', 18.5: '', 19.5: '', 20.5: '21', 21.5: '', 22.5: '', 23.5: '', 24.5: '25', 25.5: '', 26.5: '', 27.5: '', 28.5: '29', 29.5: '', 30.5: '', 31.5: '', 32.5: '33'}
    if need_y_labels:
        p.yaxis.axis_label = y_axis_label
    else:
        p.yaxis.axis_label = ""

    if need_x_labels:
        p.xaxis.axis_label = "Grid Line (16th Note)"
    else:
        p.xaxis.axis_label = ""

    # thick grid lines at 0, 4, 8, 12, 16

    p.xgrid.ticker = [0.5, 4.5, 8.5, 12.5, 16.5]
    p.x_range = Range1d(-0.1, 16)
    # horizontal infinite line at min and max
    min_dice = min(dice_scores_per_step)
    max_dice = max(dice_scores_per_step)
    p.line(x=[0, 16], y=[min_dice, min_dice], line_width=1, line_color='grey', line_dash='dashed', legend_label=f'min: {min_dice:.2f}')
    p.line(x=[0, 16], y=[max_dice, max_dice], line_width=1, line_color='black', line_dash='dashed', legend_label=f'max: {max_dice:.2f}')

    # draw vertical lines at 0, 4, 8, 12, 16
    p.line(x=[0.5, 0.5], y=[0, 1.4], line_width=2, line_color='black')
    p.line(x=[4.5, 4.5], y=[0, 1.4], line_width=2, line_color='black')
    p.line(x=[8.5, 8.5], y=[0, 1.4], line_width=2, line_color='black')
    p.line(x=[12.5, 12.5], y=[0, 1.4], line_width=2, line_color='black')

    # single row legend
    p.legend.location = "top_center"
    p.legend.orientation = "horizontal"
    p.legend.click_policy="hide"
    p.legend.label_text_font_size = "8pt"
    p.legend.spacing = 8
    p.legend.glyph_height = 15
    p.legend.glyph_width = 5
    p.legend.label_height = 10
    p.legend.label_width = 10

    # keep legend narrow and flush with plot
    p.legend.margin = 2
    p.legend.padding =5

    # make legend background fully opaque
    p.legend.background_fill_alpha = 1.0


    # title font size
    p.title.text_font_size = '9pt'
    return p


def plot_jaccard(jaccard_scores_per_step, title, y_axis_label, width=700, height=200, need_x_labels=True, need_y_labels=True):
    p = figure(width=width, height=height, toolbar_location=None,
           title=title)

    # use colors:
    # '#084594', '#4292c6', ,

    # Histogram with four bars
    edges = np.linspace(0, 32, 33)
    p.quad(top=jaccard_scores_per_step, bottom=0, left=edges, right=edges+1,
             fill_color='#9ecae1', line_color="white")

    # change x tick labels to 0, 4, 8, 12, 16, 20, 24, 28, 32 at 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5
    p.xaxis.ticker = [0., 4., 8., 12., 16., 20., 24., 28., 32.]
    #p.xaxis.major_label_overrides = {0.24: '0', 4.5: '4', 8.5: '8', 12.5: '12', 16.5: '16', 20.5: '20', 24.5: '24', 28.5: '28', 32.5: '32'}
    # place legend at top left (in a single row)
    # place legend outside of plot
    p.legend.location = "top_center"
    p.legend.orientation = "horizontal"
    p.legend.click_policy="hide"
    p.legend.label_text_font_size = "8pt"
    p.legend.spacing = 8
    p.legend.glyph_height = 15
    p.legend.glyph_width = 5
    p.legend.label_height = 10
    p.legend.label_width = 10
    p.legend.margin = 2
    p.legend.padding =5

    # adjust y range to -0.2, 1
    p.y_range = Range1d(0, 1.2)
    p.yaxis.ticker = [0., 0.2, 0.4, 0.6, 0.8, 1.0]
    p.yaxis.major_label_overrides = {0.0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1.0'}
    p.xaxis.ticker = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5]
    p.xaxis.major_label_overrides = {0.5: '1', 1.5: '', 2.5: '', 3.5: '', 4.5: '5', 5.5: '', 6.5: '', 7.5: '', 8.5: '9', 9.5: '', 10.5: '', 11.5: '', 12.5: '13', 13.5: '', 14.5: '', 15.5: '', 16.5: '17', 17.5: '', 18.5: '', 19.5: '', 20.5: '21', 21.5: '', 22.5: '', 23.5: '', 24.5: '25', 25.5: '', 26.5: '', 27.5: '', 28.5: '29', 29.5: '', 30.5: '', 31.5: '', 32.5: '33'}
    if need_y_labels:
        p.yaxis.axis_label = y_axis_label
    else:
        p.yaxis.axis_label = ""

    if need_x_labels:
        p.xaxis.axis_label = "Grid Line (16th Note)"
    else:
        p.xaxis.axis_label = ""

    # thick grid lines at 0, 4, 8, 12, 16

    p.xgrid.ticker = [0.5, 4.5, 8.5, 12.5, 16.5]
    p.x_range = Range1d(-0.1, 16)
    # horizontal infinite line at min and max
    min_dice = min(jaccard_scores_per_step)
    max_dice = max(jaccard_scores_per_step)
    p.line(x=[0, 16], y=[min_dice, min_dice], line_width=1, line_color='grey', line_dash='dashed', legend_label=f'min: {min_dice:.2f}')
    p.line(x=[0, 16], y=[max_dice, max_dice], line_width=1, line_color='black', line_dash='dashed', legend_label=f'max: {max_dice:.2f}')

    # draw vertical lines at 0, 4, 8, 12, 16
    p.line(x=[0.5, 0.5], y=[0, 1.4], line_width=2, line_color='black')
    p.line(x=[4.5, 4.5], y=[0, 1.4], line_width=2, line_color='black')
    p.line(x=[8.5, 8.5], y=[0, 1.4], line_width=2, line_color='black')
    p.line(x=[12.5, 12.5], y=[0, 1.4], line_width=2, line_color='black')

    # single row legend
    p.legend.location = "top_center"
    p.legend.orientation = "horizontal"
    p.legend.click_policy="hide"
    p.legend.label_text_font_size = "8pt"
    p.legend.spacing = 8
    p.legend.glyph_height = 15
    p.legend.glyph_width = 5
    p.legend.label_height = 10
    p.legend.label_width = 10

    # keep legend narrow and flush with plot
    p.legend.margin = 2
    p.legend.padding =5

    # make legend background fully opaque
    p.legend.background_fill_alpha = 1.0


    # title font size
    p.title.text_font_size = '9pt'
    return p
