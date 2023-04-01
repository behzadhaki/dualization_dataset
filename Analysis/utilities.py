from hvo_sequence import HVO_Sequence, midi_to_hvo_sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING, DUALIZATION_ROLAND_HAND_DRUM, ROLAND_REDUCED_MAPPING_HEATMAPS
import os, glob
from eval.GrooveEvaluator import Evaluator
from bokeh.models import Tabs
import numpy as np
import pandas as pd


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