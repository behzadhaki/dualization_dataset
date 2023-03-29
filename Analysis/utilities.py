from hvo_sequence import HVO_Sequence, midi_to_hvo_sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING, DUALIZATION_ROLAND_HAND_DRUM, ROLAND_REDUCED_MAPPING_HEATMAPS
import os, glob
from eval.GrooveEvaluator import Evaluator
from bokeh.models import Tabs


# search for all midi files in the root path
def get_repetition_files_as_hvo_seqs(root_path, print_missed_files=False, extra_filter=None):
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
            hvo_sequences.append(hvo_seq)
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

    return hvo_sequences

def get_original_files_as_hvo_seqs(root_path, print_missed_files=False, extra_filter=None):
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
                          resize_witdth_ratio=1, resize_height_ratio=1):
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

    hvo_sequences = get_repetition_files_as_hvo_seqs(root_path)

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
                          resize_witdth_ratio=1, resize_height_ratio=1):
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

    hvo_sequences = get_repetition_files_as_hvo_seqs(root_path, extra_filter=extra_filter)

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
