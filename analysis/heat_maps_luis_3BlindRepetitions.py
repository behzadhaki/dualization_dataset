from analysis.utils import midi_to_LeftRightHVO, midi_to_HVO, extract_style_from, get_dualization_heatmap_from_midis
from GrooveEvaluator.feature_extractor import Feature_Extractor_From_HVO_SubSets
from GrooveEvaluator.plotting_utils import velocity_timing_heatmaps_scatter_plotter, global_features_plotter, separate_figues_by_tabs

from bokeh.io import output_file, show, save

import os, glob

import sys
sys.path.insert(1, "../../")
sys.path.insert(1, "../")

############ QUICK GUIDE FOR LOADING DATA
midi_folder = 'processed_data/InterDrummer_Repetitions/luis/' \
                'drummer1-eval_session-4_soul-groove4_80_beat_4-4_best_2bar_segment_6'

# Plotting Dualization
hvo_seq_dualized = midi_to_LeftRightHVO(os.path.join(midi_folder, "repetition_2.mid"))
# hvo_seq.hvo
hvo_seq_dualized.to_html_plot(show_figure=False)

# Plotting Original
hvo_seq_original = midi_to_HVO(os.path.join(midi_folder, "original.mid"))
# hvo_seq.hvo
hvo_seq_original.to_html_plot(show_figure=False)

############ QUICK GUIDE FOR LOADING DATA
# get drummers and filenames
data_folder = "processed_data/InterDrummer_Repetitions/luis"

get_dualization_heatmap_from_midis(data_folder, save_path="heatmaps/LuisSessionAB")









# Master Folder for each of the 72 files
full_paths = glob.glob(os.path.join(root_data_dir, drummers[0], "*"))
unique_files_tested_from_gmd = [x.split("/")[-1] for x in full_paths]

# Dicts to load/organize data
original_patterns_per_style = {}
dualized_patterns_per_style = {}

# Load original and dualized midis and convert to HVO_Sequence
for ix, folder_name in enumerate(unique_files_tested_from_gmd):
    style = extract_style_from(folder_name)

    if style not in original_patterns_per_style.keys():
        original_patterns_per_style[style] = []
        dualized_patterns_per_style[style] = []

    original_patterns_per_style[style].append(midi_to_HVO(os.path.join(full_paths[ix], "original.mid")))
    dualized_patterns_per_style[style].append(midi_to_LeftRightHVO(os.path.join(full_paths[ix], "repetition_0.mid")))
    dualized_patterns_per_style[style].append(midi_to_LeftRightHVO(os.path.join(full_paths[ix], "repetition_1.mid")))
    dualized_patterns_per_style[style].append(midi_to_LeftRightHVO(os.path.join(full_paths[ix], "repetition_2.mid")))


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

regroup_by_drum_voice = False
original_heatmaps_dict, original_scatters_dict = feature_extractors_for_originals.get_velocity_timing_heatmap_dicts(
    s=(4, 10),
    bins=[32*8, 127],
    regroup_by_drum_voice=regroup_by_drum_voice)

dualized_heatmaps_dict, dualized_scatters_dict = feature_extractors_for_dualizeds.get_velocity_timing_heatmap_dicts(
    s=(4, 10),
    bins=[32*8, 127],
    regroup_by_drum_voice=regroup_by_drum_voice)

mixed_heatmaps = {}
mixed_scatters_dict = {}
mixed_heatmaps.update(original_heatmaps_dict)
mixed_scatters_dict.update(original_scatters_dict)
for style in mixed_heatmaps.keys():
    for dualized_voice in dualized_heatmaps_dict[style].keys():
        mixed_heatmaps[style][dualized_voice] = dualized_heatmaps_dict[style][dualized_voice]
        mixed_scatters_dict[style][dualized_voice] = dualized_scatters_dict[style][dualized_voice]

#feature_dicts_grouped = feature_extractors_for_subsets.get_global_features_dicts()


output_file("{}.html".format("temp_heat"))

number_of_loops_per_subset_dict = {tag: len(original_patterns_per_style[tag]) for tag in list(set(original_patterns_per_style.keys()))}

p = velocity_timing_heatmaps_scatter_plotter(
    mixed_heatmaps,
    mixed_scatters_dict,
    number_of_loops_per_subset_dict=number_of_loops_per_subset_dict,
    organized_by_drum_voice=regroup_by_drum_voice,  # denotes that the first key in heatmap and dict corresponds to drum voices
    title_prefix="",
    plot_width=1200, plot_height_per_set=100, legend_fnt_size="8px",
    synchronize_plots=True,
    downsample_heat_maps_by=1
)


# Assign the panels to Tabs
tabs = separate_figues_by_tabs(p, tab_titles=list(mixed_heatmaps.keys()))

show(tabs)