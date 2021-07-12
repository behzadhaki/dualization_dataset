from analysis.utils import midi_to_LeftRightHVO, midi_to_HVO, extract_style_from, get_dualization_heatmap_from_midis, get_inter_drummer_heatmaps
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


## Plot the heatmaps
"""get_dualization_heatmap_from_midis(data_folder, save_path="heatmaps/LuisSessionAB_Separataed_BY_Style", separate_by_style=True)
get_dualization_heatmap_from_midis(data_folder, save_path="heatmaps/LuisSessionAB_ALL_STYLES_MIXED", separate_by_style=False)"""


get_inter_drummer_heatmaps("processed_data/InterDrummer_Repetitions/", ["genis", "ignasi", "luis", "morgan", "pau"],
                           save_path="heatmaps/InterDrummer_Separataed_BY_Style", separate_by_style=True, regroup_by_drum_voice=False)
get_inter_drummer_heatmaps("processed_data/InterDrummer_Repetitions/", ["genis", "ignasi", "luis", "morgan", "pau"],
                           save_path="heatmaps/InterDrummer_ALL_STYLES_MIXED", separate_by_style=False, regroup_by_drum_voice=False)