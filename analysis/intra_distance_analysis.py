from analysis.utils import midi_to_LeftRightHVO, midi_to_123format, cosine_distance, hamming_distance, convert_hvo_sequence_to_123format

from hvo_sequence.hvo_seq import HVO_Sequence
import matplotlib.pyplot as plt

import os, glob

midi_filename = 'processed_data/InterDrummer_Repetitions/genis/' \
                'drummer7-session3-131_soul_105_beat_4-4_best_2bar_segment_9/repetition_2.mid'

# Plotting Dualization
hvo_seq = midi_to_LeftRightHVO(midi_filename)
hvo_seq.to_html_plot(show_figure=False)

# Convert to 123 format
squeezed_hits = midi_to_123format(midi_filename)
squeezed_hits

# get drummers and filenames
root_data_dir = "processed_data/InterDrummer_Repetitions"
drummers = ["genis", "ignasi", "morgan", "pau"]
original_fname = "original.mid"
repetition_fnames = ["repetition_0.mid", "repetition_1.mid", "repetition_2.mid"]

unique_files_tested_from_gmd = [x.split("/")[-1] for x in glob.glob(os.path.join(root_data_dir, drummers[0], "*"))]

# Load Data
hvo_seq_data = {x: {} for x in unique_files_tested_from_gmd}
hits_data = {x: {} for x in unique_files_tested_from_gmd}
squeezed_123format_data = {x: {} for x in unique_files_tested_from_gmd}
squeezed_velocity_groove_data = {x: {} for x in unique_files_tested_from_gmd}

intra_drummer_cosine_distances = {x: {} for x in unique_files_tested_from_gmd}
intra_drummer_hamming_distances = {x: {} for x in unique_files_tested_from_gmd}

# Load Data
for drummer in drummers:

    for unique_file_tested_from_gmd in unique_files_tested_from_gmd:

        hvo_seqs_repeated = []  # contains the hvo_sequence instances
        hits_repeated = []  # each entry has shape (32, 2)
        squeezed_123_repeated = []  # converted to 1, 2, 3 format
        squeezed_velocity_groove_repeated = []  # tappified left/right (only velocity)

        cosine_distances = []
        hamming_distances = []

        for repetition_fname in repetition_fnames:
            midi_filename = os.path.join(root_data_dir, drummer, unique_file_tested_from_gmd, repetition_fname)
            hvo_seq = midi_to_LeftRightHVO(midi_filename)
            hvo_seqs_repeated.append(hvo_seq)
            hits_repeated.append(hvo_seq.get("h"))
            squeezed_123_repeated.append(convert_hvo_sequence_to_123format(hvo_seq))
            squeezed_velocity_groove_repeated.append(hvo_seq.flatten_voices(voice_idx=0)[:, 2])

        hvo_seq_data[unique_file_tested_from_gmd][drummer] = hvo_seqs_repeated
        hits_data[unique_file_tested_from_gmd][drummer] = hits_repeated
        squeezed_123format_data[unique_file_tested_from_gmd][drummer] = squeezed_123_repeated
        squeezed_velocity_groove_data[unique_file_tested_from_gmd][drummer] = squeezed_velocity_groove_repeated

        # calculate cosine distances
        for idx0 in range(len(squeezed_123_repeated)):
            for idx1 in range(idx0+1, len(squeezed_123_repeated)):
                cosine_distances.append(cosine_distance(squeezed_123_repeated[idx0], squeezed_123_repeated[idx1]))
                hamming_distances.append(hamming_distance(squeezed_velocity_groove_repeated[idx0], squeezed_velocity_groove_repeated[idx1]))

        intra_drummer_cosine_distances[unique_file_tested_from_gmd][drummer] = cosine_distances
        intra_drummer_hamming_distances[unique_file_tested_from_gmd][drummer] = hamming_distances


########################################
########################################
# INTRA DRUMMER CALCULATIONS
########################################
########################################

# Compile all distances for each drummer
cosine_distances_compiled = {drummer: [] for drummer in drummers}
hamming_distances_compiled = {drummer: [] for drummer in drummers}

for drummer in drummers:
    compiled_cosine = []
    compiled_hamming = []
    for track in unique_files_tested_from_gmd:
        compiled_cosine.extend(intra_drummer_cosine_distances[track][drummer])
        compiled_hamming.extend(intra_drummer_hamming_distances[track][drummer])

    cosine_distances_compiled[drummer] = compiled_cosine
    hamming_distances_compiled[drummer] = compiled_hamming


# Plot intra-distances for cosine
data = [cosine_distances_compiled[drummer] for drummer in drummers]
fig1, ax1 = plt.subplots()
ax1.set_title('Intra-drummer cosine distances')
ax1
ax1.boxplot(data)
plt.show()

# Plot intra-distances for hamming
data = [hamming_distances_compiled[drummer] for drummer in drummers]
fig2, ax2 = plt.subplots()
ax2.set_title('Intra-drummer hamming distances')
ax2
ax2.boxplot(data)
plt.show()



########################################
########################################
# Inter DRUMMER CALCULATIONS
########################################
########################################
inter_cosine_distances = {x: [] for x in unique_files_tested_from_gmd}
inter_hamming_distances = {x: [] for x in unique_files_tested_from_gmd}

compiled_123_formats_per_track = {}
compiled_velocity_grooves_per_track = {}
for track in unique_files_tested_from_gmd:
    compiled_123_formats = []
    compiled_velocity_grooves = []
    for drummer in drummers:
        compiled_123_formats.extend(squeezed_123format_data[track][drummer])
        compiled_velocity_grooves.extend(squeezed_velocity_groove_data[track][drummer])

    cosine_distances_inter = []
    hamming_distances_inter = []

    # calculate cosine distances
    for idx0 in range(len(compiled_123_formats)):
        for idx1 in range(idx0 + 1, len(compiled_123_formats)):
            cosine_distances_inter.append(cosine_distance(compiled_123_formats[idx0], compiled_123_formats[idx1]))
            hamming_distances_inter.append(
                hamming_distance(compiled_velocity_grooves[idx0], compiled_velocity_grooves[idx1]))

    inter_cosine_distances[track] = cosine_distances_inter
    inter_hamming_distances[track] = hamming_distances_inter

# Plot inter-distances for cosine
data = [inter_cosine_distances[track] for track in unique_files_tested_from_gmd]
fig3, ax3 = plt.subplots()
ax1.set_title('Inter-drummer cosine distances')
ax3
ax3.boxplot(data)
plt.show()



# Plot inter-distances for hamming
data = [inter_hamming_distances[track] for track in unique_files_tested_from_gmd]
fig3, ax3 = plt.subplots()
ax1.set_title('Inter-drummer hamming distances')
ax3
ax3.boxplot(data)
plt.show()