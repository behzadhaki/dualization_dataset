from analysis.utils import midi_to_LeftRightHVO, midi_to_123format

from hvo_sequence.hvo_seq import HVO_Sequence

midi_filename = 'processed_data/InterDrummer_Repetitions/genis/' \
                'drummer7-session3-131_soul_105_beat_4-4_best_2bar_segment_9/repetition_2.mid'

# Plotting Dualization
hvo_seq = midi_to_LeftRightHVO(midi_filename)
hvo_seq.to_html_plot(show_figure=False)

# Convert to 123 format
squeezed_hits = midi_to_123format(midi_filename)
squeezed_hits