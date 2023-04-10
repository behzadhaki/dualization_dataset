from API import DualizationDatasetAPI

FullDualizationDataset = DualizationDatasetAPI(midi_folder="midi_files")

FullDualizationDataset.ThreeRepetitionSubset.MultipleParticipantSubset.piano_rolls()