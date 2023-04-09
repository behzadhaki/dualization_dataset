from hvo_sequence import HVO_Sequence, midi_to_hvo_sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING, DUALIZATION_ROLAND_HAND_DRUM, \
    ROLAND_REDUCED_MAPPING_HEATMAPS
import os
import glob
import copy
from eval.GrooveEvaluator import Evaluator
from bokeh.models import Tabs
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import Panel, Range1d, HoverTool

import typing


# ---------------------------------------------------------------------------
# -----------                Collect Data                     ---------------
# ---------------------------------------------------------------------------
#
# def collect_data(midi_folder, filename=None):
#     """
#     Collects all data into a dataframe and saves it to a csv file if filename is not None
#
#
#     The collected data will have the following columns:
#     - Test Number (int) - 000 to 345
#     - Test Type (str) - Simple Complex OR Three Random Repetitions
#     - Multiple Participants (bool) - True if more than one participant played the test
#     - Participant 1 (bool) - True if participant 1 played the test
#     - Participant 2 (bool) - True if participant 2 played the test
#     - Participant 3 (bool) - True if participant 3 played the test
#     - Participant 4 (bool) - True if participant 4 played the test
#     - Style (str) - Style of the test
#     - Tempo (float) - Tempo of the test
#     - GMD Drummer (str) - The Groove Midi Datasert Drummer (not to be confused with the participant)
#     - GMD Performance Session (str) - Performance Session in the Groove Midi Dataset
#     - GMD Segment Type (str) - Segment Type in the Groove Midi Dataset (e.g. Beat, Fill) -> so far only Beat
#     - GMD Segment Meter (str) - Segment Meter in the Groove Midi Dataset (e.g. 4/4) -> so far only 4/4
#
#     @param midi_folder:  folder containing all midi files (can be nested)
#     @param filename:  filename of the csv file to save the data to
#     @return: dataframe containing all data
#     """
#
#     # Search for all original.mid files in all subfolders
#     file_list = glob.glob(f'{midi_folder}/**/original.mid', recursive=True)
#     test_folders = sorted([os.path.dirname(f) for f in file_list], key=lambda x: x.split(os.sep)[-1])
#
#     csv_data = pd.DataFrame()
#
#     for current_test_folder in test_folders:
#         # row data
#         row_data = {
#             "Test Number": None,
#             "Test Type": None,
#             "Multiple Participants": None,
#             "Participant 1": None,
#             "Participant 2": None,
#             "Participant 3": None,
#             "Participant 4": None,
#             "Style": None,
#             "Tempo": None,
#             "GMD Drummer": None,
#             "GMD Performance Session": None,
#             "GMD Segment Type": None,
#             "GMD Segment Meter": None,
#             "Selected 2Bars From Start": None,
#             "Folder Path": current_test_folder,
#         }
#
#         # Test Number
#         TestNumber = current_test_folder.split(os.sep)[-1].split(" ")[0].split("[")[-1]
#         row_data["Test Number"] = TestNumber
#
#         # type
#         type = current_test_folder.split(os.sep)[1].replace(
#             "Repetitions", "Three Random Repetitions").replace(
#             "SimpleComplex", "Simple Complex")
#         row_data["Test Type"] = type
#
#         # check if multiple participants
#         folder = current_test_folder.split(os.sep)[-1]
#         Plist = folder.split(" ")[1].split("]")[0]
#         isMultipleParticipants = len(Plist) > 2
#         row_data["Multiple Participants"] = isMultipleParticipants
#
#         # Get all repetitions
#         row_data["Participant 1"] = True if "P1" in Plist else False
#         row_data["Participant 2"] = True if "P2" in Plist else False
#         row_data["Participant 3"] = True if "P3" in Plist else False
#         row_data["Participant 4"] = True if "P4" in Plist else False
#         GMD_String = folder.split(" ")[-1]
#
#         # GMD drummer
#         GMD_drummer = GMD_String.split("-")[0]
#         row_data["GMD Drummer"] = GMD_drummer
#
#         # GMD Performance Session
#         GMD_String = "-".join(GMD_String.split("-")[1:])
#         row_data["GMD Performance Session"] = GMD_String.split("_")[0]
#
#         # Style
#         row_data["Style"] =  GMD_String.split("_")[1]
#
#         # tempo
#         row_data["Tempo"] = float(GMD_String.split("_")[2])
#
#         # GMD Segment Type
#         row_data["GMD Segment Type"] = GMD_String.split("_")[3]
#
#         # GMD Segment Meter
#         row_data["GMD Segment Meter"]  = GMD_String.split("_")[4]
#
#         # GMD Segment Number
#         row_data["Selected 2Bars From Start"] = int(GMD_String.split("_")[-1])
#
#         # append row data to csv data
#         csv_data = csv_data.append(row_data, ignore_index=True)
#
#     if filename is not None:
#         # save csv
#         if not filename.endswith(".csv"):
#             filename = filename + ".csv"
#
#         csv_data.to_csv(filename, index=False)
#         print("Saved csv to {}".format(filename))
#
#     return csv_data


class DualizationDatasetAPI():
    def __init__(self, midi_folder=None):
        self.__midi_folder = ""
        self.__test_folders = []
        self.__test_numbers = []
        self.__wasTestedOnP1 = []
        self.__wasTestedOnP2 = []
        self.__wasTestedOnP3 = []
        self.__wasTestedOnP4 = []
        self.__wasTestedOnMultipleParticipants = []
        self.__testType = []
        self.__testNumber = []
        self.__style = []
        self.__tempo = []
        self.__gmdDrummer = []
        self.__gmdPerformanceSession = []
        self.__gmdSegmentType = []
        self.__gmdSegmentMeter = []
        self.__selected2BarsFromStart = []
        self.__dualizedMidifolderPath = []

        if midi_folder is not None:
            self.populate_fields(midi_folder)

    def populate_fields(self, midi_folder):
        self.__midi_folder = midi_folder
        tests = glob.glob(f'{midi_folder}/**/original.mid', recursive=True)
        assert len(tests) > 0, "No folders found in {}".format(midi_folder)
        self.__test_folders = sorted([os.path.dirname(f) for f in tests], key=lambda x: x.split(os.sep)[-1])
        print("Found {} tested patterns".format(len(self.__test_folders)))
        self.__test_numbers = [t.split(os.sep)[-1].split(" ")[0].split("[")[-1] for t in self.__test_folders]
        self.__wasTestedOnP1 = [True if "P1" in t else False for t in self.__test_folders]
        self.__wasTestedOnP2 = [True if "P2" in t else False for t in self.__test_folders]
        self.__wasTestedOnP3 = [True if "P3" in t else False for t in self.__test_folders]
        self.__wasTestedOnP4 = [True if "P4" in t else False for t in self.__test_folders]
        self.__wasTestedOnMultipleParticipants = [True if len(t.split(" ")[1].split("]")[0]) > 2 else False
                                                  for t in self.__test_folders]
        self.__testType = [t.split(os.sep)[1].replace(
            "Repetitions", "Three Random Repetitions").replace(
            "SimpleComplex", "Simple Complex") for t in self.__test_folders]

        self.__testNumber = [t.split(os.sep)[-1].split(" ")[0].split("[")[-1] for t in self.__test_folders]
        self.__style = [t.split(os.sep)[-1].split(" ")[-1].split("_")[1].lower() for t in self.__test_folders]
        self.__tempo = [float(t.split(os.sep)[-1].split(" ")[-1].split("_")[2]) for t in self.__test_folders]
        self.__gmdDrummer = [t.split(os.sep)[-1].split(" ")[-1].split("-")[0] for t in self.__test_folders]
        self.__gmdPerformanceSession = [t.split(os.sep)[-1].split(" ")[-1].split("-")[1].split("_")[0]
                                        for t in self.__test_folders]
        self.__gmdSegmentType = [t.split(os.sep)[-1].split(" ")[-1].split("_")[3] for t in self.__test_folders]
        self.__gmdSegmentMeter = [t.split(os.sep)[-1].split(" ")[-1].split("_")[4] for t in self.__test_folders]
        self.__selected2BarsFromStart = [int(t.split(os.sep)[-1].split(" ")[-1].split("_")[-1]) for t in
                                         self.__test_folders]
        self.__dualizedMidifolderPath = [os.path.join(os.path.dirname(t), "dualized") for t in self.__test_folders]

    def copy(self):
        return copy.deepcopy(self)

    def __remove_datapoints(self, should_keep_):
        """ Removes datapoints from the dataset
        @param should_keep_:  [True, False, True, ...] indicating which datapoints to keep
        @return: None
        """
        for key in self.__dict__.keys():
            if key.startswith("_DatasetAPI__"):
                if isinstance(self.__dict__[key], list):
                    self.__dict__[key] = [self.__dict__[key][i] for i in range(len(should_keep_)) if should_keep_[i]]

    @property
    def summary_dataframe(self):
        summary = pd.DataFrame()
        summary["Test Number"] = self.__testNumber
        summary["Test Type"] = self.__testType
        summary["Was Tested On Multiple Participants"] = self.__wasTestedOnMultipleParticipants
        summary["Was Tested On P1"] = self.__wasTestedOnP1
        summary["Was Tested On P2"] = self.__wasTestedOnP2
        summary["Was Tested On P3"] = self.__wasTestedOnP3
        summary["Was Tested On P4"] = self.__wasTestedOnP4
        summary["Style"] = self.__style
        summary["Tempo"] = self.__tempo
        summary["GMD Drummer"] = self.__gmdDrummer
        summary["GMD Performance Session"] = self.__gmdPerformanceSession
        summary["GMD Segment Type"] = self.__gmdSegmentType
        summary["GMD Segment Meter"] = self.__gmdSegmentMeter
        summary["Selected 2Bars From Start"] = self.__selected2BarsFromStart
        summary["Dualized Midifolder Path"] = self.__dualizedMidifolderPath
        summary["Test Folder"] = self.__test_folders
        return summary

    @property
    def summary_dict(self):
        return self.summary_dataframe.to_dict(orient="records")

    def save_summary(self, filename):
        summary = self.summary_dataframe
        if not filename.endswith(".csv"):
            filename = filename + ".csv"
        summary.to_csv(filename, index=False)
        print("Saved csv to {}".format(filename))

    @property
    def MultipleParticipantSubset(self):
        """ Returns a subset of dataset for which the drum patterns were tested on at least two participants
        @return: DualizationDatasetAPI object"""
        new_dataset = self.copy()
        indices = self.summary_dataframe["Was Tested On Multiple Participants"] == True
        new_dataset.__remove_datapoints(indices)
        return new_dataset

    @property
    def SingleParticipantSubset(self):
        """ Returns a subset of dataset for which the drum patterns were tested on a single participant only
        @return: DualizationDatasetAPI object"""
        new_dataset = self.copy()
        indices = self.summary_dataframe["Was Tested On Multiple Participants"] == False
        new_dataset.__remove_datapoints(indices)
        return new_dataset

    @property
    def ThreeRepetitionSubset(self):
        """ Returns a subset object with only the tests in which a given drum pattern was presented
        three times randomly to the participants without letting them know that they had already
        dualized the pattern before
        @return: DualizationDatasetAPI object"""
        new_dataset = self.copy()
        indices = self.summary_dataframe["Test Type"] == "Three Random Repetitions"
        new_dataset.__remove_datapoints(indices)
        return new_dataset

    @property
    def SimpleComplexSubset(self):
        """ Returns a subset object with only the tests in which the participants were
        asked to provide a simple and a complex dualization for a given drum pattern
        @return: DualizationDatasetAPI object"""
        new_dataset = self.copy()
        indices = self.summary_dataframe["Test Type"] == "Simple Complex"
        new_dataset.__remove_datapoints(indices)
        return new_dataset

    @property
    def P1Tests(self):
        """ Returns a subset object with only the tests that were performed on P1
        @return: DualizationDatasetAPI object """
        new_dataset = self.copy()
        indices = self.summary_dataframe["Was Tested On P1"] == True
        new_dataset.__remove_datapoints(indices)
        return new_dataset

    @property
    def P2Tests(self):
        """ Returns a subset object with only the tests that were performed on P2
        @return: DualizationDatasetAPI object"""
        new_dataset = self.copy()
        indices = self.summary_dataframe["Was Tested On P2"] == True
        new_dataset.__remove_datapoints(indices)
        return new_dataset

    @property
    def P3Tests(self):
        """ Returns a subset object with only the tests that were performed on P3
        @return: DualizationDatasetAPI object"""
        new_dataset = self.copy()
        indices = self.summary_dataframe["Was Tested On P3"] == True
        new_dataset.__remove_datapoints(indices)
        return new_dataset

    @property
    def P4Tests(self):
        """ Returns a subset object with only the tests that were performed on P4
        @return: DualizationDatasetAPI object"""
        new_dataset = self.copy()
        indices = self.summary_dataframe["Was Tested On P4"] == True
        new_dataset.__remove_datapoints(indices)
        return new_dataset

    def get_subset_matching_styles(self, style, hard_match=False):
        """
        Filters the dataset by style (or styles) and returns a new dataset.
        The matching ignores case.
        @param style:
        @param hard_match:
        @return:
        """
        new_dataset = self.copy()
        if isinstance(style, str):
            style = [style]

        style = [s.lower() for s in style]

        if hard_match:
            indices = self.summary_dataframe["Style"].isin(style)
        else:
            indices = self.summary_dataframe["Style"].str.lower().isin([s.lower() for s in style])
        new_dataset.__remove_datapoints(indices)
        return new_dataset

    def get_subset_within_tempo_range(self, min_, max_):
        """
        Filters the dataset by tempo range and returns a new dataset.
        @param min_:  minimum tempo
        @param max_:  maximum tempo
        @return:  new dataset
        """
        new_dataset = self.copy()
        indices = (self.summary_dataframe["Tempo"] >= min_) & (self.summary_dataframe["Tempo"] <= max_)
        new_dataset.__remove_datapoints(indices)
        return new_dataset

    def __format_test_number(self, test_number, check_if_exists=True):
        """
        Formats the test number to 3 digits (e.g. 1 -> 001)
        @return:
        """
        df = self.summary_dataframe
        test_number = f"{test_number:03d}" if isinstance(test_number, int) else test_number
        if test_number not in df["Test Number"].values:
            raise Warning(f"Test number {test_number} not found in dataset. Select from {df['Test Number'].values}")
        return test_number

    def get_participants_attempted_test_number(self, test_number):
        """
        Returns a list of participants who attempted a given test number
        @param test_number (int or str):
                test sample id potentially from 1 to 345 (if str, must be 3 digits, "001" to "345")
        @return:  list of participants
        """
        df = self.summary_dataframe
        test_number = self.__format_test_number(test_number)
        row = df.loc[df['Test Number'] == str(test_number)]
        participants = []
        if row["Was Tested On P1"].values[0]:
            participants.append(1)
        if row["Was Tested On P2"].values[0]:
            participants.append(2)
        if row["Was Tested On P3"].values[0]:
            participants.append(3)
        if row["Was Tested On P4"].values[0]:
            participants.append(4)
        return participants

    def get_test_numbers(self):
        """
        Returns a list of test numbers
        @return:  list of test numbers
        """
        return self.summary_dataframe["Test Number"].values

    def get_folder_path_for_test_number(self, test_number):
        """
        Returns the folder path for a given test number
        @param test_number (int or str):
                test sample id potentially from 1 to 345 (if str, must be 3 digits, "001" to "345")
        @return:  path to folder containing the test number
        """
        df = self.summary_dataframe
        test_number = self.__format_test_number(test_number)
        row = df.loc[df['Test Number'] == str(test_number)]
        return row["Test Folder"].values[0]

    def get_tested_drum_pattern_path(self, test_number):
        """
        Returns the midi file for the drum pattern that was tested
        @param test_number (int or str):
                test sample id potentially from 1 to 345 (if str, must be 3 digits, "001" to "345")
        @return:  midi file
        """
        return os.path.join(self.get_folder_path_for_test_number(test_number), "original.mid")

    def get_test_type_for_test_number(self, test_number):
        """
        Returns the test type for a given test number
        @param test_number (int or str):
                test sample id potentially from 1 to 345 (if str, must be 3 digits, "001" to "345")
        @return:  test type
        """
        df = self.summary_dataframe
        test_number = self.__format_test_number(test_number)
        row = df.loc[df['Test Number'] == str(test_number)]
        return row["Test Type"].values[0]

    def get_style_for_test_number(self, test_number):
        """
        Returns the style for a given test number
        @param test_number (int or str):
                test sample id potentially from 1 to 345 (if str, must be 3 digits, "001" to "345")
        @return:  style
        """
        df = self.summary_dataframe
        test_number = self.__format_test_number(test_number)
        row = df.loc[df['Test Number'] == str(test_number)]
        return row["Style"].values[0]

    def get_tempo_for_test_number(self, test_number):
        """
        Returns the tempo for a given test number
        @param test_number (int or str):
                test sample id potentially from 1 to 345 (if str, must be 3 digits, "001" to "345")
        @return:  tempo
        """
        df = self.summary_dataframe
        test_number = self.__format_test_number(test_number)
        row = df.loc[df['Test Number'] == str(test_number)]
        return row["Tempo"].values[0]

    def get_participant_dualizations_for_test_number(self, test_number, participant):
        """
        Returns a DualizationsForParticipant object for a given test number and participant
        @param test_number (int or str):
                test sample id potentially from 1 to 345 (if str, must be 3 digits, "001" to "345")
        @return:  DualizationsForParticipant object
        """
        test_number = self.__format_test_number(test_number)
        if participant not in self.get_participants_attempted_test_number(test_number):
            raise ValueError(f"Participant {participant} did not attempt test number {test_number}")
        dualizationData = DualizationsForParticipant()
        dualizationData.populate_attributes(self, test_number, participant)
        return dualizationData


class DualizationsForParticipant():
    def __int__(self):
        self.__FolderPath = None          # path to the folder containing the test
        self.__TestNumber = None          # 001, 002, 003, ... (str)
        self.__Participant = None         # 1, 2, 3 or 4 (int) for the participants P1, P2, P3 or P4
        self.__TestType = None            # "Three Random Repetitions" or "Simple Complex"
        self.__otherParticipants = None   # list of other participants if multiple participants tried the test pattern
        self.__rep1 = None                # 1st repetition if test type is "Three Random Repetitions"
        self.__rep2 = None                # 2nd repetition if test type is "Three Random Repetitions"
        self.__rep3 = None                # 3rd repetition if test type is "Three Random Repetitions"
        self.__simple = None              # simple if test type is "Simple Complex"
        self.__complex = None             # complex if test type is "Simple Complex"
        self.__style = None               # style of the test pattern
        self.__tempo = None               # tempo of the test pattern
        self.__original = None            # original test pattern

    def populate_attributes(self, dualizationDatasetAPI, number, participant):
        """
        Populates the attributes of the DualizationsForParticipant class
        @param dualizationDatasetAPI:  DualizationDatasetAPI object
        @param number:  test number
        @param participant:  participant number
        @return:
        """
        self.FolderPath = dualizationDatasetAPI.get_folder_path_for_test_number(number)
        self.TestNumber = number
        self.Participant = participant
        self.TestType = dualizationDatasetAPI.get_test_type_for_test_number(number)
        self.otherParticipants = dualizationDatasetAPI.get_participants_attempted_test_number(number)
        # remove the current participant from the list of other participants
        self.otherParticipants.remove(participant)
        self.style = dualizationDatasetAPI.get_style_for_test_number(number)
        self.tempo = dualizationDatasetAPI.get_tempo_for_test_number(number)
        if self.TestType == "Three Random Repetitions":
            self.rep1 = os.path.join(self.FolderPath, f"Participant_{participant}_repetition_0.mid")
            self.rep2 = os.path.join(self.FolderPath, f"Participant_{participant}_repetition_1.mid")
            self.rep3 = os.path.join(self.FolderPath, f"Participant_{participant}_repetition_2.mid")
        elif self.TestType == "Simple Complex":
            self.simple = os.path.join(self.FolderPath, f"Participant_{participant}_simple.mid")
            self.complex = os.path.join(self.FolderPath, f"Participant_{participant}_complex.mid")


    #get_test_type_for_test_number, get_style_for_test_number, get_tempo_for_test_number
    @property
    def FolderPath(self):
        return self.__FolderPath

    @FolderPath.setter
    def FolderPath(self, value):
        self.__FolderPath = value

    @property
    def TestNumber(self):
        return self.__TestNumber

    @TestNumber.setter
    def TestNumber(self, value):
        self.__TestNumber = value

    @property
    def Participant(self):
        return self.__Participant

    @Participant.setter
    def Participant(self, value):
        assert value in [1, 2, 3, 4], "Participant must be 1, 2, 3 or 4"
        self.__Participant = value

    @property
    def TestType(self):
        return self.__TestType

    @TestType.setter
    def TestType(self, value):
        assert value in ["Three Random Repetitions", "Simple Complex"], "TestType must be 'Three Random Repetitions' or 'Simple Complex'"
        self.__TestType = value

    @property
    def otherParticipants(self):
        return self.__otherParticipants

    @otherParticipants.setter
    def otherParticipants(self, value):
        if isinstance(value, int):
            value = [value]
        assert isinstance(value, list), "otherParticipants must be a list of integers"
        assert min(value) >= 1 and max(value) <= 4, "otherParticipants must be a list of integers between 1 and 4"
        self.__otherParticipants = value

    @property
    def rep1(self):
        if self.TestType == "Three Random Repetitions":
            return self.__rep1
        else:
            raise AttributeError("rep1 is not defined for TestType 'Simple Complex'")

    @rep1.setter
    def rep1(self, value):
        if self.TestType == "Three Random Repetitions":
            assert value.endswith(".mid"), " must assign a midi file (xyz.mid)"
            self.__rep1 = Pattern(value, self.style, self.tempo, self.__TestNumber)
        else:
            raise AttributeError("rep1 field is not available for TestType 'Simple Complex'"
                                 "use simple and complex instead or change TestType to 'Three Random Repetitions'")

    @property
    def rep2(self):
        if self.TestType == "Three Random Repetitions":
            return self.__rep2
        else:
            raise AttributeError("rep2 is not defined for TestType 'Simple Complex'")

    @rep2.setter
    def rep2(self, value):
        if self.TestType == "Three Random Repetitions":
            assert value.endswith(".mid"), " must assign a midi file (xyz.mid)"
            self.__rep2 = Pattern(value, self.style, self.tempo, self.__TestNumber)
        else:
            raise AttributeError("rep2 field is not available for TestType 'Simple Complex'"
                                 "use simple and complex instead or change TestType to 'Three Random Repetitions'")

    @property
    def rep3(self):
        if self.TestType == "Three Random Repetitions":
            return self.__rep3
        else:
            raise AttributeError("rep3 is not defined for TestType 'Simple Complex'")


    @rep3.setter
    def rep3(self, value):
        if self.TestType == "Three Random Repetitions":
            assert value.endswith(".mid"), " must assign a midi file (xyz.mid)"
            self.__rep3 = Pattern(value, self.style, self.tempo, self.__TestNumber)
        else:
            raise AttributeError("rep3 field is not available for TestType 'Simple Complex'"
                                 "use simple and complex instead or change TestType to 'Three Random Repetitions'")

    @property
    def simple(self):
        if self.TestType == "Simple Complex":
            return self.__simple
        else:
            raise AttributeError("simple is not defined for TestType 'Three Random Repetitions'")

    @simple.setter
    def simple(self, value):
        if self.TestType == "Simple Complex":
            assert value.endswith(".mid"), " must assign a midi file (xyz.mid)"
            self.__simple = Pattern(value, self.style, self.tempo, self.__TestNumber)
        else:
            raise AttributeError("simple field is not available for TestType 'Three Random Repetitions'"
                                 "use rep1, rep2 and rep3 instead or change TestType to 'Simple Complex'")

    @property
    def complex(self):
        if self.TestType == "Simple Complex":
            return self.__complex
        else:
            raise AttributeError("complex is not defined for TestType 'Three Random Repetitions'")

    @complex.setter
    def complex(self, value):
        if self.TestType == "Simple Complex":
            assert value.endswith(".mid"), " must assign a midi file (xyz.mid)"
            self.__complex = Pattern(value, self.style, self.tempo, self.__TestNumber)
        else:
            raise AttributeError("complex field is not available for TestType 'Three Random Repetitions'"
                                 "use rep1, rep2 and rep3 instead or change TestType to 'Simple Complex'")

    @property
    def original(self):
        return self.__original

    @original.setter
    def original(self, value):
        assert value.endswith(".mid"), " must assign a midi file (xyz.mid)"
        self.__original = Pattern(value, self.style, self.tempo, self.__TestNumber)

    @property
    def style(self):
        return self.__style

    @style.setter
    def style(self, value):
        self.__style = value

    @property
    def tempo(self):
        return self.__tempo

    @tempo.setter
    def tempo(self, value):
        self.__tempo = value


class Pattern:
    """Class to store a drum pattern or a dualization"""
    def __init__(self, midi_file_path, style, tempo, test_number):
        self.__path = midi_file_path
        self.__hvo_sequence = None
        self.__style = style
        self.__tempo = tempo
        self.__test_number = test_number

    @property
    def path(self):
        return self.__path

    @property
    def hvo_sequence(self):
        if self.__hvo_sequence is None:
            dmap = ROLAND_REDUCED_MAPPING_HEATMAPS if "original" in self.path else DUALIZATION_ROLAND_HAND_DRUM
            self.__hvo_sequence = midi_to_hvo_sequence(filename=self.__path,
                                                       drum_mapping=dmap)
            self.__hvo_sequence.adjust_length(32)
            self.__hvo_sequence.metadata.update({
                "Style": self.__style,
                "Tempo": self.__tempo,
                "Test Number": self.__test_number
            })
        return self.__hvo_sequence


if __name__ == "__main__":
    # summary = collect_data("midi_files", "./midi_files/summary.csv")

    # print(summary)
    dataset = DualizationDatasetAPI(midi_folder="midi_files")

    # print(dataset.summary_dict)

    dataset.save_summary("midi_files/summary2.csv")

    new_dataset = dataset.get_subset_matching_styles("Jazz", hard_match=True)
    new_dataset.save_summary("midi_files/summary2jazz.csv")

    dataset.get_participants_attempted_test_number("100")
    dataset.get_test_numbers()

    dataset.get_folder_path_for_test_number("100")

    dataset.get_tested_drum_pattern_path("100")

    dualizations = dataset.get_participant_dualizations_for_test_number(test_number=100, participant=1)

    dataset.get_tested_drum_pattern_path("100")