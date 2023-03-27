import os
import shutil
import glob

# find all files in processed_data/Repetitions/tested_with_four_participants/Participant_x/[needed folders]

participant_1_paths = glob.glob('processed_data/SimpleVsComplex_Repetitions/Participant_1/*', recursive=True)
participant_2_paths = glob.glob('processed_data/SimpleVsComplex_Repetitions/Participant_2/*', recursive=True)

participant_1_ids = [x.split('/')[-1] for x in participant_1_paths]
participant_2_ids = [x.split('/')[-1] for x in participant_2_paths]


shared_examples = []
participant_1_only = []

for file in participant_1_ids:
    sample_id = file.split('/')[-1]
    # check if sample_id can be found in the other participants (search
    if sample_id in participant_2_ids:
        shared_examples.append(sample_id)
    else:
        participant_1_only.append(sample_id)

# create a folder for each shared example in processed_data/Repetitions/tested_with_four_participants/
for example in shared_examples:
    os.makedirs('processed_data/SimpleComplex/tested_with_two_participants/' + example, exist_ok=True)

for example in participant_1_only:
    os.makedirs('processed_data/SimpleComplex/tested_with_Participant_1_Only/' + example, exist_ok=True)

for example in shared_examples:
    for id in [1, 2, 3, 4]:
        files = glob.glob(f'processed_data/SimpleVsComplex_Repetitions/Participant_{id}/{example}/*.mid', recursive=True)
        for file in files:
            shutil.copy(file, f'processed_data/SimpleComplex/tested_with_two_participants/{example}')
            # rename file to include participant id
            if 'original' not in file:
                os.rename(f'processed_data/SimpleComplex/tested_with_two_participants/{example}/{file.split("/")[-1]}',
                          f'processed_data/SimpleComplex/tested_with_two_participants/{example}/Participant_{id}_{file.split("/")[-1].split(".")[0]}.mid')

for example in participant_1_only:
    files = glob.glob(f'processed_data/SimpleVsComplex_Repetitions/Participant_1/{example}/*.mid', recursive=True)
    for file in files:
        shutil.copy(file, f'processed_data/SimpleComplex/tested_with_Participant_1_Only/{example}')
        # rename file to include participant id
        if 'original' not in file:
            os.rename(f'processed_data/SimpleComplex/tested_with_Participant_1_Only/{example}/{file.split("/")[-1]}',
                      f'processed_data/SimpleComplex/tested_with_Participant_1_Only/{example}/Participant_1_{file.split("/")[-1].split(".")[0]}.mid')