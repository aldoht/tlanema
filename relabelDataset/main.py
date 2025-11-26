import pandas as pd
import os

TRAIN_DIR = './train'
TEST_DIR = './test'
VALID_DIR = './valid'

# CSV files
train_csv = TRAIN_DIR + '/_classes.csv'
test_csv = TEST_DIR + '/_classes.csv'
valid_csv = VALID_DIR + '/_classes.csv'

# Maps
LETTER_TO_LABEL_MAP = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'O': 12,
    'P': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'U': 17,
    'V': 18,
    'W': 19,
    'Y': 20
}
LABEL_TO_LETTER_MAP = {v:k for k,v in LETTER_TO_LABEL_MAP.items()}

# DataFrames
def csv_to_nondummydataset(csv_file: str) -> pd.DataFrame:
  print(f'Creating dataframe for {csv_file}')

  df = pd.read_csv(csv_file)
  filenames = df.filename
  df = df.drop('filename', axis='columns')

  new_names = {}
  for col_name, col in df.items():
    new_names[col_name] = 'class_'+col_name
  df = df.rename(columns=new_names)

  final_df = pd.from_dummies(df, sep='_')
  final_df = final_df.join(filenames)

  return final_df

def create_dir_for_letter(dir_letter: str) -> bool:
    directory_name = './' + dir_letter + '/'
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
        return True
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")
        return True
    except PermissionError:
        print(f"Permission denied: Unable to create '{directory_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return False

train_df = csv_to_nondummydataset(train_csv)
valid_df = csv_to_nondummydataset(valid_csv)
test_df = csv_to_nondummydataset(test_csv)

# Directories
for path, df in [(TRAIN_DIR, train_df), (VALID_DIR, valid_df), (TEST_DIR, test_df)]:
    for idx, series in df.iterrows():
        letter = series['class']
        filename = series['filename']
        # quit()
        if create_dir_for_letter(letter):
            os.rename(path + '/' + filename, './' + letter + '/' + filename)
        else:
            print(f'Saltando la letra {letter} en {filename}')
print('Fin del script. Adios =)')