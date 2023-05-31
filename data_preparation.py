import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import neptune

RAW_DATA_FOLDER = Path('data/identification_experiment_raw/')
OUTPUT_DATA_FOLDER = Path('data/identification_experiment/')

JIRA = 'IDS-002'

if __name__ == '__main__':

    OUTPUT_DATA_FOLDER.mkdir(exist_ok=True, parents=True)

    for file in RAW_DATA_FOLDER.glob('*.csv'):
        run = neptune.init_run(
            project="kirill.ussenko/System-identification",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MDc3N2YyYy0zNGNjLTQwMGQtOTZiOC0zMmMwNTNiMTdmNDEifQ=="
        )

        run["train/raw_data"].track_files(str(RAW_DATA_FOLDER))
        run["train/prepared_data"].track_files(str(OUTPUT_DATA_FOLDER))

        run["JIRA"] = JIRA
        run["preparation"] = "train_test_split"

        df = pd.read_csv(file)
        df.columns = ['time', 'u', 'y', 'k', 't']
        train, test = train_test_split(df, test_size=0.2, shuffle=False)

        train.to_csv(OUTPUT_DATA_FOLDER / f'{file.name.replace(".csv", "")}_train.csv', index=False)
        test.to_csv(OUTPUT_DATA_FOLDER / f'{file.name.replace(".csv", "")}_test.csv', index=False)

        run[f"train/control"].extend(train['u'].tolist(), steps=train['time'].tolist())
        run[f"test/control"].extend(test['u'].tolist(), steps=test['time'].tolist())

        run[f"train/output"].extend(train['y'].tolist(), steps=train['time'].tolist())
        run[f"test/output"].extend(test['y'].tolist(), steps=test['time'].tolist())

        run[f"train/k"].extend(train['k'].tolist(), steps=train['time'].tolist())
        run[f"test/k"].extend(test['k'].tolist(), steps=test['time'].tolist())

        run[f"train/T"].extend(train['t'].tolist(), steps=train['time'].tolist())
        run[f"test/T"].extend(test['t'].tolist(), steps=test['time'].tolist())

        run.stop()
