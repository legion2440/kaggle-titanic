from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
GENDER_SUBMISSION_PATH = DATA_DIR / "gender_submission.csv"

RANDOM_STATE = 42
TARGET = "Survived"
ID_COLUMN = "PassengerId"
