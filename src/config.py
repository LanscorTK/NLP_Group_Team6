from pathlib import Path

# Project root is the parent of the src directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = ROOT_DIR / "external"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# Raw data file
MBS_RAW_FILE = RAW_DATA_DIR / "tripadvisor_mbs_review_from201501_v2.xlsx"

# Processed data files
MBS_FILTERED_REVIEWS = PROCESSED_DATA_DIR / "mbs_filtered_reviews.csv"
MBS_SENTENCES = PROCESSED_DATA_DIR / "mbs_sentences.csv"

# SemEval data
SEMEVAL_DIR = EXTERNAL_DIR / "semeval2019task9"
SEMEVAL_TRAIN = SEMEVAL_DIR / "V1.4_Training.csv"
SEMEVAL_TEST_LABELED = SEMEVAL_DIR / "SubtaskA_EvaluationData_labeled.csv"

# Annotation paths
MBS_CALIBRATION_SHEET = PROCESSED_DATA_DIR / "mbs_calibration_100.csv"
MBS_ANNOTATED_FULL = PROCESSED_DATA_DIR / "mbs_annotated_full.csv"
MBS_ANNOTATED_TRAIN = PROCESSED_DATA_DIR / "mbs_annotated_train.csv"
MBS_ANNOTATED_TEST = PROCESSED_DATA_DIR / "mbs_annotated_test.csv"

# Output directories
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"

# Random Seed for reproducibility
RANDOM_SEED = 42

# Preprocessing parameters
MIN_SENTENCE_TOKENS = 4
MAX_SENTENCE_TOKENS = 100
MIN_ASCII_RATIO = 0.5  # for language filtering

# Suggestion signal word lists (for EDA and regex baseline)
MODAL_VERBS = ["should", "would", "could", "might", "need to", "ought to"]
IMPERATIVE_SIGNALS = ["please", "recommend", "suggest", "consider", "try", "make sure"]
CONDITIONAL_SIGNALS = ["i wish", "if only", "it would be nice", "hopefully"]

# --- Phase 4: Model Development ---

# Regex baseline — refined patterns (more specific than bare modal verbs to reduce FPs)
REGEX_PATTERNS = [
    # Modal suggestions (specific phrases, not bare modals)
    r"\bshould\b", r"\bwould be better\b", r"\bcould improve\b", r"\bcould be better\b",
    r"\bcould use\b", r"\bmight want\b", r"\bneed to\b", r"\bought to\b",
    r"\bshould be able\b",
    # Imperative / request
    r"\bplease\b", r"\brecommend\b", r"\bsuggest\b", r"\bconsider\b",
    r"\btry to\b", r"\bmake sure\b",
    # Conditional / wish
    r"\bi wish\b", r"\bif only\b", r"\bit would be nice\b", r"\bhopefully\b",
    r"\bwould like\b", r"\bwould love\b",
    # Additional from SemEval baseline
    r"\bi hope\b", r"\bi want\b", r"\bgo for\b",
]

# BERT hyperparameters
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LENGTH = 128
BERT_BATCH_SIZE = 16
BERT_EPOCHS_STAGE1 = 3
BERT_LR_STAGE1 = 2e-5
BERT_EPOCHS_STAGE2 = 3
BERT_LR_STAGE2 = 1e-5
BERT_WEIGHT_DECAY = 0.01
BERT_WARMUP_RATIO = 0.1

# --- Phase 5: Topic Modeling ---

MBS_PREDICTIONS = PROCESSED_DATA_DIR / "mbs_predictions.csv"

HOTEL_ASPECTS = {
    "room": ["room", "bed", "pillow", "bathroom", "shower", "towel", "view",
             "balcony", "minibar", "closet", "clean", "noise", "wall"],
    "food & beverage": ["breakfast", "restaurant", "food", "dining", "buffet",
                        "coffee", "bar", "drink", "lunch", "dinner", "menu"],
    "service": ["staff", "service", "check-in", "checkout", "concierge",
                "reception", "housekeeping", "friendly", "helpful"],
    "facilities": ["pool", "gym", "spa", "infinity", "casino", "elevator",
                   "lift", "wifi", "internet", "parking", "shuttle"],
    "location": ["location", "mall", "shopping", "walk", "mrt", "taxi",
                 "airport", "nearby", "area", "district"],
    "value": ["price", "cost", "expensive", "worth", "value", "money",
              "charge", "fee", "pay", "overpriced"],
}
