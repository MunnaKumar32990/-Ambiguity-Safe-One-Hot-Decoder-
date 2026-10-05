"""
presets.py
==========
Preset experiment configurations for the web frontend.

Each preset contains training data, test data, ground truth labels,
and encoder configuration — the same scenarios used in the experiments/
directory, packaged as JSON-serializable dictionaries.
"""

PRESETS = {
    "gender_binary": {
        "name": "Gender (Binary — Canonical Example)",
        "description": (
            "The canonical research problem: Female/Male training with "
            "an Unknown test value. drop='if_binary' causes Female to be "
            "dropped, creating a collision with unseen categories."
        ),
        "train_data": [["Female"], ["Male"]],
        "test_data": [["Female"], ["Male"], ["Unknown"]],
        "ground_truth": [["Female"], ["Male"], ["Unknown"]],
        "feature_names": ["Gender"],
        "drop": "if_binary",
        "handle_unknown": "ignore",
        "expected_insight": (
            "Female and Unknown both encode to [0]. sklearn silently "
            "returns 'Female' for Unknown — our system flags both as AMBIGUOUS."
        ),
    },
    "employment_status": {
        "name": "Employment Status (Binary Variant)",
        "description": (
            "Active/Inactive binary feature with an unseen 'Terminated' "
            "value. Same ambiguity pattern as the Gender example."
        ),
        "train_data": [["Active"], ["Inactive"]],
        "test_data": [["Active"], ["Inactive"], ["Terminated"]],
        "ground_truth": [["Active"], ["Inactive"], ["Terminated"]],
        "feature_names": ["Status"],
        "drop": "if_binary",
        "handle_unknown": "ignore",
        "expected_insight": (
            "Active (dropped) and Terminated (unseen) collide at [0]. "
            "Demonstrates the problem is not specific to gender data."
        ),
    },
    "colors_no_drop": {
        "name": "Colors (3-Class, No Drop)",
        "description": (
            "Red/Green/Blue with an unseen Yellow. drop=None means no "
            "category is dropped, so the all-zeros vector uniquely "
            "indicates an unseen category (UNKNOWN, not AMBIGUOUS)."
        ),
        "train_data": [["Red"], ["Green"], ["Blue"]],
        "test_data": [["Red"], ["Green"], ["Blue"], ["Yellow"]],
        "ground_truth": [["Red"], ["Green"], ["Blue"], ["Yellow"]],
        "feature_names": ["Color"],
        "drop": None,
        "handle_unknown": "ignore",
        "expected_insight": (
            "No drop configured → all-zeros uniquely means unseen. "
            "Yellow is UNKNOWN (not AMBIGUOUS). Known colors are all SAFE."
        ),
    },
    "colors_drop_first": {
        "name": "Colors (3-Class, Drop First)",
        "description": (
            "Red/Green/Blue with drop='first'. The first category (Blue, "
            "alphabetically sorted) is dropped, creating ambiguity with "
            "any unseen category."
        ),
        "train_data": [["Red"], ["Green"], ["Blue"]],
        "test_data": [["Red"], ["Green"], ["Blue"], ["Yellow"]],
        "ground_truth": [["Red"], ["Green"], ["Blue"], ["Yellow"]],
        "feature_names": ["Color"],
        "drop": "first",
        "handle_unknown": "ignore",
        "expected_insight": (
            "Blue (dropped first alphabetically) and Yellow (unseen) both "
            "map to all-zeros → AMBIGUOUS. Shows the problem extends beyond "
            "binary features."
        ),
    },
    "gender_city_multicolumn": {
        "name": "Gender + City (Multi-Column)",
        "description": (
            "Two features: Gender (binary) and City (3-class). With "
            "drop='if_binary', only Gender gets a drop (binary). City "
            "has no drop (not binary). Shows per-feature ambiguity isolation."
        ),
        "train_data": [
            ["Female", "Delhi"],
            ["Female", "Hyderabad"],
            ["Male", "Delhi"],
            ["Male", "Chennai"],
        ],
        "test_data": [
            ["Female", "Delhi"],
            ["Male", "Hyderabad"],
            ["Unknown", "Delhi"],
            ["Female", "Bangalore"],
            ["Unknown", "Bangalore"],
        ],
        "ground_truth": [
            ["Female", "Delhi"],
            ["Male", "Hyderabad"],
            ["Unknown", "Delhi"],
            ["Female", "Bangalore"],
            ["Unknown", "Bangalore"],
        ],
        "feature_names": ["Gender", "City"],
        "drop": "if_binary",
        "handle_unknown": "ignore",
        "expected_insight": (
            "Gender ambiguity doesn't contaminate City analysis. "
            "Per-feature isolation means City can be SAFE even when "
            "Gender is AMBIGUOUS in the same row."
        ),
    },
    "gender_no_drop": {
        "name": "Gender (Binary, No Drop — Control)",
        "description": (
            "Same Gender data but with drop=None. Without dropping a "
            "category, there is no collision — unknowns are clearly "
            "UNKNOWN, not AMBIGUOUS. Serves as a control experiment."
        ),
        "train_data": [["Female"], ["Male"]],
        "test_data": [["Female"], ["Male"], ["Unknown"]],
        "ground_truth": [["Female"], ["Male"], ["Unknown"]],
        "feature_names": ["Gender"],
        "drop": None,
        "handle_unknown": "ignore",
        "expected_insight": (
            "No drop → no ambiguity. Female=[1,0], Male=[0,1], "
            "Unknown=[0,0]. All-zeros uniquely means unseen → UNKNOWN."
        ),
    },
    "nt1_binary_collision": {
        "name": "NT-1: Binary Dropped Category Collision",
        "description": (
            "Mandatory Negative Test 1: An unseen gender category ('NonBinary') "
            "collides with the dropped known category ('Female') under drop='if_binary'. "
            "sklearn silently outputs 'Female'. Our system flags AMBIGUOUS."
        ),
        "train_data": [["Female"], ["Male"]],
        "test_data": [["Female"], ["Male"], ["NonBinary"]],
        "ground_truth": [["Female"], ["Male"], ["NonBinary"]],
        "feature_names": ["Gender"],
        "drop": "if_binary",
        "handle_unknown": "ignore",
        "expected_insight": (
            "sklearn silently misdecodes 'NonBinary' as 'Female'. "
            "Ambiguity-Safe layer intercepts the all-zeros vector and withholds the erroneous label."
        ),
    },
    "nt2_multicolumn_compound": {
        "name": "NT-2: Multi-Column Compound Dropped Collision",
        "description": (
            "Mandatory Negative Test 2: Multiple columns have dropped categories ('first'). "
            "A test row with unseen categories across all columns results in an all-zeros vector "
            "for each feature, causing sklearn to fabricate a fictitious valid record."
        ),
        "train_data": [
            ["High_Risk", "Credit_OK", "North"],
            ["Low_Risk", "Credit_Poor", "South"],
            ["Med_Risk", "Credit_Good", "East"],
        ],
        "test_data": [
            ["High_Risk", "Credit_OK", "North"],
            ["Unseen_Risk", "Unseen_Credit", "Unseen_Region"],
        ],
        "ground_truth": [
            ["High_Risk", "Credit_OK", "North"],
            ["Unseen_Risk", "Unseen_Credit", "Unseen_Region"],
        ],
        "feature_names": ["Risk_Level", "Credit_Tier", "Region"],
        "drop": "first",
        "handle_unknown": "ignore",
        "expected_insight": (
            "Baseline decodes unseen row into [High_Risk, Credit_Good, East] silently. "
            "Ambiguity-Safe decoder detects compound ambiguity across all columns independently."
        ),
    },
    "nt4_missing_values": {
        "name": "NT-4: Missing Values (NaN / None) Injection",
        "description": (
            "Mandatory Negative Test 4: Missing data values (NaN, None) ingested as inputs. "
            "When drop='first', missing values encode to zeros and silently decode to the first category."
        ),
        "train_data": [["Grade_A"], ["Grade_B"], ["Grade_C"]],
        "test_data": [["Grade_A"], ["None"], ["NaN"], ["Grade_X"]],
        "ground_truth": [["Grade_A"], ["None"], ["NaN"], ["Grade_X"]],
        "feature_names": ["Grade"],
        "drop": "first",
        "handle_unknown": "ignore",
        "expected_insight": (
            "Missing values (None/NaN) are mapped to zeros and mistakenly assigned 'Grade_A' by sklearn. "
            "Our system detects the collision and flags AMBIGUOUS."
        ),
    },
    "adult_census_demographics": {
        "name": "Real-World: Adult Census Demographics",
        "description": (
            "Realistic high-dimensional categorical features adapted from the UCI Adult Census dataset: "
            "Workclass, Education, MaritalStatus, Occupation, and Sex. Evaluates multi-attribute ambiguity."
        ),
        "train_data": [
            ["Private", "Bachelors", "Never-married", "Tech-support", "Female"],
            ["Self-emp", "Masters", "Married-civ", "Exec-managerial", "Male"],
            ["State-gov", "HS-grad", "Divorced", "Adm-clerical", "Female"],
            ["Federal-gov", "Doctorate", "Married-civ", "Prof-specialty", "Male"],
        ],
        "test_data": [
            ["Private", "Bachelors", "Never-married", "Tech-support", "Female"],
            ["Self-emp", "Masters", "Married-civ", "Exec-managerial", "Male"],
            ["Gig-Economy", "Some-College", "Separated", "Freelance", "Non-Binary"],
            ["Private", "Doctorate", "Never-married", "Cybersecurity", "Female"],
        ],
        "ground_truth": [
            ["Private", "Bachelors", "Never-married", "Tech-support", "Female"],
            ["Self-emp", "Masters", "Married-civ", "Exec-managerial", "Male"],
            ["Gig-Economy", "Some-College", "Separated", "Freelance", "Non-Binary"],
            ["Private", "Doctorate", "Never-married", "Cybersecurity", "Female"],
        ],
        "feature_names": ["Workclass", "Education", "MaritalStatus", "Occupation", "Sex"],
        "drop": "first",
        "handle_unknown": "ignore",
        "expected_insight": (
            "Unseen job and gender categories trigger column-isolated ambiguity alerts without corrupting "
            "unambiguous features like Private or Bachelors."
        ),
    },
}


def get_preset_names():
    """Return a list of preset keys and their display names."""
    return [
        {"key": key, "name": preset["name"], "description": preset["description"]}
        for key, preset in PRESETS.items()
    ]


def get_preset(key):
    """Return a single preset by key, or None if not found."""
    return PRESETS.get(key)
