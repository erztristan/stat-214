import pandas as pd
import numpy as np
import json

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw dataset by removing unnecesary features, 
    checking format, handling missing values and context-based data validation.

    Parameters:
    -----------
    df : pd.DataFrame
        The raw DataFrame loaded from the dataset.

    Returns:
    --------
    pd.DataFrame
        The cleaned dataset.
    """
    # Step 1: Remove unnecessary features
    df = remove_unnecessary_columns(df)

    # Step 2: Check all the values are in the predermined range
    assess_values_range(df, "../data/possible_values.json")

    # Step 3: Handling missing values
    df = fill_missing_values(df)

    # Step 4: Context-Based Data Validation and Downsizing
    df = logic_check(df)

    print("The data cleaning process has been completed.")

    return df


def remove_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes columns that are deemed irrelevant to the analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        The input dataset containing all variables.

    Returns:
    --------
    pd.DataFrame
        The dataset with unnecessary columns removed.
    """
    columns_to_drop = ['PatNum', 'EmplType', 'Certification', "Ethnicity"]
    
    # Ensure the columns exist before dropping to prevent errors
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    print("Completed the removal of unnecessary columns.")

    return df


def assess_values_range(df: pd.DataFrame, json_file: str) -> None:
    """
    Assesses and validates unique values in each column by comparing them against predefined possible values
    from a JSON file. If any values fall outside the allowed range, they are printed. If all values are valid,
    a confirmation message is displayed.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to be analyzed.
    json_file : str
        Path to the JSON file containing the possible values for each column.

    Returns:
    --------
    None
        The function prints the validation results.
    """

    # Load the possible values from JSON
    with open(json_file, "r", encoding="utf-8") as file:
        possible_values = json.load(file)

    all_valid = True  # Flag to track if all values are valid

    for column in df.columns:
        if column not in possible_values:
            continue  # Skip columns not in the JSON file

        output_lines = [f"\nValidation for '{column}' column:"]

        # Get unique non-null values in the dataset
        unique_values = set(df[column].dropna().unique())

        # Get allowed values from JSON and convert them to strings for uniformity
        allowed_values = set(str(val) for val in possible_values[column])

        # Convert dataset values to strings for comparison
        unique_values_str = {str(int(val)) if isinstance(val, (np.integer, np.float64)) and val.is_integer() else str(val)
                             for val in unique_values}

        # Identify invalid values
        invalid_values = unique_values_str - allowed_values

        # If "Num" is in allowed values, allow all integers (negative, zero, positive)
        if "Num" in allowed_values:
            invalid_values = {val for val in unique_values if not (isinstance(val, (int, np.integer)) or (isinstance(val, float) and val.is_integer()))}
        else:
            invalid_values = unique_values_str - allowed_values

        # Output results only if there are invalid values
        if invalid_values:
            output_lines.append("Some values do not match the predefined categories:")
            output_lines.append(str(invalid_values))
            all_valid = False  # At least one invalid value exists

        # Print results only if there is an issue
        if invalid_values:
            for line in output_lines:
                print(line)

    # If no invalid values were found in any column, print a success message
    if all_valid:
        print("All values across all columns fall within the predefined categories.")


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values in the dataset based on predefined assumptions and logic.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset containing missing values.

    Returns:
    --------
    pd.DataFrame
        The dataset with missing values handled appropriately.
    """

    ### This section was intended to be included in the report but was omitted due to space constraints. 
    ### It serves as a footnote to provide a more detailed explanation of how missing values were handled.
    """
    Handles missing values in the dataset based on structured rules to maintain logical consistency.

    1. Retains meaningful information by differentiating between absence, irrelevance, and true missing values.
    2. Assumes absence (0) for missing values in binary medical condition features.
    3. Infers missing values from related features where possible.
    4. Ensures logical consistency across dependent variables.

    Rules applied:
    
    # Rule 1: Assign "Not Applicable" (92.0) for features that are not relevant.
    df.loc[df['FeatureX'].isna(), 'FeatureX'] = 92.0  # Example placeholder

    # Rule 2: Assume absence (0) for missing values in binary features.
    # Example: Vomiting is recorded only if observed, so missing values mean no vomiting.
    df.loc[df['Vomit'].isna(), 'Vomit'] = 0

    # Rule 3: Infer missing values from related features.
    # Example: If the total GCS score is missing but its components are available, recalculate it.
    df.loc[df['GCSTotal'].isna(), 'GCSTotal'] = (
        df['GCSEye'].fillna(0) + df['GCSVerbal'].fillna(0) + df['GCSMotor'].fillna(0)
    )

    # Example: If Age in Months is missing, compute from Age in Years.
    df.loc[df['AgeInMonth'].isna(), 'AgeInMonth'] = df['AgeInYears'] * 12

    # Rule 4: Ensure logical consistency between related features.
    # Example: If no neurological deficits, then all related features should be "Not Applicable" (92).
    df.loc[df['NeuroD'] == 0, ['NeuroDMotor', 'NeuroDSensory']] = 92
    """

    # Replace NaN values in the 'InjuryMech' column with 90.0 (Other Mechanism)
    df['InjuryMech'] = df['InjuryMech'].fillna(90.0)

    # Replace NaN values in the 'High_impact_InjSev' with 2.0 (Moderate)
    df['High_impact_InjSev'] = df['High_impact_InjSev'].fillna(2.0)

    # Assume no amnesia (0) for missing values.
    df["Amnesia_verb"] = df["Amnesia_verb"].fillna(0)

    ## Loss of consciousness
    # If 0 < LocLen < 5, assume LOCSeparate = 1, else 0.
    df["LOCSeparate"] = df["LOCSeparate"].fillna(df["LocLen"].apply(lambda x: 1 if 0 < x < 5 else 0))
    # Replace missing LOC duration 92.
    df['LocLen'] = df['LocLen'].fillna(92.0)

    ## Post-Traumatic Seizure Handling
    # Assume no seizure (0) for missing values.
    df["Seiz"] = df["Seiz"].fillna(0)
    # If 'Seiz' is 0 (No Seizure), mark 'SeizOccur' as 92 (Not Applicable)
    df["SeizOccur"] = df["SeizOccur"].fillna(92.0)
    # If 'Seiz' is 0 (No Seizure), mark 'SeizLen' as 92 (Not Applicable)
    df["SeizLen"] = df["SeizLen"].fillna(92.0)

    ## Parental Perception of Normal Behavior
    # If missing, assume the parent thinks the child is behaving normally (1).
    df["ActNorm"] = df["ActNorm"].fillna(1)

    ## Headache Handling
    # Assume no headache (0) for missing values.
    df["HA_verb"] = df["HA_verb"].fillna(0)
    # If 'HA_verb' is 0 (No Headache) or pre-verbal/non-verba, mark 'HASeverity' as 92 (Not Applicable)
    # If 'HA_verb' is 1 (Headache) and "HASeverity" is missing, mark 'HASeverity' as 92 (Not Applicable)
    df.loc[df["HA_verb"] == 0, "HASeverity"] = 92.0
    df.loc[df["HA_verb"] == 91, "HASeverity"] = 92.0
    df["HASeverity"] = df["HASeverity"].fillna(92.0)
    # If 'HA_verb' is 0 (No Headache) or pre-verbal/non-verba, mark 'HAStart' as 92 (Not Applicable)
    # If 'HA_verb' is 1 (Headache) and "HAStart" is missing, mark 'HAStart' as 92 (Not Applicable)
    df.loc[df["HA_verb"] == 0, "HAStart"] = 92.0
    df.loc[df["HA_verb"] == 91, "HAStart"] = 92.0
    df["HAStart"] = df["HAStart"].fillna(92.0)

    ## Vomiting Handling
    # Assume no vomiting (0) for missing values.
    df["Vomit"] = df["Vomit"].fillna(0)
    # If 'Vomit' is 0 (No Vomiting), mark 'VomitNbr' as 92 (Not Applicable)
    # If 'Vomit' is 1 (Vomiting) and "VomitNbr" is missing, mark 'VomitNbr' as 92 (Not Applicable)
    df["VomitNbr"] = df["VomitNbr"].fillna(92.0)
    df.loc[df["Vomit"] == 0, "VomitNbr"] = 92.0
    # If 'Vomit' is 0 (No Vomiting), mark 'VomitStart' as 92 (Not Applicable)
    # If 'Vomit' is 1 (Vomiting) and "VomitStart" is missing, mark 'VomitStart' as 92 (Not Applicable)
    df["VomitStart"] = df["VomitStart"].fillna(92.0)
    df.loc[df["Vomit"] == 0, "VomitStart"] = 92.0
    # If 'Vomit' is 0 (No Vomiting), mark 'VomitLast' as 92 (Not Applicable)
    # If 'Vomit' is 1 (Vomiting) and "VomitLast" is missing, mark 'VomitLast' as 92 (Not Applicable)
    df["VomitLast"] = df["VomitLast"].fillna(92.0)
    df.loc[df["Vomit"] == 0, "VomitLast"] = 92.0

    ## Dizziness Handling
    # Assume no dizziness (0) for missing values.
    df["Dizzy"] = df["Dizzy"].fillna(0)

    ## Intubation, Paralysis, and Sedation Handling
    # Assume no intubation (0), no paralysis (0), and no sedation (0) for missing values.
    df["Intubated"] = df["Intubated"].fillna(0)
    df["Paralyzed"] = df["Paralyzed"].fillna(0)
    df["Sedated"] = df["Sedated"].fillna(0)

    ## GCS Component Handling (Eye, Verbal, Motor)
    # Fill missing values with the most common (mode) score for each category
    df["GCSEye"] = df["GCSEye"].fillna(df["GCSEye"].mode()[0])
    df["GCSVerbal"] = df["GCSVerbal"].fillna(df["GCSVerbal"].mode()[0])
    df["GCSMotor"] = df["GCSMotor"].fillna(df["GCSMotor"].mode()[0])

    ## GCS Total Score Handling
    # If missing, calculate as sum of individual GCS components
    df["GCSTotal"] = df["GCSTotal"].fillna(df["GCSEye"] + df["GCSVerbal"] + df["GCSMotor"])

    ## GCS Group Handling
    # If GCSTotal is missing, set it to 1 (3-13) or 2 (14-15) based on total score
    df["GCSGroup"] = df["GCSGroup"].fillna(df["GCSTotal"].apply(lambda x: 1 if x < 14 else 2))

    ## Altered Mental Status (AMS) Handling
    # Assume no altered mental status (0) for missing values.
    df["AMS"] = df["AMS"].fillna(0)

    ## AMS Subcategories Handling
    # If AMS is 0 (No Altered Mental Status), mark AMS subcategories as 92 (Not Applicable)
    ams_subcategories = ["AMSAgitated", "AMSSleep", "AMSSlow", "AMSRepeat", "AMSOth"]
    for col in ams_subcategories:
        df[col] = df[col].fillna(92.0)
        df.loc[df["AMS"] == 0, col] = 92.0


    ## Palpable Skull Fracture Handling
    # Assume no palpable skull fracture (0) for missing values.
    df["SFxPalp"] = df["SFxPalp"].fillna(0)
    # If SFxPalp is 0 (No) or 2 (Unclear Exam), mark SFxPalpDepress as 92 (Not Applicable)
    df["SFxPalpDepress"] = df["SFxPalpDepress"].fillna(92.0)
    df.loc[df["SFxPalp"].isin([0, 2]), "SFxPalpDepress"] = 92.0

    ## Anterior Fontanelle Bulging Handling
    # Assume no bulging (0) for missing values.
    df["FontBulg"] = df["FontBulg"].fillna(0)

    ## Basilar Skull Fracture Handling
    # Assume no basilar skull fracture (0) for missing values.
    df["SFxBas"] = df["SFxBas"].fillna(0)

    # If SFxBas is 0 (No), mark SFxBasHem, SFxBasOto, SFxBasPer, SFxBasRet, SFxBasRhi as 92 (Not Applicable)
    basilar_fracture_subcategories = ["SFxBasHem", "SFxBasOto", "SFxBasPer", "SFxBasRet", "SFxBasRhi"]
    for col in basilar_fracture_subcategories:
        df[col] = df[col].fillna(92.0)
        df.loc[df["SFxBas"] == 0, col] = 92.0

    ## Scalp Hematoma Handling
    # Assume no scalp hematoma (0) for missing values.
    df["Hema"] = df["Hema"].fillna(0)
    # If 'Hema' is 0 (No), mark 'HemaLoc' and 'HemaSize' as 92 (Not Applicable).
    df["HemaLoc"] = df["HemaLoc"].fillna(92.0)
    df.loc[df["Hema"] == 0, "HemaLoc"] = 92.0
    df["HemaSize"] = df["HemaSize"].fillna(92.0)
    df.loc[df["Hema"] == 0, "HemaSize"] = 92.0

    ## Trauma Above the Clavicles Handling
    # Assume no trauma above the clavicles (0) for missing values.
    df["Clav"] = df["Clav"].fillna(0)
    # If 'Clav' is 0 (No), mark all related trauma regions as 92 (Not Applicable).
    clavicle_trauma_columns = ["ClavFace", "ClavNeck", "ClavFro", "ClavOcc", "ClavPar", "ClavTem"]
    for col in clavicle_trauma_columns:
        df[col] = df[col].fillna(92.0)
        df.loc[df["Clav"] == 0, col] = 92.0

    ## Neurological Deficits Handling
    # Assume no neurological deficit (0) for missing values.
    df["NeuroD"] = df["NeuroD"].fillna(0)
    # If 'NeuroD' is 0 (No), mark all neurological deficit subcategories as 92 (Not Applicable).
    neuro_deficit_columns = ["NeuroDMotor", "NeuroDSensory", "NeuroDCranial", "NeuroDReflex", "NeuroDOth"]
    for col in neuro_deficit_columns:
        df[col] = df[col].fillna(92.0)
        df.loc[df["NeuroD"] == 0, col] = 92.0

    ## Other (Non-Head) Substantial Injury Handling
    # Assume no substantial injury (0) for missing values.
    df["OSI"] = df["OSI"].fillna(0)

    # If 'OSI' is 0 (No), mark all related substantial injury subcategories as 92 (Not Applicable).
    osi_subcategories = ["OSIExtremity", "OSICut", "OSICspine", "OSIFlank", "OSIAbdomen", "OSIPelvis", "OSIOth"]
    for col in osi_subcategories:
        df[col] = df[col].fillna(92.0)
        df.loc[df["OSI"] == 0, col] = 92.0

    ## Drug/Alcohol Intoxication Handling
    # Assume no suspicion of alcohol/drug intoxication (0) for missing values.
    df["Drugs"] = df["Drugs"].fillna(0)

    ## CT Scan Ordered Handling
    # Assume no head CT, skull x-ray, or head MRI was planned (0) for missing values.
    df["CTForm1"] = df["CTForm1"].fillna(0)

    ## If 'CTForm1' is 0 (No), mark all CT indications as 92 (Not Applicable).
    ct_indication_columns = [
        "IndAge", "IndAmnesia", "IndAMS", "IndClinSFx", "IndHA", "IndHema",
        "IndLOC", "IndMech", "IndNeuroD", "IndRqstMD", "IndRqstParent", "IndRqstTrauma",
        "IndSeiz", "IndVomit", "IndXraySFx", "IndOth", "CTSed", "CTSedAgitate", 
        "CTSedAge", "CTSedRqst", "CTSedOth"
    ]
    for col in ct_indication_columns:
        df[col] = df[col].fillna(92.0)
        df.loc[df["CTForm1"] == 0, col] = 92.0

    ## Age Handling
    # If 'AgeInMonth' is missing, fill with 'AgeInYears'*12
    df["AgeInMonth"] = df["AgeInMonth"].fillna(df["AgeinYears"]*12)
    # If 'AgeinYears' is missing, fill with 'AgeInMonth'*12
    df["AgeinYears"] = df["AgeinYears"].fillna(df["AgeinYears"]//12)
    # If 'AgeTwoPlus' is missing, compute based on 'AgeinYears'
    df["AgeTwoPlus"] = df["AgeTwoPlus"].fillna(df["AgeinYears"].apply(lambda x: 1 if x < 2 else 2))

    # Demographic Information Handling
    # Add new category 90 (No Record), if the values are missing.
    df["Gender"] = df["Gender"].fillna(90.0)
    df["Race"] = df["Race"].fillna(90.0)

    ## ED Observation Handling
    # Assume no ED observation (0) for missing values.
    df["Observed"] = df["Observed"].fillna(0)

    ## ED Disposition Handling
    # Assume Home (1) for missing values.
    df["EDDisposition"] = df["EDDisposition"].fillna(1)

    ## TBI Outcome Handling
    # Assume no death due to TBI (0) for missing values.
    df["DeathTBI"] = df["DeathTBI"].fillna(0)

    # Assume no hospitalization for 2+ nights due to head injury (0) for missing values.
    df["HospHead"] = df["HospHead"].fillna(0)

    # Assume no hospitalization for 2+ nights with TBI on CT (0) for missing values.
    df["HospHeadPosCT"] = df["HospHeadPosCT"].fillna(0)

    # Assume no intubation >24 hours for head trauma (0) for missing values.
    df["Intub24Head"] = df["Intub24Head"].fillna(0)

    # Assume no neurosurgery (0) for missing values.
    df["Neurosurgery"] = df["Neurosurgery"].fillna(0)

    ## Clinically-Important TBI (ciTBI) Handling
    # If missing, determine based on definition: ciTBI is present if any of the following are true:
    # (1) Neurosurgery performed
    # (2) Intubated > 24 hours for head trauma
    # (3) Death due to TBI or in the ED
    # (4) Hospitalized for ≥2 nights due to head injury and had a TBI on CT
    df["PosIntFinal"] = df["PosIntFinal"].fillna(
        df.apply(lambda row: 1 if (row["Neurosurgery"] == 1 or 
                                    row["Intub24Head"] == 1 or 
                                    row["DeathTBI"] == 1 or 
                                    row["HospHeadPosCT"] == 1) else 0, axis=1)
    )

    print("Completed filling missing values.")
    return df


def logic_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs context-based validation by:
    1. Removing redundant variables.
    2. Identifying and eliminating records with logical inconsistencies.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to be validated.

    Returns:
    --------
    pd.DataFrame
        The cleaned dataset with inconsistencies and redundancies removed.
    """

    # 1️⃣ Remove Redundant Variables (e.g., Age Representation)
    redundant_columns = ["AgeinYears", "AgeTwoPlus"]  # Keep "AgeInMonth" as it holds the most information
    df = df.drop(columns=[col for col in redundant_columns if col in df.columns], errors="ignore")
    print(f"Removed redundant columns: {redundant_columns}")

    # 2️⃣ Identify Logically Inconsistent Records

    # Define logical inconsistency conditions
    inconsistency_conditions = [
        # If Hema (Scalp Hematoma) is present, HemaLoc and HemaSize must not be 'Not Applicable' (92)
        (df["Hema"] == 1) & ((df["HemaLoc"] == 92) | (df["HemaSize"] == 92)),

        # If Seizure (Seiz) is present, SeizOccur and SeizLen should not be 'Not Applicable' (92)
        (df["Seiz"] == 1) & ((df["SeizOccur"] == 92) | (df["SeizLen"] == 92)),

        # If Vomiting (Vomit) is present, VomitNbr, VomitStart, and VomitLast should not be 'Not Applicable' (92)
        (df["Vomit"] == 1) & ((df["VomitNbr"] == 92) | (df["VomitStart"] == 92) | (df["VomitLast"] == 92)),

        # If Neurological Deficits (NeuroD) is present, subcategories should not be 'Not Applicable' (92)
        (df["NeuroD"] == 1) & (
            (df["NeuroDMotor"] == 92) | 
            (df["NeuroDSensory"] == 92) | 
            (df["NeuroDCranial"] == 92) | 
            (df["NeuroDReflex"] == 92) | 
            (df["NeuroDOth"] == 92)
        ),

        # If Skull Fracture (SFxBas) is present, its subcategories should not be 'Not Applicable' (92)
        (df["SFxBas"] == 1) & (
            (df["SFxBasHem"] == 92) | 
            (df["SFxBasOto"] == 92) | 
            (df["SFxBasPer"] == 92) | 
            (df["SFxBasRet"] == 92) | 
            (df["SFxBasRhi"] == 92)
        )
    ]

    # Combine all conditions to filter invalid rows
    inconsistent_rows = df[sum(inconsistency_conditions) > 0]
    print(f"Removing {len(inconsistent_rows)} records due to logical inconsistencies.")

    # Remove inconsistent rows
    df = df[~(sum(inconsistency_conditions) > 0)].reset_index(drop=True)

    print("Completed logic-based validation and downsizing.")
    return df

