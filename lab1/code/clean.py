import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from typing import Dict, List, Union


# Maps database column names to descriptive variable names
def rename_columns(df):
    """Maps original column names to more descriptive names"""
    rename_dict = {
        "PatNum": "patient_id",
        "EmplType": "physician_position",
        "Certification": "physician_certification",
        "InjuryMech": "mechanism_injury",
        "High_impact_InjSev": "injury_severity_high_impact",
        "Amnesia_verb": "amnesia_present",
        "LOCSeparate": "loss_of_consciousness",
        "LocLen": "unconscious_duration",
        "Seiz": "posttraumatic_seizure",
        "SeizOccur": "time_of_seizure",
        "SeizLen": "duration_of_seizure",
        "ActNorm": "acting_normal_after_injury",
        "HA_verb": "headache_reported",
        "HASeverity": "headache_intensity",
        "HAStart": "headache_onset_time",
        "Vomit": "vomiting_reported",
        "VomitNbr": "vomiting_episode_count",
        "VomitStart": "vomiting_onset_time",
        "VomitLast": "last_vomiting_time",
        "Dizzy": "dizziness_reported",
        "Intubated": "intubation_status",
        "Paralyzed": "pharmacologically_paralyzed",
        "Sedated": "pharmacologically_sedated",
        "GCSEye": "gcs_eye",
        "GCSVerbal": "gcs_verbal",
        "GCSMotor": "gcs_motor",
        "GCSTotal": "gcs_total",
        "GCSGroup": "gcs_group",
        "AMS": "altered_mental_status",
        "AMSAgitated": "ams_agitated",
        "AMSSleep": "ams_sleepy",
        "AMSSlow": "ams_slow_response",
        "AMSRepeat": "ams_repetitive_questions",
        "AMSOth": "ams_other",
        "SFxPalp": "palpable_skull_fracture",
        "SFxPalpDepress": "palpable_skull_fracture_depressed",
        "FontBulg": "fontanelle_bulging",
        "SFxBas": "basilar_skull_fracture_signs",
        "SFxBasHem": "basilar_skull_fracture_hemotympanum",
        "SFxBasOto": "basilar_skull_fracture_otorrhea",
        "SFxBasPer": "basilar_skull_fracture_periorbital_ecchymosis",
        "SFxBasRet": "basilar_skull_fracture_retroauricular_ecchymosis",
        "SFxBasRhi": "basilar_skull_fracture_rhinorrhea",
        "Hema": "scalp_hematoma_present",
        "HemaLoc": "scalp_hematoma_location",
        "HemaSize": "scalp_hematoma_size",
        "Clav": "trauma_above_clavicles",
        "ClavFace": "trauma_face",
        "ClavNeck": "trauma_neck",
        "ClavFro": "trauma_scalp_frontal",
        "ClavOcc": "trauma_scalp_occipital",
        "ClavPar": "trauma_scalp_parietal",
        "ClavTem": "trauma_scalp_temporal",
        "NeuroD": "neurological_deficit_present",
        "NeuroDMotor": "neurological_deficit_motor",
        "NeuroDSensory": "neurological_deficit_sensory",
        "NeuroDCranial": "neurological_deficit_cranial_nerve",
        "NeuroDReflex": "neurological_deficit_reflexes",
        "NeuroDOth": "neurological_deficit_other",
        "OSI": "other_substantial_injury",
        "OSIExtremity": "other_injury_extremity",
        "OSICut": "other_injury_laceration",
        "OSICspine": "other_injury_cspine",
        "OSIFlank": "other_injury_chest_back_flank",
        "OSIAbdomen": "other_injury_abdomen",
        "OSIPelvis": "other_injury_pelvis",
        "OSIOth": "other_injury_other",
        "Drugs": "drug_or_alcohol_suspicion",
        "CTForm1": "head_ct_planned",
        "IndAge": "ct_reason_age",
        "IndAmnesia": "ct_reason_amnesia",
        "IndAMS": "ct_reason_ams",
        "IndClinSFx": "ct_reason_clinical_skull_fracture",
        "IndHA": "ct_reason_headache",
        "IndHema": "ct_reason_scalp_hematoma",
        "IndLOC": "ct_reason_loss_of_consciousness",
        "IndMech": "ct_reason_mechanism",
        "IndNeuroD": "ct_reason_neurological_deficit",
        "IndRqstMD": "ct_reason_request_md",
        "IndRqstParent": "ct_reason_request_parent",
        "IndRqstTrauma": "ct_reason_request_trauma",
        "IndSeiz": "ct_reason_seizure",
        "IndVomit": "ct_reason_vomiting",
        "IndXraySFx": "ct_reason_xray_skull_fracture",
        "IndOth": "ct_reason_other",
        "CTSed": "sedation_for_ct",
        "CTSedAgitate": "sedation_for_ct_agitated",
        "CTSedAge": "sedation_for_ct_age",
        "CTSedRqst": "sedation_for_ct_tech_request",
        "CTSedOth": "sedation_for_ct_other",
        "AgeInMonth": "age_in_months",
        "AgeinYears": "age_in_years",
        "AgeTwoPlus": "age_category",
        "Gender": "gender",
        "Ethnicity": "ethnicity",
        "Race": "race",
        "Observed": "patient_observed_in_ed",
        "EDDisposition": "ed_disposition",
        "CTDone": "head_ct_done",
        "EDCT": "head_ct_performed_in_ed",
        "PosCT": "tbi_detected_on_ct",
        "Finding1": "cerebellar_hemorrhage",
        "Finding2": "cerebral_contusion",
        "Finding3": "cerebral_edema",
        "Finding4": "cerebral_hemorrhage",
        "Finding5": "skull_diastasis",
        "Finding6": "epidural_hematoma",
        "Finding7": "extra_axial_hematoma",
        "Finding8": "intraventricular_hemorrhage",
        "Finding9": "midline_shift",
        "Finding10": "pneumocephalus",
        "Finding11": "skull_fracture",
        "Finding12": "subarachnoid_hemorrhage",
        "Finding13": "subdural_hematoma",
        "Finding14": "traumatic_infarction",
        "Finding20": "diffuse_axonal_injury",
        "Finding21": "herniation",
        "Finding22": "shear_injury",
        "Finding23": "sigmoid_sinus_thrombosis",
        "DeathTBI": "death_due_to_tbi",
        "HospHead": "hospitalized_due_to_head_injury",
        "HospHeadPosCT": "hospitalized_head_injury_tbi_ct",
        "Intub24Head": "intubated_over_24hrs",
        "Neurosurgery": "neurosurgery",
        "PosIntFinal": "clinically_important_tbi",
    }
    return df.rename(columns=rename_dict)


def rename_columns_plotting(df):
    rename_dict = {
        "patient_id": "Patient ID",
        "physician_position": "Physician Position",
        "physician_certification": "Physician Certification",
        "mechanism_injury": "Mechanism of Injury",
        "injury_severity_high_impact": "High-Impact Injury Severity",
        "amnesia_present": "Amnesia Present",
        "loss_of_consciousness": "Loss of Consciousness",
        "unconscious_duration": "Duration of Unconsciousness",
        "posttraumatic_seizure": "Post-Traumatic Seizure",
        "time_of_seizure": "Time of Seizure",
        "duration_of_seizure": "Duration of Seizure",
        "acting_normal_after_injury": "Acting Normal After Injury",
        "headache_reported": "Headache Reported",
        "headache_intensity": "Headache Intensity",
        "headache_onset_time": "Headache Onset Time",
        "vomiting_reported": "Vomiting Reported",
        "vomiting_episode_count": "Number of Vomiting Episodes",
        "vomiting_onset_time": "Vomiting Onset Time",
        "last_vomiting_time": "Last Vomiting Episode",
        "dizziness_reported": "Dizziness Reported",
        "intubation_status": "Intubation Status",
        "pharmacologically_paralyzed": "Pharmacologically Paralyzed",
        "pharmacologically_sedated": "Pharmacologically Sedated",
        "gcs_eye": "GCS Eye Response",
        "gcs_verbal": "GCS Verbal Response",
        "gcs_motor": "GCS Motor Response",
        "gcs_total": "Total GCS Score",
        "gcs_group": "GCS Group",
        "altered_mental_status": "Altered Mental Status",
        "ams_agitated": "AMS - Agitated",
        "ams_sleepy": "AMS - Sleepy",
        "ams_slow_response": "AMS - Slow Response",
        "ams_repetitive_questions": "AMS - Repetitive Questions",
        "ams_other": "AMS - Other",
        "palpable_skull_fracture": "Palpable Skull Fracture",
        "palpable_skull_fracture_depressed": "Depressed Skull Fracture",
        "fontanelle_bulging": "Bulging Fontanelle",
        "basilar_skull_fracture_signs": "Signs of Basilar Skull Fracture",
        "basilar_skull_fracture_hemotympanum": "Hemotympanum",
        "basilar_skull_fracture_otorrhea": "CSF Otorrhea",
        "basilar_skull_fracture_periorbital_ecchymosis": "Periorbital Ecchymosis",
        "basilar_skull_fracture_retroauricular_ecchymosis": "Retroauricular Ecchymosis",
        "basilar_skull_fracture_rhinorrhea": "CSF Rhinorrhea",
        "scalp_hematoma_present": "Scalp Hematoma Present",
        "scalp_hematoma_location": "Scalp Hematoma Location",
        "scalp_hematoma_size": "Scalp Hematoma Size",
        "trauma_above_clavicles": "Trauma Above Clavicles",
        "trauma_face": "Facial Trauma",
        "trauma_neck": "Neck Trauma",
        "trauma_scalp_frontal": "Frontal Scalp Trauma",
        "trauma_scalp_occipital": "Occipital Scalp Trauma",
        "trauma_scalp_parietal": "Parietal Scalp Trauma",
        "trauma_scalp_temporal": "Temporal Scalp Trauma",
        "neurological_deficit_present": "Neurological Deficit Present",
        "neurological_deficit_motor": "Motor Neurological Deficit",
        "neurological_deficit_sensory": "Sensory Neurological Deficit",
        "neurological_deficit_cranial_nerve": "Cranial Nerve Deficit",
        "neurological_deficit_reflexes": "Reflex Deficit",
        "neurological_deficit_other": "Other Neurological Deficit",
        "other_substantial_injury": "Other Substantial Injury",
        "other_injury_extremity": "Extremity Injury",
        "other_injury_laceration": "Laceration",
        "other_injury_cspine": "Cervical Spine Injury",
        "other_injury_chest_back_flank": "Chest/Back/Flank Injury",
        "other_injury_abdomen": "Abdominal Injury",
        "other_injury_pelvis": "Pelvic Injury",
        "other_injury_other": "Other Injury",
        "drug_or_alcohol_suspicion": "Drug/Alcohol Suspicion",
        "head_ct_planned": "Head CT Planned",
        "ct_reason_age": "CT Reason: Age",
        "ct_reason_amnesia": "CT Reason: Amnesia",
        "ct_reason_ams": "CT Reason: Altered Mental Status",
        "ct_reason_clinical_skull_fracture": "CT Reason: Clinical Skull Fracture",
        "ct_reason_headache": "CT Reason: Headache",
        "ct_reason_scalp_hematoma": "CT Reason: Scalp Hematoma",
        "ct_reason_loss_of_consciousness": "CT Reason: Loss of Consciousness",
        "ct_reason_mechanism": "CT Reason: Mechanism of Injury",
        "ct_reason_neurological_deficit": "CT Reason: Neurological Deficit",
        "ct_reason_request_md": "CT Reason: Physician Request",
        "ct_reason_request_parent": "CT Reason: Parent Request",
        "ct_reason_request_trauma": "CT Reason: Trauma Team Request",
        "ct_reason_seizure": "CT Reason: Seizure",
        "ct_reason_vomiting": "CT Reason: Vomiting",
        "ct_reason_xray_skull_fracture": "CT Reason: Skull Fracture on X-Ray",
        "ct_reason_other": "CT Reason: Other",
        "sedation_for_ct": "Sedation for CT",
        "sedation_for_ct_agitated": "Sedation for CT - Agitated",
        "sedation_for_ct_age": "Sedation for CT - Age Consideration",
        "sedation_for_ct_tech_request": "Sedation for CT - Technician Request",
        "sedation_for_ct_other": "Sedation for CT - Other",
        "age_in_months": "Age in Months",
        "age_in_years": "Age in Years",
        "age_category": "Age Category",
        "gender": "Gender",
        "ethnicity": "Ethnicity",
        "race": "Race",
        "patient_observed_in_ed": "Patient Observed in ED",
        "ed_disposition": "ED Disposition",
        "head_ct_done": "Head CT Done",
        "head_ct_performed_in_ed": "Head CT Performed in ED",
        "tbi_detected_on_ct": "TBI Detected on CT",
        "cerebellar_hemorrhage": "Cerebellar Hemorrhage",
        "cerebral_contusion": "Cerebral Contusion",
        "cerebral_edema": "Cerebral Edema",
        "cerebral_hemorrhage": "Cerebral Hemorrhage",
        "skull_diastasis": "Skull Diastasis",
        "epidural_hematoma": "Epidural Hematoma",
        "extra_axial_hematoma": "Extra-Axial Hematoma",
        "intraventricular_hemorrhage": "Intraventricular Hemorrhage",
        "midline_shift": "Midline Shift",
        "pneumocephalus": "Pneumocephalus",
        "skull_fracture": "Skull Fracture",
        "subarachnoid_hemorrhage": "Subarachnoid Hemorrhage",
        "subdural_hematoma": "Subdural Hematoma",
        "traumatic_infarction": "Traumatic Infarction",
        "diffuse_axonal_injury": "Diffuse Axonal Injury",
        "herniation": "Herniation",
        "shear_injury": "Shear Injury",
        "sigmoid_sinus_thrombosis": "Sigmoid Sinus Thrombosis",
        "death_due_to_tbi": "Death Due to TBI",
        "hospitalized_due_to_head_injury": "Hospitalized Due to Head Injury",
        "intubated_over_24hrs": "Intubated Over 24 Hours",
        "neurosurgery": "Neurosurgery",
        "clinically_important_tbi": "Clinically Important TBI",
    }
    return df.rename(columns=rename_dict)


def validate_dataframe_values(df):
    """Checks dataframe values against allowed values and returns validation results"""

    allowed_values = {
        "patient_id": "numeric",
        "physician_position": [1, 2, 3, 4, 5],
        "physician_certification": [1, 2, 3, 4, 90],
        "mechanism_injury": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 90],
        "injury_severity_high_impact": [1, 2, 3],
        "amnesia_present": [0, 1, 91],
        "loss_of_consciousness": [0, 1, 2],
        "unconscious_duration": [1, 2, 3, 4, 92],
        "posttraumatic_seizure": [0, 1],
        "time_of_seizure": [1, 2, 3, 92],
        "duration_of_seizure": [1, 2, 3, 4, 92],
        "acting_normal_after_injury": [0, 1],
        "headache_reported": [0, 1, 91],
        "headache_intensity": [1, 2, 3, 92],
        "headache_onset_time": [1, 2, 3, 4, 92],
        "vomiting_reported": [0, 1],
        "vomiting_episode_count": [1, 2, 3, 92],
        "vomiting_onset_time": [1, 2, 3, 4, 92],
        "last_vomiting_time": [1, 2, 3, 92],
        "dizziness_reported": [0, 1],
        "intubation_status": [0, 1],
        "pharmacologically_paralyzed": [0, 1],
        "pharmacologically_sedated": [0, 1],
        "gcs_eye": [1, 2, 3, 4],
        "gcs_verbal": [1, 2, 3, 4, 5],
        "gcs_motor": [1, 2, 3, 4, 5, 6],
        "gcs_total": "numeric",
        "gcs_group": [1, 2],
        "altered_mental_status": [0, 1],
        "ams_agitated": [0, 1, 92],
        "ams_sleepy": [0, 1, 92],
        "ams_slow_response": [0, 1, 92],
        "ams_repetitive_questions": [0, 1, 92],
        "ams_other": [0, 1, 92],
        "palpable_skull_fracture": [0, 1, 2],
        "palpable_skull_fracture_depressed": [0, 1, 92],
        "fontanelle_bulging": [0, 1],
        "basilar_skull_fracture_signs": [0, 1],
        "basilar_skull_fracture_hemotympanum": [0, 1, 92],
        "basilar_skull_fracture_otorrhea": [0, 1, 92],
        "basilar_skull_fracture_periorbital_ecchymosis": [0, 1, 92],
        "basilar_skull_fracture_retroauricular_ecchymosis": [0, 1, 92],
        "basilar_skull_fracture_rhinorrhea": [0, 1, 92],
        "scalp_hematoma_present": [0, 1],
        "scalp_hematoma_location": [1, 2, 3, 92],
        "scalp_hematoma_size": [1, 2, 3, 92],
        "trauma_above_clavicles": [0, 1],
        "trauma_face": [0, 1, 92],
        "trauma_neck": [0, 1, 92],
        "trauma_scalp_frontal": [0, 1, 92],
        "trauma_scalp_occipital": [0, 1, 92],
        "trauma_scalp_parietal": [0, 1, 92],
        "trauma_scalp_temporal": [0, 1, 92],
        "neurological_deficit_present": [0, 1],
        "neurological_deficit_motor": [0, 1, 92],
        "neurological_deficit_sensory": [0, 1, 92],
        "neurological_deficit_cranial_nerve": [0, 1, 92],
        "neurological_deficit_reflexes": [0, 1, 92],
        "neurological_deficit_other": [0, 1, 92],
        "other_substantial_injury": [0, 1],
        "other_injury_extremity": [0, 1, 92],
        "other_injury_laceration": [0, 1, 92],
        "other_injury_cspine": [0, 1, 92],
        "other_injury_chest_back_flank": [0, 1, 92],
        "other_injury_abdomen": [0, 1, 92],
        "other_injury_pelvis": [0, 1, 92],
        "other_injury_other": [0, 1, 92],
        "drug_or_alcohol_suspicion": [0, 1],
        "head_ct_planned": [0, 1],
        "ct_reason_age": [0, 1, 92],
        "ct_reason_amnesia": [0, 1, 92],
        "ct_reason_ams": [0, 1, 92],
        "ct_reason_clinical_skull_fracture": [0, 1, 92],
        "ct_reason_headache": [0, 1, 92],
        "ct_reason_scalp_hematoma": [0, 1, 92],
        "ct_reason_loss_of_consciousness": [0, 1, 92],
        "ct_reason_mechanism": [0, 1, 92],
        "ct_reason_neurological_deficit": [0, 1, 92],
        "ct_reason_request_md": [0, 1, 92],
        "ct_reason_request_parent": [0, 1, 92],
        "ct_reason_request_trauma": [0, 1, 92],
        "ct_reason_seizure": [0, 1, 92],
        "ct_reason_vomiting": [0, 1, 92],
        "ct_reason_xray_skull_fracture": [0, 1, 92],
        "ct_reason_other": [0, 1, 92],
        "sedation_for_ct": [0, 1, 92],
        "sedation_for_ct_agitated": [0, 1, 92],
        "sedation_for_ct_age": [0, 1, 92],
        "sedation_for_ct_tech_request": [0, 1, 92],
        "sedation_for_ct_other": [0, 1, 92],
        "age_in_months": "numeric",
        "age_in_years": "numeric",
        "age_category": [1, 2],
        "gender": [1, 2],
        "ethnicity": [1, 2],
        "race": [1, 2, 3, 4, 5, 90],
        "patient_observed_in_ed": [0, 1],
        "ed_disposition": [1, 2, 3, 4, 5, 6, 7, 8, 90],
        "head_ct_done": [0, 1],
        "head_ct_performed_in_ed": [0, 1, 92],
        "tbi_detected_on_ct": [0, 1, 92],
        "cerebellar_hemorrhage": [0, 1, 92],
        "cerebral_contusion": [0, 1, 92],
        "cerebral_edema": [0, 1, 92],
        "cerebral_hemorrhage": [0, 1, 92],
        "skull_diastasis": [0, 1, 92],
        "epidural_hematoma": [0, 1, 92],
        "extra_axial_hematoma": [0, 1, 92],
        "intraventricular_hemorrhage": [0, 1, 92],
        "midline_shift": [0, 1, 92],
        "pneumocephalus": [0, 1, 92],
        "skull_fracture": [0, 1, 92],
        "subarachnoid_hemorrhage": [0, 1, 92],
        "subdural_hematoma": [0, 1, 92],
        "traumatic_infarction": [0, 1, 92],
        "diffuse_axonal_injury": [0, 1, 92],
        "herniation": [0, 1, 92],
        "shear_injury": [0, 1, 92],
        "sigmoid_sinus_thrombosis": [0, 1, 92],
        "death_due_to_tbi": [0, 1],
        "hospitalized_due_to_head_injury": [0, 1],
        "hospitalized_head_injury_tbi_ct": [0, 1],
        "intubated_over_24hrs": [0, 1],
        "neurosurgery": [0, 1],
        "clinically_important_tbi": [0, 1],
    }

    validation_results = {}

    for column, valid_values in allowed_values.items():
        if column not in df.columns:
            validation_results[column] = ["Column not found in dataframe"]
            continue

        if valid_values == "numeric":
            non_numeric = df[
                ~df[column].isna()
                & (~df[column].astype(str).str.replace(".", "").str.isnumeric())
            ]
            if not non_numeric.empty:
                validation_results[column] = list(non_numeric[column].unique())
            continue

        invalid_values = list(set(df[column].dropna().unique()) - set(valid_values))
        if invalid_values:
            validation_results[column] = invalid_values

    return validation_results


def validate_dataset(df):
    """Validates dataset against allowed values and prints validation report"""
    try:
        results = validate_dataframe_values(df)

        if results:
            print("\nInvalid values found:")
            for column, invalid_values in results.items():
                print(f"\nColumn '{column}':")
                print(f"Invalid values: {invalid_values}")

                if column in df.columns and invalid_values != [
                    "Column not found in dataframe"
                ]:
                    invalid_mask = df[column].isin(invalid_values)
                    invalid_count = invalid_mask.sum()
                    total_count = len(df)
                    print(
                        f"Number of invalid values: {invalid_count} ({(invalid_count / total_count) * 100:.2f}% of total)"
                    )
        else:
            print("All values are valid!")
    except Exception as e:
        print(f"Error validating dataset: {str(e)}")


def fill_gcs_scores(row, medians):
    """Updates missing Glasgow Coma Scale scores maintaining eye + verbal + motor = total"""
    gcs_cols = ["gcs_eye", "gcs_verbal", "gcs_motor", "gcs_total"]
    scores = [row[col] for col in gcs_cols]
    missing_count = sum(pd.isnull(score) for score in scores)

    if missing_count == 0:
        return row

    eye, verbal, motor, total = scores

    # Single missing value: calculate from other components
    if missing_count == 1:
        if pd.isnull(total):
            row["gcs_total"] = eye + verbal + motor
        elif pd.isnull(eye):
            row["gcs_eye"] = total - verbal - motor
        elif pd.isnull(verbal):
            row["gcs_verbal"] = total - eye - motor
        elif pd.isnull(motor):
            row["gcs_motor"] = total - eye - verbal
        return row

    # Multiple missing values: use medians and scale to match total
    eye = medians["gcs_eye"] if pd.isnull(eye) else eye
    verbal = medians["gcs_verbal"] if pd.isnull(verbal) else verbal
    motor = medians["gcs_motor"] if pd.isnull(motor) else motor

    if pd.isnull(total):
        total = eye + verbal + motor
    else:
        sum_components = eye + verbal + motor
        if sum_components != 0:
            scale = total / sum_components
            eye = round(eye * scale)
            verbal = round(verbal * scale)
            motor = round(motor * scale)
            motor += total - (eye + verbal + motor)

    row[gcs_cols] = [eye, verbal, motor, total]
    return row


def impute_gcs_columns(df):
    """Fills missing GCS values using known relationships and medians"""
    # Perfect GCS score imputation
    perfect_score = df["gcs_total"] == 15
    df.loc[perfect_score & df["gcs_eye"].isna(), "gcs_eye"] = 4
    df.loc[perfect_score & df["gcs_verbal"].isna(), "gcs_verbal"] = 5
    df.loc[perfect_score & df["gcs_motor"].isna(), "gcs_motor"] = 6

    medians = {
        col: df[col].median(skipna=True)
        for col in ["gcs_eye", "gcs_verbal", "gcs_motor"]
    }
    return df.apply(lambda row: fill_gcs_scores(row, medians), axis=1)


def impute_group_data(df, main_col, subcols, default_value=92, main_default=0):
    """Imputes related columns with default values and updates main indicator"""
    df[subcols] = df[subcols].fillna(default_value)
    df.loc[
        df[main_col].isna() & df[subcols].eq(default_value).all(axis=1), main_col
    ] = main_default
    return df


def convert_numeric_columns(df):
    """Converts string columns to integers where possible"""
    for col in df.columns:
        try:
            numeric_col = pd.to_numeric(df[col], errors="raise")
            if ((numeric_col.dropna() % 1) == 0).all():
                df[col] = numeric_col.astype("Int64")
        except Exception:
            continue
    return df


def clean_data():
    """Main function to clean and preprocess the TBI dataset"""
    # Load and preprocess data
    df = pd.read_csv("../data/TBI PUD 10-08-2013.csv", dtype=str)
    df = rename_columns(df)
    df = convert_numeric_columns(df)
    df = impute_gcs_columns(df)

    # Define groups of related columns for imputation
    group_imputations = {
        "vomiting_reported": [
            "vomiting_episode_count",
            "vomiting_onset_time",
            "last_vomiting_time",
        ],
        "headache_reported": ["headache_intensity", "headache_onset_time"],
        "altered_mental_status": [
            "ams_agitated",
            "ams_sleepy",
            "ams_slow_response",
            "ams_repetitive_questions",
            "ams_other",
        ],
        "palpable_skull_fracture": ["palpable_skull_fracture_depressed"],
        "basilar_skull_fracture_signs": [
            "basilar_skull_fracture_hemotympanum",
            "basilar_skull_fracture_otorrhea",
            "basilar_skull_fracture_periorbital_ecchymosis",
            "basilar_skull_fracture_retroauricular_ecchymosis",
            "basilar_skull_fracture_rhinorrhea",
        ],
        "scalp_hematoma_present": ["scalp_hematoma_location", "scalp_hematoma_size"],
        "trauma_above_clavicles": [
            "trauma_face",
            "trauma_neck",
            "trauma_scalp_frontal",
            "trauma_scalp_occipital",
            "trauma_scalp_parietal",
            "trauma_scalp_temporal",
        ],
        "neurological_deficit_present": [
            "neurological_deficit_motor",
            "neurological_deficit_sensory",
            "neurological_deficit_cranial_nerve",
            "neurological_deficit_reflexes",
            "neurological_deficit_other",
        ],
        "other_substantial_injury": [
            "other_injury_extremity",
            "other_injury_laceration",
            "other_injury_cspine",
            "other_injury_chest_back_flank",
            "other_injury_abdomen",
            "other_injury_pelvis",
            "other_injury_other",
        ],
        "sedation_for_ct": [
            "sedation_for_ct_agitated",
            "sedation_for_ct_age",
            "sedation_for_ct_tech_request",
            "sedation_for_ct_other",
        ],
        "head_ct_planned": [
            "ct_reason_age",
            "ct_reason_amnesia",
            "ct_reason_ams",
            "ct_reason_clinical_skull_fracture",
            "ct_reason_headache",
            "ct_reason_scalp_hematoma",
            "ct_reason_loss_of_consciousness",
            "ct_reason_mechanism",
            "ct_reason_neurological_deficit",
            "ct_reason_request_md",
            "ct_reason_request_parent",
            "ct_reason_request_trauma",
            "ct_reason_seizure",
            "ct_reason_vomiting",
            "ct_reason_xray_skull_fracture",
            "ct_reason_other",
        ],
    }

    # Impute grouped columns
    for main_col, subcols in group_imputations.items():
        df = impute_group_data(df, main_col, subcols)

    # Impute individual columns with specific values
    single_imputations = {
        "mechanism_injury": 90,
        "injury_severity_high_impact": 2,
        "amnesia_present": 91,
        "loss_of_consciousness": 0,
        "unconscious_duration": 92,
        "posttraumatic_seizure": 0,
        "time_of_seizure": 92,
        "duration_of_seizure": 92,
        "acting_normal_after_injury": 90,
        "intubation_status": 0,
        "pharmacologically_paralyzed": 0,
        "pharmacologically_sedated": 0,
        "drug_or_alcohol_suspicion": 0,
        "patient_observed_in_ed": 0,
        "fontanelle_bulging": 0,
        "ed_disposition": 90,
        "gender": 1,
        "physician_position": 3,
    }

    for col, value in single_imputations.items():
        df[col] = df[col].fillna(value)

    # Drop rows with missing critical data
    critical_columns = [
        "clinically_important_tbi",
        "death_due_to_tbi",
        "hospitalized_due_to_head_injury",
        "intubated_over_24hrs",
        "neurosurgery",
    ]
    df = df.dropna(subset=critical_columns)

    # Remove unnecessary columns
    df = df.drop(columns=["dizziness_reported", "ethnicity", "race"])

    # Save processed data
    df.to_csv("../data/dataframe_after_cleaning.csv", index=False)

    return df


clean_data()
