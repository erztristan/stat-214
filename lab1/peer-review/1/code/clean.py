import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def clean_data(df):
    # Exclude patients with `GCSTotal` 3–13 and missing primary outcome. 
    df = df[~((df.GCSTotal<=13)|(df.PosIntFinal.isna()))]
    # Drop columns based on the discussion in the EDA part
    df.drop(['DeathTBI','HospHeadPosCT','Intub24Head','GCSGroup','EmplType','Certification','GCSEye', 'GCSVerbal',
              'GCSMotor', 'Intubated','Paralyzed','Sedated','AgeinYears', 'HospHead','Dizzy','Ethnicity'],
              inplace=True,axis=1, errors='ignore')
    
    # Adjust the HA_verb and Amnesia_verb
    df.loc[(df['HA_verb']==91)|(df['Amnesia_verb']==91), 'Amnesia_verb'] = 91
    df.loc[(df['HA_verb']==91)|(df['Amnesia_verb']==91), 'HA_verb'] = 91
    df.loc[df['AgeInMonth']<12, 'HA_verb'] = 91
    df.loc[df['AgeInMonth']<12, 'Amnesia_verb'] = 91

    # Transfer values from 1/2 to 0/1
    df.loc[df['AgeInMonth']>=24, 'AgeTwoPlus'] = 1
    df.loc[df['AgeInMonth']<24, 'AgeTwoPlus'] = 0
    df.loc[df['Gender']==1, 'Gender'] = 0
    df.loc[df['Gender']==2, 'Gender'] = 1
    # df.loc[df['Ethnicity']==1, 'Ethnicity'] = 0
    # df.loc[df['Ethnicity']==2, 'Ethnicity'] = 1

    # Adjust the contradiction in Ind variables
    df.loc[(df.Seiz==0)&(df.IndSeiz==1),'IndSeiz'] = 0
    df.loc[(df.Vomit==0)&(df.IndVomit==1),'IndVomit'] = 0
    df.loc[(df.HA_verb==0)&(df.IndHA==1),'IndHA'] = 0
    df.loc[(df.Hema==0)&(df.IndHema==1),'IndHema'] = 0

    # Add new variables to demonstrate findings in age and outcome
    df['age_indicator'] = (df['AgeInMonth'] <= 12).astype(int)
    df['age_indicator_inter'] = df['age_indicator'] * df['AgeInMonth']

    # Drop variables that cannot be used as an indicator for CT test
    df.drop([c for c in df.columns if 'Ind' in c], axis=1, inplace=True, errors='ignore')
    df.drop([c for c in df.columns if 'Finding' in c], axis=1, inplace=True, errors='ignore')
    df.drop([c for c in df.columns if 'CT'==c[:2]], axis=1, inplace=True, errors='ignore')
    df.drop(['PatNum', 'EDCT', 'PosCT', 'EDDisposition','Observed'], axis=1, inplace=True, errors='ignore')
    
    # Old and new names
    rename_dict = {
        "InjuryMech": "InjuryMechanism",
        "High_impact_InjSev": "InjurySeverity",
        "Amnesia_verb": "HasAmnesia",
        "LOCSeparate": "LossOfConsciousness",
        "LocLen": "LOC_Duration",
        "Seiz": "PostTraumaticSeizure",
        "SeizOccur": "SeizureTiming",
        "SeizLen": "SeizureDuration",
        "ActNorm": "ParentalAssessment",
        "HA_verb": "HeadachePresent",
        "HASeverity": "HeadacheSeverity",
        "HAStart": "HeadacheOnset",
        "Vomit": "VomitingPresent",
        "VomitNbr": "VomitingEpisodes",
        "VomitStart": "VomitingOnset",
        "VomitLast": "LastVomitingEpisode",
        "Dizzy": "DizzinessPresent",
        "GCSTotal": "GCS_Total",
        "AMS": "AlteredMentalStatus",
        "AMSAgitated": "AMS_Agitated",
        "AMSSleep": "AMS_Sleepy",
        "AMSSlow": "AMS_SlowResponse",
        "AMSRepeat": "AMS_RepetitiveQuestions",
        "AMSOth": "AMS_Other",
        "SFxPalp": "PalpableSkullFracture",
        "SFxPalpDepress": "DepressedSkullFracture",
        "FontBulg": "FontanelleBulging",
        "SFxBas": "BasilarSkullFracture",
        "SFxBasHem": "BasilarFracture_Hemotympanum",
        "SFxBasOto": "BasilarFracture_CSF_Otorrhea",
        "SFxBasPer": "BasilarFracture_PeriorbitalEcchymosis",
        "SFxBasRet": "BasilarFracture_RetroauricularEcchymosis",
        "SFxBasRhi": "BasilarFracture_CSF_Rhinorrhea",
        "Hema": "ScalpHematoma",
        "HemaLoc": "HematomaLocation",
        "HemaSize": "HematomaSize",
        "Clav": "TraumaAboveClavicle",
        "ClavFace": "TraumaFace",
        "ClavNeck": "TraumaNeck",
        "ClavFro": "TraumaScalpFrontal",
        "ClavOcc": "TraumaScalpOccipital",
        "ClavPar": "TraumaScalpParietal",
        "ClavTem": "TraumaScalpTemporal",
        "NeuroD": "NeurologicalDeficit",
        "NeuroDMotor": "NeuroDeficit_Motor",
        "NeuroDSensory": "NeuroDeficit_Sensory",
        "NeuroDCranial": "NeuroDeficit_CranialNerve",
        "NeuroDReflex": "NeuroDeficit_Reflexes",
        "NeuroDOth": "NeuroDeficit_Other",
        "OSI": "OtherSubstantialInjury",
        "OSIExtremity": "SubstantialInjury_Extremity",
        "OSICut": "SubstantialInjury_Laceration",
        "OSICspine": "SubstantialInjury_CSpine",
        "OSIFlank": "SubstantialInjury_ChestBackFlank",
        "OSIAbdomen": "SubstantialInjury_Abdomen",
        "OSIPelvis": "SubstantialInjury_Pelvis",
        "OSIOth": "SubstantialInjury_Other",
        "Drugs": "SuspectedIntoxication",
        "AgeInMonth": "AgeInMonth",
        "AgeTwoPlus": "AgeTwoPlus",
        "Gender": "Gender",
        "Ethnicity": "Ethnicity",
        "Race": "Race",
        "Neurosurgery": "NeurosurgeryRequired",
        "PosIntFinal": "PosIntFinal",
        "age_indicator": "AgeOneLess",
        "age_indicator_inter": "AgeOneLess_Inter"
    }

    df.rename(columns=rename_dict, inplace=True)

    numeric_vars = ['AgeInMonth','AgeOneLess_Inter']
    y_var = 'PosIntFinal'
    cate_vars = [i for i in df.columns if i not in numeric_vars and i!=y_var]

    # Dummy variables
    df = pd.get_dummies(df, columns=cate_vars, dummy_na=True, drop_first=True)
    # Transfer from Bool to Int
    df.loc[:,df.dtypes == np.bool] = df.loc[:,df.dtypes == np.bool].astype(int)
    nan_cols = np.array([c for c in df.columns if '_nan' in c])
    # Drop dummy variables that always equal 0
    df.drop(nan_cols[df[nan_cols].sum(0) == 0], axis=1, inplace=True, errors='ignore')
    # Drop dummy nan variables that has multicolinearity problem
    df.drop(['High_impact_InjSev_nan'], axis=1, inplace=True, errors='ignore')

    # Drop dummy 91/92 variables that has multicolinearity problem
    notapplicable_cols = np.array([c for c in df.columns if '91' in c or '92' in c])
    cols_keep_91 = ['HasAmnesia_91.0', 'LOC_Duration_92.0', 'SeizureTiming_92.0', 'HeadacheSeverity_92.0', 'VomitingEpisodes_92.0', 'AMS_Agitated_92.0',
                     'DepressedSkullFracture_92.0', 'BasilarFracture_Hemotympanum_92.0', 'HematomaLocation_92.0', 'TraumaFace_92.0', 'NeuroDeficit_Motor_92.0', 'SubstantialInjury_Extremity_92.0']
    cols_drop_91 = [str(c) for c in notapplicable_cols if c not in cols_keep_91]
    df.drop(cols_drop_91, axis=1, inplace=True, errors='ignore')

    # Drop other variables that has multicolinearity problem based on VIF scores
    vars_high_corr = ['InjuryMechanism_6.0',
                        'InjuryMechanism_7.0',
                        'InjuryMechanism_nan',
                        'InjurySeverity_2.0',
                        'InjurySeverity_3.0',
                        'LossOfConsciousness_1.0',
                        'LossOfConsciousness_2.0',
                        'LOC_Duration_92.0',
                        'PostTraumaticSeizure_1.0',
                        'SeizureTiming_92.0',
                        'HeadachePresent_1.0',
                        'HeadacheSeverity_92.0',
                        'HeadacheOnset_2.0',
                        'HeadacheOnset_nan',
                        'VomitingPresent_1.0',
                        'VomitingEpisodes_92.0',
                        'VomitingOnset_2.0',
                        'VomitingOnset_3.0',
                        'VomitingOnset_4.0',
                        'AlteredMentalStatus_1.0',
                        'AMS_Agitated_92.0',
                        'PalpableSkullFracture_1.0',
                        'DepressedSkullFracture_92.0',
                        'BasilarSkullFracture_1.0',
                        'BasilarFracture_Hemotympanum_92.0',
                        'ScalpHematoma_1.0',
                        'HematomaLocation_92.0',
                        'TraumaAboveClavicle_1.0',
                        'TraumaFace_92.0',
                        'NeurologicalDeficit_1.0',
                        'NeuroDeficit_Motor_92.0',
                        'OtherSubstantialInjury_1.0',
                        'SubstantialInjury_Extremity_92.0',
                        'AgeOneLess_1.0'
                    ]
    df.drop(vars_high_corr, axis=1, inplace=True, errors='ignore')

    return df