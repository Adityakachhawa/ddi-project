import pandas as pd
import numpy as np
import pickle
import os
import hashlib
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

def main():
    # Load datasets
    drugs_df = pd.read_csv("drugs.csv").drop(columns=["smile"])
    events_df = pd.read_csv("events.csv")

    # Process risk levels
    def map_risk_level(label):
        if label <= 30: 
            return 'low'
        elif label <= 60: 
            return 'moderate'
        else: 
            return 'severe'
    
    events_df['risk_level'] = events_df['label'].apply(map_risk_level)

    # ========== DRUG FEATURES ==========
    target_encoder = MultiLabelBinarizer()
    enzyme_encoder = MultiLabelBinarizer()

    # Process target and enzyme features
    target_features = target_encoder.fit_transform(drugs_df['target'].str.split('|'))
    enzyme_features = enzyme_encoder.fit_transform(drugs_df['enzyme'].str.split('|'))
    drug_features = np.hstack([target_features, enzyme_features])

    # Create drug mapping
    drug_mapping = {name: idx for idx, name in enumerate(drugs_df['name'])}

    # ========== RISK ENCODER ==========
    label_encoder_risk = LabelEncoder()
    label_encoder_risk.fit(events_df['risk_level'])

    # ========== INTERACTION MAPPINGS ==========
    # Preserve original order from events.csv
    interaction_list = events_df['label_text'].unique().tolist()
    
    # Simple language mapping with fallback
    base_mapping = {
    "the anticoagulant activities increase": "Increased risk of bleeding",
    "the risk or severity of bleeding and hemorrhage increase": "Higher chance of severe bleeding",
    "the risk or severity of bleeding increase": "Higher chance of bleeding",
    "the therapeutic efficacy increase": "Improved effectiveness of treatment",
    "the anticoagulant activities decrease": "Reduced blood-thinning effect",
    "the therapeutic efficacy decrease": "Decreased effectiveness of treatment",
    "the risk or severity of hemorrhage increase": "Higher risk of internal bleeding",
    "the antiplatelet activities increase": "Increased risk of excessive bleeding",
    "the risk or severity of adverse effects increase": "Greater risk of side effects",
    "the risk or severity of hyponatremia increase": "Higher risk of low sodium levels",
    "the antihypertensive activities decrease": "Reduced blood pressure-lowering effect",
    "the risk or severity of hypertension increase": "Higher risk of high blood pressure",
    "the excretion rate which could result in a higher serum level decrease": "Slower drug removal, increasing drug levels",
    "the excretion rate which could result in a lower serum level and potentially a reduction in efficacy increase": "Faster drug removal, reducing effectiveness",
    "the hypertensive activities increase": "Increased blood pressure",
    "the immunosuppressive activities increase": "Weakened immune response",
    "the serum concentration increase": "Higher drug levels in the blood",
    "the nephrotoxic activities increase": "Increased risk of kidney damage",
    "the excretion decrease": "Slower drug elimination",
    "the serum concentration decrease": "Lower drug levels in the blood",
    "the neurotoxic activities increase": "Higher risk of nerve damage",
    "an increase in the absorption resulting in an increased serum concentration and potentially a worsening of adverse effects cause": "More drug absorption, leading to stronger effects and side effects",
    "the metabolism increase": "Faster drug breakdown",
    "the metabolism decrease": "Slower drug breakdown",
    "a decrease in the absorption resulting in a reduced serum concentration and potentially a decrease in efficacy cause": "Less drug absorption, reducing effectiveness",
    "the risk or severity of myelosuppression increase": "Higher risk of bone marrow suppression",
    "the risk or severity of renal failure increase": "Higher risk of kidney failure",
    "the risk or severity of hyperkalemia increase": "Higher risk of high potassium levels",
    "the risk or severity of nephrotoxicity increase": "Higher risk of kidney toxicity",
    "the risk or severity of liver damage increase": "Higher risk of liver damage",
    "the risk or severity of renal failure and hypertension increase": "Higher risk of kidney failure and high blood pressure",
    "the risk or severity of myopathy rhabdomyolysis and myoglobinuria increase": "Higher risk of muscle breakdown and kidney damage",
    "the risk or severity of seizure increase": "Higher risk of seizures",
    "the hepatotoxic activities increase": "Higher risk of liver toxicity",
    "the risk or severity of hypokalemia increase": "Higher risk of low potassium levels",
    "the central nervous system depressant (CNS depressant) activities increase": "Increased sedation and drowsiness",
    "the sedative activities increase": "Higher chance of drowsiness",
    "the serotonergic activities increase": "Increased serotonin activity",
    "the risk or severity of serotonin syndrome increase": "Higher risk of serotonin syndrome (dangerous condition caused by excess serotonin)",
    "the risk or severity of CNS depression increase": "Higher risk of excessive drowsiness and sedation",
    "the risk or severity of sedation increase": "Higher chance of sleepiness",
    "the risk or severity of hypoglycemia increase": "Higher risk of low blood sugar",
    "the thrombogenic activities increase": "Higher risk of blood clot formation",
    "the risk or severity of neutropenia increase": "Higher risk of low white blood cells",
    "the risk or severity of QTc prolongation increase": "Higher risk of heart rhythm problems",
    "the risk or severity of gastrointestinal bleeding increase": "Higher risk of stomach bleeding",
    "the risk or severity of hypotension increase": "Higher risk of low blood pressure",
    "the risk or severity of orthostatic hypotension and syncope increase": "Higher risk of dizziness and fainting",
    "the risk or severity of hypotension and orthostatic hypotension increase": "Higher risk of low blood pressure and dizziness",
    "the hypotensive activities increase": "Stronger blood pressure-lowering effect",
    "the antihypertensive activities increase": "Stronger blood pressure-lowering effect",
    "the risk or severity of renal failure hyperkalemia and hypertension increase": "Higher risk of kidney failure, high potassium, and high blood pressure",
    "the risk or severity of renal failure hypotension and hyperkalemia increase": "Higher risk of kidney failure, low blood pressure, and high potassium",
    "the hyperkalemic activities increase": "Increased potassium levels in the blood",
    "the risk or severity of angioedema increase": "Higher risk of severe swelling (angioedema)",
    "the orthostatic hypotensive activities increase": "Higher risk of dizziness when standing",
    "the neuromuscular blocking activities increase": "Increased muscle relaxation effect",
    "the bradycardic activities increase": "Lower heart rate",
    "the neuromuscular blocking activities decrease": "Reduced muscle relaxation effect",
    "the risk or severity of hyperglycemia increase": "Higher risk of high blood sugar",
    "the risk or severity of tendinopathy increase": "Higher risk of tendon damage",
    "the hypokalemic activities increase": "Higher risk of low potassium levels",
    "the risk or severity of fluid retention increase": "Higher risk of swelling due to fluid retention",
    "the risk or severity of gastrointestinal irritation increase": "Higher risk of stomach discomfort",
    "the risk or severity of myopathy and weakness increase": "Higher risk of muscle weakness",
    "the risk or severity of edema formation increase": "Higher risk of swelling",
    "the risk or severity of electrolyte imbalance increase": "Higher risk of imbalance in body salts",
    "the anticholinergic activities increase": "Increased drying effects (dry mouth, constipation)",
    "the analgesic activities increase": "Stronger pain relief",
    "the sedative and stimulatory activities decrease": "Reduced sedative and stimulant effects",
    "the stimulatory activities decrease": "Reduced stimulant effect",
    "the hypoglycemic activities increase": "Higher risk of low blood sugar",
    "the risk or severity of tachycardia increase": "Higher risk of fast heartbeat",
    "the risk or severity of extrapyramidal symptoms increase": "Higher risk of movement disorders",
    "the risk or severity of sedation and somnolence increase": "Higher risk of sleepiness",
    "the risk or severity of hypertension decrease": "Lower risk of high blood pressure",
    "the vasoconstricting activities increase": "Increased narrowing of blood vessels",
    "the arrhythmogenic activities increase": "Higher risk of abnormal heart rhythms",
    "the risk or severity of bradycardia increase": "Higher risk of slow heartbeat",
    "the risk or severity of QTc prolongation decrease": "Lower risk of heart rhythm problems",
    "the QTc prolonging activities increase": "Higher risk of heart rhythm problems",
    "the risk or severity of neutropenia and thrombocytopenia increase": "Higher risk of low white blood cells and platelets",
    "the risk or severity of hyperthermia and oligohydrosis increase": "Higher risk of overheating and reduced sweating",
    "the hypoglycemic activities decrease": "Lower risk of low blood sugar",
    "the serum concentration of the active metabolites increase": "Higher levels of active drug in the blood",
    "the risk or severity of cardiac arrhythmia increase": "Higher risk of irregular heartbeat",
    "the vasodilatory activities increase": "Increased blood vessel relaxation",
    "the absorption decrease": "Less drug absorption in the body",
    }
    
    # Generate final mapping with fallbacks
    simple_language_mapping = {
        text: base_mapping.get(text, f"Technical: {text}")
        for text in interaction_list
    }

    # ========== VALIDATION CHECKS ==========
    try:
        target_index = interaction_list.index("the excretion rate which could result in a higher serum level decrease")
        print(f"✅ Critical interaction found at index: {target_index}")
    except ValueError:
        print("❌ Critical interaction missing from list!")

    assert drug_features.shape[0] == len(drugs_df), "Drug feature count mismatch!"
    assert len(interaction_list) == len(simple_language_mapping), "Mapping length mismatch!"

    # Create resources directory
    os.makedirs("resources", exist_ok=True)

    # ========== SAVE RESOURCES ==========
    # Save core drug data
    with open("resources/drug_mapping.pkl", "wb") as f:
        pickle.dump(drug_mapping, f)
    
    np.save("resources/drug_features.npy", drug_features)

    # Save encoders
    with open("resources/target_encoder.pkl", "wb") as f:
        pickle.dump(target_encoder, f)
    
    with open("resources/enzyme_encoder.pkl", "wb") as f:
        pickle.dump(enzyme_encoder, f)

    # Save risk system
    with open("resources/label_encoder_risk.pkl", "wb") as f:
        pickle.dump(label_encoder_risk, f)

    # Save interaction system
    with open("resources/interaction_list.pkl", "wb") as f:
        pickle.dump(interaction_list, f)
    
    with open("resources/simple_language_mapping.pkl", "wb") as f:
        pickle.dump(simple_language_mapping, f)

    # Save version info
    def file_hash(file_path):
        return hashlib.md5(open(file_path, 'rb').read()).hexdigest()
    
    version_info = {
        "drugs_hash": file_hash("drugs.csv"),
        "events_hash": file_hash("events.csv"),
        "created_at": pd.Timestamp.now().isoformat(),
        "interaction_count": len(interaction_list)
    }
    
    with open("resources/version_info.pkl", "wb") as f:
        pickle.dump(version_info, f)

    # Final validation
    print(f"\n✅ Saved resources for {len(drug_mapping)} drugs and {len(interaction_list)} interactions")
    print(f"🔍 Sample validation - Index 67: {interaction_list[67]}")
    print(f"   Mapped to: {simple_language_mapping[interaction_list[67]]}")
    print(f"📦 Version hash: {version_info['events_hash']}")

if __name__ == "__main__":
    main()