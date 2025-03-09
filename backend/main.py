from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import torch
import torch.nn as nn
from unidecode import unidecode
from fuzzywuzzy import process
import logging
from typing import Dict, List, Union
import os
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DDI-API")

app = FastAPI(title="Drug Interaction API", version="2.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load resources
RESOURCE_PATH = "./resources"

def load_resource(filename: str):
    """Safe resource loading with error handling"""
    try:
        with open(os.path.join(RESOURCE_PATH, filename), "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Server configuration error")

try:
    # Load core resources
    drug_mapping = load_resource("drug_mapping.pkl")
    label_encoder_risk = load_resource("label_encoder_risk.pkl")
    interaction_list = load_resource("interaction_list.pkl")
    simple_language_mapping = load_resource("simple_language_mapping.pkl")
    drug_features = np.load(os.path.join(RESOURCE_PATH, "drug_features.npy"))
    
    # Load feature encoders and version info
    target_encoder = load_resource("target_encoder.pkl")
    enzyme_encoder = load_resource("enzyme_encoder.pkl")
    version_info = load_resource("version_info.pkl")

except HTTPException:
    raise

# Validate resource consistency
expected_features = len(target_encoder.classes_) + len(enzyme_encoder.classes_)
if drug_features.shape[1] != expected_features:
    logger.critical(f"Feature dimension mismatch! Expected {expected_features}, got {drug_features.shape[1]}")
    raise HTTPException(status_code=500, detail="Server configuration error")

# Model definition
class MultiTaskDDIClassifier(nn.Module):
    def __init__(self, input_size, num_classes_risk, num_classes_interaction):
        super().__init__()
        self.shared_fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.risk_output = nn.Linear(512, num_classes_risk)
        self.interaction_output = nn.Linear(512, num_classes_interaction)

    def forward(self, x):
        shared_representation = self.shared_fc(x)
        return self.risk_output(shared_representation), self.interaction_output(shared_representation)

# Initialize model
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskDDIClassifier(
        input_size=drug_features.shape[1] * 2,
        num_classes_risk=len(label_encoder_risk.classes_),
        num_classes_interaction=len(interaction_list)
    )
    model.load_state_dict(torch.load(
        os.path.join(RESOURCE_PATH, "multi_task_ddi_classifier_2.pth"),
        map_location=device
    ), strict=True)
    model.eval()
    model.to(device)
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise HTTPException(status_code=500, detail="Model initialization failed")

DRUG_SYNONYMS = {
    "eliquis": "apixaban",
    "tylenol": "acetaminophen",
    "advil": "ibuprofen",
    "vicodin": "hydrocodone",
    "nexium": "esomeprazole",
    "xarelto": "rivaroxaban",
    "plavix": "clopidogrel",
}

SEVERITY_MAP = {
    "LOW": {"color": "#10b981", "action": "Monitor therapy"},
    "MODERATE": {"color": "#f59e0b", "action": "Consider alternative"},
    "SEVERE": {"color": "#ef4444", "action": "Contraindicated"},
}

def normalize_name(name: str) -> str:
    return unidecode(name).lower().strip().replace("-", " ").replace("  ", " ")

def find_best_match(name: str) -> str:
    original = normalize_name(name)
    if original in DRUG_SYNONYMS:
        return DRUG_SYNONYMS[original]
    match, score = process.extractOne(original, drug_mapping.keys())
    return match if score > 90 else original

def predict_interaction(drugA: str, drugB: str) -> dict:
    """Wrapper function for model prediction"""
    try:
        if drugA not in drug_mapping or drugB not in drug_mapping:
            raise KeyError(f"Drug not found: {drugA if drugA not in drug_mapping else drugB}")
            
        drugA_idx = drug_mapping[drugA]
        drugB_idx = drug_mapping[drugB]
        
        if drugA_idx >= drug_features.shape[0] or drugB_idx >= drug_features.shape[0]:
            raise ValueError("Drug index out of bounds")

        features = np.concatenate([drug_features[drugA_idx], drug_features[drugB_idx]])
        
        if len(features) != (drug_features.shape[1] * 2):
            raise ValueError(f"Feature size mismatch. Expected {drug_features.shape[1]*2}, got {len(features)}")

        with torch.no_grad():
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            output_risk, output_interaction = model(input_tensor)
            
            risk_prob = torch.softmax(output_risk, dim=1)[0]
            interaction_prob = torch.softmax(output_interaction, dim=1)[0]
            
            risk_pred = torch.argmax(risk_prob).item()
            interaction_pred = torch.argmax(interaction_prob).item()
            
        return {
            "risk_level": label_encoder_risk.inverse_transform([risk_pred])[0].upper(),
            "risk_confidence": risk_prob[risk_pred].item(),
            "interaction_idx": interaction_pred,
            "interaction_confidence": interaction_prob[interaction_pred].item()
        }
        
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Drug not found: {e.args[0]}")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/predict")
async def predict(drugA: str, drugB: str):
    try:
        matched_A = find_best_match(drugA.lower().strip())
        matched_B = find_best_match(drugB.lower().strip())

        if matched_A not in drug_mapping:
            raise HTTPException(status_code=404, detail=f"Drug '{drugA}' not found")
        if matched_B not in drug_mapping:
            raise HTTPException(status_code=404, detail=f"Drug '{drugB}' not found")

        prediction = predict_interaction(matched_A, matched_B)
        
        interaction_text = interaction_list[prediction["interaction_idx"]]
        interaction_desc = simple_language_mapping.get(interaction_text, "Unknown interaction")
        
        return {
            "drugs": [matched_A, matched_B],
            "risk": prediction["risk_level"],
            "risk_confidence": round(prediction["risk_confidence"], 4),
            "interaction": interaction_desc,
            "interaction_confidence": round(prediction["interaction_confidence"], 4),
            "severity": SEVERITY_MAP.get(prediction["risk_level"], {}).get("action", "Unknown"),
            "color": SEVERITY_MAP.get(prediction["risk_level"], {}).get("color", "#94a3b8")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process prediction request")

@app.get("/version")
async def get_version():
    return {
        "api_version": app.version,
        "data_version": version_info["created_at"],
        "drugs_hash": version_info["drugs_hash"],
        "events_hash": version_info["events_hash"],
        "interaction_count": version_info["interaction_count"]
    }

@app.get("/debug/interaction/{index}")
async def debug_interaction(index: int):
    if index >= len(interaction_list):
        raise HTTPException(status_code=404, detail="Index out of range")
    return {
        "index": index,
        "technical": interaction_list[index],
        "description": simple_language_mapping.get(interaction_list[index], "Unknown interaction")
    }

@app.get("/check-drug/{drug_name}")
async def check_drug(drug_name: str):
    try:
        normalized = normalize_name(drug_name)
        match = find_best_match(normalized)
        exists = match in drug_mapping
        return {
            "input": drug_name,
            "normalized": normalized,
            "matched_name": match,
            "exists": exists,
            "message": f"Drug {'found' if exists else 'not found'} in database"
        }
    except Exception as e:
        logger.error(f"Drug check error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process drug check request")

@app.get("/all-drugs")
async def list_drugs(limit: int = 100, search: str = None, threshold: int = 50):
    try:
        drugs = list(drug_mapping.keys())
        if search:
            search_term = normalize_name(search)
            matches = process.extract(search_term, drugs, limit=limit)
            matches.sort(key=lambda x: x[1], reverse=True)
            drugs = [match[0] for match in matches if match[1] > threshold]
        return {"drugs": drugs[:limit]}
    except Exception as e:
        logger.error(f"Drug listing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve drug list")

@app.get("/", include_in_schema=False)
async def health_check():
    return {"status": "active", "version": app.version}

@app.get("/debug/interaction/67")
async def debug_interaction_67():
    return {
        "index": 67,
        "technical": interaction_list[67],
        "description": simple_language_mapping[interaction_list[67]]
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


# 2ND API CODE

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import pickle
# import numpy as np
# import torch
# import torch.nn as nn
# from unidecode import unidecode
# from fuzzywuzzy import process
# import logging
# from typing import Dict, List, Union
# import os
# from fastapi.responses import JSONResponse
# from typing import Optional

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger("DDI-API")

# app = FastAPI(title="Drug Interaction API", version="1.3.0")

# # CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load resources
# RESOURCE_PATH = "./resources"

# def load_resource(filename: str):
#     """Safe resource loading with error handling"""
#     try:
#         with open(os.path.join(RESOURCE_PATH, filename), "rb") as f:
#             return pickle.load(f)
#     except Exception as e:
#         logger.error(f"Failed to load {filename}: {str(e)}")
#         raise HTTPException(status_code=500, detail="Server configuration error")

# # Add to resource loading section
# try:
#     # Existing loads
#     drug_mapping = load_resource("drug_mapping.pkl")
#     label_encoder_risk = load_resource("label_encoder_risk.pkl")
#     interaction_list = load_resource("interaction_list.pkl")
#     simple_language_mapping = load_resource("simple_language_mapping.pkl")
#     drug_features = np.load(os.path.join(RESOURCE_PATH, "drug_features.npy"))
    
#     # New critical additions
#     target_encoder = load_resource("target_encoder.pkl")
#     enzyme_encoder = load_resource("enzyme_encoder.pkl")
#     version_info = load_resource("version_info.pkl")
# except HTTPException:
#     raise
# # Add after loading resources
# expected_features = len(target_encoder.classes_) + len(enzyme_encoder.classes_)
# if drug_features.shape[1] != expected_features:
#     logger.critical(f"Drug feature dimension mismatch! Expected {expected_features}, got {drug_features.shape[1]}")
#     raise HTTPException(status_code=500, detail="Server configuration error")

# # Updated Model definition
# class MultiTaskDDIClassifier(nn.Module):
#     def __init__(self, input_size, num_classes_risk, num_classes_interaction):
#         super().__init__()
#         self.shared_fc = nn.Sequential(
#             nn.Linear(input_size, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#         )
#         self.risk_output = nn.Linear(512, num_classes_risk)
#         self.interaction_output = nn.Linear(512, num_classes_interaction)

#     def forward(self, x):
#         shared_representation = self.shared_fc(x)
#         return self.risk_output(shared_representation), self.interaction_output(shared_representation)

# # Initialize model
# # Initialize model
# try:
#     input_size = drug_features.shape[1] * 2
#     num_classes_risk = len(label_encoder_risk.classes_)
#     num_classes_interaction = len(interaction_list)
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Add this line
#     model = MultiTaskDDIClassifier(input_size, num_classes_risk, num_classes_interaction)
#     model.load_state_dict(torch.load(
#         os.path.join(RESOURCE_PATH, "multi_task_ddi_classifier.pth"),
#         map_location=device  # Update this
#     ))
#     model.eval()
#     model.to(device)  # Add this line
# except Exception as e:
#     logger.error(f"Model loading failed: {str(e)}")
#     raise HTTPException(status_code=500, detail="Model initialization failed")

# # Existing drug name processing remains the same
# DRUG_SYNONYMS = {
#     "eliquis": "apixaban",
#     "tylenol": "acetaminophen",
#     "advil": "ibuprofen",
#     "vicodin": "hydrocodone",
#     "nexium": "esomeprazole",
#     "xarelto": "rivaroxaban",
#     "plavix": "clopidogrel",
# }

# SEVERITY_MAP = {
#     "LOW": {"color": "#10b981", "action": "Monitor therapy"},
#     "MODERATE": {"color": "#f59e0b", "action": "Consider alternative"},
#     "SEVERE": {"color": "#ef4444", "action": "Contraindicated"},
# }

# def normalize_name(name: str) -> str:
#     return unidecode(name).lower().strip().replace("-", " ").replace("  ", " ")

# def find_best_match(name: str) -> str:
#     original = normalize_name(name)
    
#     if original in DRUG_SYNONYMS:
#         return DRUG_SYNONYMS[original]
    
#     match, score = process.extractOne(original, drug_mapping.keys())
#     return match if score > 90 else original

# def predict_interaction(drugA: str, drugB: str) -> dict:
#     """Wrapper function for model prediction"""
#     try:
#         # Get indices from drug mapping
#         drugA_idx = drug_mapping[drugA]
#         drugB_idx = drug_mapping[drugB]
        
#         # Create feature vector (ensure same order as training)
#         features = np.concatenate([
#             drug_features[drugA_idx],
#             drug_features[drugB_idx]
#         ])
        
#         # Convert to tensor and predict
#         with torch.no_grad():
#             input_tensor = torch.tensor(features, 
#                                       dtype=torch.float32).unsqueeze(0).to(device)
#             output_risk, output_interaction = model(input_tensor)
            
#             # Process outputs
#             risk_prob = torch.softmax(output_risk, dim=1)[0]
#             interaction_prob = torch.softmax(output_interaction, dim=1)[0]
            
#             risk_pred = torch.argmax(risk_prob).item()
#             interaction_pred = torch.argmax(interaction_prob).item()
            
#         return {
#             "risk_level": label_encoder_risk.inverse_transform([risk_pred])[0].upper(),
#             "risk_confidence": risk_prob[risk_pred].item(),
#             "interaction_idx": interaction_pred,
#             "interaction_confidence": interaction_prob[interaction_pred].item()
#         }
        
#     except KeyError as e:
#         raise HTTPException(status_code=404, detail=f"Drug not found: {e.args[0]}")

# @app.get("/predict", response_model=Dict[str, Union[str, List[str], float]])
# async def predict(drugA: str, drugB: str) -> Dict:
#     try:
#         matched_A = find_best_match(drugA.lower().strip())
#         matched_B = find_best_match(drugB.lower().strip())

#         # Validate drugs exist
#         if matched_A not in drug_mapping:
#             raise HTTPException(status_code=404, detail=f"Drug '{drugA}' not found")
#         if matched_B not in drug_mapping:
#             raise HTTPException(status_code=404, detail=f"Drug '{drugB}' not found")

#         # Get prediction
#         prediction = predict_interaction(matched_A, matched_B)
        
#         # Get interaction text
#         interaction_text = interaction_list[prediction["interaction_idx"]]
#         interaction_desc = simple_language_mapping.get(interaction_text, "Unknown interaction")
        
#         return {
#             "drugs": [matched_A, matched_B],
#             "risk": prediction["risk_level"],
#             "risk_confidence": round(prediction["risk_confidence"], 4),
#             "interaction": interaction_desc,
#             "interaction_confidence": round(prediction["interaction_confidence"], 4),
#             "severity": SEVERITY_MAP.get(prediction["risk_level"], {}).get("action", "Unknown"),
#             "color": SEVERITY_MAP.get(prediction["risk_level"], {}).get("color", "#94a3b8")
#         }

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Prediction error: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Failed to process prediction request")

# @app.get("/check-drug/{drug_name}", response_model=Dict[str, Union[str, bool]])
# async def check_drug(drug_name: str) -> Dict:
#     """Check drug existence in database"""
#     try:
#         normalized = normalize_name(drug_name)
#         match = find_best_match(normalized)
#         exists = match in drug_mapping
#         return {
#             "input": drug_name,
#             "normalized": normalized,
#             "matched_name": match,
#             "exists": exists,
#             "message": f"Drug {'found' if exists else 'not found'} in database"
#         }
#     except Exception as e:
#         logger.error(f"Drug check error: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail="Failed to process drug check request"
#         )

# @app.get("/all-drugs", response_model=Dict[str, List[str]])
# async def list_drugs(limit: int = 100, search: str = None, threshold: int = 50, min_length: int = 2) -> Dict:
#     """List available drugs with optional search filter and fuzzy matching"""
#     try:
#         drugs = list(drug_mapping.keys())
#         if search:
#             if len(search) < min_length:
#                 return {"drugs": []}  # Return empty list if query is too short
#             search_term = normalize_name(search)
#             # Use fuzzy matching to find the best matches
#             matches = process.extract(search_term, drugs, limit=limit)
#             # Sort by match score (descending)
#             matches.sort(key=lambda x: x[1], reverse=True)
#             drugs = [match[0] for match in matches if match[1] > threshold]  # Filter by match score
#         return {"drugs": drugs[:limit]}
#     except Exception as e:
#         logger.error(f"Drug listing error: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail="Failed to retrieve drug list"
#         )

# @app.get("/", include_in_schema=False)
# async def health_check():
#     return {"status": "active", "version": app.version}

# @app.exception_handler(HTTPException)
# async def http_exception_handler(request, exc):
#     return JSONResponse(
#         status_code=exc.status_code,
#         content={"error": exc.detail}
#     )

# @app.exception_handler(Exception)
# async def generic_exception_handler(request, exc):
#     logger.error(f"Unexpected error: {str(exc)}")
#     return JSONResponse(
#         status_code=500,
#         content={"error": "Internal server error"}
#     )







# 1ST API CODE

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import pickle
# import numpy as np
# import torch
# import torch.nn as nn
# from unidecode import unidecode
# from fuzzywuzzy import process
# import logging
# from typing import Dict, List, Union
# import os
# from fastapi.responses import JSONResponse

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger("DDI-API")

# app = FastAPI(title="Drug Interaction API", version="1.2.0")

# # CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load resources
# RESOURCE_PATH = "./resources"

# def load_resource(filename: str):
#     """Safe resource loading with error handling"""
#     try:
#         with open(os.path.join(RESOURCE_PATH, filename), "rb") as f:
#             return pickle.load(f)
#     except Exception as e:
#         logger.error(f"Failed to load {filename}: {str(e)}")
#         raise HTTPException(status_code=500, detail="Server configuration error")

# try:
#     drug_mapping = load_resource("drug_mapping.pkl")
#     label_encoder = load_resource("label_encoder.pkl")
#     interaction_mapping = load_resource("interaction_mapping.pkl")
#     drug_features = np.load(os.path.join(RESOURCE_PATH, "drug_features.npy"))
# except HTTPException:
#     raise

# # Model definition
# class DrugInteractionModel(nn.Module):
#     def __init__(self, input_size: int, output_size: int = 3):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_size, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, output_size)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.fc(x)

# # Initialize model
# try:
#     input_size = drug_features.shape[1] * 2
#     model = DrugInteractionModel(input_size)
#     model.load_state_dict(torch.load(
#         os.path.join(RESOURCE_PATH, "ddi_model.pth"),
#         map_location="cpu"
#     ))
#     model.eval()
# except Exception as e:
#     logger.error(f"Model loading failed: {str(e)}")
#     raise HTTPException(status_code=500, detail="Model initialization failed")

# # Drug name processing (FIXED SYNONYM DIRECTIONS)
# DRUG_SYNONYMS = {
#     # Brand -> Generic mappings
#     "eliquis": "apixaban",
#     "tylenol": "acetaminophen",
#     "advil": "ibuprofen",
#     "vicodin": "hydrocodone",
#     "nexium": "esomeprazole",
#     "xarelto": "rivaroxaban",
#     "plavix": "clopidogrel",
# }

# SEVERITY_MAP = {
#     "LOW": {"color": "#10b981", "action": "Monitor therapy"},
#     "MODERATE": {"color": "#f59e0b", "action": "Consider alternative"},
#     "SEVERE": {"color": "#ef4444", "action": "Contraindicated"},
# }

# def normalize_name(name: str) -> str:
#     """Standardize drug name format for matching"""
#     return unidecode(name).lower().strip().replace("-", " ").replace("  ", " ")

# def find_best_match(name: str) -> str:
#     """Enhanced fuzzy matching with logging"""
#     original = normalize_name(name)
    
#     # Check synonyms first
#     if original in DRUG_SYNONYMS:
#         generic_name = DRUG_SYNONYMS[original]
#         logger.info(f"Matched synonym: {original} → {generic_name}")
#         return generic_name
    
#     # Fuzzy match with quality threshold
#     match, score = process.extractOne(original, drug_mapping.keys())
    
#     if score > 90:  # Increased threshold for better accuracy
#         logger.info(f"Fuzzy matched: {original} → {match} (score: {score})")
#         return match
    
#     logger.warning(f"No good match found for: {original}")
#     return original

# @app.get("/predict", response_model=Dict[str, Union[str, List[str], float]])
# async def predict(drugA: str, drugB: str) -> Dict:
#     """Predict drug interaction with enhanced safety checks"""
#     try:
#         # Validate input parameters
#         if not drugA or not drugB:
#             raise HTTPException(status_code=400, detail="Both drug names are required")
        
#         # Process drug names with improved matching
#         matched_A = find_best_match(drugA)
#         matched_B = find_best_match(drugB)

#         # Validate matches exist in database
#         validation_errors = []
#         if matched_A not in drug_mapping:
#             validation_errors.append(f"Drug '{drugA}' not found. Closest match: '{matched_A}'")
#         if matched_B not in drug_mapping:
#             validation_errors.append(f"Drug '{drugB}' not found. Closest match: '{matched_B}'")
        
#         if validation_errors:
#             raise HTTPException(status_code=404, detail=" | ".join(validation_errors))

#         # Get feature indices
#         idx_a = drug_mapping[matched_A]
#         idx_b = drug_mapping[matched_B]
        
#         # Create feature vector
#         features = np.concatenate([drug_features[idx_a], drug_features[idx_b]])
        
#         if features.shape[0] != input_size:
#             raise HTTPException(
#                 status_code=500,
#                 detail="Feature dimension mismatch"
#             )

#         # Model prediction
#         with torch.no_grad():
#             input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
#             output = model(input_tensor)
#             pred = torch.argmax(output).item()
#             confidence = torch.softmax(output, dim=1)[0][pred].item()

#         risk_level = label_encoder.inverse_transform([pred])[0].upper()
        
#         return {
#             "drugs": [matched_A, matched_B],
#             "risk": risk_level,
#             "confidence": round(confidence, 4),
#             "description": interaction_mapping.get(pred, "No detailed description available"),
#             "severity": SEVERITY_MAP.get(risk_level, {}).get("action", "Unknown"),
#             "color": SEVERITY_MAP.get(risk_level, {}).get("color", "#94a3b8")
#         }

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Prediction error: {str(e)}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail="Failed to process prediction request"
#         )

# @app.get("/check-drug/{drug_name}", response_model=Dict[str, Union[str, bool]])
# async def check_drug(drug_name: str) -> Dict:
#     """Check drug existence in database"""
#     try:
#         normalized = normalize_name(drug_name)
#         match = find_best_match(normalized)
#         exists = match in drug_mapping
#         return {
#             "input": drug_name,
#             "normalized": normalized,
#             "matched_name": match,
#             "exists": exists,
#             "message": f"Drug {'found' if exists else 'not found'} in database"
#         }
#     except Exception as e:
#         logger.error(f"Drug check error: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail="Failed to process drug check request"
#         )

# @app.get("/all-drugs", response_model=Dict[str, List[str]])
# async def list_drugs(limit: int = 100, search: str = None, threshold: int = 50, min_length: int = 2) -> Dict:
#     """List available drugs with optional search filter and fuzzy matching"""
#     try:
#         drugs = list(drug_mapping.keys())
#         if search:
#             if len(search) < min_length:
#                 return {"drugs": []}  # Return empty list if query is too short
#             search_term = normalize_name(search)
#             # Use fuzzy matching to find the best matches
#             matches = process.extract(search_term, drugs, limit=limit)
#             # Sort by match score (descending)
#             matches.sort(key=lambda x: x[1], reverse=True)
#             drugs = [match[0] for match in matches if match[1] > threshold]  # Filter by match score
#         return {"drugs": drugs[:limit]}
#     except Exception as e:
#         logger.error(f"Drug listing error: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail="Failed to retrieve drug list"
#         )

# @app.get("/", include_in_schema=False)
# async def health_check():
#     return {"status": "active", "version": app.version}

# @app.exception_handler(HTTPException)
# async def http_exception_handler(request, exc):
#     return JSONResponse(
#         status_code=exc.status_code,
#         content={"error": exc.detail}
#     )

# @app.exception_handler(Exception)
# async def generic_exception_handler(request, exc):
#     logger.error(f"Unexpected error: {str(exc)}")
#     return JSONResponse(
#         status_code=500,
#         content={"error": "Internal server error"}
#     )