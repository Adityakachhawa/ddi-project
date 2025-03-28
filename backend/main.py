import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import torch
import torch.nn as nn
from unidecode import unidecode
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import logging
from typing import Dict, List, Union
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DDI-API")

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "resources")

app = FastAPI(title="Drug Interaction API", version="2.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ddi-project-git-main-aditya-kachhawas-projects.vercel.app",
        "https://ddi-project.vercel.app",
        "https://ddiaadi.vercel.app/",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ======== END OF CORS UPDATE ========
# Add this right after CORS configuration
from fastapi import Request
from fastapi.responses import RedirectResponse

# @app.middleware("http")
# async def https_redirect(request: Request, call_next):
#     if request.url.scheme == "http":
#         url = request.url.replace(scheme="https")
#         return RedirectResponse(url)
#     return await call_next(request)

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
    # A-C
    "angiox": "bivalirudin",
    "angiomax": "bivalirudin",
    "ddavp": "desmopressin",
    "neoral": "cyclosporine",
    "sandimmune": "cyclosporine",
    "vitaminb12": "cyanocobalamin",
    "vitaminc": "ascorbicacid",
    "rocaltrol": "calcitriol",
    "vitaminb2": "riboflavin",
    "vitamind2": "ergocalciferol",
    "vitamind3": "cholecalciferol",
    "vitaminb9": "folicacid",
    "vitaminb6": "pyridoxine",
    "luvox": "fluvoxamine",
    "diovan": "valsartan",
    "altace": "ramipril",
    "nasalide": "flunisolide",
    "adderall": "amphetamine",
    "nicorette": "nicotine",
    "evoxac": "cevimeline",
    "ativan": "lorazepam",
    "brevibloc": "esmolol",
    "velcade": "bortezomib",
    "adipex": "phentermine",
    "ultram": "tramadol",
    "betoptic": "betaxolol",
    "diflucan": "fluconazole",
    "tamiflu": "oseltamivir",
    "ery-tab": "erythromycin",
    "anectine": "succinylcholine",
    "viagra": "sildenafil",
    "daraprim": "pyrimethamine",
    "zithromax": "azithromycin",
    "ticlid": "ticlopidine",
    "sanctura": "trospium",
    "proamatine": "midodrine",
    "protonix": "pantoprazole",
    "demadex": "torasemide",
    "celexa": "citalopram",
    "relpax": "eletriptan",
    "avelox": "moxifloxacin",
    "viracept": "nelfinavir",
    "amaryl": "glimepiride",
    "crixivan": "indinavir",
    "mevacor": "lovastatin",
    "restoril": "temazepam",
    # D-G
    "dostinex": "cabergoline",
    "dilantin": "phenytoin",
    "lotrimin": "clotrimazole",
    "agrylin": "anagrelide",
    "lopressor": "metoprolol",
    "toprolxl": "metoprolol",
    "requip": "ropinirole",
    "topamax": "topiramate",
    "theo-24": "theophylline",
    "acova": "argatroban",
    "cytomel": "liothyronine",
    "norpace": "disopyramide",
    "xylocaine": "lidocaine",
    "tavist": "clemastine",
    "effexor": "venlafaxine",
    "strattera": "atomoxetine",
    "mscontin": "morphine",
    "marcaine": "bupivacaine",
    "viread": "tenofovirdisoproxil",
    "nembutal": "pentobarbital",
    "depakote": "valproicacid",
    "zomig": "zolmitriptan",
    "tylenol": "acetaminophen",
    "paracetamol": "acetaminophen",
    "dolo 650": "acetaminophen",
    "iressa": "gefitinib",
    "migranal": "dihydroergotamine",
    "elavil": "amitriptyline",
    "tasmar": "tolcapone",
    "fml": "fluorometholone",
    "nitropress": "nitroprusside",
    "dilaudid": "hydromorphone",
    "indocin": "indomethacin",
    "dolophine": "methadone",
    "zyprexa": "olanzapine",
    "tenormin": "atenolol",
    "elidel": "pimecrolimus",
    "prilosec": "omeprazole",
    "cardizem": "diltiazem",
    "uroxatral": "alfuzosin",
    "onfi": "clobazam",
    "rogaine": "minoxidil",
    "megace": "megestrolacetate",
    "parafon": "chlorzoxazone",
    "lariam": "mefloquine",
    "kuvan": "sapropterin",
    "navelbine": "vinorelbine",
    "clozaril": "clozapine",
    "unisom": "doxylamine",
    "planb": "levonorgestrel",
    "vistide": "cidofovir",
    "remeron": "mirtazapine",
    "timoptic": "timolol",
    "aloxi": "palonosetron",
    "norvasc": "amlodipine",
    "dyrenium": "triamterene",
    "sudafedpe": "phenylephrine",
    "lanoxin": "digoxin",
    "nimotop": "nimodipine",
    "qvar": "beclomethasonedipropionate",
    "soma": "carisoprodol",
    "prometrium": "progesterone",
    "nexavar": "sorafenib",
    "grifulvinv": "griseofulvin",
    "sular": "nisoldipine",
    "lunesta": "eszopiclone",
    "xanax": "alprazolam",
    "avandia": "rosiglitazone",
    "seconal": "secobarbital",
    "aldactone": "spironolactone",
    "ritalin": "methylphenidate",
    "ambien": "zolpidem",
    "famvir": "famciclovir",
    "viroptic": "trifluridine",
    "compazine": "prochlorperazine",
    "periactin": "cyproheptadine",
    "inomax": "nitricoxide",
    "zyloprim": "allopurinol",
    "gemzar": "gemcitabine",
    "celestone": "betamethasone",
    "ellence": "epirubicin",
    "chloromycetin": "chloramphenicol",
    "prevacid": "lansoprazole",
    "synthroid": "levothyroxine",
    "demerol": "meperidine",
    "claritin": "loratadine",
    "tofranil": "imipramine",
    "relafen": "nabumetone",
    "toradol": "ketorolac",
    "penetrex": "enoxacin",
    "qualiquin": "quinine",
    "mobiflex": "tenoxicam",
    "marinol": "dronabinol",
    "singulair": "montelukast",
    "prozac": "fluoxetine",
    "sarafem": "fluoxetine",  # Alternative brand
    "brevital": "methohexital",
    "cymbalta": "duloxetine",
    "thorazine": "chlorpromazine",
    "evista": "raloxifene",
    "celebrex": "celecoxib",
    "alphagan": "brimonidine",
    "dycill": "dicloxacillin",
    "cesamet": "nabilone",
    "sorine": "sotalol",
    "buspar": "buspirone",
    "glyset": "miglitol",
    "comtan": "entacapone",
    "retrovir": "zidovudine",
    "enablex": "darifenacin",
    "oxycontin": "oxycodone",
    "eulexin": "flutamide",
    "tolectin": "tolmetin",
    "tagamet": "cimetidine",
    "haldol": "haloperidol",
    "norvir": "ritonavir",
    "vesprin": "triflupromazine",
    "amicar": "aminocaproicacid",
    "delsym": "dextromethorphan",
    "albenza": "albendazole",
    "mavik": "trandolapril",
    "cancidas": "caspofungin",
    "ocupress": "carteolol",
    "dibucaine": "cinchocaine",
    "zanidip": "lercanidipine",
    "tarceva": "erlotinib",
    "cytoxan": "cyclophosphamide",
    "mesantoin": "mephenytoin",
    "vioxx": "rofecoxib",  # Withdrawn
    "cipro": "ciprofloxacin",
    "fareston": "toremifene",
    "pamelor": "nortriptyline",
    "oncovin": "vincristine",
    "lotensin": "benazepril",
    "asendin": "amoxapine",
    "adrucil": "fluorouracil",
    "accupril": "quinapril",
    "accolate": "zafirlukast",
    "propyl-thyracil": "propylthiouracil",
    "feldene": "piroxicam",
    "lamictal": "lamotrigine",
    "atarax": "hydroxyzine",
    "vistaril": "hydroxyzine",  # Alternative brand
    "tracleer": "bosentan",
    "trexall": "methotrexate",
    "tegretol": "carbamazepine",
    "stugeron": "cinnarizine",
    "velban": "vinblastine",
    "inderal": "propranolol",
    "fen-phen": "fenfluramine",  # Withdrawn combo
    "catapres": "clonidine",
    "thiosulfil": "sulfamethizole",
    "valtrex": "valaciclovir",
    "geocillin": "carbenicillin",
    "bextra": "valdecoxib",  # Withdrawn
    "vfend": "voriconazole",
    "axid": "nizatidine",
    "voltaren": "diclofenac",
    "flovent": "fluticasonepropionate",
    "dopergin": "lisuride",
    "cardura": "doxazosin",
    "synalar": "fluocinoloneacetonide",
    "antepar": "piperazine",
    "zarontin": "ethosuximide",
    "midamor": "amiloride",
    "normodyne": "labetalol",
    "pentothal": "thiopental",
    "stromectol": "ivermectin",
    "depo-provera": "medroxyprogesteroneacetate",
    "propulsid": "cisapride",  # Restricted
    "clinoril": "sulindac",
    "unipen": "nafcillin",
    "aralen": "chloroquine",
    "zebeta": "bisoprolol",
    "camoquin": "amodiaquine",
    "mycobutin": "rifabutin",
    "gleevec": "imatinib",
    "kenalog": "triamcinolone",
    "oxandrin": "oxandrolone",
    "cardene": "nicardipine",
    "prolixin": "fluphenazine",
    "androderm": "testosterone",
    "sustiva": "efavirenz",
    "niaspan": "niacin",
    "tranxene": "clorazepicacid",
    "wytensin": "guanabenz",
    "clolar": "clofarabine",
    "precedex": "dexmedetomidine",
    "deltasone": "prednisone",
    "ativan": "clofibrate",  # Note: Ativan is actually lorazepam - corrected below
    "hismanal": "astemizole",  # Withdrawn
    "adenocard": "adenosine",
    "zocor": "simvastatin",
    "alimta": "pemetrexed",
    "vermox": "mebendazole",
    "darvon": "dextropropoxyphene",  # Withdrawn
    "lysodren": "mitotane",
    "depo-estradiol": "estrone",
    "desyrel": "trazodone",
    "calan": "verapamil",
    "flumethasone": "flumethasone",  # Rarely branded
    "nilandron": "nilutamide",
    "epipen": "epinephrine",
    "imitrex": "sumatriptan",
    "diabinese": "chlorpropamide",
    "emend": "aprepitant",
    "razadyne": "galantamine",
    "nolvadex": "tamoxifen",
    "floropryl": "isoflurophate",
    "cozaar": "losartan",
    "mellaril": "thioridazine",
    "coumadin": "warfarin",
    "versed": "midazolam",
    "tobi": "tobramycin",
    "trovan": "trovafloxacin",  # Restricted
    "florinef": "fludrocortisone",
    "cellcept": "mycophenolatemofetil",
    "dalmane": "flurazepam",
    "cerubidine": "daunorubicin",
    "lasix": "furosemide",
    "cafergot": "ergotamine",  # Combo drug
    "zanaflex": "tizanidine",
    "macrodantin": "nitrofurantoin",
    "sermion": "nicergoline",
    "inspra": "eplerenone",
    "agenerase": "amprenavir",
    "revia": "naltrexone",
    "rescriptor": "delavirdine",
    "flomax": "tamsulosin",
    "sufenta": "sufentanil",
    "epivir": "lamivudine",
    "hetrazan": "diethylcarbamazine",
    "ansaid": "flurbiprofen",
    "apokyn": "apomorphine",
    "paxil": "paroxetine",
    "aygestin": "norethisterone",
    "hepsera": "adefovirdipivoxil",
    "optimine": "azatadine",
    "bonefos": "clodronicacid",
    "novocain": "procaine",
    "aldara": "imiquimod",
    "surmontil": "trimipramine",
    "nitrostat": "nitroglycerin",
    "mintezol": "thiabendazole",
    "starlix": "nateglinide",
    "risperdal": "risperidone",
    "nexium": "esomeprazole",
    "antivert": "meclizine",
    "nebupent": "pentamidine",
    "rilutek": "riluzole",
    "cortef": "hydrocortisone",
    "zyflo": "zileuton",
    "provigil": "modafinil",
    "desferal": "deferoxamine",
    "lodine": "etodolac",
    "elestat": "epinastine",
    "parnate": "tranylcypromine",
    "forane": "isoflurane",
    "retin-a": "tretinoin",
    "anzemet": "dolasetron",
    "plavix": "clopidogrel",
    "sumycin": "tetracycline",
    "merrem": "meropenem",
    "camptosar": "irinotecan",
    "tapazole": "methimazole",
    "asmanex": "mometasone",
    "vepesid": "etoposide",
    "trileptal": "oxcarbazepine",
    "rulide": "roxithromycin",
    "nardil": "phenelzine",
    "estrace": "estradiol",
    "ponstel": "mefenamicacid",
    "zovirax": "acyclovir",
    "aleve": "naproxen",  # OTC brand
    "aceon": "perindopril",
    "pelamine": "tripelennamine",
    "mysoline": "primidone",
    "atacand": "candesartancilexetil",
    "tazorac": "tazarotene",
    "alfenta": "alfentanil",
    "cantor": "minaprine",
    "trental": "pentoxifylline",
    "lozol": "indapamide",
    "akineton": "biperiden",
    "rebetol": "ribavirin",
    "butazolidin": "phenylbutazone",
    "duragesic": "fentanyl",
    "mobic": "meloxicam",
    "eradacil": "rosoxacin",
    "diprivan": "propofol",
    "diamox": "acetazolamide",
    "cialis": "tadalafil",
    "antabuse": "disulfiram",
    "levsin": "levomenthol",
    "cinobac": "cinoxacin",
    "valium": "diazepam",
    "stelazine": "trifluoperazine",
    "ceclor": "cefaclor",
    "mifeprex": "mifepristone",
    "imodium": "loperamide",
    "tolinase": "tolazamide",
    "dobutrex": "dobutamine",
    "serax": "oxazepam",
    "aricept": "donepezil",
    "lamprene": "clofazimine",
    "cordran": "flurandrenolide",
    "cystagon": "cysteamine",
    "meghabarbital": "methylphenobarbital",
    "trilafon": "perphenazine",
    "dtic-dome": "dacarbazine",
    "sudafed": "pseudoephedrine",
    "lamisil": "terbinafine",
    "orapred": "prednisolone",
    "dolobid": "diflunisal",
    "levitra": "vardenafil",
    "zantac": "ranitidine",  # Withdrawn in many markets
    "prograf": "tacrolimus",
    "didrex": "benzphetamine",
    "aptin": "alprenolol",
    "yutopar": "ritodrine",
    "trusopt": "dorzolamide",
    "profenal": "suprofen",
    "brethine": "terbutaline",
    "vaprisol": "conivaptan",
    "lotemax": "loteprednoletabonate",
    "fluanxol": "flupentixol",
    "rapamune": "sirolimus",
    "emtriva": "emtricitabine",
    "accupril": "quinapril",  # Duplicate entry
    "clomid": "clomifene",
    "isordil": "isosorbidedinitrate",
    "actonel": "risedronicacid",
    "bumex": "bumetanide",
    "kytril": "granisetron",
    "sulfabid": "sulfapyridine",
    "vexol": "rimexolone",
    "halcion": "triazolam",
    "antizol": "fomepizole",  # Note: Fomepizole is for ethanol/methanol poisoning
    "edecrin": "etacrynicacid",
    "zofran": "ondansetron",
    "lumigan": "bimatoprost",
    "gabitril": "tiagabine",
    "cocaine": "cocaine",  # Rarely prescribed medically
    "quinaglute": "quinidine",
    "zonegran": "zonisamide",
    "zemplar": "paricalcitol",
    "prandin": "repaglinide",
    "dibotin": "phenformin",  # Withdrawn
    "symmetrel": "amantadine",
    "flagyl": "metronidazole",
    "axert": "almotriptan",
    "suboxone": "buprenorphine",  # Combo with naloxone
    "flexeril": "cyclobenzaprine",
    "tegison": "etretinate",  # Withdrawn
    "pepcid": "famotidine",
    "vidaza": "azacitidine",
    "aptivus": "tipranavir",
    "serentil": "mesoridazine",
    "ludiomil": "maprotiline",
    "salacid": "salicylicacid",
    "serevent": "salmeterol",
    "vagantin": "methantheline",
    "hivid": "zalcitabine",  # Discontinued
    "aspirin": "acetylsalicylicacid",
    "marcoumar": "phenprocoumon",
    "faslodex": "fulvestrant",
    "felbatol": "felbamate",
    "inh": "isoniazid",
    "amerge": "naratriptan",
    "maxalt": "rizatriptan",
    "vicodin": "hydrocodone",  # Combo with acetaminophen
    "ortho-cyclen": "norgestimate",
    "medrol": "methylprednisolone",
    "visken": "pindolol",
    "sonata": "zaleplon",
    "xibrom": "bromfenac",
    "micardis": "telmisartan",
    "aldomet": "methyldopa",
    "lotronex": "alosetron",
    "astelin": "azelastine",
    "zetia": "ezetimibe",
    "maxaquin": "lomefloxacin",
    "cyclogyl": "cyclopentolate",
    "rozerem": "ramelteon",
    "accutane": "isotretinoin",
    "foradil": "formoterol",
    "cytosar-u": "cytarabine",
    "intropin": "dopamine",
    "aromasin": "exemestane",
    "imuran": "azathioprine",
    "neurontin": "gabapentin",
    "adriamycin": "doxorubicin",
    "frova": "frovatriptan",
    "proventil": "salbutamol",
    "chloroprocaine": "chloroprocaine",
    "chiron": "levobupivacaine",
    "hydrea": "hydroxyurea",
    "femara": "letrozole",
    "vagistat": "tioconazole",
    "orudis": "ketoprofen",
    "metopirone": "metyrapone",
    "sensipar": "cinacalcet",
    "temovate": "clobetasolpropionate",
    "colazal": "balsalazide",
    "bactrim": "sulfamethoxazole",  # Combination drug
    "diabeta": "glyburide",
    "tenex": "guanfacine",
    "imdur": "isosorbidemononitrate",
    "naqua": "trichlormethiazide",
    "plendil": "felodipine",
    "cellcept": "mycophenolicacid",
    "nizoral": "ketoconazole",
    "penthrane": "methoxyflurane",
    "avapro": "irbesartan",
    "hycamtin": "topotecan",
    "benemid": "probenecid",
    "purinethol": "mercaptopurine",
    "pronestyl": "procainamide",
    "detrol": "tolterodine",
    "eldepryl": "selegiline",
    "tricor": "fenofibrate",
    "thalomid": "thalidomide",
    "namenda": "memantine",
    "tequin": "gatifloxacin",
    "rifadin": "rifampicin",
    "amitiza": "lubiprostone",
    "lidex": "fluocinonide",
    "ziagen": "abacavir",
    "motrin": "ibuprofen",
    "penicillin g": "benzylpenicillin",
    "baypress": "nitrendipine",
    "tonocard": "tocainide",
    "noroxin": "norfloxacin",
    "amoxil": "amoxicillin",
    "ditropan": "oxybutynin",
    "isuprel": "isoprenaline",
    "circadin": "melatonin",
    "glucotrol": "glipizide",
    "klonopin": "clonazepam",
    "phenergan": "promethazine",
    "primalan": "mequitazine",
    "reyataz": "atazanavir",
    "pexion": "perhexiline",
    "benadryl": "diphenhydramine",
    "lipitor": "atorvastatin",
    "zelnorm": "tegaserod",  # Withdrawn
    "sabril": "vigabatrin",
    "xenical": "orlistat",
    "pilocar": "pilocarpine",
    "anbesol": "benzocaine",
    "primaquine": "primaquine",  # Generic used as brand
    "lescol": "fluvastatin",
    "arava": "leflunomide",
    "crestor": "rosuvastatin",
    "orap": "pimozide",
    "xeloda": "capecitabine",
    "atabrine": "quinacrine",
    "zoloft": "sertraline",
    "meridia": "sibutramine",  # Withdrawn
    "modrastane": "trilostane",
    "monistat": "miconazole",
    "chlor-trimeton": "chlorpheniramine",
    "procardia": "nifedipine",
    "arfonad": "trimethaphan",
    "mepron": "atovaquone",
    "cordarone": "amiodarone",
    "proglycem": "diazoxide",
    "diamicron": "gliclazide",
    "mytelase": "ambenonium",
    "orinase": "tolbutamide",
    "avodart": "dutasteride",
    "spectazole": "econazole",
    "casodex": "bicalutamide",
    "aciphex": "rabeprazole",
    "dermatop": "prednicarbate",
    "paludrine": "proguanil",
    "actos": "pioglitazone",
    "coreg": "carvedilol",
    "levaquin": "levofloxacin",
    "anturane": "sulfinpyrazone",
    "mycamine": "micafungin",
    "sinequan": "doxepin",
    "ethyol": "amifostine",
    "serzone": "nefazodone",  # Withdrawn
    "norpramin": "desipramine",
    "surital": "thiamylal",
    "factive": "gemifloxacin",
    "wellbutrin": "bupropion",
    "halothane": "halothane",  # Generic name used
    "ciloxan": "ciprofloxacin",  # Ophthalmic
    "pletal": "cilostazol",
    "sporanox": "itraconazole",
    "matulane": "procarbazine",
    "trisenox": "arsenictrioxide",
    "aurorix": "moclobemide",
    "kantrex": "kanamycin",
    "norflex": "orphenadrine",
    "luminal": "phenobarbital",
    "lexapro": "escitalopram",
    "marezine": "cyclizine",
    "idamycin": "idarubicin",
    "ifex": "ifosfamide",
    "rythmol": "propafenone",
    "narcan": "naloxone",
    "motilium": "domperidone",
    "permax": "pergolide",  # Withdrawn
    "cleocin": "clindamycin",
    "prozac": "fluoxetine",  # Previously listed
    "dilaudid": "hydromorphone",  # Previously listed
    "sectral": "acebutolol",
    "azopt": "brinzolamide",
    "tambocor": "flecainide",
    "emcyt": "estramustine",
    "imovane": "zopiclone",
    "tubarine": "tubocurarine",
    "parlodel": "bromocriptine",
    "priftin": "rifapentine",
    "novantrone": "mitoxantrone",
    "ceenu": "lomustine",
    "zagam": "sparfloxacin",
    "betagan": "levobunolol",
    "biaxin": "clarithromycin",
    "rocephin": "ceftriaxone",
    "antizol": "fomepizole",
    "optipranolol": "metipranolol",
    "prosom": "estazolam",
    "propecia": "finasteride",
    "arimidex": "anastrozole",
    "halofantrine": "halfan",
    "xifaxan": "rifaximin",
    "ketalar": "ketamine",
    "entocort": "budesonide",
    "theolair": "aminophylline",
    "seroquel": "quetiapine",
    "orlaam": "levacetylmethadol",
    "enkaid": "encainide",
    "taxol": "paclitaxel",
    "invirase": "saquinavir",
    "reglan": "metoclopramide",
    "decadron": "dexamethasone",
    "larodopa": "levodopa",
    "sevorane": "sevoflurane",
    "abilify": "aripiprazole",
    "lopid": "gemfibrozil",
    "anafranil": "clomipramine",
    "vascor": "bepridil",
    "pavulon": "pancuronium",
    "taxotere": "docetaxel",
    "glurenorm": "gliquidone",
    "glufast": "mitiglinide",
    "ergometrine": "ergotrate",
    "sprycel": "dasatinib",
    "vyvanse": "lisdexamfetamine",
    "altabax": "retapamulin",
    "tykerb": "lapatinib",
    "verdeso": "desonide",
    "januvia": "sitagliptin",
    "dacogen": "decitabine",
    "noxafil": "posaconazole",
    "prezista": "darunavir",
    "invega": "paliperidone",
    "sutent": "sunitinib",
    "brovana": "arformoterol",
    "apresoline": "hydralazine",
    "aranelle": "nelarabine",
    "prexige": "lumiracoxib",  # Withdrawn
    "trazodone": "desyrel",  # Previously listed
    "coreg": "carvedilol",  # Duplicate
    "tenormin": "atenolol",  # Previously listed
    "zestril": "lisinopril",  # Not in list
    "eliquis": "apixaban",
    "bosulif": "bosutinib",
    "inlyta": "axitinib",
    "ampyra": "dalfampridian",
    "xalkori": "crizotinib",
    "ella": "ulipristal",
    "gilenya": "fingolimod",
    "victrelis": "boceprevir",
    "dificid": "fidaxomicin",
    "cometriq": "cabozantinib",
    "jakafi": "ruxolitinib",
    "zelboraf": "vemurafenib",
    "tradjenta": "linagliptin",
    "fycompa": "perampanel",
    "myrbetriq": "mirabegron",
    "xeljanz": "tofacitinib",
    "stivarga": "regorafenib",
    "tudorza": "aclidinium",
    "xtandi": "enzalutamide",
    "iclusig": "ponatinib",
    "sirturo": "bedaquiline",
    "breo ellipta": "fluticasonefuroate",  # Combination
    "invokana": "canagliflozin",
    "pomalyst": "pomalidomide",
    "mekinist": "trametinib",
    "tafinlar": "dabrafenib",
    "fetzima": "levomilnacipran",
    "tivicay": "dolutegravir",
    "adempas": "riociguat",
    "opsumit": "macitentan",
    "luzu": "luliconazole",
    "sovaldi": "sofosbuvir",
    "tekturna": "aliskiren",
    "zontivity": "vorapaxar",
    "imbruvica": "ibrutinib",
    "zygen": "idelalisib",
    "epidiolex": "cannabidiol",
    "zykadia": "ceritinib",
    "trintellix": "vortioxetine",
    "ibrance": "palbociclib",
    "lynparza": "olaparib",
    "nexletol": "bempedoicacid",
    "nurtec": "rimegepant",
    "avigan": "favipiravir",
    "aliqopa": "copanlisib",
    "reblozyl": "luspatercept",  # Note: Not in list
    "vyndaqel": "tafamidis",  # Not in list
    "valturna": "aliskiren",
    "valdoxan": "agomelatine",
    "rydapt": "midostaurin",
    "farydak": "panobinostat",
    "nubeqa": "darolutamide",
    "vitrakvi": "larotrectinib",
    "veklury": "remdesivir",
    "oxbryta": "voxelotor",
    "brukinsa": "zanubrutinib",
    "rinvoq": "upadacitinib",
    "pemazyre": "pemigatinib",
    "ayvakit": "avapritinib",
    "ubrogepant": "ubrogepant",
    "trikafta": "elexacaftor",  # Combination
    "retevmo": "selpercatinib",
    "ozempic": "semaglutide",
    "wegovy": "semaglutide",
    "descovy": "tenofoviralafenamide",  # Combination
    "biktarvy": "bictegravir",  # Combination
    "olumiant": "baricitinib",
    "spravato": "esketamine",
    "steGLATRO": "ertugliflozin",
    "nerlynx": "neratinib",
    "isturisa": "osilodrostat",
    "yupelri": "revefenacin",
    "zulresso": "brexanolone",
    "erleada": "apalutamide",
    "austedo": "deutetrabenazine",
    "balversa": "erdafitinib",
    "jevtana": "cabazitaxel",
    "qulipta": "atogepant",  # Not in list
    "leqvio": "inclisiran",  # Not in list
    "muraglitazar": "muraglitazar",  # Withdrawn (no brand)
    "repaire": "repinotan",  # Never marketed
    "alvodax": "deramciclane",  # Experimental
    "dynepo": "mitemcinal",  # Withdrawn
}

SEVERITY_MAP = {
    "LOW": {"color": "#10b981", "action": "Monitor therapy"},
    "MODERATE": {"color": "#f59e0b", "action": "Consider alternative"},
    "SEVERE": {"color": "#ef4444", "action": "Contraindicated"},
}

def normalize_name(name: str) -> str:
    return unidecode(name).lower().strip().replace("-", " ").replace("  ", " ")

def find_best_match(name: str) -> str:
    """Enhanced drug matching with synonym resolution and validation"""
    original = normalize_name(name)
    
    # Normalize drug_mapping keys for consistent lookup
    drug_mapping_normalized = {normalize_name(k): k for k in drug_mapping.keys()}
    
    # 1. Check direct synonyms first
    if original in DRUG_SYNONYMS:
        generic_name = DRUG_SYNONYMS[original]
        normalized_generic = normalize_name(generic_name)
        if normalized_generic in drug_mapping_normalized:
            return drug_mapping_normalized[normalized_generic]  # Return original case from drug_mapping
        logger.warning(f"Synonym {original} maps to missing generic: {generic_name}")

    # 2. Check if input is already a known generic
    if original in drug_mapping_normalized:
        return drug_mapping_normalized[original]

    # 3. Search in synonyms' generic names
    for brand, generic in DRUG_SYNONYMS.items():
        if original == normalize_name(generic):
            normalized_generic = normalize_name(generic)
            if normalized_generic in drug_mapping_normalized:
                return drug_mapping_normalized[normalized_generic]
            logger.warning(f"Generic {generic} from synonym not in drug_mapping")
            break

    # 4. Fuzzy match with combined dictionary (brands + generics)
    combined_names = list(drug_mapping.keys()) + list(DRUG_SYNONYMS.keys())
    match, score = process.extractOne(original, combined_names, scorer=fuzz.QRatio)
    
    if score > 85:
        final_name = DRUG_SYNONYMS.get(match, match)
        normalized_final = normalize_name(final_name)
        if normalized_final in drug_mapping_normalized:
            return drug_mapping_normalized[normalized_final]
        logger.warning(f"Fuzzy match {match} resolved to invalid generic: {final_name}")

    logger.error(f"No valid match found for {original}")
    raise ValueError(f"No valid match found for {original}")

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
        matched_A = find_best_match(drugA)
        matched_B = find_best_match(drugB)

        # Check if the matched names exist in drug_mapping
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
        logger.info(f"Received drug check request for: {drug_name}")

        # Normalize and find match
        normalized = normalize_name(drug_name)
        mapped_name = find_best_match(normalized)

        # Ensure mapped_name is checked correctly
        normalized_mapped = normalize_name(mapped_name)
        exists = normalized_mapped in {normalize_name(k): v for k, v in drug_mapping.items()}

        logger.info(f"Final matched name: {mapped_name} (Exists in database: {exists})")

        return {
            "input": drug_name,
            "normalized": normalized,
            "matched_name": mapped_name,
            "exists": exists,
            "message": f"Drug {'found' if exists else 'not found'} in database"
        }

    except Exception as e:
        logger.error(f"Unexpected error in /check-drug/: {str(e)}", exc_info=True)
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

@app.get("/test-synonym/{drug_name}")
async def test_synonym(drug_name: str):
    normalized = normalize_name(drug_name)
    best_match = find_best_match(normalized)
    return {"input": drug_name, "normalized": normalized, "best_match": best_match}

app.get("/debug/synonyms")
async def debug_synonyms():
    return {"total_synonyms": len(DRUG_SYNONYMS), "example": dict(list(DRUG_SYNONYMS.items())[:10])}

@app.get("/synonym-suggestions/{query}")
async def synonym_suggestions(query: str):
    try:
        normalized_query = normalize_name(query)
        drug_mapping_normalized = {normalize_name(k): k for k in drug_mapping.keys()}
        results = []
        for term, generic in DRUG_SYNONYMS.items():
            normalized_generic = normalize_name(generic)
            if normalized_generic not in drug_mapping_normalized:
                continue
            term_normalized = normalize_name(term)
            # Stricter matching
            if term_normalized.startswith(normalized_query) or fuzz.partial_ratio(normalized_query, term_normalized) > 90:
                results.append({"display": f"{term} (Brand of {generic})", "value": generic})
            if normalized_query in normalized_generic:
                brands = [k for k, v in DRUG_SYNONYMS.items() if normalize_name(v) == normalized_generic]
                results.append({"display": f"{generic} (Generic for: {', '.join(brands)})", "value": generic})
        seen = set()
        final_results = [r for r in results if not (r['value'] in seen or seen.add(r['value']))][:10]
        return {"suggestions": final_results}
    except Exception as e:
        logger.error(f"Synonym suggestion error: {str(e)}")
        raise HTTPException(500, "Suggestion service unavailable")


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
# Add this at the end of your API code
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))