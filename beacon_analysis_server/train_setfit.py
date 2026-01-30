from datasets import Dataset
import polars as pl
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer

from beacon_analysis_server.config import MODELS_DIR

train_examples = [
    {
        "text": "Nagkaroon ng brownout matapos ang malakas na ulan at hangin.",
        "label": "Weather Event",
    },
    {
        "text": "Nawala ang kuryente kasabay ng malakas na hangin at ulan.",
        "label": "Weather Event",
    },
    {
        "text": "Biglang nag-blackout habang umuulan ng malakas.",
        "label": "Weather Event",
    },
    {
        "text": "Walang kuryente matapos ang tuloy-tuloy na ulan.",
        "label": "Weather Event",
    },
    {
        "text": "Brownout po sa aming lugar dahil sa masamang panahon.",
        "label": "Weather Event",
    },
    {
        "text": "Nawalan ng kuryente matapos bumagsak ang puno dahil sa hangin.",
        "label": "Weather Event",
    },
    {
        "text": "Namatay ang kuryente kasabay ng malakas na kulog at kidlat.",
        "label": "Lightning Damage",
    },
    {
        "text": "Nawalan ng kuryente pagkatapos ng malakas na kidlat.",
        "label": "Lightning Damage",
    },
    {
        "text": "May malakas na kulog bago tuluyang mawala ang kuryente.",
        "label": "Lightning Damage",
    },
    {
        "text": "Tinamaan ng kidlat ang linya kaya nag-blackout.",
        "label": "Lightning Damage",
    },
    {
        "text": "Kasabay ng thunder at kidlat ay nawala ang kuryente.",
        "label": "Lightning Damage",
    },
    {
        "text": "May narinig na putok sa poste bago mag-brownout.",
        "label": "Primary Line Damage",
    },
    {
        "text": "Nag-flicker ang ilaw bago tuluyang mawala ang kuryente.",
        "label": "Primary Line Damage",
    },
    {
        "text": "Biglang nawala ang kuryente sa buong street.",
        "label": "Primary Line Damage",
    },
    {
        "text": "May spark sa poste bago mawalan ng ilaw.",
        "label": "Primary Line Damage",
    },
    {
        "text": "Buong purok ay sabay-sabay nawalan ng kuryente.",
        "label": "Primary Line Damage",
    },
    {
        "text": "Patay ang lahat ng streetlight sa amin.",
        "label": "Primary Line Damage",
    },
    {
        "text": "May ilaw pero mahina ang boltahe.",
        "label": "Primary Line Damage",
    },
    {
        "text": "Nawalan ng kuryente habang may ginagawang kalsada sa tapat ng bahay.",
        "label": "Construction Damage",
    },
    {
        "text": "Biglang nawala ang kuryente habang may road repair.",
        "label": "Construction Damage",
    },
    {
        "text": "Nawalan ng kuryente habang may inaayos na poste.",
        "label": "Construction Damage",
    },
    {
        "text": "May ginagawang hukay sa kalsada bago nag-blackout.",
        "label": "Construction Damage",
    },
    {
        "text": "Biglang nawala ang kuryente kagabi at hanggang ngayon ay wala pa rin.",
        "label": "Restoration Delay",
    },
    {
        "text": "Mahigit tatlong oras nang walang kuryente.",
        "label": "Restoration Delay",
    },
    {
        "text": "Wala pa ring kuryente kahit tapos na ang maintenance.",
        "label": "Restoration Delay",
    },
    {
        "text": "Matagal nang brownout at wala pang update.",
        "label": "Restoration Delay",
    },
    {
        "text": "Bigla na lang nawalan, di ko alam kung bakit.",
        "label": "Unspecified Electrical Fault",
    },
    {
        "text": "Parang may nangyari pero di malinaw.",
        "label": "Unspecified Electrical Fault",
    },
    {
        "text": "May narinig ako sa labas tapos patay na yung ilaw.",
        "label": "Unspecified Electrical Fault",
    },
    {
        "text": "Mukhang may problema ulit sa kuryente.",
        "label": "Unspecified Electrical Fault",
    },
    {
        "text": "Nag off lang bigla tapos wala na.",
        "label": "Unspecified Electrical Fault",
    },
]

# train_examples = [
#     # Weather Event (8)
#     {
#         "text": "Nagkaroon ng brownout matapos ang malakas na ulan at hangin.",
#         "label": "Weather Event",
#     },
#     {
#         "text": "Namatay ang kuryente kasabay ng malakas na kulog at kidlat.",
#         "label": "Weather Event",
#     },
#     {"text": "Biglang nag-blackout habang umuulan ng malakas.", "label": "Weather Event"},
#     {
#         "text": "Nawala ang kuryente kasabay ng malakas na hangin at ulan.",
#         "label": "Weather Event",
#     },
#     {
#         "text": "Brownout po sa aming lugar simula pa kaninang madaling-araw.",
#         "label": "Weather Event",
#     },
#     {
#         "text": "Nawalan ng kuryente matapos bumagsak ang puno malapit sa linya.",
#         "label": "Weather Event",
#     },
#     {"text": "Walang kuryente matapos ang malakas na ulan.", "label": "Weather Event"},
#     {"text": "Nawalan ng kuryente pagkatapos ng malakas na kidlat.", "label": "Weather Event"},
#     # Lightning Damage (8)
#     {
#         "text": "Namatay ang kuryente kasabay ng malakas na kulog at kidlat.",
#         "label": "Lightning Damage",
#     },
#     {"text": "Nawalan ng kuryente pagkatapos ng malakas na kidlat.", "label": "Lightning Damage"},
#     {"text": "Nawalan ng kuryente kasabay ng malakas na kulog.", "label": "Lightning Damage"},
#     {"text": "Biglang nawala ang kuryente kahit walang ulan.", "label": "Lightning Damage"},
#     {
#         "text": "Nag-flicker ang ilaw bago tuluyang mawala ang kuryente.",
#         "label": "Lightning Damage",
#     },
#     {"text": "May narinig na putok sa poste bago mag-brownout.", "label": "Lightning Damage"},
#     {"text": "Nawala ang kuryente bandang hatinggabi.", "label": "Lightning Damage"},
#     {"text": "Biglang nag-blackout nang walang anumang babala.", "label": "Lightning Damage"},
#     # Primary Line Damage (8)
#     {"text": "May narinig na putok sa poste bago mag-brownout.", "label": "Primary Line Damage"},
#     {
#         "text": "Biglang nag-flicker ang ilaw bago tuluyang mawala ang kuryente.",
#         "label": "Primary Line Damage",
#     },
#     {
#         "text": "Nawalan ng kuryente habang may ginagawa sa poste ng kuryente.",
#         "label": "Primary Line Damage",
#     },
#     {
#         "text": "Walang kuryente sa buong street namin, pati mga streetlight ay patay.",
#         "label": "Primary Line Damage",
#     },
#     {"text": "Buong subdivision ay walang kuryente ngayon.", "label": "Primary Line Damage"},
#     {
#         "text": "Buong compound po namin ay walang kuryente simula kaninang umaga.",
#         "label": "Primary Line Damage",
#     },
#     {"text": "Buong purok ay apektado ng brownout.", "label": "Primary Line Damage"},
#     {"text": "Patay ang lahat ng ilaw sa street namin.", "label": "Primary Line Damage"},
#     # Construction Damage (8)
#     {
#         "text": "Nawalan ng kuryente habang may ginagawang kalsada sa tapat ng bahay.",
#         "label": "Construction Damage",
#     },
#     {
#         "text": "Nawalan ng kuryente habang may road repair sa lugar.",
#         "label": "Construction Damage",
#     },
#     {
#         "text": "Nawalan ng kuryente habang may ginagawang repair sa poste.",
#         "label": "Construction Damage",
#     },
#     {
#         "text": "Walang kuryente habang may ginagawang kalsada sa tapat.",
#         "label": "Construction Damage",
#     },
#     {"text": "Biglang nawala ang kuryente habang nagluluto.", "label": "Construction Damage"},
#     {
#         "text": "Nawalan ng kuryente habang nagcha-charge ng cellphone.",
#         "label": "Construction Damage",
#     },
#     {"text": "Walang kuryente sa buong apartment building namin.", "label": "Construction Damage"},
#     {
#         "text": "Nawalan ng kuryente habang may ginagawa sa poste ng kuryente.",
#         "label": "Construction Damage",
#     },
#     # Restoration Delay (8)
#     {
#         "text": "Biglang nawala ang kuryente kaninang gabi at hanggang ngayon ay hindi pa bumabalik.",
#         "label": "Restoration Delay",
#     },
#     {
#         "text": "Walang kuryente sa buong lugar simula pa kahapon ng hapon.",
#         "label": "Restoration Delay",
#     },
#     {
#         "text": "Wala pa ring kuryente kahit tapos na ang inanunsyong maintenance.",
#         "label": "Restoration Delay",
#     },
#     {
#         "text": "Matagal nang walang kuryente at wala pang update kung kailan babalik.",
#         "label": "Restoration Delay",
#     },
#     {
#         "text": "Mahigit tatlong oras nang walang kuryente sa lugar namin.",
#         "label": "Restoration Delay",
#     },
#     {
#         "text": "Walang kuryente simula pa kagabi at hanggang ngayon ay wala pa rin.",
#         "label": "Restoration Delay",
#     },
#     {
#         "text": "Namatay ang kuryente at hindi pa rin naaayos hanggang ngayon.",
#         "label": "Restoration Delay",
#     },
#     {
#         "text": "Walang kuryente at walang update mula sa electric company.",
#         "label": "Restoration Delay",
#     },
# ]

train_df = pl.DataFrame(train_examples)
train_ds = Dataset.from_polars(train_df)

model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds,
    loss_class=CosineSimilarityLoss,
    batch_size=8,
    num_iterations=20,
    num_epochs=1,
)
trainer.train()
trainer.model.save_pretrained(f"{MODELS_DIR}/setfit_v2")
