# from transformers import AutoModelForSequenceClassification, AutoTokenizer

# model_name = "dost-asti/RoBERTa-tl-sentiment-analysis"
# local_path = "./models/itanong_roberta"

# # Download and save
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

# tokenizer.save_pretrained(local_path)
# model.save_pretrained(local_path)

import polars as pl

from beacon_analysis_server.config import DATA_DIR

df = pl.read_excel(
    f"{DATA_DIR}/raw/filipino-wordsentiment-dictionary/PH Word Sentiments Full.xlsx"
)

df.write_csv(f"{DATA_DIR}/processed/filipino-wordsentiment-dictionary/PH Word Sentiments Full.csv")

(
    "Nawalan po kami ng kuryente bandang alas-8 ng gabi. Pati mga streetlight sa lugar namin ay patay.",
)
("Wala pa ring kuryente mula kaninang umaga. Marami ring bahay sa paligid ang apektado.",)
("Paulit-ulit po ang brownout ngayong araw. Maya-maya ay nawawala tapos bumabalik din.",)
(
    "Bigla pong nag-blackout matapos ang malakas na ulan. May narinig na malakas na tunog sa malapit na transformer.",
)
("Simula alas-2 ng madaling-araw ay wala na pong kuryente hanggang ngayon.",)
("Bahagyang brownout lang po. May ilaw pero walang kuryente ang mga saksakan.",)
("Nawalan ng kuryente habang malakas ang hangin. Posibleng may naputol na linya ng kuryente.",)
("Bigla na lang pong nawalan ng kuryente kaninang hapon. Wala pong abiso beforehand.",)
("Buong block po namin ay walang kuryente, pati streetlights ay hindi gumagana.",)
("Nanghihina at nag-flicker ang ilaw bago tuluyang mawala ang kuryente kagabi.",)
("May narinig kaming parang pagsabog malapit sa poste bago mawalan ng kuryente.",)
("Wala pa ring kuryente mula tanghali. Hindi na rin makapagbukas ang mga tindahan sa lugar.",)
("Ilang beses pong nawawala ang kuryente ngayong araw, ngayon tuluyan na itong nawala.",)
("Nawalan ng kuryente habang may thunderstorm. Posibleng tinamaan ng kidlat.",)
("Maraming kabahayan sa barangay namin ang apektado ng brownout.",)
("Madaling-araw pa po nagsimula ang brownout. Hindi gumagana ang ref at internet.",)
(
    "Biglang nawalan ng kuryente habang gumagamit ng appliances. Ayos naman ang circuit breaker sa bahay.",
)
("Sa bahay lang po namin walang kuryente, pero may kuryente ang mga kapitbahay.",)
(
    "Nagkaroon ng brownout bandang alas-6 ng gabi at mahigit dalawang oras na pong wala hanggang ngayon.",
)
("Natapos na raw ang scheduled maintenance pero wala pa ring kuryente sa aming lugar.",)
