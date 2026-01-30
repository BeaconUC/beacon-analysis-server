# from transformers import AutoModelForSequenceClassification, AutoTokenizer

# model_name = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
# local_path = "./models/zero-shot-rca-model"

# # Download and save
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

# tokenizer.save_pretrained(local_path)
# model.save_pretrained(local_path)

# from transformers import pipeline

# from beacon_analysis_server.config import MODELS_DIR

# model_name = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
# classifier = pipeline("zero-shot-classification", model=model_name)

# save_directory = f"{MODELS_DIR}/models/zero-shot-model"
# classifier.save_pretrained(save_directory)

# print(f"Model saved to {save_directory}")

# import polars as pl

# from beacon_analysis_server.config import DATA_DIR

# df = pl.read_excel(
#     f"{DATA_DIR}/raw/filipino-wordsentiment-dictionary/PH Word Sentiments Full.xlsx"
# )

# df.write_csv(f"{DATA_DIR}/processed/filipino-wordsentiment-dictionary/PH Word Sentiments Full.csv")

# (
#     "Nawalan po kami ng kuryente bandang alas-8 ng gabi. Pati mga streetlight sa lugar namin ay patay.",
# )
# ("Wala pa ring kuryente mula kaninang umaga. Marami ring bahay sa paligid ang apektado.",)
# ("Paulit-ulit po ang brownout ngayong araw. Maya-maya ay nawawala tapos bumabalik din.",)
# (
#     "Bigla pong nag-blackout matapos ang malakas na ulan. May narinig na malakas na tunog sa malapit na transformer.",
# )
# ("Simula alas-2 ng madaling-araw ay wala na pong kuryente hanggang ngayon.",)
# ("Bahagyang brownout lang po. May ilaw pero walang kuryente ang mga saksakan.",)
# ("Nawalan ng kuryente habang malakas ang hangin. Posibleng may naputol na linya ng kuryente.",)
# ("Bigla na lang pong nawalan ng kuryente kaninang hapon. Wala pong abiso beforehand.",)
# ("Buong block po namin ay walang kuryente, pati streetlights ay hindi gumagana.",)
# ("Nanghihina at nag-flicker ang ilaw bago tuluyang mawala ang kuryente kagabi.",)
# ("May narinig kaming parang pagsabog malapit sa poste bago mawalan ng kuryente.",)
# ("Wala pa ring kuryente mula tanghali. Hindi na rin makapagbukas ang mga tindahan sa lugar.",)
# ("Ilang beses pong nawawala ang kuryente ngayong araw, ngayon tuluyan na itong nawala.",)
# ("Nawalan ng kuryente habang may thunderstorm. Posibleng tinamaan ng kidlat.",)
# ("Maraming kabahayan sa barangay namin ang apektado ng brownout.",)
# ("Madaling-araw pa po nagsimula ang brownout. Hindi gumagana ang ref at internet.",)
# (
#     "Biglang nawalan ng kuryente habang gumagamit ng appliances. Ayos naman ang circuit breaker sa bahay.",
# )
# ("Sa bahay lang po namin walang kuryente, pero may kuryente ang mga kapitbahay.",)
# (
#     "Nagkaroon ng brownout bandang alas-6 ng gabi at mahigit dalawang oras na pong wala hanggang ngayon.",
# )
# ("Natapos na raw ang scheduled maintenance pero wala pa ring kuryente sa aming lugar.",)

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from beacon_analysis_server.config import MODELS_DIR

models = [
    # "dost-asti/RoBERTa-tl-sentiment-analysis",
    # "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
    f"{MODELS_DIR}/roberta-sentiment-custom"
]

for m in models:
    model = ORTModelForSequenceClassification.from_pretrained(m, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(m, fix_mistral_regex=True)

    model.save_pretrained(f"{MODELS_DIR}/roberta_sentiment_custom")
    tokenizer.save_pretrained(f"{MODELS_DIR}/roberta_sentiment_custom")

# import onnx
# from onnxruntime.quantization import QuantType, quantize_dynamic, shape_inference

# from beacon_analysis_server.config import MODELS_DIR

# models = [
#     "itanong_roberta",
#     #  "minilmv2"
# ]


# def quantize_for_haswell(model_path):
#     input_fp32 = f"{model_path}/model_optimized.onnx"
#     preprocessed_path = f"{model_path}/model_optimized_pre.onnx"
#     output_quant = f"{model_path}/model_optimized_pre_quantized.onnx"

#     shape_inference.quant_pre_process(
#         input_fp32, preprocessed_path, skip_symbolic_shape_inference=False
#     )

#     quantize_dynamic(
#         model_input=preprocessed_path,
#         model_output=output_quant,
#         weight_type=QuantType.QInt8,
#         extra_options={
#             "DefaultTensorType": onnx.TensorProto.FLOAT,
#             "EnableQuantizeScaleShift": True,
#         },
#     )


# for m in models:
#     model_input = f"{MODELS_DIR}/{m}"
#     # model_output = f"{MODELS_DIR}/{m}_quantized.onnx"
#     quantize_for_haswell(model_path=model_input)

# from optimum.onnxruntime import ORTOptimizer
# from optimum.onnxruntime.configuration import OptimizationConfig

# from beacon_analysis_server.config import MODELS_DIR

# models = ["itanong_roberta", "minilmv2"]


# def optimize_for_haswell(model_path):
#     optimizer = ORTOptimizer.from_pretrained(model_path, file_names=["model.onnx"])
#     optimization_config = OptimizationConfig(optimization_level=99)

#     optimizer.optimize(
#         save_dir=model_path,
#         optimization_config=optimization_config,
#     )


# for m in models:
#     model_input = f"{MODELS_DIR}/{m}"
#     optimize_for_haswell(model_path=model_input)

# import onnx

# from beacon_analysis_server.config import MODELS_DIR

# onnx_models = ["model", "model_optimized", "model_quantized", "model_quantized_optimized"]

# for m in onnx_models:
#     model = onnx.load(f"{MODELS_DIR}/minilmv2/{m}.onnx")
#     nodes = [n.op_type for n in model.graph.node]

#     print(f"Model: {m}")
#     print(f"QLinearMatMul: {nodes.count('QLinearMatMul')}")
#     print(f"QAttention: {nodes.count('QAttention')}")
#     print(f"MatMulInteger: {nodes.count('MatMulInteger')}")
#     print(f"DynamicQuantizeLinear: {nodes.count('DynamicQuantizeLinear')}\n")
