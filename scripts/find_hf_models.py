import os

try:
    from huggingface_hub import HfApi
except ImportError:
    path = "found_models.txt"
    with open(path, "w") as f:
        f.write("ERROR: huggingface_hub not installed")
    print("huggingface_hub not installed")
    exit(1)

api = HfApi()

with open("found_models.txt", "w", encoding="utf-8") as f:
    f.write("Searching for PP-StructureV3 models...\n")
    models = api.list_models(search="PP-StructureV3", author="PaddlePaddle")
    for model in models:
        f.write(f"Found model: {model.modelId}\n")

    f.write("\nSearching for SLANet models...\n")
    models4 = api.list_models(search="SLANet", author="PaddlePaddle")
    for model in models4:
        f.write(f"Found model: {model.modelId}\n")

    f.write("\nSearching for Table models...\n")
    models5 = api.list_models(search="Table", author="PaddlePaddle")
    for model in models5:
        f.write(f"Found model: {model.modelId}\n")

    f.write("\nSearching for structure models...\n")
    models2 = api.list_models(search="structure", author="PaddlePaddle")
    for model in models2:
        f.write(f"Found model: {model.modelId}\n")

    f.write("\nSearching for layout models...\n")
    models3 = api.list_models(search="layout", author="PaddlePaddle")
    for model in models3:
        f.write(f"Found model: {model.modelId}\n")
