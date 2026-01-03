import os

files = [
    "public/models/paddle/structure_v3/layout.onnx",
    "public/models/paddle/structure_v3/table_structure.onnx",
]

for f in files:
    if os.path.exists(f):
        print(f"{f}: {os.path.getsize(f)} bytes")
    else:
        print(f"{f}: MISSING")
