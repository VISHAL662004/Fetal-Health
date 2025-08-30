import os, json

# Get absolute path to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct correct path to data/fetal_parameters.json
json_path = os.path.join(BASE_DIR, "data", "fetal_parameters.json")

with open(json_path) as f:
    knowledge_base = json.load(f)

def lookup_parameter(param: str):
    param = param.lower()
    if param in knowledge_base:
        return knowledge_base[param]
    return None
