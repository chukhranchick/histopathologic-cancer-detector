import json


def fetch_json(path: str) -> dict:
    with open(path, "r") as file:
        json_file = json.load(file)
        file.close()
    return json_file


def save_json(json_dict: dict, path: str):
    with open(path, "w") as file:
        json.dump(json_dict, file)
        file.close()
