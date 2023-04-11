import json


def fetch_json(path: str) -> dict:
    try:
        with open(path, "r") as file:
            json_file = json.load(file)
    except Exception as e:
        print(f"Error while reading json file: {e}")
        print(f"Returning empty dict")
        json_file = {}
    finally:
        file.close()
    return json_file


def save_json(json_dict: dict, path: str) -> None:
    try:
        with open(path, "w") as file:
            json.dump(json_dict, file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error while saving json file: {e}")
    finally:
        file.close()
