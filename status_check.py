from datetime import datetime, timedelta
import os
import json

def update_status_json(id, task, result):
    status_file = f"temporary/{id}/status.json"
    if not os.path.exists(status_file):
        status = {"task1": False, "task2": False, "task3": False}
    else:
        with open(status_file, 'r') as file:
            status = json.load(file)

    status[task] = result
    with open(status_file, 'w') as file:
        json.dump(status, file)
    return status

def check_all_tasks_completed(status):
    return all(status.values())

def get_normalized_filename(id, task, original_filename):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_extension = os.path.splitext(original_filename)[1]
    return f"{id}_{task}_{timestamp}{file_extension}"