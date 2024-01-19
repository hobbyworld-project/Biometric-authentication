from datetime import datetime, timedelta
import os
import json

# Function to update the status of a task in a JSON file
def update_status_json(id, task, result):
    # Define the path to the status file based on the id
    status_file = f"temporary/{id}/status.json"
    
    # Check if the status file exists
    if not os.path.exists(status_file):
        # Initialize the status dictionary if the file does not exist
        status = {"task1": False, "task2": False, "task3": False}
    else:
        # If the file exists, load the current status from the file
        with open(status_file, 'r') as file:
            status = json.load(file)

    # Update the status of the specified task with the result
    status[task] = result
    
    # Write the updated status back to the file
    with open(status_file, 'w') as file:
        json.dump(status, file)
    
    # Return the updated status dictionary
    return status

# Function to check if all tasks are completed
def check_all_tasks_completed(status):
    # Check if all values in the status dictionary are True
    return all(status.values())

# Function to generate a normalized filename for a task
def get_normalized_filename(id, task, original_filename):
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # Extract the file extension from the original filename
    file_extension = os.path.splitext(original_filename)[1]
    # Return a normalized filename using the id, task, timestamp, and file extension
    return f"{id}_{task}_{timestamp}{file_extension}"
