import csv
import json
import os

def process_csv_file(csv_file_path, messages):
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        system_message = ""

        for row in csv_reader:
            id = ''

            if row["role"] == "user":
                id = f"user_id: {row['user_id']}. "
            # Process the content without the column name
            content = id + row["content"].replace('\n', ' ').replace('\r', ' ').strip()

            if row["role"] == "system":
                system_message += content + " "
                continue

            if system_message:
                messages.append({"role": "system", "content": system_message.strip()})
                system_message = ""

            message = {
                "role": row["role"],
                "content": content
            }
            if row["role"] == "assistant":
                message["weight"] = 1 if row["flag"].lower() == "true" else 0

            messages.append(message)

        if system_message:
            messages.append({"role": "system", "content": system_message.strip()})


def csv_to_jsonl(input_dir, jsonl_file_path):
    messages = []

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.startswith('flagged_') and filename.endswith('.csv') and not filename.endswith('_condensed.csv'):
                csv_file_path = os.path.join(root, filename)
                process_csv_file(csv_file_path, messages)

    with open(jsonl_file_path, 'w') as jsonl_file:
        jsonl_file.write(json.dumps({"messages": messages}) + '\n')


if __name__ == "__main__":
    input_dir = 'test_script'  # Replace with the path to your directory
    jsonl_file_path = 'converted_output_2.jsonl'  # Replace with the desired output JSONL file path
    csv_to_jsonl(input_dir, jsonl_file_path)
