import csv
import json


def csv_to_jsonl(csv_file_path, jsonl_file_path):
    with open(csv_file_path, 'r') as csv_file, open(jsonl_file_path, 'w') as jsonl_file:
        csv_reader = csv.DictReader(csv_file)
        messages = []
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

        jsonl_file.write(json.dumps({"messages": messages}) + '\n')

if __name__ == "__main__":
    csv_file_path = 'flagged_20240705.csv'  # Replace with the path to your CSV file
    jsonl_file_path = 'converted_output.jsonl'  # Replace with the desired output JSONL file path
    csv_to_jsonl(csv_file_path, jsonl_file_path)
