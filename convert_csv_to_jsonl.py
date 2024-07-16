import csv
import json

import csv
import json


def csv_to_jsonl(csv_file_path, jsonl_file_path):
    # with open(csv_file_path, 'r') as csv_file, open(jsonl_file_path, 'w') as jsonl_file:
    #     csv_reader = csv.DictReader(csv_file)
    #     current_conversation = []
    #     system_message = ""
    #
    #     for row in csv_reader:
    #         # Concatenate the content column without the column name and other columns with their names
    #         content = row["content"].replace('\n', ' ').replace('\r', ' ').strip()
    #         additional_content = " ".join(
    #             [f"{key}: {str(value)}" for key, value in row.items() if
    #              key not in ["content", "Conv_id", "time", "role", "user_id"]]
    #         )
    #         if row["role"] == "user":
    #             id = str(row["user_id"])
    #             id_content = f"user_id: {id}"
    #             full_content = f"{content} {id_content} {additional_content}".strip()
    #         else:
    #             full_content = f"{content} {additional_content}".strip()
    #
    #         if row["role"] == "system":
    #             system_message += full_content + " "
    #             continue
    #
    #         if system_message:
    #             current_conversation.append({"role": "system", "content": system_message.strip()})
    #             system_message = ""
    #
    #         message = {
    #             "role": row["role"],
    #             "content": full_content
    #         }
    #         if row["role"] == "assistant":
    #             message["weight"] = 1 if row["flag"].lower() == "true" else 0
    #
    #         current_conversation.append(message)
    #
    #         if row["role"] == "assistant":
    #             jsonl_file.write(json.dumps({"messages": current_conversation}) + '\n')
    #             current_conversation = []
    #
    #     if system_message:
    #         current_conversation.append({"role": "system", "content": system_message.strip()})
    #         jsonl_file.write(json.dumps({"messages": current_conversation}) + '\n')

    with open(csv_file_path, 'r') as csv_file, open(jsonl_file_path, 'w') as jsonl_file:
        csv_reader = csv.DictReader(csv_file)
        current_conversation = []
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
                current_conversation.append({"role": "system", "content": system_message.strip()})
                system_message = ""

            message = {
                "role": row["role"],
                "content": content
            }
            if row["role"] == "assistant":
                message["weight"] = 1 if row["flag"].lower() == "true" else 0

            current_conversation.append(message)

            if row["role"] == "assistant":
                jsonl_file.write(json.dumps({"messages": current_conversation}) + '\n')
                current_conversation = []

        if system_message:
            current_conversation.append({"role": "system", "content": system_message.strip()})
            jsonl_file.write(json.dumps({"messages": current_conversation}) + '\n')

if __name__ == "__main__":
    csv_file_path = 'flagged_20240705.csv'  # Replace with the path to your CSV file
    jsonl_file_path = 'converted_output.jsonl'  # Replace with the desired output JSONL file path
    csv_to_jsonl(csv_file_path, jsonl_file_path)
