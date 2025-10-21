import json
import argparse

def convert_jsonl_to_tsv(input):
    topic = {}
    with open(input, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            id = data.get('id', None)
            text = data.get('text', data.get('request', ''))
            topic[id] = text

    output = input.replace('.jsonl', '.tsv')
    with open(output, 'w', encoding='utf-8') as f_out:
        for id, text in topic.items():
            f_out.write(f"{id}\t{text}\n")

    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    args = parser.parse_args()
    output = convert_jsonl_to_tsv(args.input)
    print(f"The .tsv file is saved to {output}")
