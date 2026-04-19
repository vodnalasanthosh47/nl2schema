# filtering with control over the probability threshold

import json
import fasttext

PATH_TO_DATA_FOLDER = "../../data/schemapile"

MODEL_PATH = "../../utils/lid.176.bin"
PROB_THRESHOLD = 0.8
MARGIN_THRESHOLD = 0.15

model = fasttext.load_model(MODEL_PATH)


def get_json_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.loads(f.read())


def normalize_identifiers(identifiers):

    text = ' '.join(identifiers)

    text = text.lower()

    text = text.replace('_', ' ')

    # expand common abbreviations to improve language detection signal
    text = (
        text.replace(" dept ", " department ")
        .replace(" addr ", " address ")
        .replace(" loc ", " location ")
        .replace(" desc ", " description ")
        .replace(" info ", " information ")
        .replace(" num ", " number ")
        .replace(" amt ", " amount ")
        .replace(" qty ", " quantity ")
        .replace(" emp ", " employee ")
        .replace(" cust ", " customer ")
        .replace(" prod ", " product ")
    )

    return text



def detect_language_fasttext(text, k=2):

    labels, probs = model.predict(text, k=k)

    languages = [
        label.replace("__label__", "")
        for label in labels
    ]

    return languages, probs



def filter_schemas_by_language(
    schemas,
    language="en",
    prob_threshold=PROB_THRESHOLD,
    margin_threshold=MARGIN_THRESHOLD
):

    filtered_schemas = []

    failed_schemas = []

    for i, schema in enumerate(schemas):

        print(f"Detecting language for schema {i+1}...")

        identifiers = []

        for table_name, table_data in schema["tables"].items():

            identifiers.append(table_name)

            for column_name in table_data["COLUMNS"].keys():

                identifiers.append(column_name)

        text = normalize_identifiers(identifiers)

        try:

            languages, probs = detect_language_fasttext(text, k=2)

            top_lang = languages[0]
            top_prob = probs[0]

            second_lang = languages[1] if len(languages) > 1 else None
            second_prob = probs[1] if len(probs) > 1 else 0.0

            margin = top_prob - second_prob

            print(
                f"Top: {top_lang} ({top_prob:.3f}), "
                f"Second: {second_lang} ({second_prob:.3f}), "
                f"Margin: {margin:.3f}"
            )

            if (
                top_lang == language
                and top_prob >= prob_threshold
                and margin >= margin_threshold
            ):

                filtered_schemas.append(i+1)

            else:

                failed_schemas.append(i+1)

                print(f"Rejected text sample: {text[:150]}")

        except Exception as e:

            print(f"Error detecting language for schema {i+1}: {e}")

            failed_schemas.append(i+1)

            print(f"Text sample: {text[:150]}")

    return filtered_schemas, failed_schemas



def main():

    schemapile = get_json_from_file(
        f"{PATH_TO_DATA_FOLDER}/processed/schemapile-pruned-sample200.json"
    )

    schemas = schemapile["schemas"]

    filtered_schema_indices, failed_schema_indices = filter_schemas_by_language(
        schemas,
        language="en",
        prob_threshold=0.2,
        margin_threshold=0.1
    )

    print("\nSchemas detected as English:")
    print(filtered_schema_indices)

    print("\nSchemas rejected:")
    print(failed_schema_indices)
    print(len(failed_schema_indices))


if __name__ == "__main__":
    main()