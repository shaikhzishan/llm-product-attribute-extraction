import json
import os
import time
import random
import re
import csv
from pathlib import Path

# Reproducibility
SEED = 42
random.seed(SEED)

# Dataset
DATASET_PATH = Path("mave_sample_dataset.json")

# Experiment settings
ATTRIBUTES = ["brand", "color", "size", "material"]

CATEGORIES = [
    "Electronics",
    "Clothing",
    "Home",
    "Sports",
    "Food"
]

MODELS = [
    "gpt-4o-mini",
    "gemini-3.8-flash"
]

PROMPT_STRATEGIES = [
    "ZERO_SHOT",
    "FEW_SHOT",
    "SCHEMA_GUIDED",
    "DEFINITION_AUGMENTED"
]

TEMPERATURE = 0

print("Configuration loaded.")
print("Dataset:", DATASET_PATH)
print("Attributes:", ATTRIBUTES)
print("Models:", MODELS)
print("Prompt strategies:", PROMPT_STRATEGIES)


# Load dataset
def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Dataset should contain a list of products.")

    print(f"Loaded {len(data)} products.")

    return data


data = load_dataset()


# Prompt generation
def build_zero_shot_prompt(product):
    product_text = f"""
Title: {product["title"]}
Description: {product["description"]}
"""

    return f"""
Extract the following product attributes from the product information:

- brand
- color
- size
- material

Return the result as a JSON object with exactly these four fields.
If an attribute is not available, use null.

Product:
{product_text}
"""


def build_few_shot_prompt(product):
    product_text = f"""
Title: {product["title"]}
Description: {product["description"]}
"""

    examples = """
Example 1:

Product:
Title: Nike Air Max Running Shoes
Description: Black running shoes made with mesh upper. Size 10.

Output:
{
    "brand": "Nike",
    "color": "Black",
    "size": "10",
    "material": "Mesh"
}


Example 2:

Product:
Title: Samsung 55 Inch Smart TV
Description: 55-inch LED television with a black plastic body.

Output:
{
    "brand": "Samsung",
    "color": "Black",
    "size": "55 Inch",
    "material": "Plastic"
}
"""

    return f"""
Extract the following product attributes:

- brand
- color
- size
- material

Return only a JSON object with these four fields.
If an attribute is not available, use null.

{examples}

Now extract the attributes from this product:

{product_text}
"""


def build_schema_guided_prompt(product):
    product_text = f"""
Title: {product["title"]}
Description: {product["description"]}
"""

    return f"""
Extract product attributes from the following product information.

Attributes:
- brand
- color
- size
- material

The output must follow this exact JSON structure:

{{
    "brand": null,
    "color": null,
    "size": null,
    "material": null
}}

Replace null values with the extracted values when they are available.
Do not add any additional fields.
Return only valid JSON.

Product:
{product_text}
"""


def build_definition_augmented_prompt(product):
    product_text = f"""
Title: {product["title"]}
Description: {product["description"]}
"""

    definitions = """
Attribute definitions:

brand:
The manufacturer or company associated with the product.

color:
The color or primary color of the product.

size:
The stated size, dimensions, capacity, or size designation of the product.

material:
The material or physical composition used to make the product.
"""

    return f"""
Extract product attributes using the attribute definitions below.

{definitions}

Return a JSON object containing:

- brand
- color
- size
- material

If an attribute cannot be found in the product information, return null.
Do not guess or invent information.
Return only valid JSON.

Product:
{product_text}
"""


def build_prompt(product, strategy):
    if strategy == "ZERO_SHOT":
        return build_zero_shot_prompt(product)

    if strategy == "FEW_SHOT":
        return build_few_shot_prompt(product)

    if strategy == "SCHEMA_GUIDED":
        return build_schema_guided_prompt(product)

    if strategy == "DEFINITION_AUGMENTED":
        return build_definition_augmented_prompt(product)

    raise ValueError(f"Unknown prompt strategy: {strategy}")


# Model clients
def get_openai_client():
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    return OpenAI(api_key=api_key)


def get_gemini_client():
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    return genai.Client(api_key=api_key)


def call_model(prompt, model):
    if model == "gpt-4o-mini":
        client = get_openai_client()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=TEMPERATURE
        )

        return response.choices[0].message.content

    if model == "gemini-3.8-flash":
        client = get_gemini_client()

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": TEMPERATURE
            }
        )

        return response.text

    raise ValueError(f"Unknown model: {model}")


# Parse model output
def parse_json_response(response_text):
    if not response_text:
        return None

    response_text = response_text.strip()

    # Remove markdown code fences
    response_text = re.sub(
        r"^```(?:json)?\s*",
        "",
        response_text,
        flags=re.IGNORECASE
    )

    response_text = re.sub(
        r"\s*```$",
        "",
        response_text
    )

    response_text = response_text.strip()

    # Try parsing the complete response first
    try:
        result = json.loads(response_text)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # Try extracting the first JSON object
    match = re.search(r"\{.*\}", response_text, re.DOTALL)

    if match:
        try:
            result = json.loads(match.group(0))

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

    return None


def clean_prediction(prediction):
    if not isinstance(prediction, dict):
        return {
            attribute: None
            for attribute in ATTRIBUTES
        }

    cleaned = {}

    for attribute in ATTRIBUTES:
        cleaned[attribute] = prediction.get(attribute)

    return cleaned


# Evaluation
def normalize_value(value):
    if value is None:
        return None

    return str(value).strip().lower()


def evaluate_prediction(prediction, product):
    scores = {}

    for attribute in ATTRIBUTES:
        predicted = normalize_value(
            prediction.get(attribute)
        )

        expected = normalize_value(
            product.get(attribute)
        )

        scores[attribute] = int(predicted == expected)

    correct = sum(scores.values())
    total = len(ATTRIBUTES)

    accuracy = correct / total

    return {
        "scores": scores,
        "correct": correct,
        "total": total,
        "accuracy": accuracy
    }


# Model call with retry handling
def call_model_with_retry(prompt, model, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = call_model(prompt, model)
            return response

        except Exception as error:
            print(
                f"Model call failed "
                f"(attempt {attempt + 1}/{max_retries}): {error}"
            )

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return None


# Results
RESULTS_PATH = Path("experiment_results.csv")


def save_result(result, path=RESULTS_PATH):
    file_exists = path.exists()

    fieldnames = [
        "product_id",
        "category",
        "model",
        "prompt_strategy",
        "brand_prediction",
        "color_prediction",
        "size_prediction",
        "material_prediction",
        "brand_correct",
        "color_correct",
        "size_correct",
        "material_correct",
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]

    with open(
        path,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)


def make_result_row(
    product,
    model,
    prompt_strategy,
    prediction,
    evaluation
):
    scores = evaluation["scores"]

    return {
        "product_id": product["product_id"],
        "category": product["category"],
        "model": model,
        "prompt_strategy": prompt_strategy,

        "brand_prediction": prediction.get("brand"),
        "color_prediction": prediction.get("color"),
        "size_prediction": prediction.get("size"),
        "material_prediction": prediction.get("material"),

        "brand_correct": scores["brand"],
        "color_correct": scores["color"],
        "size_correct": scores["size"],
        "material_correct": scores["material"],

        "accuracy": evaluation["accuracy"]
    }


# Checkpoint / resume
def load_completed_results(path=RESULTS_PATH):
    completed = set()

    if not path.exists():
        return completed

    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            key = (
                int(row["product_id"]),
                row["model"],
                row["prompt_strategy"]
            )

            completed.add(key)

    return completed




# Main experiment runner
def run_experiment(
    products=None,
    models=None,
    prompt_strategies=None,
    dry_run=False
):
    if products is None:
        products = data

    if models is None:
        models = MODELS

    if prompt_strategies is None:
        prompt_strategies = PROMPT_STRATEGIES

    completed = load_completed_results()

    total_runs = (
        len(products)
        * len(models)
        * len(prompt_strategies)
    )

    current_run = 0

    print(f"Total experiment runs: {total_runs}")

    for product in products:

        for model in models:

            for strategy in prompt_strategies:

                current_run += 1

                key = (
                    product["product_id"],
                    model,
                    strategy
                )

                if key in completed:
                    print(
                        f"[{current_run}/{total_runs}] "
                        f"Skipping completed: {key}"
                    )
                    continue

                prompt = build_prompt(
                    product,
                    strategy
                )

                print(
                    f"[{current_run}/{total_runs}] "
                    f"{model} | {strategy} | "
                    f"Product {product['product_id']}"
                )

                response_text = None
                prediction = None
                error = None

                if dry_run:

                    prediction = {
                        attribute: product.get(attribute)
                        for attribute in ATTRIBUTES
                    }

                    response_text = json.dumps(
                        prediction
                    )

                else:

                    try:
                        response_text = call_model_with_retry(
                            prompt,
                            model
                        )

                        if not response_text:
                            failure_type = classify_failure(
                                response_text=response_text,
                                prediction=None
                            )

                            save_failure(
                                product,
                                model,
                                strategy,
                                failure_type,
                                response_text=response_text
                            )

                            continue

                        prediction = parse_json_response(
                            response_text
                        )

                        failure_type = classify_failure(
                            response_text=response_text,
                            prediction=prediction
                        )

                        if failure_type is not None:

                            save_failure(
                                product,
                                model,
                                strategy,
                                failure_type,
                                response_text=response_text
                            )

                            continue

                        prediction = clean_prediction(
                            prediction
                        )

                    except Exception as exc:

                        error = exc

                        save_failure(
                            product,
                            model,
                            strategy,
                            "API_ERROR",
                            response_text=response_text,
                            error=error
                        )

                        continue

                evaluation = evaluate_prediction(
                    prediction,
                    product
                )

                f1_result = calculate_f1(
                    prediction,
                    product
                )

                result = make_result_row(
                    product,
                    model,
                    strategy,
                    prediction,
                    evaluation
                )

                result["precision"] = f1_result["precision"]
                result["recall"] = f1_result["recall"]
                result["f1"] = f1_result["f1"]

                if not dry_run:
                    save_result(result)

                completed.add(key)

    cost_tracker.save_cost_log()

    print("Experiment finished.")
