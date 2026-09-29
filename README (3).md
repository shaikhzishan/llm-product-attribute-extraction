# LLM Product Attribute Extraction

A practical experiment for extracting structured product attributes from product titles and descriptions using Large Language Models (LLMs).

## Author

**Shaikh Zishan**

## Project Overview

This project evaluates how different LLMs and prompting strategies perform when extracting product attributes such as:

- Brand
- Color
- Size
- Material

## Models

- GPT-4o-mini
- Gemini 2.5 Flash

## Prompting Strategies

1. Zero-shot
2. Few-shot
3. Schema-guided
4. Definition-augmented

## Dataset

The project uses `mave_sample_dataset.json`, containing 200 product records across Electronics, Clothing, Home, Sports, and Food.

## Experiment Setup

- 200 products
- 2 models
- 4 prompting strategies
- 1,600 total model predictions

The experiment uses random seed 42 and temperature 0.

## Evaluation

Predictions are compared with the ground-truth product attributes.

The project records:

- Attribute-level correctness
- Accuracy
- Precision
- Recall
- F1 score

## Project Features

- Multiple LLM providers
- Prompt strategy comparison
- JSON response parsing
- Retry handling
- Failure tracking
- Result checkpointing
- Incremental CSV saving
- Cost tracking
- Resume support

## Project Structure

```text
llm-product-attribute-extraction/
├── mave_sample_dataset.json
├── run_experiments.py
├── .gitignore
└── README.md
```

## Setup

Install the required packages:

```bash
pip install openai google-genai pydantic pandas numpy scikit-learn matplotlib seaborn tqdm requests
```

## API Keys

Set the required API keys as environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export GEMINI_API_KEY="your_gemini_api_key"
```

Never commit API keys to GitHub.

## Running the Experiment

From the project directory:

```bash
python run_experiments.py
```

For testing without API calls:

```python
run_experiment(
    products=data[:2],
    models=MODELS,
    prompt_strategies=PROMPT_STRATEGIES,
    dry_run=True
)
```

## Output

Experiment outputs are saved as CSV files and can be used to analyze model performance, prompting strategies, attribute-level accuracy, failures, and API usage.

## Author

Shaikh Zishan