# Multilingual Word Taxonomy Generator

## Implementation Notes

### Processing Optimizations

- The script implements batch processing to optimize database operations and API calls
- Uses configurable batch sizes to balance between performance and API rate limits
- SQLite database is periodically optimized with automated VACUUM and REINDEX operations

### Natural Language Processing

- Utilizes NLTK's SnowballStemmer for word normalization across different languages
- Implements similarity scoring using Python's SequenceMatcher for string comparisons
- Uses sequence matching to identify similar categories, reducing redundant API calls
- Configurable similarity threshold (default: 0.8) for category matching

### Database and Storage

- SQLite is used for persistent storage with optimized indexes for fast lookups
- Maintains separate indexes for case-insensitive searches and language-specific queries
- Implements efficient category mapping across multiple languages
- Stores confidence scores for each word-category association

### Data Management

- Automatically detects and reuses existing similar categories to maintain consistency
- Category translations are cached to minimize redundant API calls
- Export functionality provides comprehensive data access for analysis and verification
- Supports incremental updates without duplicating categories

## Technical Requirements

- Python 3.12+
- SQLite3
- uv package manager

### Development Setup

1. Install uv (if not already installed)

2. Create and activate virtual environment:

    ```bash
    uv venv .venv --python=3.12
    source .venv/bin/activate  # On Unix/macOS
    ```

3. Install dependencies:

    ```bash
    uv pip install -e .
    ```

## Supported Language Models

The script supports multiple LLM providers:

- Ollama (default, local deployment)
- OpenAI
- Anthropic

## Command Line Interface

The flags available:

- `--provider`: LLM provider to use (ollama, openai, or anthropic)
        - Required for processing mode
        - Optional for export-only mode

- `--model`: Model name for the chosen provider
        - Optional, defaults:
          - Ollama: mistral-nemo
          - OpenAI: gpt-4o-mini
          - Anthropic: claude-3-5-sonnet-20241022

- `--input-language`: Source language code [default: en]
        - Supported: de, en, fr, es, it
- `--output-languages`: Target language codes [default: de]
        - Can specify multiple: --output-languages de fr es
- `--input-file`: Path to input file containing words
        - Optional, supports CSV or TXT format
        - If not provided, uses default word set
- `--export-file`: Path for exporting categorized words to CSV
        - Optional for processing mode
        - Required for export-only mode

## Default models for each provider

- Ollama: `mistral-nemo`
- OpenAI: `gpt-4o-mini`
- Anthropic: `claude-3-5-sonnet-20241022`

To use a different model:

```bash
taxonomy --provider ollama --model your-model-name --input-language en --output-languages de fr
```

## File Input Processing

Process words from an input file:

```bash
taxonomy --provider ollama --input-file words.csv --input-language en --output-languages de fr
```

## Data Export

Export existing categorizations:

```bash
taxonomy --export-file output.csv --input-language en --output-languages de fr
```

## Combined Processing and Export

Process new words and export results:

```bash
taxonomy --provider ollama --input-file words.csv --export-file output.csv --input-language en --output-languages de fr
```

## Input File Formats

The script accepts two input formats:

### CSV Format

Simple CSV file with one word per line:

```csv
word
house
computer
book
```

### Text Format

Plain text file with one word per line:

```text
house
computer
book
```

## Output Format

The export functionality generates a CSV file containing:

- Word
- Category
- Language
- Confidence Score

## Environment Configuration

Required environment variables:

```bash
DB_PATH=path/to/database.db    # SQLite database location
BATCH_SIZE=20                  # Processing batch size
```

Additional provider-specific variables:

```bash
# For OpenAI
OPENAI_API_KEY=your_key

# For Anthropic
ANTHROPIC_API_KEY=your_key

# For Ollama
OLLAMA_BASE_URL=http://localhost:11434
```

## License

MIT License
