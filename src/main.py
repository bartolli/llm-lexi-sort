import asyncio
import os
import argparse
from dotenv import load_dotenv
from src.word_reader import WordReader
from src.data_exporter import DataExporter
from src.taxonomy_manager import TaxonomyManager
from src.llm_manager import LLMManager
from typing import List

from rich.console import Console
console = Console()

# Load environment variables for API keys and other configs
load_dotenv()

async def export_data(db_path: str, output_file: str, input_language: str, output_languages: List[str]):
    """Handle export-only mode"""
    try:
        exporter = DataExporter(db_path)
        console.print(f"[cyan]Exporting categorized words to: {output_file}[/cyan]")
        
        success = exporter.export_categorized_words(
            output_file=output_file,
            input_language=input_language,
            output_languages=output_languages
        )
        
        if success:
            console.print(f"[green]Successfully exported data to {output_file}[/green]")
        else:
            console.print(f"[red]No data found to export for the specified languages[/red]")
            
    except Exception as e:
        console.print(f"[red]Error during export: {e}[/red]")
        
async def main(provider: str = None, model: str = None, input_language: str = "en", 
              output_languages: List[str] = ["de"], input_file: str = None,
              export_file: str = None):
    # Get configuration from environment
    db_path = os.getenv('DB_PATH', 'test_taxonomy.db')
    
    # If only export_file is provided, run in export-only mode
    if export_file and not any([provider, input_file]):
        await export_data(db_path, export_file, input_language, output_languages)
        return
        
    # Regular processing mode
    batch_size = int(os.getenv('BATCH_SIZE', 20))

    # Initialize managers
    taxonomy_mgr = TaxonomyManager(
        db_path, 
        input_language=input_language, 
        output_languages=output_languages,
        provider=provider,
        batch_size=batch_size,
    )
    llm = LLMManager(provider=provider, model=model, language_code=input_language)
    
    # Get input words
    if input_file:
        try:
            console.print(f"[cyan]Reading words from file: {input_file}[/cyan]")
            test_words = WordReader.read_words(input_file)
            console.print(f"[green]Successfully loaded {len(test_words)} unique words[/green]")
        except Exception as e:
            console.print(f"[red]Error reading input file: {e}[/red]")
            return
    else:
        # Fall back to default example words
        test_words = {
            "de": ["Haus", "Katze", "Schule", "Buch", "Tisch"],
            "en": ["house", "cat", "school", "book", "table"],
            "fr": ["maison", "chat", "école", "livre", "table"],
            "es": ["casa", "gato", "escuela", "libro", "mesa"],
            "it": ["casa", "gatto", "scuola", "libro", "tavolo"]
        }.get(input_language, ["house", "cat", "school", "book", "table"])
    
    console.print(f"Using {provider.upper()} provider")
    console.print(f"Input Language: {input_language}")
    console.print(f"Output Languages: {', '.join(output_languages)}")
    console.print(f"Batch size: {batch_size}")
    
    # Regular processing logic
    uncategorized_words = []
    for word in test_words:
        if not await taxonomy_mgr.process_word(word):
            uncategorized_words.append(word)
    
    if uncategorized_words:
        console.print(f"\n[cyan]Requesting categories for {len(uncategorized_words)} new words...[/cyan]")
        categories = await llm.categorize_words(uncategorized_words)
        console.print("[yellow]Categories received:[/yellow]", categories)
        
        # Process and store new categorizations
        for word, category in categories.items():
            await taxonomy_mgr.process_word(word, category)
            
    # If export file is specified, export after processing
    if export_file:
        await export_data(db_path, export_file, input_language, output_languages)

def cli():
    parser = argparse.ArgumentParser(description='Multilingual Word Taxonomy Generator')
    
    # Provider is now optional (not required for export-only mode)
    parser.add_argument('--provider', type=str, choices=['ollama', 'openai', 'anthropic'],
                      help='LLM provider to use (ollama, openai, or anthropic)')
    parser.add_argument('--model', type=str, 
                      help='Model name for the chosen provider (default: mistral-nemo for ollama, gpt-4o-mini for openai)')
    parser.add_argument('--input-language', type=str, default='en',
                      help='Input language code (default: en). Supported: de, en, fr, es, it')
    parser.add_argument('--output-languages', type=str, nargs='+', default=['de'],
                      help='Output language codes (default: de). Can specify multiple, e.g., --output-languages de fr es')
    parser.add_argument('--input-file', type=str,
                      help='Path to input file containing words (CSV or TXT format)')
    parser.add_argument('--export-file', type=str,
                      help='Path for exporting categorized words to CSV')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.export_file and not args.provider:
        parser.error("Either --provider or --export-file must be specified")
        
    if args.provider and not args.export_file:
        if args.provider not in ['ollama', 'openai', 'anthropic']:
            parser.error("--provider must be one of: ollama, openai, anthropic")
    
    asyncio.run(main(
        provider=args.provider,
        model=args.model,
        input_language=args.input_language,
        output_languages=args.output_languages,
        input_file=args.input_file,
        export_file=args.export_file
    ))

if __name__ == "__main__":
    cli()