# src/word_reader.py

from pathlib import Path
from typing import List
import csv
import logging

logger = logging.getLogger('WordReader')

class WordReader:
    """Simple reader for word list files"""
    
    @staticmethod
    def read_words(file_path: str, encoding: str = 'utf-8') -> List[str]:
        """
        Read words from a file (CSV or TXT)
        
        Args:
            file_path: Path to the input file
            encoding: File encoding (default: utf-8)
            
        Returns:
            List of unique, non-empty words
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        words = set()  # Using set to ensure uniqueness
        
        try:
            with open(path, 'r', encoding=encoding) as f:
                if path.suffix.lower() == '.csv':
                    # Try reading as CSV first
                    reader = csv.reader(f)
                    for row in reader:
                        if row:  # Skip empty rows
                            word = row[0].strip()  # Take first column
                            if word and not word.lower() == 'word':  # Skip header if present
                                words.add(word)
                else:
                    # Read as plain text, one word per line
                    for line in f:
                        word = line.strip()
                        if word:  # Skip empty lines
                            words.add(word)
                            
            word_list = sorted(list(words))  # Convert to sorted list
            logger.info(f"Successfully read {len(word_list)} unique words from {file_path}")
            return word_list
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise