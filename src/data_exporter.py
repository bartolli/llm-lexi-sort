# src/data_exporter.py

import csv
from pathlib import Path
import sqlite3
import logging
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger('DataExporter')

class DataExporter:
    """Exports categorized words from SQLite database to CSV"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path

    def export_categorized_words(self, 
                               output_file: str, 
                               input_language: str,
                               output_languages: Optional[List[str]] = None,
                               encoding: str = 'utf-8') -> bool:
        """
        Export categorized words to CSV file
        
        Args:
            output_file: Path for the output CSV file
            input_language: Source language code
            output_languages: List of target language codes (optional)
            encoding: Output file encoding (default: utf-8)
            
        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query based on requested languages
                query = """
                    SELECT DISTINCT 
                        wc.word,
                        c.name as category,
                        c.output_language,
                        wc.confidence_score
                    FROM word_categories wc
                    JOIN categories c ON wc.category_id = c.id
                    WHERE wc.input_language = ?
                """
                params = [input_language]
                
                if output_languages:
                    query += " AND c.output_language IN ({})".format(
                        ','.join('?' * len(output_languages))
                    )
                    params.extend(output_languages)
                
                query += " ORDER BY wc.word, c.output_language"
                
                # Fetch results
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                if not results:
                    logger.warning(f"No data found for language(s): {input_language}")
                    return False
                
                # Write to CSV
                with open(output_path, 'w', newline='', encoding=encoding) as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow(['Word', 'Category', 'Language', 'Confidence'])
                    
                    # Write data
                    writer.writerows(results)
                
                logger.info(f"Successfully exported {len(results)} entries to {output_file}")
                return True
                
        except Exception as e:
            logger.error(f"Error exporting data to {output_file}: {e}")
            return False