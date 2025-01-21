from typing import List, Dict, Optional
import sqlite3
import logging
import pandas as pd
from pydantic import BaseModel
from nltk.stem.snowball import SnowballStemmer
from difflib import SequenceMatcher
from rich.console import Console
from src.llm_manager import LLMManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('TaxonomyManager')
console = Console()

class Category(BaseModel):
    """Pydantic model for a taxonomy category"""
    name: str
    description: str
    parent_category: Optional[str] = None
    subcategories: Optional[List['Category']] = None
    input_language: str
    output_language: str

class TaxonomyStructure(BaseModel):
    """Pydantic model for the complete taxonomy structure"""
    categories: List[Category]
    total_categories: int
    total_words: int
    statistics: Dict[str, int]
    input_language: str
    output_languages: List[str]

class TextAnalysis(BaseModel):
    keywords: List[str]
    suggested_categories: List[str]
    confidence_score: float
    summary: str
    input_language: str
    output_language: str

class TaxonomyManager:
    def __init__(self, db_path: str, input_language: str = "en", output_languages: List[str] = ["de"], provider: str = "openai", batch_size: int = 100):
        self.db_path = db_path
        self.input_language = input_language
        self.output_languages = output_languages
        self.stemmer = SnowballStemmer(self._get_stemmer_language(input_language))
        self.llm = LLMManager(provider=provider, language_code=input_language)
        self.batch_size = batch_size
        self._operation_count = 0
        self._last_optimization = 0
        self.setup_database()


    def should_optimize(self) -> bool:
        """Determine if database optimization should be performed"""
        operations_since_last = self._operation_count - self._last_optimization
        
        # Optimize if we've processed at least one full batch since last optimization
        if operations_since_last >= self.batch_size:
            logger.info(f"Triggering optimization after {operations_since_last} operations")
            return True
            
        return False
    
    def _get_stemmer_language(self, language_code: str) -> str:
        """Maps ISO language codes to NLTK stemmer languages"""
        language_map = {
            "de": "german",
            "en": "english",
            "fr": "french",
            "es": "spanish",
            "it": "italian",
        }
        return language_map.get(language_code, "english")  # Default to English if unsupported

    def setup_database(self):
        """Initialize the database with required tables and indexes for optimal performance"""
        with sqlite3.connect(self.db_path) as conn:
            # Create categories table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS categories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    parent_category TEXT,
                    input_language TEXT NOT NULL,
                    output_language TEXT NOT NULL,
                    base_category TEXT NOT NULL,  -- Store the English base category name
                    UNIQUE(name, input_language, output_language)
                )
            """)
            
            # Create word-category mapping table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS word_categories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT NOT NULL,
                    category_id INTEGER NOT NULL,
                    input_language TEXT NOT NULL,
                    output_language TEXT NOT NULL,
                    confidence_score FLOAT,
                    FOREIGN KEY(category_id) REFERENCES categories(id),
                    UNIQUE(word, category_id, input_language, output_language)
                )
            """)
            
            # Create indexes for frequently accessed columns
            
            # Index for category name lookups (case-insensitive search optimization)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_categories_name 
                ON categories(name COLLATE NOCASE)
            """)
            
            # Index for base category lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_categories_base 
                ON categories(base_category COLLATE NOCASE)
            """)
            
            # Composite index for language-specific category lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_categories_lang 
                ON categories(input_language, output_language, name)
            """)
            
            # Index for word lookups in word_categories
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_word_categories_word 
                ON word_categories(word COLLATE NOCASE)
            """)
            
            # Composite index for word-language combinations
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_word_categories_lang 
                ON word_categories(input_language, output_language, word)
            """)
            
            logger.info(f"Database initialized at {self.db_path} with optimized indexes")


    def calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using various metrics"""
        console.print(f"[cyan]Calculating similarity between '{str1}' and '{str2}'[/cyan]")
        
        # Stem both strings
        stemmed1 = self.stemmer.stem(str1.lower())
        stemmed2 = self.stemmer.stem(str2.lower())
        console.print(f"[yellow]Stemmed strings: '{stemmed1}' and '{stemmed2}'[/yellow]")
        
        # Get sequence similarity
        sequence_similarity = SequenceMatcher(None, stemmed1, stemmed2).ratio()
        console.print(f"[yellow]Sequence similarity: {sequence_similarity:.3f}[/yellow]")
        
        # Exact stem match gets highest score
        if stemmed1 == stemmed2:
            similarity = 1.0
            console.print(f"[yellow]Exact stem match found between '{str1}' and '{str2}'[/yellow]")
        else:
            similarity = sequence_similarity
            console.print(f"[green]Using sequence similarity: {similarity:.3f}[/green]")
            
        console.print(f"[cyan]Final similarity score: {similarity:.3f}[/cyan]")
        return similarity

    async def find_similar_category(self, category_name: str, output_language: str, threshold: float = 0.8) -> Optional[str]:
        """Find existing similar categories using optimized queries"""
        console.print(f"[green]Entering find_similar_category with category='{category_name}', language='{output_language}'[/green]")
        with sqlite3.connect(self.db_path) as conn:
            existing_categories = pd.read_sql("""
                SELECT DISTINCT name 
                FROM categories 
                WHERE input_language = ? AND output_language = ?
                -- Using idx_categories_lang for better performance
                ORDER BY name
            """, conn, params=(self.input_language, output_language))['name'].tolist()

            console.print(f"[green]Found {len(existing_categories)} existing categories: {existing_categories}[/green]")
            
            if not existing_categories:
                return None
                
            console.print(f"[yellow]Searching for similar categories to: {category_name}[/yellow]")
            
            exact_match = conn.execute("""
                SELECT name 
                FROM categories 
                WHERE name = ? COLLATE NOCASE 
                AND output_language = ?
                -- Using idx_categories_name for better performance
                LIMIT 1
            """, (category_name, output_language)).fetchone()
            
            if exact_match:
                console.print(f"[green]Found exact match: {exact_match[0]}[/green]")
                return exact_match[0]
            
            stemmed_category = self.stemmer.stem(category_name.lower())
            max_similarity = 0
            most_similar = None
            
            for existing in existing_categories:
                stemmed_existing = self.stemmer.stem(existing.lower())
                similarity = self.calculate_similarity(stemmed_category, stemmed_existing)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar = existing
            
            if max_similarity >= threshold:
                console.print(f"[green]Found similar category: {most_similar} with similarity {max_similarity:.3f}[/green]")
                return most_similar
                
            console.print(f"[yellow]No similar category found. Highest similarity was {max_similarity:.3f} with '{most_similar}'[/yellow]")
            return None
    
    async def process_word(self, word: str, category: str = None, confidence_score: float = 1.0):
        """Process a word and its category, with batch optimization"""
        try:
            # First check if we already have this word categorized
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                existing = cursor.execute("""
                    SELECT c.name, c.base_category
                    FROM word_categories wc
                    JOIN categories c ON wc.category_id = c.id
                    WHERE wc.word = ? AND wc.input_language = ?
                    LIMIT 1
                """, (word, self.input_language)).fetchone()
                
                if existing and not category:
                    logger.info(f"Found existing categorization for '{word}': {existing[0]} (base: {existing[1]})")
                    return True
            
            # If we have a category (from LLM), process it
            if category:
                result = await self._process_word_internal(word, category, confidence_score)
            else:
                logger.info(f"No existing category found for word '{word}' and no category provided")
                return False
            
            # Increment operation count
            self._operation_count += 1
            
            # Check if optimization is needed
            if self.should_optimize():
                logger.info("Starting database optimization...")
                try:
                    self.optimize_database()
                    self._last_optimization = self._operation_count
                    logger.info(f"Database optimized after {self._operation_count} total operations")
                except Exception as e:
                    logger.error(f"Optimization error (continuing processing): {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing word {word}: {e}")
            return False
        
    async def _process_word_internal(self, word: str, category: str, confidence_score: float = 1.0):
        """Internal method for word processing logic"""
        base_category = category
        
        for output_lang in self.output_languages:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                try:
                    # Execute in a transaction
                    cursor.execute("BEGIN")
                    
                    existing_category = cursor.execute("""
                        SELECT id, name 
                        FROM categories 
                        WHERE base_category = ? COLLATE NOCASE
                        AND input_language = ? 
                        AND output_language = ?
                        LIMIT 1
                    """, (base_category, self.input_language, output_lang)).fetchone()

                    if existing_category:
                        category_id, category_name = existing_category
                        logger.info(f"Using existing category '{category_name}' for '{base_category}' in {output_lang}")
                    else:
                        # Try to find similar category before creating new one
                        similar_category = await self.find_similar_category(base_category, output_lang)
                        if similar_category:
                            existing_category = cursor.execute("""
                                SELECT id, name 
                                FROM categories 
                                WHERE name = ?
                                AND input_language = ? 
                                AND output_language = ?
                                LIMIT 1
                            """, (similar_category, self.input_language, output_lang)).fetchone()
                            category_id, category_name = existing_category
                            logger.info(f"Using similar category '{category_name}' for '{base_category}' in {output_lang}")
                        else:
                            # Create new translation if needed
                            if output_lang != self.input_language:
                                category_name = await self.llm.translate_category(base_category, output_lang)
                                logger.info(f"Created new category '{category_name}' for '{base_category}' in {output_lang}")
                            else:
                                category_name = base_category
                        
                        cursor.execute("""
                            INSERT INTO categories 
                            (name, description, input_language, output_language, base_category) 
                            VALUES (?, ?, ?, ?, ?)
                            RETURNING id
                        """, (
                            category_name,
                            f"Translation of {base_category}",
                            self.input_language,
                            output_lang,
                            base_category
                        ))
                        category_id = cursor.fetchone()[0]
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO word_categories 
                        (word, category_id, input_language, output_language, confidence_score) 
                        VALUES (?, ?, ?, ?, ?)
                    """, (word, category_id, self.input_language, output_lang, confidence_score))
                    
                    # Commit the transaction
                    conn.commit()
                    logger.info(f"Mapped word '{word}' to category '{category_name}' in {output_lang}")
                    
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error processing word {word} for language {output_lang}: {e}")
                    raise
        
        return True

    def optimize_database(self):
        """Perform database optimization operations with separate connections for VACUUM"""
        try:
            # First connection for ANALYZE and REINDEX
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN")
                try:
                    # Update SQLite statistics
                    cursor.execute("ANALYZE")
                    # Rebuild indexes
                    cursor.execute("REINDEX")
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error during ANALYZE/REINDEX: {e}")
                    raise
            
            # Separate connection for VACUUM (cannot be in a transaction)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("VACUUM")
                logger.info("Database optimization completed successfully")
                
        except Exception as e:
            logger.error(f"Failed to optimize database: {e}")
            raise

    def get_optimization_stats(self) -> dict:
        """Get current optimization statistics"""
        return {
            "total_operations": self._operation_count,
            "operations_since_optimization": self._operation_count - self._last_optimization,
            "batch_size": self.batch_size,
            "optimizations_performed": self._last_optimization // self.batch_size
        }

    async def add_category(self, category: Category) -> bool:
        """Add new category if it doesn't exist"""
        with sqlite3.connect(self.db_path) as conn:
            try:
                # Check if category exists for this input/output language combination
                exists = conn.execute("""
                    SELECT COUNT(*) FROM categories 
                    WHERE name = ? AND input_language = ? AND output_language = ?
                """, (category.name, category.input_language, category.output_language)).fetchone()[0] > 0
                
                if exists:
                    logger.info(f"Category {category.name} already exists for {category.output_language}")
                    return True
                    
                conn.execute("""
                    INSERT INTO categories 
                    (name, description, parent_category, input_language, output_language) 
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    category.name, 
                    category.description, 
                    category.parent_category, 
                    category.input_language, 
                    category.output_language
                ))
                logger.info(f"Added new category: {category.name} for {category.output_language}")
                return True
            except sqlite3.IntegrityError:
                logger.warning(f"Failed to add category {category.name}: database constraint")
                return False
            except Exception as e:
                logger.error(f"Error adding category {category.name}: {e}")
                return False

    def get_structured_taxonomy(self) -> TaxonomyStructure:
        """Returns a structured representation of the taxonomy using Pydantic models"""
        with sqlite3.connect(self.db_path) as conn:
            # Get categories by language
            categories_by_lang = {}
            for lang in self.output_languages:
                categories = conn.execute("""
                    SELECT DISTINCT name 
                    FROM categories 
                    WHERE output_language = ?
                    ORDER BY name
                """, (lang,)).fetchall()
                categories_by_lang[lang] = [cat[0] for cat in categories]
            
            # Get total unique words
            total_words = conn.execute("""
                SELECT COUNT(DISTINCT word) 
                FROM word_categories
            """).fetchone()[0]
            
            # Calculate statistics
            statistics = {
                'total_categories': sum(len(cats) for cats in categories_by_lang.values()) // len(self.output_languages),
                'total_words': total_words
            }
            for lang in self.output_languages:
                statistics[f'{lang}_categories'] = len(categories_by_lang[lang])
            
            # Create category objects
            categories = []
            for lang, cat_names in categories_by_lang.items():
                for name in cat_names:
                    categories.append(Category(
                        name=name,
                        description="",
                        input_language=self.input_language,
                        output_language=lang
                    ))
            
            return TaxonomyStructure(
                categories=categories,
                total_categories=statistics['total_categories'],
                total_words=total_words,
                statistics=statistics,
                input_language=self.input_language,
                output_languages=self.output_languages
            )
            