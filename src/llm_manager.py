from abc import ABC, abstractmethod
import json
from typing import List, Dict, Optional
from pydantic import BaseModel
from anthropic import Anthropic
from openai import OpenAI, OpenAIError
import ollama
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

class WordCategory(BaseModel):
    word: str
    category: str

class CategoryResponse(BaseModel):
    categories: dict[str, str]

class Step(BaseModel):
    word: str
    reasoning: str
    category: str

class TaxonomyReasoning(BaseModel):
    steps: list[Step]
    categories: dict[str, str]

class LLMProvider(ABC):
    @abstractmethod
    async def generate_categories(self, words: List[str]) -> Dict[str, str]:
        """Generate categories for a list of words"""
        pass
        
    @abstractmethod
    async def translate(self, text: str, target_language: str) -> str:
        """
        Translate text to target language
        Args:
            text: Text to translate
            target_language: Target language code (e.g., 'en', 'de')
        Returns:
            Translated text
        """
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    async def generate_categories(self, words: List[str]) -> Dict[str, str]:
        prompt = self._create_prompt(words)
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Du bist ein deutscher Sprachexperte, der sich auf die Kategorisierung von Wörtern spezialisiert hat."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )

            if completion.choices and completion.choices[0].message.content:
                response_content = completion.choices[0].message.content.strip()
                try:
                    parsed_response = TaxonomyReasoning.model_validate_json(response_content)
                    return parsed_response.categories
                except json.JSONDecodeError:
                    print(f"Failed to parse response as JSON: {response_content}")
                    return {}
            return {}

        except OpenAIError as e:
            print(f"OpenAI API error: {e}")
            return {}

    async def translate(self, text: str, target_language: str) -> str:
        """Translate text using OpenAI"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "system",
                    "content": "You are a translator. Translate the text to the target language. Return only the translated text."
                }, {
                    "role": "user",
                    "content": f"Translate to {target_language}: {text}"
                }],
                temperature=0.3
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def _create_prompt(self, words: List[str]) -> str:
        return f"""
        Kategorisiere diese deutschen Wörter in sinnvolle Kategorien.
        Erkläre für jedes Wort deine Gedanken und ordne es dann einer passenden Kategorie zu.
        Wichtig: Die Kategorien müssen auf Deutsch sein.
        
        Wörter: {', '.join(words)}
        
        Antworte im folgenden JSON-Format:
        {{
            "steps": [
                {{
                    "word": "Haus",
                    "reasoning": "Ein Haus ist eine Struktur, die als Wohnraum oder für andere Zwecke genutzt wird. Es gehört zur Kategorie der Gebäude.",
                    "category": "Gebäude"
                }},
                {{
                    "word": "Katze",
                    "reasoning": "Eine Katze ist ein Säugetier und Haustier. Sie gehört eindeutig zur Kategorie der Tiere.",
                    "category": "Tiere"
                }}
            ],
            "categories": {{
                "Haus": "Gebäude",
                "Katze": "Tiere"
            }}
        }}
        """
        
class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.client = Anthropic(api_key=api_key)
        self.model = model

    async def generate_categories(self, words: List[str]) -> Dict[str, str]:
        prompt = self._create_prompt(words)
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                system="Du bist ein deutscher Sprachexperte, der sich auf die Kategorisierung von Wörtern spezialisiert hat.",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            try:
                response_data = json.loads(message.content[0].text)
                return response_data.get('categories', {})
            except json.JSONDecodeError:
                print(f"Failed to parse response as JSON: {message.content}")
                return {}
                
        except Exception as e:
            print(f"Error generating categories with Anthropic: {e}")
            return {}

    async def translate(self, text: str, target_language: str) -> str:
        """Translate text using Anthropic"""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                system="You are a translator. Translate the text to the target language. Return only the translated text.",
                messages=[{
                    "role": "user",
                    "content": f"Translate to {target_language}: {text}"
                }]
            )
            return message.content[0].text.strip()
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def _create_prompt(self, words: List[str]) -> str:
        return f"""
        Kategorisiere diese deutschen Wörter in sinnvolle Kategorien.
        Erkläre für jedes Wort deine Gedanken und ordne es dann einer passenden Kategorie zu.
        Wichtig: Die Kategorien müssen auf Deutsch sein.
        
        Wörter: {', '.join(words)}
        
        Antworte im folgenden JSON-Format:
        {{
            "steps": [
                {{
                    "word": "Haus",
                    "reasoning": "Ein Haus ist eine Struktur, die als Wohnraum oder für andere Zwecke genutzt wird. Es gehört zur Kategorie der Gebäude.",
                    "category": "Gebäude"
                }},
                {{
                    "word": "Katze",
                    "reasoning": "Eine Katze ist ein Säugetier und Haustier. Sie gehört eindeutig zur Kategorie der Tiere.",
                    "category": "Tiere"
                }}
            ],
            "categories": {{
                "Haus": "Gebäude",
                "Katze": "Tiere"
            }}
        }}
        """
        
class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = None, model: str = "mistral-nemo"):
        if base_url:
            os.environ['OLLAMA_HOST'] = base_url
        self.model = model
        print(f"Initializing Ollama with model: {model}")
        if base_url:
            print(f"Using Ollama host: {base_url}")

    async def generate_categories(self, words: List[str]) -> Dict[str, str]:
        import ollama
        prompt = self._create_prompt(words)
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Du bist ein deutscher Sprachexperte, der sich auf die Kategorisierung von Wörtern spezialisiert hat."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                format=TaxonomyReasoning.model_json_schema(),
            )
            
            try:
                # Parse the response using Pydantic model
                parsed_response = TaxonomyReasoning.model_validate_json(response.message.content)
                return parsed_response.categories
            except Exception as parse_error:
                print(f"Failed to parse response: {parse_error}")
                print(f"Raw response: {response.message.content}")
                return {}
                
        except Exception as e:
            print(f"Error generating categories with Ollama: {e}")
            return {}

    async def translate(self, text: str, target_language: str) -> str:
        """Translate text using Ollama"""
        import ollama
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": f"You are a translator. Translate the input text to {target_language}. Return only the translated word without explanation."
                    },
                    {"role": "user", "content": text}
                ],
            )
            return response.message.content.strip()
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def _create_prompt(self, words: List[str]) -> str:
        return f"""
        Kategorisiere diese deutschen Wörter in sinnvolle Kategorien.
        Erkläre für jedes Wort deine Gedanken und ordne es dann einer passenden Kategorie zu.
        Wichtig: Die Kategorien müssen auf Deutsch sein.
        
        Wörter: {', '.join(words)}
        
        Beispielformat der Antwort:
        {{
            "steps": [
                {{
                    "word": "Haus",
                    "reasoning": "Ein Haus ist eine Struktur, die als Wohnraum oder für andere Zwecke genutzt wird. Es gehört zur Kategorie der Gebäude.",
                    "category": "Gebäude"
                }},
                {{
                    "word": "Katze",
                    "reasoning": "Eine Katze ist ein Säugetier und Haustier. Sie gehört eindeutig zur Kategorie der Tiere.",
                    "category": "Tiere"
                }}
            ],
            "categories": {{
                "Haus": "Gebäude",
                "Katze": "Tiere"
            }}
        }}
        """


class LLMManager:
    def __init__(self, provider: str = "ollama", **kwargs):
        self.provider = provider
        self.language_code = kwargs.get('language_code', 'en')
        self.language_names = {
            "de": "German",
            "en": "English",
            "fr": "French",
            "es": "Spanish",
            "it": "Italian",
        }
        self._setup_provider(provider, **kwargs)
        
    def _setup_provider(self, provider: str, **kwargs):
        """Initialize the appropriate LLM provider"""
        if provider == "openai":
            api_key = kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key is required")
            model = kwargs.get('model') or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
            self.provider = OpenAIProvider(api_key=api_key, model=model)
        elif provider == "anthropic":
            api_key = kwargs.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key is required")
            model = kwargs.get('model') or os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022')
            self.provider = AnthropicProvider(api_key=api_key, model=model)
        elif provider == "ollama":
            base_url = kwargs.get('base_url') or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            model = kwargs.get('model') or os.getenv('OLLAMA_MODEL', 'mistral-nemo')
            self.provider = OllamaProvider(base_url=base_url if base_url else None, model=model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def translate_category(self, category: str, target_language: str) -> str:
        """
        Translate a category name to the target language using the current provider
        Args:
            category: Category name to translate
            target_language: Target language code (e.g., 'en', 'de')
        Returns:
            Translated category name
        """
        try:
            translated = await self.provider.translate(
                text=category,
                target_language=target_language
            )
            return translated.strip()
        except Exception as e:
            print(f"Translation error for {category} to {target_language}: {e}")
            return category  # Fallback to original category name

    async def categorize_words(self, words: List[str]) -> Dict[str, str]:
        """Categorize a list of words using the current provider"""
        return await self.provider.generate_categories(words)

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Example with Ollama
        llm = LLMManager(provider="ollama", model="mistral-nemo")
        test_words = ["Haus", "Katze", "Schule"]
        results = await llm.categorize_words(test_words)
        print("Ollama results:", results)

    asyncio.run(main())