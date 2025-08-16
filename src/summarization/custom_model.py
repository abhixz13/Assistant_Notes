"""
Custom Model Processing Module

This module provides AI-powered content enhancement using custom models
via OpenAI-compatible APIs (like Ollama) to transform raw transcripts 
into comprehensive, scholar-level notes.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

from utils.config import ConfigManager
from utils.logger import get_logger
from .prompts import PromptManager


class CustomModelProcessor:
    """
    AI Scholar Processor using custom models via OpenAI-compatible APIs.
    
    Transforms raw transcripts into comprehensive, structured notes
    that enable deep understanding and 60-minute speaking capability.
    """
    
    def __init__(self, model_override: Optional[str] = None):
        """Initialize the custom model processor with model selection."""
        self.logger = get_logger(__name__)
        self.config = ConfigManager()
        
        # Load configuration
        self.summarization_config = self.config.get_summarization_config()
        self.custom_model_config = self.config.get_custom_model_config()
        self.notes_config = self.config.get_notes_config()
        
        # Get model choice from settings.yaml or CLI override
        config_model = self.summarization_config.get('model', 'deepseek')
        selected_model = model_override or config_model
        
        # Configure model based on selection
        self._configure_model(selected_model)
        
        # Processing configuration
        self.max_tokens = self.summarization_config.get('max_tokens', 2048)
        self.temperature = self.summarization_config.get('temperature', 0.7)
        self.context_length = self.summarization_config.get('context_length', 8192)
        
        # Enhanced prompt configuration
        self.concept_extraction_tokens = self.summarization_config.get('concept_extraction_tokens', 512)
        self.title_tokens = self.summarization_config.get('title_tokens', 100)
        
        self.logger.info(f"Initialized Custom Model Processor with {self.model_name}")
        self.logger.info(f"API URL: {self.api_url}")
        self.logger.info(f"Chat Model: {self.chat_model}")
        
        # Topic-based summary configuration
        self.topic_summary_tokens = self.summarization_config.get('topic_summary_tokens', 600)
        self.topic_summary_max_workers = self.summarization_config.get('topic_summary_max_workers', 3)
        
        # Expertise level configuration
        self.default_expertise_level = self.summarization_config.get('default_expertise_level', 'moderate')
        
        # Windowed topic extraction configuration
        self.window_size = self.summarization_config.get('topic_window_size', 3000)
        self.window_overlap = self.summarization_config.get('topic_window_overlap', 500)
        self.topics_per_window = self.summarization_config.get('topics_per_window', 5)
        self.max_final_topics = self.summarization_config.get('max_final_topics', 12)

    def _configure_model(self, selected_model: str):
        """Configure model-specific settings based on selection from settings.yaml."""
        self.logger.info(f"Configuring model: {selected_model}")
        
        if selected_model == 'gpt-4o-mini':
            # OpenAI GPT-4o-mini configuration
            self.model_name = 'GPT-4o-mini'
            self.api_url = 'https://api.openai.com/v1'
            self.chat_model = 'gpt-4o-mini'
            self.api_key = self.config.get_env('OPENAI_API_KEY')
            self.api_type = 'openai'
            
            if not self.api_key or self.api_key.startswith('sk-placeholder'):
                self.logger.warning("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
                
        elif selected_model == 'mixtral':
            # Mixtral local configuration
            self.model_name = 'Mixtral-8x7B'
            self.model_path = self.summarization_config.get('model_path', 'models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf')
            self.api_type = 'local'
            self.api_url = None
            self.chat_model = 'mixtral'
            self.api_key = None
            
        else:  # deepseek or default (DeepSeek via Ollama)
            self.model_name = self.custom_model_config.get('name', 'Deepseek-coder')
            self.api_url = self.custom_model_config.get('api', {}).get('url', 'http://localhost:11434/v1')
            self.api_type = self.custom_model_config.get('api', {}).get('type', 'openai-compatible')
            self.chat_model = self.custom_model_config.get('chat', {}).get('model', 'deepseek-coder:latest')
            self.api_key = None
    
    def _make_api_request(self, messages: List[Dict], max_tokens: int = None) -> Optional[str]:
        """
        Make a request to the OpenAI-compatible API.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens for the response
            
        Returns:
            Response text or None if failed
        """
        try:
            # Handle local Mixtral differently
            if hasattr(self, 'api_type') and self.api_type == 'local':
                return self._make_local_request(messages, max_tokens)
            
            headers = {
                'Content-Type': 'application/json',
            }
            
            # Add OpenAI authentication
            if hasattr(self, 'api_type') and self.api_type == 'openai':
                if hasattr(self, 'api_key') and self.api_key:
                    headers['Authorization'] = f'Bearer {self.api_key}'
                else:
                    self.logger.error("OpenAI API key not found")
                    return None
            
            payload = {
                'model': self.chat_model,
                'messages': messages,
                'temperature': self.temperature,
                'max_tokens': max_tokens or self.max_tokens,
                'stream': False
            }
            
            self.logger.debug(f"Making API request to {self.api_url} using {self.api_type}")
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                self.logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error making API request: {e}")
            return None
    
    def _make_local_request(self, messages: List[Dict], max_tokens: int = None) -> Optional[str]:
        """Handle local Mixtral model requests."""
        try:
            # Import and use MixtralProcessor for local processing
            from .llama_local import MixtralProcessor
            mixtral = MixtralProcessor()
            
            # Convert messages to text for local processing
            prompt = "\n".join([msg["content"] for msg in messages if msg["role"] == "user"])
            response = mixtral._generate_text(prompt, max_tokens or self.max_tokens)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error with local Mixtral processing: {e}")
            return None
    
    def enhance_transcript(self, transcript: Dict, expertise_level: str = "moderate") -> Dict:
        """
        Enhance a raw transcript with AI-generated context and structure.
        
        Args:
            transcript: Raw transcript dictionary with segments and metadata
            
        Returns:
            Enhanced transcript with structured notes
        """
        try:
            self.logger.info(f"Starting transcript enhancement with custom model (expertise level: {expertise_level})")
            
            # Extract key information
            full_text = transcript.get('full_text', '')
            segments = transcript.get('segments', [])
            metadata = transcript.get('metadata', {})
            
            if not full_text.strip():
                self.logger.warning("No text content found in transcript")
                return self._fallback_enhance_transcript(transcript)
            
            # Generate enhanced notes
            enhanced_notes = self._generate_scholar_notes(full_text, segments, metadata, expertise_level)
            
            # Update transcript with enhanced content
            transcript['enhanced_notes'] = enhanced_notes
            transcript['processing_timestamp'] = datetime.now().isoformat()
            transcript['model_used'] = self.model_name
            
            self.logger.info("✅ Transcript enhancement completed successfully")
            return transcript
            
        except Exception as e:
            self.logger.error(f"Error enhancing transcript: {e}")
            return self._fallback_enhance_transcript(transcript)
    
    def _generate_scholar_notes(self, full_text: str, segments: List[Dict], metadata: Dict, expertise_level: str = "moderate") -> Dict:
        """
        Generate comprehensive scholar-level notes from transcript.
        
        Args:
            full_text: Complete transcript text
            segments: Individual transcript segments
            metadata: Transcript metadata
            
        Returns:
            Dictionary containing structured notes
        """
        try:
            # Generate title
            title = self._generate_title(full_text, metadata)
            
            # Generate key topics for better summary context
            topics = self._extract_topics(full_text, title)
            
            # Generate topic-based summaries in parallel
            self.logger.info(f"Generating topic-based summaries for {len(topics)} topics with {expertise_level} expertise level...")
            topic_summaries = self._generate_topic_summaries(topics, segments, expertise_level)
            
            # Assemble topic summaries into comprehensive summary
            summary = self._assemble_topic_summaries(topic_summaries)
            
            return {
                'title': title,
                'summary': summary,
                'metadata': {
                    'model_used': self.model_name,
                    'processing_timestamp': datetime.now().isoformat(),
                    'expertise_level': expertise_level,
                    'topics': topics,
                    'topic_summaries': topic_summaries,  # Store individual topic summaries
                    'original_metadata': metadata
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating scholar notes: {e}")
            return self._generate_fallback_notes(full_text, metadata)
    
    def _build_topic_text(self, segments: List[Dict], topic: str, max_chars: int = 6000) -> str:
        """
        Build topic-focused text by filtering segments relevant to the topic.
        
        Args:
            segments: List of transcript segments
            topic: Topic to filter for
            max_chars: Maximum characters to include
            
        Returns:
            Concatenated text relevant to the topic
        """
        try:
            topic_lc = topic.lower()
            relevant_segments = []
            
            # Find segments that mention the topic
            for segment in segments:
                segment_text = segment.get('text', '').lower()
                if topic_lc in segment_text:
                    relevant_segments.append(segment.get('text', ''))
            
            # If no direct matches, use first few segments as fallback
            if not relevant_segments:
                fallback_text = " ".join([s.get('text', '') for s in segments[:10]])
                return fallback_text[:max_chars]
            
            # Combine relevant segments
            topic_text = " ".join(relevant_segments)
            return topic_text[:max_chars]
            
        except Exception as e:
            self.logger.error(f"Error building topic text for '{topic}': {e}")
            return ""
    
    def _get_expertise_prompt(self, expertise_level: str) -> Dict[str, str]:
        """
        Get expertise-level specific prompts from PromptManager.
        
        Args:
            expertise_level: User expertise level (beginner/moderate/expert)
            
        Returns:
            Dict containing system message and template for the expertise level
        """
        level = (expertise_level or self.default_expertise_level).strip().lower()
        if level not in {"beginner", "moderate", "expert"}:
            level = "moderate"
            
        expertise_prompts = PromptManager.get_expertise_prompts()
        return expertise_prompts.get(level, expertise_prompts["moderate"])
    
    def _clean_ai_response(self, response: str) -> str:
        """
        Clean AI response to remove meta-commentary, apologies, and internal thoughts.
        
        Args:
            response: Raw AI response
            
        Returns:
            Cleaned response
        """
        if not response:
            return ""
            
        # Remove common AI prefixes and meta-commentary
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip lines that contain AI meta-commentary
            if (line.startswith("I'm sorry") or 
                line.startswith("Sure, here") or 
                line.startswith("Here are") or
                line.startswith("This question is beyond") or
                line.startswith("Could you please") or
                line.startswith("Are we talking about") or
                line.startswith("If you have any questions") or
                "meta-commentary" in line.lower() or
                "internal thoughts" in line.lower()):
                continue
                
            # Remove numbered lists that are just outlines
            if line.startswith("1.") and "brief summaries" in line.lower():
                continue
            if line.startswith("2.") and "brief summaries" in line.lower():
                continue
            if line.startswith("3.") and "brief summaries" in line.lower():
                continue
                
            # Keep meaningful content
            if line and len(line) > 10:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _generate_topic_summary(self, topic: str, topic_text: str, expertise_level: str = "moderate") -> str:
        """
        Generate a focused summary for a specific topic.
        
        Args:
            topic: The topic to summarize
            topic_text: Text content relevant to the topic
            expertise_level: User expertise level
            
        Returns:
            Focused summary for the topic
        """
        try:
            # Get topic summary prompt from PromptManager
            prompt_config = PromptManager.get_topic_summary_prompt(topic, topic_text, expertise_level)
            
            messages = [
                {"role": "system", "content": prompt_config["system"]},
                {"role": "user", "content": prompt_config["template"]}
            ]
            
            response = self._make_api_request(messages, self.topic_summary_tokens)
            if response:
                # Clean the response to remove AI meta-commentary
                cleaned_response = self._clean_ai_response(response.strip())
                return cleaned_response
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Error generating topic summary for '{topic}': {e}")
            return ""
    
    def _generate_topic_summaries(self, topics: List[str], segments: List[Dict], expertise_level: str = "moderate") -> Dict[str, str]:
        """
        Generate summaries for all topics in parallel.
        
        Args:
            topics: List of topics to summarize
            segments: Transcript segments
            
        Returns:
            Dictionary mapping topics to their summaries
        """
        results = {}
        if not topics:
            return results
            
        try:
            with ThreadPoolExecutor(max_workers=self.topic_summary_max_workers) as executor:
                # Submit all topic summary tasks
                future_to_topic = {}
                for topic in topics:
                    topic_text = self._build_topic_text(segments, topic)
                    if topic_text:
                        future = executor.submit(self._generate_topic_summary, topic, topic_text, expertise_level)
                        future_to_topic[future] = topic
                
                # Collect results as they complete
                for future in as_completed(future_to_topic):
                    topic = future_to_topic[future]
                    try:
                        summary = future.result()
                        if summary and summary.strip():
                            results[topic] = summary
                        else:
                            self.logger.warning(f"Empty summary generated for topic: {topic}")
                    except Exception as e:
                        self.logger.error(f"Failed to generate summary for topic '{topic}': {e}")
                        results[topic] = ""
            
            self.logger.info(f"Generated summaries for {len(results)} topics")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in parallel topic summary generation: {e}")
            return {}
    
    
    def _assemble_topic_summaries(self, topic_summaries: Dict[str, str]) -> str:
        """
        Assemble individual topic summaries into a structured comprehensive summary.
        
        Args:
            topic_summaries: Dictionary of topic summaries
            
        Returns:
            Structured comprehensive summary
        """
        try:
            if not topic_summaries:
                return "No topic summaries available."
            
            # Create a more cohesive, structured summary
            summary_parts = []
            
            # Start with a brief overview
            topics_list = list(topic_summaries.keys())
            if len(topics_list) == 1:
                summary_parts.append(f"This discussion explores {topics_list[0].lower()} in depth, covering key concepts, methodologies, and practical applications.")
            else:
                main_topics = ", ".join([topic.lower() for topic in topics_list[:-1]])
                last_topic = topics_list[-1].lower()
                summary_parts.append(f"This discussion explores {main_topics}, and {last_topic}, providing comprehensive insights into these interconnected areas of AI and technology.")
            
            summary_parts.append("")
            
            # Add structured topic sections
            for i, (topic, summary) in enumerate(topic_summaries.items(), 1):
                if summary and summary.strip():
                    # Format the topic section
                    summary_parts.append(f"{topic.upper()}:")
                    summary_parts.append(f"{summary}")
                    summary_parts.append("")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Error assembling topic summaries: {e}")
            return "Error assembling topic summaries."
    
    def _extract_topics(self, text: str, title: str) -> List[str]:
        """Extract key topics from the full transcript using sliding windows."""
        try:
            self.logger.info(f"Extracting topics from text length: {len(text)} characters")
            self.logger.info(f"Title: {title}")
            
            # Use windowed extraction for comprehensive coverage
            return self._extract_topics_windowed(text, title)
       
        except Exception as e:
            self.logger.error(f"Error extracting topics: {e}")
            return []
    
    def _extract_topics_windowed(self, text: str, title: str) -> List[str]:
        """Extract topics using sliding windows across the full transcript."""
        try:
            text_length = len(text)
            all_topics = []
            
            # Create sliding windows
            windows = self._create_sliding_windows(text)
            self.logger.info(f"Created {len(windows)} windows for text of {text_length} characters")
            
            # Extract topics from each window
            for i, window_text in enumerate(windows):
                self.logger.info(f"Processing window {i+1}/{len(windows)} ({len(window_text)} chars)")
                window_topics = self._extract_topics_from_window(window_text, title, i+1)
                all_topics.extend(window_topics)
            
            # Deduplicate and rank topics
            final_topics = self._deduplicate_and_rank_topics(all_topics, title)
            
            self.logger.info(f"Final topics ({len(final_topics)}): {final_topics}")
            return final_topics
            
        except Exception as e:
            self.logger.error(f"Error in windowed topic extraction: {e}")
            return []
    
    def _create_sliding_windows(self, text: str) -> List[str]:
        """Create overlapping windows across the text."""
        windows = []
        start = 0
        
        while start < len(text):
            end = min(start + self.window_size, len(text))
            window = text[start:end]
            windows.append(window)
            
            # Move to next window with overlap
            start += (self.window_size - self.window_overlap)
            
            # Stop if we've covered the entire text
            if end >= len(text):
                break
        
        return windows
    
    def _extract_topics_from_window(self, window_text: str, title: str, window_num: int) -> List[str]:
        """Extract topics from a single window."""
        try:
            # Get topic extraction prompt from PromptManager
            prompt_config = PromptManager.get_topic_extraction_prompt()
            
            # Format the prompt with window data
            prompt = prompt_config["template"].format(
                title=title,
                text_preview=window_text
            )

            messages = [
                {"role": "system", "content": prompt_config["system"]},
                {"role": "user", "content": prompt}
            ]

            response = self._make_api_request(messages, self.concept_extraction_tokens)
            if response:
                topics = self._parse_topics_response(response)
                # Limit topics per window
                limited_topics = topics[:self.topics_per_window]
                self.logger.info(f"Window {window_num} extracted {len(limited_topics)} topics: {limited_topics}")
                return limited_topics
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error extracting topics from window {window_num}: {e}")
            return []
    
    def _parse_topics_response(self, response: str) -> List[str]:
        """Parse topics from AI response."""
        try:
            response_clean = response.strip()
            
            # Try to extract JSON array from the response
            start_idx = response_clean.find('[')
            end_idx = response_clean.find(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_part = response_clean[start_idx:end_idx + 1]
                topics = json.loads(json_part)
                return [str(t).strip() for t in topics if str(t).strip()]
            else:
                # Try direct JSON parsing
                topics = json.loads(response_clean)
                return [str(t).strip() for t in topics if str(t).strip()]
                
        except json.JSONDecodeError:
            # Fallback: extract quoted strings
            topic_matches = re.findall(r'"([^"]+)"', response)
            if topic_matches:
                return [t.strip() for t in topic_matches if t.strip() and len(t.strip()) > 3]
        
        return []
    
    def _deduplicate_and_rank_topics(self, all_topics: List[str], title: str) -> List[str]:
        """Deduplicate topics and rank by semantic similarity to title."""
        if not all_topics:
            return []
        
        # Remove exact duplicates while preserving order
        unique_topics = []
        seen = set()
        
        for topic in all_topics:
            topic_lower = topic.lower().strip()
            if topic_lower not in seen:
                unique_topics.append(topic)
                seen.add(topic_lower)
        
        # Score topics by semantic similarity to title and order
        scored_topics = []
        for i, topic in enumerate(unique_topics):
            similarity_score = self._calculate_semantic_similarity(topic, title)
            order_score = 1.0 / (i + 1)  # Earlier topics get higher scores
            combined_score = (similarity_score * 0.7) + (order_score * 0.3)
            
            scored_topics.append((topic, combined_score, similarity_score, order_score))
        
        # Sort by combined score (descending)
        scored_topics.sort(key=lambda x: x[1], reverse=True)
        
        # Log scoring details
        self.logger.info("Topic scoring details:")
        for topic, combined, similarity, order in scored_topics[:self.max_final_topics]:
            self.logger.info(f"  '{topic}': similarity={similarity:.3f}, order={order:.3f}, combined={combined:.3f}")
        
        # Return top topics
        final_topics = [topic for topic, _, _, _ in scored_topics[:self.max_final_topics]]
        return final_topics
    
    def _calculate_semantic_similarity(self, topic: str, title: str) -> float:
        """Calculate semantic similarity between topic and title."""
        try:
            # Normalize text
            topic_clean = re.sub(r'[^\w\s]', '', topic.lower())
            title_clean = re.sub(r'[^\w\s]', '', title.lower())
            
            # Word overlap similarity
            topic_words = set(topic_clean.split())
            title_words = set(title_clean.split())
            
            if not topic_words or not title_words:
                return 0.0
            
            # Jaccard similarity (intersection over union)
            intersection = len(topic_words.intersection(title_words))
            union = len(topic_words.union(title_words))
            jaccard_score = intersection / union if union > 0 else 0.0
            
            # String similarity using SequenceMatcher
            sequence_score = SequenceMatcher(None, topic_clean, title_clean).ratio()
            
            # Combined similarity (weighted average)
            combined_similarity = (jaccard_score * 0.6) + (sequence_score * 0.4)
            
            return combined_similarity
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity for '{topic}' vs '{title}': {e}")
            return 0.0
        
    def _generate_title(self, text: str, metadata: Dict) -> str:
        """Generate a descriptive title for the content."""
        try:
            # Get title generation prompt from PromptManager
            prompt_config = PromptManager.get_title_generation_prompt()
            
            # Format the prompt with actual data
            prompt = prompt_config["template"].format(
                text_preview=text[:1000]
            )
            
            messages = [
                {"role": "system", "content": prompt_config["system"]},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_api_request(messages, self.title_tokens)
            
            if response:
                return response.strip().strip('"').strip("'")
            
            return metadata.get('title', 'Technical Discussion')
            
        except Exception as e:
            self.logger.error(f"Error generating title: {e}")
            return metadata.get('title', 'Technical Discussion')
    


    

    
    def _generate_fallback_notes(self, text: str, metadata: Dict) -> Dict:
        """Generate fallback notes when AI processing fails."""
        return {
            'title': metadata.get('title', 'Technical Discussion'),
            'summary': f"Summary of technical discussion with {len(text.split())} words.",
            'metadata': {
                'model_used': f"{self.model_name} (fallback)",
                'processing_timestamp': datetime.now().isoformat(),
                'original_metadata': metadata
            }
        }
    
    def _fallback_enhance_transcript(self, transcript: Dict) -> Dict:
        """Fallback method when enhancement fails."""
        self.logger.warning("Using fallback transcript enhancement")
        
        transcript['enhanced_notes'] = {
            'title': 'Technical Discussion',
            'summary': 'Content processing completed with basic enhancement.',
            'metadata': {
                'model_used': f"{self.model_name} (fallback)",
                'processing_timestamp': datetime.now().isoformat(),
                'original_metadata': transcript.get('metadata', {})
            }
        }
        
        return transcript
    
    def get_status(self) -> Dict:
        """Get processor status and configuration."""
        return {
            'model_name': self.model_name,
            'api_url': self.api_url,
            'chat_model': self.chat_model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'context_length': self.context_length,
            'status': 'ready' if self.api_url else 'not_configured'
        } 