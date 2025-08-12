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
from typing import Dict, List, Optional, Any
from datetime import datetime
import requests

from utils.config import ConfigManager
from utils.logger import get_logger


class CustomModelProcessor:
    """
    AI Scholar Processor using custom models via OpenAI-compatible APIs.
    
    Transforms raw transcripts into comprehensive, structured notes
    that enable deep understanding and 60-minute speaking capability.
    """
    
    def __init__(self):
        """Initialize the custom model processor."""
        self.logger = get_logger(__name__)
        self.config = ConfigManager()
        
        # Load configuration
        self.custom_model_config = self.config.get_custom_model_config()
        self.summarization_config = self.config.get_summarization_config()
        self.notes_config = self.config.get_notes_config()
        
        # Model configuration
        self.model_name = self.custom_model_config.get('name', 'Deepseek-coder')
        self.api_url = self.custom_model_config.get('api', {}).get('url', 'http://localhost:11434/v1')
        self.api_type = self.custom_model_config.get('api', {}).get('type', 'openai-compatible')
        self.chat_model = self.custom_model_config.get('chat', {}).get('model', 'deepseek-coder:latest')
        
        # Processing configuration
        self.max_tokens = self.summarization_config.get('max_tokens', 2048)
        self.temperature = self.summarization_config.get('temperature', 0.7)
        self.context_length = self.summarization_config.get('context_length', 8192)
        
        # Enhanced prompt configuration
        self.concept_extraction_tokens = self.summarization_config.get('concept_extraction_tokens', 512)
        self.summary_tokens = self.summarization_config.get('summary_tokens', 300)
        # Removed: discussion_points_tokens
        self.additional_details_tokens = self.summarization_config.get('additional_details_tokens', 1500)
        self.title_tokens = self.summarization_config.get('title_tokens', 100)
        
        self.logger.info(f"Initialized Custom Model Processor with {self.model_name}")
        self.logger.info(f"API URL: {self.api_url}")
        self.logger.info(f"Chat Model: {self.chat_model}")
    
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
            headers = {
                'Content-Type': 'application/json',
            }
            
            payload = {
                'model': self.chat_model,
                'messages': messages,
                'temperature': self.temperature,
                'max_tokens': max_tokens or self.max_tokens,
                'stream': False
            }
            
            self.logger.debug(f"Making API request to {self.api_url}")
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
    
    def enhance_transcript(self, transcript: Dict) -> Dict:
        """
        Enhance a raw transcript with AI-generated context and structure.
        
        Args:
            transcript: Raw transcript dictionary with segments and metadata
            
        Returns:
            Enhanced transcript with structured notes
        """
        try:
            self.logger.info("Starting transcript enhancement with custom model")
            
            # Extract key information
            full_text = transcript.get('full_text', '')
            segments = transcript.get('segments', [])
            metadata = transcript.get('metadata', {})
            
            if not full_text.strip():
                self.logger.warning("No text content found in transcript")
                return self._fallback_enhance_transcript(transcript)
            
            # Generate enhanced notes
            enhanced_notes = self._generate_scholar_notes(full_text, segments, metadata)
            
            # Update transcript with enhanced content
            transcript['enhanced_notes'] = enhanced_notes
            transcript['processing_timestamp'] = datetime.now().isoformat()
            transcript['model_used'] = self.model_name
            
            self.logger.info("✅ Transcript enhancement completed successfully")
            return transcript
            
        except Exception as e:
            self.logger.error(f"Error enhancing transcript: {e}")
            return self._fallback_enhance_transcript(transcript)
    
    def _generate_scholar_notes(self, full_text: str, segments: List[Dict], metadata: Dict) -> Dict:
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
            
            # Generate summary with topics context
            summary = self._generate_summary(full_text, topics)
            
            # Generate additional details
            additional_details = self._generate_additional_details(full_text)
            
            return {
                'title': title,
                'summary': summary,
                'additional_details': additional_details,
                'metadata': {
                    'model_used': self.model_name,
                    'processing_timestamp': datetime.now().isoformat(),
                    'topics': topics,
                    'original_metadata': metadata
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating scholar notes: {e}")
            return self._generate_fallback_notes(full_text, metadata)
    
    def _extract_topics(self, text: str, title: str) -> List[str]:
        """Extract 5-8 key topics from the transcript using title as context."""
        try:
            system_msg = """
            You are an expert AI/ML analyst. Your role is to read a transcript of an AI/ML discussion and identify the main topics in clear, concise bullet points.
            
            Output requirements:
            - List 3–6 topics in logical order.
            - Each topic should be a short phrase (max 12 words).
            - Avoid vague labels — use precise, descriptive wording.
            - Topics should cover technical, business, and application aspects mentioned.
            - Do not write summaries or paragraphs — only list the topics.
            """
            
            prompt = f"""
            Given the following title and transcript, list the 5-8 most important topics covered.
            - Focus on AI/ML concepts, methods, tools, and subtopics explicitly discussed
            - Be concise: return only a JSON array of short topic phrases

            Title: {title}

            Transcript: {text[:3000]}...

            JSON array:
            """

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ]

            response = self._make_api_request(messages, self.concept_extraction_tokens)
            if response:
                try:
                    topics = json.loads(response.strip())
                    if isinstance(topics, list):
                        # Normalize topics to strings and limit count
                        topics = [str(t).strip() for t in topics if str(t).strip()]
                        return topics[:8]
                except json.JSONDecodeError:
                    pass
            # Fallback: naive extraction using frequent nouns-like words as placeholders
            return []
        except Exception as e:
            self.logger.error(f"Error extracting topics: {e}")
            return []
    
    def _generate_title(self, text: str, metadata: Dict) -> str:
        """Generate a descriptive title for the content."""
        try:
            prompt = f"""
            Generate a concise, descriptive title (5-8 words) for this technical content.
            The title should capture the main topic or theme.
            
            Content: {text[:1000]}...
            
            Title:
            """
            
            messages = [
                {"role": "system", "content": "You are a technical business writer with an expertise in AI/ML and Product Management. Generate concise, descriptive titles."},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_api_request(messages, self.title_tokens)
            
            if response:
                return response.strip().strip('"').strip("'")
            
            return metadata.get('title', 'Technical Discussion')
            
        except Exception as e:
            self.logger.error(f"Error generating title: {e}")
            return metadata.get('title', 'Technical Discussion')
    
    def _generate_summary(self, text: str, topics: List[str]) -> str:
        """Generate a comprehensive, topic-structured summary of the content."""
        
        try:
            topics_str = ", ".join(topics[:8]) if topics else "No topics provided"

            prompt = f"""
            You will write a professional, multi-paragraph summary of AI/ML content
            based on the transcript and the identified topics below.

            Identified topics to emphasize: {topics_str}

            **Instructions:**
            1. Use the identified topics as section headings in the summary.
            2. For each topic, write one well-developed paragraph that:
            - Explains the concept in clear, professional language
            - Integrates relevant details from the transcript
            - Includes specific technical methods, tools, datasets, or metrics mentioned
            - Describes real-world applications and business implications
            - Notes any challenges, limitations, or future directions
            3. Maintain a professional, educational tone suitable for both technical and business audiences.
            4. Avoid bullet points — write continuous prose under each heading.
            5. The overall summary should be 3–4 paragraphs total (one per topic).

            **Example Output Style:**
            Topic: AI Agents
            AI agents are autonomous systems capable of perceiving their environment, reasoning about possible actions, and executing multi-step tasks without constant human oversight. In the discussion, speakers explained how modern AI agents often use large language models (LLMs) as reasoning engines, allowing them to plan workflows, integrate with APIs, and adapt their decisions based on context. Examples included automated customer support agents that integrate with CRM tools and data analytics agents that pull insights from live streaming data. The panel also highlighted enabling technologies such as vector databases for persistent memory, orchestration frameworks like LangChain, and reinforcement learning for optimizing decision-making. While these agents promise significant productivity gains, challenges remain in ensuring reliability, security, and ethical alignment when deployed in dynamic environments.

            Transcript to summarize:
            {text[:4000]}
            """

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI/ML domain expert and technical writer. "
                        "Your role is to produce clear, structured, and professional summaries "
                        "that synthesize technical discussions into topic-based multi-paragraph reports. "
                        "You must balance technical accuracy with accessible explanations."
                    )
                },
                {"role": "user", "content": prompt}
            ]

            response = self._make_api_request(messages, self.summary_tokens)
            if response:
                return response.strip()
            return "Summary of the technical discussion."
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return "Summary of the technical discussion."

    
    def _generate_additional_details(self, text: str) -> Dict:
        """Generate additional technical details and resources."""
        try:
            prompt = f"""
            Based on this technical content, provide:
            1. 2-3 related technologies or tools that complement the discussion
            2. 1-2 learning resources or references for further study
            3. 1-2 best practices or recommendations
            
            Content: {text[:3000]}...
            
            Provide the response as a JSON object with these keys:
            - related_technologies (array of strings)
            - learning_resources (array of strings)
            - best_practices (array of strings)
            """
            
            messages = [
                {"role": "system", "content": "You are a technical expert providing additional resources and best practices."},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_api_request(messages, self.additional_details_tokens)
            
            if response:
                try:
                    return json.loads(response.strip())
                except json.JSONDecodeError:
                    return self._generate_fallback_additional_details(text)
            
            return self._generate_fallback_additional_details(text)
            
        except Exception as e:
            self.logger.error(f"Error generating additional details: {e}")
            return self._generate_fallback_additional_details(text)
    
    def _generate_fallback_notes(self, text: str, metadata: Dict) -> Dict:
        """Generate fallback notes when AI processing fails."""
        return {
            'title': metadata.get('title', 'Technical Discussion'),
            'summary': f"Summary of technical discussion with {len(text.split())} words.",
            
            
            'additional_details': {
                'related_technologies': ['Documentation tools'],
                'learning_resources': ['Technical documentation'],
                'best_practices': ['Regular content review']
            },
            'metadata': {
                'model_used': f"{self.model_name} (fallback)",
                'processing_timestamp': datetime.now().isoformat(),
                'original_metadata': metadata
            }
        }
    
    # Removed: _generate_fallback_discussion_points
    
    def _generate_fallback_additional_details(self, text: str) -> Dict:
        """Fallback method for additional details."""
        return {
            'related_technologies': ['Development tools', 'Testing frameworks'],
            'learning_resources': ['Official documentation', 'Community forums'],
            'best_practices': ['Code review', 'Testing strategies']
        }
    
    def _fallback_enhance_transcript(self, transcript: Dict) -> Dict:
        """Fallback method when enhancement fails."""
        self.logger.warning("Using fallback transcript enhancement")
        
        transcript['enhanced_notes'] = {
            'title': 'Technical Discussion',
            'summary': 'Content processing completed with basic enhancement.',
            
            
            'additional_details': {
                'related_technologies': ['Tools'],
                'learning_resources': ['Documentation'],
                'best_practices': ['Review']
            },
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