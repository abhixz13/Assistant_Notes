"""
Prompts Module for AI Notes Assistant

This module contains all the critical prompts used by the CustomModelProcessor
for generating AI-powered summaries and notes from transcripts.

Each prompt is organized with clear metadata including:
- Purpose: What the prompt is designed to accomplish
- Usage: Where and how it's used in the processing pipeline
- Input: What data it expects
- Output: What format it should produce
- Expertise Level: Which user expertise level it targets (if applicable)
"""

from typing import Dict, List


class PromptManager:
    """
    Manages all prompts used in the AI notes generation system.
    
    Provides organized access to prompts with metadata and documentation
    for easy modification and maintenance.
    """
    
    @staticmethod
    def get_topic_extraction_prompt() -> Dict[str, str]:
        """
        Prompt for extracting key topics from transcript.
        
        Purpose: Identifies 8-12 specific, granular topics from the transcript
        Usage: Called by _extract_topics() in the topic extraction phase
        Input: Transcript text and title
        Output: JSON array of topic strings
        Expertise Level: N/A (same for all levels)
        
        Returns:
            Dict containing system message and user prompt template
        """
        return {
            "system":
        """Extract 3-4 specific, non-overlapping topics explicitly present in the transcript. Use short noun phrases (~3–7 words), each ≤ 80 characters, grounded in the transcript’s wording. When the transcript enumerates a named set (e.g., types, taxonomy, spectrum), include ONE umbrella topic that names that set using its wording (e.g., "Agent taxonomy: …"). Collapse near-duplicates (singular/plural, synonyms). Exclude vague buckets (e.g., "Overview", "Implementation", "Software Development") and any promo/CTA content. Order topics by first appearance. Return a JSON array of double-quoted strings ONLY—no code fences, comments, or trailing commas."

        template:
        Title: {title}

        Transcript (truncated):
        {text_preview}

        JSON array only:
        """
        }
        
    
    @staticmethod
    def get_title_generation_prompt() -> Dict[str, str]:
        """
        Prompt for generating descriptive titles.
        
        Purpose: Creates concise, descriptive titles for the content
        Usage: Called by _generate_title() in the title generation phase
        Input: Transcript text preview
        Output: 5-8 word descriptive title
        Expertise Level: N/A (same for all levels)
        
        Returns:
            Dict containing system message and user prompt template
        """
        return {
            "system": "You are a concise technical titler. Title must faithfully reflect the main topic.No clickbate, keep it neutral and specific.",
            
            "template": """Generate a descriptive title (5–8 words) grounded in the text below.
Content:
{text_preview}

Title:"""
        }
    
    @staticmethod
    def get_expertise_prompts() -> Dict[str, Dict[str, str]]:
        """
        Prompts for different expertise levels.
        
        Purpose: Provides expertise-level specific prompts for topic summarization
        Usage: Called by _get_expertise_prompt() for topic summary generation
        Input: Expertise level (beginner/moderate/expert)
        Output: System message and template for topic summarization
        Expertise Level: beginner, moderate, expert
        
        Returns:
            Dict mapping expertise levels to their respective prompts
        """
        return {
            "beginner": {
                "system": """You are a concise and cohesive summarizer. Output must be strictly grounded in the provided text.
                Do NOT invent topics, sections, methods, metrics, or trade-offs.
                Include specifics (numbers, names, dates, models) only if present.
                No meta-commentary. Use plain language; define uncommon terms briefly only when they appear.""",
            
            "template": """Audience: Beginner.
Write a concise, easy-to-read summary of the excerpt.
Derive any headings from content (optional). Use short bullets or tight sentences.
Avoid redundancy; ≤ 400 words.

{topic_text}"""
            },

            
            "moderate": {
                "system": """You are a concise and cohesive summarizer. Output must be strictly grounded in the provided text.
Do NOT invent topics, sections, methods, metrics, or trade-offs.
Include specifics only if present. No meta-commentary.""",
                "template": """Audience: Mixed technical.
Produce a concise, structured summary.
Headings optional; prefer bullets/short sentences.
Include methods/limitations only if present. ≤ 500 words.

{topic_text}"""
            },
            
            "expert": {
                "system": """You are a concise and cohesive summarizer. Output must be strictly grounded in the provided text.
Do NOT invent topics, sections, methods, metrics, or trade-offs.
Include specifics only if present. No meta-commentary.""",
                "template": """Audience: Expert.
Produce a concise, structured mini-summary.
Headings optional; prefer bullets/short sentences.
Include methods/limitations only if present. ≤ 500 words.

{topic_text}"""
            }
        }
    
    @staticmethod
    def get_topic_summary_prompt(topic: str, topic_text: str, expertise_level: str = "moderate") -> Dict[str, str]:
        """
        Prompt for generating individual topic summaries.
        
        Purpose: Creates focused summaries for specific topics using expertise-level adaptation
        Usage: Called by _generate_topic_summary() for each topic
        Input: Topic name, relevant text, expertise level
        Output: Focused summary for the topic
        Expertise Level: beginner, moderate, expert
        
        Args:
            topic: The topic to summarize
            topic_text: Text content relevant to the topic
            expertise_level: User expertise level (beginner/moderate/expert)
            
        Returns:
            Dict containing system message and user prompt
        """
        expertise_prompts = PromptManager.get_expertise_prompts()
        expertise_config = expertise_prompts.get(expertise_level.lower(), expertise_prompts["moderate"])
        
        return {
            "system": expertise_config["system"],
            "template": f"""Topic: {topic}
Summarize only what this excerpt states about the topic.
If the mention is brief, keep the summary very short.

{expertise_config['template'].replace('{topic_text}', topic_text)}"""
        }
    
    @staticmethod
    def get_prompt_metadata() -> Dict[str, Dict]:
        """
        Metadata about all prompts for documentation and maintenance.
        
        Returns:
            Dict containing metadata for each prompt type
        """
        return {
            "topic_extraction": {
                "description": "Extracts 8-12 specific topics from transcript",
                "usage": "_extract_topics() method",
                "input_format": "Transcript text and title",
                "output_format": "JSON array of topic strings",
                "critical_parameters": ["title", "text_preview"],
                "modification_notes": "Adjust topic count (8-12) or granularity requirements here"
            },
            
            "title_generation": {
                "description": "Generates descriptive titles for content",
                "usage": "_generate_title() method", 
                "input_format": "Transcript text preview (first 1000 chars)",
                "output_format": "5-8 word descriptive title",
                "critical_parameters": ["text_preview"],
                "modification_notes": "Adjust title length requirements or style here"
            },
            
            "expertise_prompts": {
                "description": "Expertise-level specific prompts for topic summarization",
                "usage": "_get_expertise_prompt() and _generate_topic_summary() methods",
                "input_format": "Topic name, relevant text, expertise level",
                "output_format": "Focused topic summary",
                "critical_parameters": ["topic", "topic_text", "expertise_level"],
                "modification_notes": "Modify content structure, tone, or complexity requirements for each expertise level"
            },
            
            "topic_summary": {
                "description": "Generates individual topic summaries with expertise adaptation",
                "usage": "_generate_topic_summary() method",
                "input_format": "Topic name, filtered text, expertise level",
                "output_format": "Professional topic summary",
                "critical_parameters": ["topic", "topic_text", "expertise_level"],
                "modification_notes": "Adjust summary structure, length, or content requirements here"
            }
        }
    
    @staticmethod
    def validate_prompt_structure(prompt_type: str, prompt_data: Dict) -> bool:
        """
        Validates that a prompt has the required structure.
        
        Args:
            prompt_type: Type of prompt being validated
            prompt_data: The prompt data to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = ["system", "template"]
        
        if not all(key in prompt_data for key in required_keys):
            return False
            
        if not prompt_data["system"] or not prompt_data["template"]:
            return False
            
        return True
    
    @staticmethod
    def get_prompt_statistics() -> Dict[str, int]:
        """
        Returns statistics about the prompts for monitoring.
        
        Returns:
            Dict with prompt statistics
        """
        expertise_prompts = PromptManager.get_expertise_prompts()
        
        return {
            "total_prompt_types": 4,
            "expertise_levels": len(expertise_prompts),
            "total_prompts": 4 + len(expertise_prompts),  # 4 main + expertise levels
            "prompt_metadata_entries": len(PromptManager.get_prompt_metadata())
        } 