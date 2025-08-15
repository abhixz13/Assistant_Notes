#!/usr/bin/env python3
"""
Run CustomModelProcessor to summarize the AI agents transcript
"""

import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.summarization.custom_model import CustomModelProcessor

def main():
    """Run the CustomModelProcessor on the available transcript"""
    
    print("🤖 Testing CustomModelProcessor AI System")
    print("=" * 50)
    
    # Initialize the CustomModelProcessor
    print("📋 Initializing CustomModelProcessor...")
    processor = CustomModelProcessor()
    
    # Check processor status
    status = processor.get_status()
    print(f"🔧 Model: {status['model_name']}")
    print(f"🌐 API URL: {status['api_url']}")
    print(f"🤖 Chat Model: {status['chat_model']}")
    print(f"📊 Status: {status['status']}")
    print()
    
    # Load the transcript
    transcript_file = "data/transcripts/What_are_AI_agents_really_about_20250811_203411.json"
    print(f"📄 Loading transcript: {transcript_file}")
    
    with open(transcript_file, 'r') as f:
        transcript = json.load(f)
    
    # Extract basic info
    metadata = transcript.get('metadata', {})
    title = metadata.get('title', 'AI Agents Discussion')
    duration = metadata.get('duration_formatted', 'Unknown')
    segments = metadata.get('segments_count', 0)
    
    print(f"📺 Title: {title}")
    print(f"⏱️ Duration: {duration}")
    print(f"📝 Segments: {segments}")
    print()
    
    # Process the transcript with different expertise levels
    expertise_levels = ["beginner", "moderate", "expert"]
    
    for expertise_level in expertise_levels:
        print(f"🧠 Processing with {expertise_level} expertise level...")
        print("-" * 40)
        
        try:
            # Enhance the transcript
            enhanced_transcript = processor.enhance_transcript(transcript, expertise_level)
            
            # Extract enhanced notes
            enhanced_notes = enhanced_transcript.get('enhanced_notes', {})
            
            if enhanced_notes:
                # Save the enhanced notes
                output_dir = Path("data/notes")
                output_dir.mkdir(exist_ok=True)
                
                # Create filename based on expertise level
                clean_title = title.replace(' ', '_').replace('?', '').replace(':', '')
                filename = f"ai_agents_{expertise_level}_summary.md"
                output_file = output_dir / filename
                
                # Format the summary
                summary_content = f"""# {enhanced_notes.get('title', title)}

## Transcript Information
- **Original Title**: {title}
- **Duration**: {duration}
- **Segments**: {segments}
- **Expertise Level**: {expertise_level.capitalize()}
- **Model Used**: {enhanced_notes.get('metadata', {}).get('model_used', 'CustomModelProcessor')}
- **Processing Time**: {enhanced_notes.get('metadata', {}).get('processing_timestamp', 'Unknown')}

## Summary

{enhanced_notes.get('summary', 'No summary generated.')}

## Topics Covered

"""
                
                # Add topics if available
                topics = enhanced_notes.get('metadata', {}).get('topics', [])
                if topics:
                    for i, topic in enumerate(topics, 1):
                        summary_content += f"{i}. {topic}\n"
                else:
                    summary_content += "No specific topics extracted.\n"
                
                # Add topic summaries if available
                topic_summaries = enhanced_notes.get('metadata', {}).get('topic_summaries', {})
                if topic_summaries:
                    summary_content += "\n## Detailed Topic Summaries\n\n"
                    for topic, summary in topic_summaries.items():
                        if summary and summary.strip():
                            summary_content += f"### {topic}\n{summary}\n\n"
                
                # Save to file
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(summary_content)
                
                print(f"✅ Summary saved: {output_file}")
                print(f"📊 File size: {output_file.stat().st_size} bytes")
                
                # Show a preview of the summary
                summary_preview = enhanced_notes.get('summary', '')[:200]
                if summary_preview:
                    print(f"📝 Preview: {summary_preview}...")
                
            else:
                print("❌ No enhanced notes generated")
                
        except Exception as e:
            print(f"❌ Error processing with {expertise_level} expertise: {e}")
        
        print()
    
    print("🎉 CustomModelProcessor testing completed!")
    print("📁 Check the data/notes/ folder for generated summaries")

if __name__ == "__main__":
    main() 