#!/usr/bin/env python3
"""
Simple audio test that works with any available audio input.
"""

import sys
import sounddevice as sd
import numpy as np
import wave
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_all_audio_devices():
    """Test recording from all available audio devices."""
    print("🎤 Testing All Audio Devices")
    print("=" * 40)
    
    try:
        # Get all devices
        devices = sd.query_devices()
        print(f"Found {len(devices)} audio devices:")
        
        for i, device in enumerate(devices):
            print(f"  {i}: {device.get('name', 'Unknown')}")
            print(f"     Inputs: {device.get('max_input_channels', 0)}")
            print(f"     Outputs: {device.get('max_output_channels', 0)}")
        
        # Test each input device
        working_devices = []
        
        for i, device in enumerate(devices):
            if device.get('max_input_channels', 0) > 0:
                print(f"\nTesting device {i}: {device.get('name', 'Unknown')}")
                
                try:
                    # Record 2 seconds from this device
                    recording = sd.rec(int(2 * 16000), samplerate=16000, channels=1, device=i)
                    sd.wait()
                    
                    # Check if we got audio
                    if np.any(recording):
                        rms_level = np.sqrt(np.mean(recording**2))
                        print(f"  ✓ Audio recorded! RMS: {rms_level:.3f}")
                        
                        if rms_level > 0.001:
                            print(f"  ✓ Device {i} has audio!")
                            working_devices.append((i, device.get('name', 'Unknown'), rms_level))
                        else:
                            print(f"  ⚠ Device {i} is too quiet")
                    else:
                        print(f"  ✗ Device {i} is silent")
                        
                except Exception as e:
                    print(f"  ✗ Device {i} failed: {e}")
        
        return working_devices
        
    except Exception as e:
        print(f"✗ Error testing devices: {e}")
        return []

def test_simple_recording():
    """Test simple recording with any working device."""
    print("\n🎤 Simple Recording Test")
    print("=" * 30)
    
    try:
        from audio.capture import AudioCapture
        from utils.config import config
        
        # Find a working device
        working_devices = test_all_audio_devices()
        
        if not working_devices:
            print("❌ No working audio devices found!")
            return False
        
        # Use the device with the highest audio level
        best_device = max(working_devices, key=lambda x: x[2])
        device_index, device_name, rms_level = best_device
        
        print(f"\nUsing best device: {device_name} (index {device_index})")
        print(f"Audio level: {rms_level:.3f}")
        
        # Create a modified config to use this device
        audio_config = config.get_audio_config()
        audio_config['blackhole_device'] = device_name
        
        # Create audio capture with the working device
        capture = AudioCapture(audio_config)
        
        # Override the device index
        capture.device_index = device_index
        
        print(f"Recording 5 seconds using device {device_index}...")
        test_file = capture.record_to_file(duration=5)
        
        if os.path.exists(test_file):
            file_size = os.path.getsize(test_file)
            print(f"✓ Recording completed: {test_file} ({file_size} bytes)")
            
            if file_size > 1000:
                print("✓ Recording appears successful!")
                
                # Test playback
                print("\n🎵 Testing playback...")
                try:
                    import soundfile as sf
                    
                    audio_data, sample_rate = sf.read(test_file)
                    print(f"Audio info: {len(audio_data)} samples, {sample_rate}Hz")
                    
                    # Play the audio
                    print("Playing back the recording...")
                    sd.play(audio_data, sample_rate)
                    sd.wait()
                    print("✓ Playback completed!")
                    
                    return True, test_file
                    
                except Exception as e:
                    print(f"✗ Playback failed: {e}")
                    return False, test_file
            else:
                print("✗ Recording failed - file too small")
                return False, test_file
        else:
            print("✗ Recording failed - no file created")
            return False, None
            
    except Exception as e:
        print(f"✗ Simple recording test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Main function to test audio recording."""
    print("🎧 Simple Audio Test")
    print("=" * 30)
    
    success, test_file = test_simple_recording()
    
    if success:
        print("\n🎉 SUCCESS! Audio recording is working!")
        print(f"Test recording saved at: {test_file}")
        
        # Ask if user wants to keep the file
        keep_file = input("\nKeep the test file? (y/n): ").lower().strip()
        if keep_file != 'y' and test_file:
            os.remove(test_file)
            print("Test file removed")
        else:
            print(f"Test file kept at: {test_file}")
        
        print("\n✅ Audio Module is working!")
        print("Next: Move to Transcription Module")
        
        return True
    else:
        print("\n❌ Audio recording test failed.")
        print("Please check your audio setup.")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 