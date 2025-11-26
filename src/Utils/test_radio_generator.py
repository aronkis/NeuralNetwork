#!/usr/bin/env python3
"""
Quick test script for radio modulation generator
Tests each modulation type individually to identify any remaining issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_radio_dataset import ModulationGenerator

def test_modulation(mod_name, mod_func, generator):
    """Test a single modulation function"""
    print(f"Testing {mod_name}...", end=" ")
    try:
        signal = mod_func(snr_db=10, freq_offset=0, phase_offset=0, carrier_freq=0)
        if signal is not None and len(signal) > 0:
            print(f"✅ SUCCESS - Generated {len(signal)} samples")
            return True
        else:
            print("❌ FAILED - No signal generated")
            return False
    except Exception as e:
        print(f"❌ FAILED - {str(e)}")
        return False

def main():
    print("🧪 Testing Radio Modulation Generator...")
    print("=" * 50)

    generator = ModulationGenerator(sample_rate=1e6, samples_per_symbol=8, num_samples=1024)

    # Test all 16 modulation types
    test_cases = [
        ("BPSK", generator.generate_bpsk),
        ("QPSK", generator.generate_qpsk),
        ("8PSK", generator.generate_8psk),
        ("16PSK", generator.generate_16psk),
        ("64PSK", generator.generate_64psk),
        ("16QAM", generator.generate_16qam),
        ("32QAM", generator.generate_32qam),
        ("64QAM", generator.generate_64qam),
        ("ASK", generator.generate_ask),
        ("2FSK", generator.generate_2fsk),
        ("4FSK", generator.generate_4fsk),
        ("8FSK", generator.generate_8fsk),
        ("GMSK", generator.generate_gmsk),
        ("MSK", generator.generate_msk),
        ("AM", generator.generate_am),
        ("FM", generator.generate_fm)
    ]

    success_count = 0
    total_count = len(test_cases)

    for mod_name, mod_func in test_cases:
        if test_modulation(mod_name, mod_func, generator):
            success_count += 1

    print("=" * 50)
    print(f"🎯 Test Results: {success_count}/{total_count} modulations working")

    if success_count == total_count:
        print("🎉 ALL TESTS PASSED! Generator ready for full dataset creation.")
        return True
    else:
        print("⚠️  Some modulations failed. Check errors above.")
        return False

if __name__ == "__main__":
    main()