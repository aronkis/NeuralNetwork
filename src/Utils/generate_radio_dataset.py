#!/usr/bin/env python3
"""
Radio Modulation Dataset Generator - 4 Modulation Types
Generates synthetic I/Q samples for 4 key modulation types: BPSK, QPSK, 16QAM, 32QAM.
SNR levels designed for easy, difficult, and hard detection scenarios.
Perfect for neural network training on radio signal classification.

Modulation Types: BPSK (PSK), QPSK, 16QAM, 32QAM
SNR Levels: Easy (15-20 dB), Difficult (5-10 dB), Hard (-5-0 dB)
"""

import numpy as np
import random
from gnuradio import gr, blocks, digital, analog, channels, filter
import os
import argparse

class ModulationGenerator:
    def __init__(self, sample_rate=1e6, samples_per_symbol=8, num_samples=2048):
        self.sample_rate = sample_rate
        self.samples_per_symbol = samples_per_symbol
        self.num_samples = num_samples
        self.symbol_rate = sample_rate / samples_per_symbol

    def generate_random_bits(self, num_symbols):
        """Generate random bit stream"""
        return np.random.randint(0, 2, num_symbols)

    def generate_bpsk(self, snr_db=15, freq_offset=0, phase_offset=0):
        """Generate BPSK modulated signal - compatible with VariableModulation.py"""
        tb = gr.top_block()

        # Number of symbols needed
        num_symbols = self.num_samples // self.samples_per_symbol + 100

        # Generate random data
        data_bits = self.generate_random_bits(num_symbols)
        data_source = blocks.vector_source_b(data_bits, False)

        # BPSK constellation: [-1+0j, 1+0j] (same as VariableModulation.py)
        bpsk_constellation = digital.constellation_calcdist([-1+0j, 1+0j], [0, 1],
                                                           4, 1, digital.constellation.AMPLITUDE_NORMALIZATION).base()
        bpsk_constellation.set_npwr(1)

        # BPSK modulator
        bpsk_mod = digital.generic_mod(
            constellation=bpsk_constellation,
            differential=True,
            samples_per_symbol=self.samples_per_symbol,
            pre_diff_code=True,
            excess_bw=0.5,
            verbose=False,
            log=False
        )

        # Channel impairments
        noise_source = analog.noise_source_c(analog.GR_GAUSSIAN, 10**(-snr_db/20.0), 0)
        adder = blocks.add_cc()

        # Frequency offset
        freq_shift = blocks.multiply_cc()
        freq_source = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE, freq_offset, 1, 0)

        # Phase offset
        phase_shift = blocks.multiply_const_cc(np.exp(1j * phase_offset))
        output_sink = blocks.vector_sink_c()

        # Connect blocks
        tb.connect(data_source, bpsk_mod)
        tb.connect(bpsk_mod, (adder, 0))
        tb.connect(noise_source, (adder, 1))
        tb.connect(adder, (freq_shift, 0))
        tb.connect(freq_source, (freq_shift, 1))
        tb.connect(freq_shift, phase_shift)
        tb.connect(phase_shift, output_sink)

        # Run flowgraph
        tb.run()
        tb.wait()

        samples = np.array(output_sink.data())
        return samples[:self.num_samples]

    def generate_qpsk(self, snr_db=15, freq_offset=0, phase_offset=0):
        """Generate QPSK modulated signal - compatible with VariableModulation.py"""
        tb = gr.top_block()

        num_symbols = self.num_samples // self.samples_per_symbol + 100
        data_bits = self.generate_random_bits(num_symbols * 2)  # 2 bits per symbol

        data_source = blocks.vector_source_b(data_bits, False)

        # QPSK constellation: [-1-1j, +1-1j, +1+1j, -1+1j] (same as VariableModulation.py)
        qpsk_constellation = digital.constellation_calcdist([-1-1j, +1-1j, +1+1j, -1+1j], [0, 1, 2, 3],
                                                           4, 1, digital.constellation.AMPLITUDE_NORMALIZATION).base()
        qpsk_constellation.set_npwr(1)

        # QPSK modulator
        qpsk_mod = digital.generic_mod(
            constellation=qpsk_constellation,
            differential=True,
            samples_per_symbol=self.samples_per_symbol,
            pre_diff_code=True,
            excess_bw=0.5,
            verbose=False,
            log=False
        )

        # Channel impairments
        noise_source = analog.noise_source_c(analog.GR_GAUSSIAN, 10**(-snr_db/20.0), 0)
        adder = blocks.add_cc()
        freq_shift = blocks.multiply_cc()
        freq_source = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE, freq_offset, 1, 0)
        phase_shift = blocks.multiply_const_cc(np.exp(1j * phase_offset))
        output_sink = blocks.vector_sink_c()

        tb.connect(data_source, qpsk_mod)
        tb.connect(qpsk_mod, (adder, 0))
        tb.connect(noise_source, (adder, 1))
        tb.connect(adder, (freq_shift, 0))
        tb.connect(freq_source, (freq_shift, 1))
        tb.connect(freq_shift, phase_shift)
        tb.connect(phase_shift, output_sink)

        tb.run()
        tb.wait()

        samples = np.array(output_sink.data())
        return samples[:self.num_samples]

    def generate_16qam(self, snr_db=15, freq_offset=0, phase_offset=0):
        """Generate 16-QAM modulated signal - compatible with VariableModulation.py"""
        tb = gr.top_block()

        num_symbols = self.num_samples // self.samples_per_symbol + 100
        data_bits = self.generate_random_bits(num_symbols * 4)  # 4 bits per symbol

        data_source = blocks.vector_source_b(data_bits, False)

        # 16-QAM constellation (same as VariableModulation.py)
        qam16_constellation = digital.constellation_calcdist(
            [-3-3j, -1-3j, 1-3j, 3-3j, -3-1j, -1-1j, 1-1j, 3-1j,
             -3+1j, -1+1j, 1+1j, 3+1j, -3+3j, -1+3j, 1+3j, 3+3j],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            4, 1, digital.constellation.AMPLITUDE_NORMALIZATION).base()
        qam16_constellation.set_npwr(1)

        # 16-QAM modulator
        qam_mod = digital.generic_mod(
            constellation=qam16_constellation,
            differential=True,
            samples_per_symbol=self.samples_per_symbol,
            pre_diff_code=True,
            excess_bw=0.5,
            verbose=False,
            log=False
        )

        # Channel impairments
        noise_source = analog.noise_source_c(analog.GR_GAUSSIAN, 10**(-snr_db/20.0), 0)
        adder = blocks.add_cc()
        freq_shift = blocks.multiply_cc()
        freq_source = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE, freq_offset, 1, 0)
        phase_shift = blocks.multiply_const_cc(np.exp(1j * phase_offset))
        output_sink = blocks.vector_sink_c()

        tb.connect(data_source, qam_mod)
        tb.connect(qam_mod, (adder, 0))
        tb.connect(noise_source, (adder, 1))
        tb.connect(adder, (freq_shift, 0))
        tb.connect(freq_source, (freq_shift, 1))
        tb.connect(freq_shift, phase_shift)
        tb.connect(phase_shift, output_sink)

        tb.run()
        tb.wait()

        samples = np.array(output_sink.data())
        return samples[:self.num_samples]


    def generate_32qam(self, snr_db=15, freq_offset=0, phase_offset=0):
        """Generate 32-QAM modulated signal - compatible with VariableModulation.py"""
        tb = gr.top_block()

        num_symbols = self.num_samples // self.samples_per_symbol + 100
        data_bits = self.generate_random_bits(num_symbols * 5)  # 5 bits per symbol

        data_source = blocks.vector_source_b(data_bits, False)

        # 32-QAM constellation (same as VariableModulation.py)
        qam32_constellation = digital.constellation_calcdist(
            [-5-3j, -3-3j, -1-3j, 1-3j, 3-3j, 5-3j, -5-1j, -3-1j, -1-1j, 1-1j, 3-1j, 5-1j,
             -5+1j, -3+1j, -1+1j, 1+1j, 3+1j, 5+1j, -5+3j, -3+3j, -1+3j, 1+3j, 3+3j, 5+3j,
             -3-5j, -1-5j, 1-5j, 3-5j, -3+5j, -1+5j, 1+5j, 3+5j],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
             24, 25, 26, 27, 28, 29, 30, 31],
            4, 1, digital.constellation.AMPLITUDE_NORMALIZATION).base()
        qam32_constellation.set_npwr(1)

        # 32-QAM modulator
        qam32_mod = digital.generic_mod(
            constellation=qam32_constellation,
            differential=True,
            samples_per_symbol=self.samples_per_symbol,
            pre_diff_code=True,
            excess_bw=0.5,
            verbose=False,
            log=False
        )

        # Channel impairments
        noise_source = analog.noise_source_c(analog.GR_GAUSSIAN, 10**(-snr_db/20.0), 0)
        adder = blocks.add_cc()
        freq_shift = blocks.multiply_cc()
        freq_source = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE, freq_offset, 1, 0)
        phase_shift = blocks.multiply_const_cc(np.exp(1j * phase_offset))
        output_sink = blocks.vector_sink_c()

        tb.connect(data_source, qam32_mod)
        tb.connect(qam32_mod, (adder, 0))
        tb.connect(noise_source, (adder, 1))
        tb.connect(adder, (freq_shift, 0))
        tb.connect(freq_source, (freq_shift, 1))
        tb.connect(freq_shift, phase_shift)
        tb.connect(phase_shift, output_sink)

        tb.run()
        tb.wait()

        samples = np.array(output_sink.data())
        return samples[:self.num_samples]


def apply_channel_impairments(signal, timing_offset, carrier_drift, amplitude_variation, sample_rate):
    """Apply realistic channel impairments to match real SDR conditions (Pluto SDR)"""

    # 1. Amplitude fading (realistic for mobile/RF channels)
    signal = signal * amplitude_variation

    # 2. Carrier frequency drift over time (realistic oscillator drift)
    t = np.arange(len(signal)) / sample_rate
    drift_phase = 2j * np.pi * carrier_drift * t**2 / 2  # Quadratic phase drift
    signal = signal * np.exp(drift_phase)

    # 3. Symbol timing offset (realistic receiver sync issues)
    if abs(timing_offset) > 0.01:  # Only apply if significant
        samples_offset = int(timing_offset * 8)  # samples_per_symbol = 8
        if samples_offset > 0:
            signal = np.pad(signal, (samples_offset, 0), mode='constant')[:-samples_offset]
        elif samples_offset < 0:
            signal = np.pad(signal, (0, -samples_offset), mode='constant')[-samples_offset:]

    # 4. Additional multipath fading (simplified Rayleigh fading)
    # Generate complex random taps for multipath
    real_taps = np.random.randn(3) * 0.1
    imag_taps = np.random.randn(3) * 0.1
    fading_taps = real_taps + 1j * imag_taps
    fading_taps[0] = 1.0  # Direct path (strongest)
    multipath_signal = np.convolve(signal, fading_taps, mode='same')

    # 5. Frequency-selective fading (different frequencies fade differently)
    # Add small frequency-dependent amplitude variations
    freq_selective = 1.0 + 0.1 * np.random.randn() * np.sin(2 * np.pi * t * 1000)  # 1kHz variation
    signal = multipath_signal * freq_selective[:len(multipath_signal)]

    return signal


def apply_pluto_realistic_impairments(signal, sample_rate):
    """
    Apply impairments that specifically match Pluto SDR real-world conditions.
    These are additional effects beyond the basic channel model.
    """
    t = np.arange(len(signal)) / sample_rate
    
    # 1. I/Q Imbalance (common in direct-conversion receivers like Pluto)
    # Amplitude imbalance: 1-5%
    # Phase imbalance: 1-5 degrees
    amp_imbalance = np.random.uniform(0.95, 1.05)
    phase_imbalance = np.random.uniform(-5, 5) * np.pi / 180  # degrees to radians
    
    I = signal.real
    Q = signal.imag
    # Apply imbalance
    I_imbalanced = I
    Q_imbalanced = amp_imbalance * (Q * np.cos(phase_imbalance) + I * np.sin(phase_imbalance))
    signal = I_imbalanced + 1j * Q_imbalanced
    
    # 2. DC Offset (common in direct-conversion receivers)
    dc_offset_i = np.random.uniform(-0.05, 0.05)
    dc_offset_q = np.random.uniform(-0.05, 0.05)
    signal = signal + dc_offset_i + 1j * dc_offset_q
    
    # 3. Phase Noise (oscillator imperfection)
    # Random walk phase noise
    phase_noise_std = np.random.uniform(0.01, 0.05)  # radians
    phase_noise = np.cumsum(np.random.randn(len(signal)) * phase_noise_std)
    # Add slow varying component
    slow_phase = 0.1 * np.sin(2 * np.pi * np.random.uniform(10, 100) * t)
    signal = signal * np.exp(1j * (phase_noise + slow_phase))
    
    # 4. Sample rate offset (slight mismatch between TX and RX clocks)
    # This causes constellation rotation over time
    sro_ppm = np.random.uniform(-10, 10)  # parts per million
    sro_phase = 2 * np.pi * sro_ppm * 1e-6 * sample_rate * t
    signal = signal * np.exp(1j * sro_phase)
    
    # 5. Non-linear distortion (PA compression, ADC non-linearity)
    # Apply mild AM-AM and AM-PM distortion
    if np.random.random() < 0.3:  # Apply 30% of the time
        amplitude = np.abs(signal)
        phase = np.angle(signal)
        # Soft compression
        compressed_amp = amplitude / (1 + 0.1 * amplitude**2)
        # AM-PM conversion
        phase_distortion = 0.05 * amplitude**2
        signal = compressed_amp * np.exp(1j * (phase + phase_distortion))
    
    return signal

def generate_dataset(output_prefix, train_samples=5000, test_samples=1000,
                    num_samples=2048, sample_rate=1e6, samples_per_symbol=8):
    """Generate dataset with 4 modulation types and varied SNR levels"""

    print(f"🚀 Generating radio modulation dataset...")
    print(f"📊 Train samples per class: {train_samples:,}")
    print(f"🧪 Test samples per class: {test_samples:,}")
    print(f"📡 Samples per signal: {num_samples}")
    print(f"📈 Total training samples: {train_samples * 4:,}")
    print(f"📉 Total test samples: {test_samples * 4:,}")
    print(f"💾 Dataset size: ~{(train_samples * 4 + test_samples * 4) * num_samples * 2 * 4 / 1e9:.1f} GB")

    generator = ModulationGenerator(sample_rate, samples_per_symbol, num_samples)

    # 4 MODULATION TYPES compatible with VariableModulation.py
    modulations = [
        'BPSK',     # Class 0: Binary Phase Shift Keying (PSK)
        'QPSK',     # Class 1: Quadrature Phase Shift Keying
        '16QAM',    # Class 2: 16-Quadrature Amplitude Modulation
        '32QAM'     # Class 3: 32-Quadrature Amplitude Modulation
    ]

    mod_functions = [
        generator.generate_bpsk,      # Class 0: BPSK
        generator.generate_qpsk,      # Class 1: QPSK
        generator.generate_16qam,     # Class 2: 16QAM
        generator.generate_32qam      # Class 3: 32QAM
    ]

    # SNR levels for Easy, Difficult, Hard detection
    snr_levels = {
        'easy': {'min': 15, 'max': 20},      # Easy: 15-20 dB (high SNR)
        'difficult': {'min': 5, 'max': 10}, # Difficult: 5-10 dB (medium SNR)
        'hard': {'min': -5, 'max': 0}       # Hard: -5-0 dB (low SNR)
    }

    print(f"\n📋 MODULATION CLASSES (4 total):")
    print(f"   Class 0: {modulations[0]} (Binary Phase Shift Keying)")
    print(f"   Class 1: {modulations[1]} (Quadrature Phase Shift Keying)")
    print(f"   Class 2: {modulations[2]} (16-Quadrature Amplitude Modulation)")
    print(f"   Class 3: {modulations[3]} (32-Quadrature Amplitude Modulation)")
    print(f"   Random Baseline:   25% accuracy (1/4 classes)")
    print(f"\n📶 SNR LEVELS:")
    print(f"   Easy:      {snr_levels['easy']['min']} to {snr_levels['easy']['max']} dB")
    print(f"   Difficult: {snr_levels['difficult']['min']} to {snr_levels['difficult']['max']} dB")
    print(f"   Hard:      {snr_levels['hard']['min']} to {snr_levels['hard']['max']} dB")

    # Arrays to store data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for mod_idx, (mod_name, mod_func) in enumerate(zip(modulations, mod_functions)):
        print(f"\nGenerating {mod_name} signals...")

        # Generate training samples
        print(f"  Training samples: {train_samples}")
        for i in range(train_samples):
            if i % 1000 == 0:
                print(f"    Generated {i}/{train_samples} training samples")

            # Randomly select SNR difficulty level for training
            difficulty = random.choice(['easy', 'difficult', 'hard'])
            snr_min = snr_levels[difficulty]['min']
            snr_max = snr_levels[difficulty]['max']
            snr_db = random.uniform(snr_min, snr_max)

            # Moderate channel impairments
            freq_offset = random.uniform(-5000, 5000)  # ±5 kHz frequency offset
            phase_offset = random.uniform(0, 2*np.pi)  # Full phase rotation range

            # Moderate realistic impairments
            timing_offset = random.uniform(-0.2, 0.2)  # ±0.2 symbol timing error
            carrier_drift = random.uniform(-50, 50)  # ±50 Hz/s carrier drift
            amplitude_variation = random.uniform(0.8, 1.2)  # 20% amplitude fading

            signal = mod_func(snr_db, freq_offset, phase_offset)

            # Apply additional realistic impairments
            signal = apply_channel_impairments(signal, timing_offset, carrier_drift,
                                             amplitude_variation, sample_rate)
            
            # Apply Pluto SDR specific impairments (I/Q imbalance, DC offset, phase noise, etc.)
            signal = apply_pluto_realistic_impairments(signal, sample_rate)

            # Ensure signal has exact length and flatten I/Q data
            signal = signal[:num_samples]  # Ensure exact length
            if len(signal) < num_samples:
                # Pad with zeros if signal is too short
                signal = np.pad(signal, (0, num_samples - len(signal)), mode='constant')

            iq_flattened = np.column_stack([signal.real, signal.imag]).flatten()

            # Verify expected shape
            expected_features = num_samples * 2
            if len(iq_flattened) != expected_features:
                print(f"Warning: {mod_name} signal has {len(iq_flattened)} features, expected {expected_features}")
                iq_flattened = iq_flattened[:expected_features]  # Truncate if too long
                if len(iq_flattened) < expected_features:
                    iq_flattened = np.pad(iq_flattened, (0, expected_features - len(iq_flattened)), mode='constant')

            train_data.append(iq_flattened)
            train_labels.append(mod_idx)

        # Generate test samples
        print(f"  Test samples: {test_samples}")
        for i in range(test_samples):
            if i % 200 == 0:
                print(f"    Generated {i}/{test_samples} test samples")

            # Test conditions - same difficulty distribution as training
            difficulty = random.choice(['easy', 'difficult', 'hard'])
            snr_min = snr_levels[difficulty]['min']
            snr_max = snr_levels[difficulty]['max']
            snr_db = random.uniform(snr_min, snr_max)

            # Moderate channel impairments (same as training)
            freq_offset = random.uniform(-5000, 5000)  # ±5 kHz frequency offset
            phase_offset = random.uniform(0, 2*np.pi)  # Full phase rotation range

            # Moderate realistic impairments
            timing_offset = random.uniform(-0.2, 0.2)  # ±0.2 symbol timing error
            carrier_drift = random.uniform(-50, 50)  # ±50 Hz/s carrier drift
            amplitude_variation = random.uniform(0.8, 1.2)  # 20% amplitude fading

            signal = mod_func(snr_db, freq_offset, phase_offset)

            # Apply additional realistic impairments
            signal = apply_channel_impairments(signal, timing_offset, carrier_drift,
                                             amplitude_variation, sample_rate)
            
            # Apply Pluto SDR specific impairments (I/Q imbalance, DC offset, phase noise, etc.)
            signal = apply_pluto_realistic_impairments(signal, sample_rate)

            # Ensure signal has exact length and flatten I/Q data
            signal = signal[:num_samples]  # Ensure exact length
            if len(signal) < num_samples:
                # Pad with zeros if signal is too short
                signal = np.pad(signal, (0, num_samples - len(signal)), mode='constant')

            iq_flattened = np.column_stack([signal.real, signal.imag]).flatten()

            # Verify expected shape
            expected_features = num_samples * 2
            if len(iq_flattened) != expected_features:
                print(f"Warning: {mod_name} test signal has {len(iq_flattened)} features, expected {expected_features}")
                iq_flattened = iq_flattened[:expected_features]  # Truncate if too long
                if len(iq_flattened) < expected_features:
                    iq_flattened = np.pad(iq_flattened, (0, expected_features - len(iq_flattened)), mode='constant')

            test_data.append(iq_flattened)
            test_labels.append(mod_idx)

    # Convert to numpy arrays
    train_data = np.array(train_data)  # Shape: (samples, features)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    # Shuffle the data
    train_indices = np.random.permutation(len(train_data))
    test_indices = np.random.permutation(len(test_data))

    train_data = train_data[train_indices]
    train_labels = train_labels[train_indices]
    test_data = test_data[test_indices]
    test_labels = test_labels[test_indices]

    # Normalize data to [-1, 1] range PER-SAMPLE (not global)
    # This matches the real-time inference normalization in gnu_eval.cc
    # Each sample is normalized by its own max absolute value
    for i in range(len(train_data)):
        max_abs = np.max(np.abs(train_data[i]))
        if max_abs > 1e-10:
            train_data[i] = train_data[i] / max_abs
    
    for i in range(len(test_data)):
        max_abs = np.max(np.abs(test_data[i]))
        if max_abs > 1e-10:
            test_data[i] = test_data[i] / max_abs

    # Save to CSV files
    print(f"\nSaving dataset to CSV files...")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Save training data and labels
    train_data_file = f"{output_prefix}_train_data.csv"
    train_labels_file = f"{output_prefix}_train_labels.csv"
    np.savetxt(train_data_file, train_data, delimiter=',', fmt='%.6f')
    np.savetxt(train_labels_file, train_labels, delimiter=',', fmt='%d')

    # Save test data and labels
    test_data_file = f"{output_prefix}_test_data.csv"
    test_labels_file = f"{output_prefix}_test_labels.csv"
    np.savetxt(test_data_file, test_data, delimiter=',', fmt='%.6f')
    np.savetxt(test_labels_file, test_labels, delimiter=',', fmt='%d')

    # Save comprehensive metadata as info file
    info_file = f"{output_prefix}_info.txt"
    with open(info_file, 'w') as f:
        f.write("RADIO MODULATION DATASET - 4 CLASSES\n")
        f.write("=" * 50 + "\n\n")

        f.write("📋 MODULATION CLASSES (4 total):\n")
        for i, mod in enumerate(modulations):
            f.write(f"  Class {i}: {mod}\n")
        f.write("\n")

        f.write("🏷️ CLASS DESCRIPTIONS:\n")
        f.write(f"  Class 0 - BPSK:  Binary Phase Shift Keying (most robust)\n")
        f.write(f"  Class 1 - QPSK:  Quadrature Phase Shift Keying\n")
        f.write(f"  Class 2 - 16QAM: 16-Quadrature Amplitude Modulation\n")
        f.write(f"  Class 3 - 32QAM: 32-Quadrature Amplitude Modulation\n\n")

        f.write("📡 SIGNAL PARAMETERS:\n")
        f.write(f"  Sample rate: {sample_rate:g} Hz\n")
        f.write(f"  Samples per symbol: {samples_per_symbol}\n")
        f.write(f"  Symbol rate: {sample_rate/samples_per_symbol:g} Hz\n")
        f.write(f"  Samples per signal: {num_samples}\n")
        f.write(f"  Features per sample: {num_samples * 2} (I/Q interleaved)\n\n")

        f.write("📊 DATASET STATISTICS:\n")
        f.write(f"  Training samples per class: {train_samples:,}\n")
        f.write(f"  Test samples per class: {test_samples:,}\n")
        f.write(f"  Total training samples: {len(train_data):,}\n")
        f.write(f"  Total test samples: {len(test_data):,}\n")
        f.write(f"  Training data shape: {train_data.shape}\n")
        f.write(f"  Test data shape: {test_data.shape}\n")
        f.write(f"  Dataset size: ~{(len(train_data) + len(test_data)) * num_samples * 2 * 4 / 1e9:.1f} GB\n\n")

        f.write("🛠️ CHANNEL CONDITIONS (VARIED DIFFICULTY):\n")
        f.write(f"  Easy SNR: {snr_levels['easy']['min']} to {snr_levels['easy']['max']} dB (high signal quality)\n")
        f.write(f"  Difficult SNR: {snr_levels['difficult']['min']} to {snr_levels['difficult']['max']} dB (medium signal quality)\n")
        f.write(f"  Hard SNR: {snr_levels['hard']['min']} to {snr_levels['hard']['max']} dB (low signal quality)\n")
        f.write("  Frequency offset: ±5 kHz (moderate)\n")
        f.write("  Phase offset: 0 to 2π (random rotation)\n")
        f.write("  Channel impairments: Moderate timing, carrier drift, amplitude fading\n\n")

        f.write("💾 DATA FORMAT:\n")
        f.write("  Each row contains I/Q samples interleaved\n")
        f.write("  [I0, Q0, I1, Q1, I2, Q2, ..., I2047, Q2047]\n")
        f.write("  where I=real part, Q=imaginary part\n")
        f.write("  Normalized to [-1, +1] range PER-SAMPLE (each sample normalized by its own max)\n\n")

        f.write("📁 FILES GENERATED:\n")
        f.write(f"  📊 {train_data_file}\n")
        f.write(f"  🏷️  {train_labels_file}\n")
        f.write(f"  📊 {test_data_file}\n")
        f.write(f"  🏷️  {test_labels_file}\n")
        f.write(f"  📋 {info_file}\n\n")

        f.write("🧠 NEURAL NETWORK GUIDELINES:\n")
        f.write(f"  Random baseline accuracy: {100/len(modulations):.1f}%\n")
        f.write("  Recommended architecture: 1D CNN with multiple layers\n")
        f.write("  Expected performance: 60-90% (depends on SNR distribution)\n")
        f.write("  Key challenges: Variable SNR levels + modulation complexity\n")
        f.write("  Strategy: Handle different SNR levels, focus on robust features\n")
        f.write("  Compatible with: VariableModulation.py constellation definitions\n")

    print(f"\n🎉 DATASET GENERATION COMPLETE!")
    print(f"📁 Files created:")
    print(f"   📊 {train_data_file}")
    print(f"   🏷️  {train_labels_file}")
    print(f"   📊 {test_data_file}")
    print(f"   🏷️  {test_labels_file}")
    print(f"   📋 {info_file}")
    print(f"\n📈 Dataset Statistics:")
    print(f"   Training data shape: {train_data.shape}")
    print(f"   Test data shape: {test_data.shape}")
    print(f"   Features per sample: {train_data.shape[1]:,}")
    print(f"   Total size: ~{(len(train_data) + len(test_data)) * num_samples * 2 * 4 / 1e9:.1f} GB")
    print(f"   Classes: {len(modulations)} modulations")
    print(f"   Random baseline: {100/len(modulations):.2f}%")

    print(f"📋 Created C++ usage example: radio_dataset_16class_usage.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 4-class radio modulation dataset (BPSK, QPSK, 16QAM, 32QAM) in CSV format")
    parser.add_argument("--output", "-o", default="radio_dataset_comprehensive",
                       help="Output file prefix (will create _train_data.csv, _train_labels.csv, etc.)")
    parser.add_argument("--train-samples", "-t", type=int, default=20000,
                       help="Training samples per class (default: 20,000)")
    parser.add_argument("--test-samples", "-s", type=int, default=2000,
                       help="Test samples per class (default: 2,000)")
    parser.add_argument("--num-samples", "-n", type=int, default=2048,
                       help="Samples per signal (default: 2048, matches VariableModulation.py)")
    parser.add_argument("--sample-rate", "-r", type=float, default=1e6,
                       help="Sample rate in Hz (default: 1 MHz)")
    parser.add_argument("--samples-per-symbol", "-p", type=int, default=8,
                       help="Samples per symbol (default: 8)")

    args = parser.parse_args()

    print("🌟" + "="*70 + "🌟")
    print("🎯 RADIO MODULATION DATASET GENERATOR")
    print("🎯 4 Modulation Types: BPSK, QPSK, 16-QAM, 32-QAM")
    print("🌟" + "="*70 + "🌟")

    # Display configuration
    total_train = args.train_samples * 4
    total_test = args.test_samples * 4
    estimated_size = (total_train + total_test) * args.num_samples * 2 * 4 / 1e9

    print(f"\n📋 CONFIGURATION:")
    print(f"   📊 Classes: 4 modulations (BPSK, QPSK, 16-QAM, 32-QAM)")
    print(f"   🎓 Training samples: {args.train_samples:,} per class ({total_train:,} total)")
    print(f"   🧪 Test samples: {args.test_samples:,} per class ({total_test:,} total)")
    print(f"   📡 Signal length: {args.num_samples} complex samples")
    print(f"   📈 Features: {args.num_samples * 2:,} per sample (I/Q interleaved)")
    print(f"   💾 Estimated size: ~{estimated_size:.1f} GB")
    print(f"   🎯 Random baseline: 25.00% accuracy")

    if estimated_size > 5.0:
        print(f"\n⚠️  WARNING: Large dataset ({estimated_size:.1f} GB)")
        print(f"   Consider reducing --train-samples or --test-samples for testing")

    print(f"\n🚀 Starting generation...")

    # Generate dataset
    generate_dataset(
        args.output,
        args.train_samples,
        args.test_samples,
        args.num_samples,
        args.sample_rate,
        args.samples_per_symbol
    )

    print(f"\n🌟" + "="*70 + "🌟")
    print(f"🎉 DATASET GENERATION COMPLETE!")
    print(f"📂 Dataset files created with prefix: {args.output}")
    print(f"🧠 Ready for AMC (Automatic Modulation Classification)!")
    print(f"🌟" + "="*70 + "🌟")

    print(f"\n💡 QUICK START:")
    print(f"   Load training data: NEURAL_NETWORK::Helpers::ReadCSVMatrix(\"{args.output}_train_data.csv\", matrix)")
    print(f"   Load training labels: NEURAL_NETWORK::Helpers::ReadCSVLabels(\"{args.output}_train_labels.csv\", labels)")
    print(f"   Expected architecture: Deep 1D CNN with 3 conv layers + 2 dense layers")
    print(f"   Target accuracy: 70-90% (vs 25% random baseline)")

    print(f"\n📚 COMPATIBLE WITH:")
    print(f"   🔬 VariableModulation.py (GNU Radio ZeroMQ live streaming)")
    print(f"   🧬 radio_train.cc (C++ 1D CNN training)")
    print(f"   🔍 radio_eval.cc (C++ inference)")
    print(f"   📊 Input: {args.num_samples} complex samples = {args.num_samples * 2} features")


"""
Overview

  The signal generator in /src/Utils/generate_radio_dataset.py is a sophisticated radio frequency (RF) modulation dataset generator built on GNU Radio. It creates synthetic I/Q
  (In-phase/Quadrature) samples for machine learning training on Automatic Modulation Classification (AMC).

  Signal Variables Analysis

  1. Core Signal Parameters

  | Parameter          | Default Value | Range/Options | Purpose                          |
  |--------------------|---------------|---------------|----------------------------------|
  | sample_rate        | 1 MHz         | Fixed         | Controls temporal resolution     |
  | samples_per_symbol | 8             | Fixed         | Oversampling factor              |
  | num_samples        | 2048          | Fixed         | Signal length (complex samples)  |
  | symbol_rate        | 125 kHz       | Derived       | sample_rate / samples_per_symbol |

  2. Modulation Types Generated

  Current Active Modulations (4 classes):

  1. BPSK (Binary Phase Shift Keying) - Class 0
    - 2 constellation points: {-1, +1}
    - 1 bit per symbol
    - Most robust against noise
  2. QPSK (Quadrature Phase Shift Keying) - Class 1
    - 4 constellation points: {-1-1j, +1-1j, +1+1j, -1+1j}
    - 2 bits per symbol
    - Good noise performance
  3. 16-QAM (16-Quadrature Amplitude Modulation) - Class 2
    - 16 constellation points (4x4 grid)
    - 4 bits per symbol
    - Higher spectral efficiency
  4. 32-QAM (32-Quadrature Amplitude Modulation) - Class 3
    - 32 constellation points (cross pattern)
    - 5 bits per symbol
    - Highest spectral efficiency, most noise sensitive

  3. Channel Impairment Variables

  | Impairment          | Range           | Impact                             |
  |---------------------|-----------------|------------------------------------|
  | SNR (Easy)          | 15-20 dB        | High signal quality                |
  | SNR (Difficult)     | 5-10 dB         | Medium signal quality              |
  | SNR (Hard)          | -5-0 dB         | Low signal quality                 |
  | Frequency Offset    | ±5 kHz          | Carrier frequency errors           |
  | Phase Offset        | 0 to 2π         | Random phase rotation              |
  | Timing Offset       | ±0.2 symbols    | Symbol synchronization errors      |
  | Carrier Drift       | ±50 Hz/sec      | Oscillator instability             |
  | Amplitude Variation | 0.8 to 1.2      | 20% fading channel effects         |

  4. Advanced Realistic Impairments

  Multipath Fading:
  - 3-tap fading model with complex coefficients
  - Direct path weight: 1.0 (strongest)
  - Multipath taps: 0.1 * random_normal()

  Frequency-Selective Fading:
  - 1 kHz sinusoidal variation with 10% depth
  - Simulates frequency-dependent amplitude changes

  5. Data Processing Pipeline

  Signal Processing Chain:

  Random Bits → Modulator → Carrier Mixing → Noise Addition →
  Impairments → Frequency Shift → Phase Shift → I/Q Output

  Output Format:
  - Shape: [samples, 4096]
  - Encoding: I/Q interleaved [I₀, Q₀, I₁, Q₁, ..., I₂₀₄₇, Q₂₀₄₇]
  - Normalization: [-1, +1] range
  - Features: 4096 per signal (2048 complex samples × 2)

  Compatibility:
  - Matches VariableModulation.py buff_size=2048 (ZeroMQ streaming)
  - Matches radio_train.cc input dimensions (4096 features)
  - Same constellation definitions as GNU Radio flowgraph

  Classification Difficulty:

  1. BPSK vs QPSK: Moderate (2-point vs 4-point)
  2. QPSK vs 16-QAM: Moderate (phase-only vs amplitude+phase)
  3. 16-QAM vs 32-QAM: Hard (similar constellation density)
  4. All at low SNR: Challenging without deep learning

  Expected Performance:
  - Best case: 85-95% accuracy (high SNR)
  - Mixed SNR: 70-85% accuracy
  - Random baseline: 25% (4 classes)

"""