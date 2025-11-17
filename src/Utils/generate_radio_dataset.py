#!/usr/bin/env python3
"""
Radio Modulation Dataset Generator
Generates synthetic I/Q samples for 4 modulation types: BPSK, QPSK, 16QAM, ASK
Perfect for testing Convolution1D neural networks on radio signals.
"""

import numpy as np
import random
from gnuradio import gr, blocks, digital, analog, channels, filter
import os
import argparse

class ModulationGenerator:
    def __init__(self, sample_rate=1e6, samples_per_symbol=8, num_samples=1024):
        self.sample_rate = sample_rate
        self.samples_per_symbol = samples_per_symbol
        self.num_samples = num_samples
        self.symbol_rate = sample_rate / samples_per_symbol

    def generate_random_bits(self, num_symbols):
        """Generate random bit stream"""
        return np.random.randint(0, 2, num_symbols)

    def generate_bpsk(self, snr_db=20, freq_offset=0, phase_offset=0, carrier_freq=0):
        """Generate BPSK modulated signal"""
        tb = gr.top_block()

        # Number of symbols needed
        num_symbols = self.num_samples // self.samples_per_symbol + 100  # Extra for filtering

        # Generate random data
        data_bits = self.generate_random_bits(num_symbols)

        # Data source
        data_source = blocks.vector_source_b(data_bits, False)

        # BPSK modulator
        bpsk_mod = digital.psk_mod(
            constellation_points=2,
            mod_code="gray",
            differential=False,
            samples_per_symbol=self.samples_per_symbol,
            excess_bw=0.35,
            verbose=False,
            log=False
        )

        # Carrier frequency modulation
        carrier_mod = blocks.multiply_cc()
        carrier_source = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE,
                                            carrier_freq, 1, 0)

        # Channel impairments with multiple noise sources
        # 1. AWGN (Additive White Gaussian Noise)
        awgn_noise = analog.noise_source_c(analog.GR_GAUSSIAN,
                                         10**(-snr_db/20.0), 0)

        # 2. Additional thermal noise (10-20% of main noise)
        thermal_noise = analog.noise_source_c(analog.GR_GAUSSIAN,
                                            10**(-snr_db/20.0) * random.uniform(0.1, 0.2), 1)

        # 3. Phase noise (small random phase variations)
        phase_noise = analog.noise_source_c(analog.GR_GAUSSIAN,
                                          0.05 * random.uniform(0.5, 1.5), 2)

        # Add all noise sources
        noise_adder1 = blocks.add_cc()
        noise_adder2 = blocks.add_cc()
        signal_adder = blocks.add_cc()

        # Frequency offset (on top of carrier)
        freq_shift = blocks.multiply_cc()
        freq_source = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE,
                                        freq_offset, 1, 0)

        # Phase offset
        phase_shift = blocks.multiply_const_cc(np.exp(1j * phase_offset))

        # Output sink
        output_sink = blocks.vector_sink_c()

        # Connect blocks with realistic RF chain
        # 1. Modulate data
        tb.connect(data_source, bpsk_mod)

        # 2. Apply carrier frequency
        tb.connect(bpsk_mod, (carrier_mod, 0))
        tb.connect(carrier_source, (carrier_mod, 1))

        # 3. Combine noise sources
        tb.connect(awgn_noise, (noise_adder1, 0))
        tb.connect(thermal_noise, (noise_adder1, 1))
        tb.connect(noise_adder1, (noise_adder2, 0))
        tb.connect(phase_noise, (noise_adder2, 1))

        # 4. Add all noise to signal
        tb.connect(carrier_mod, (signal_adder, 0))
        tb.connect(noise_adder2, (signal_adder, 1))

        # 5. Apply frequency offset
        tb.connect(signal_adder, (freq_shift, 0))
        tb.connect(freq_source, (freq_shift, 1))

        # 6. Apply phase offset and output
        tb.connect(freq_shift, phase_shift)
        tb.connect(phase_shift, output_sink)

        # Run flowgraph
        tb.run()
        tb.wait()

        # Get samples and trim to desired length
        samples = np.array(output_sink.data())
        return samples[:self.num_samples]

    def generate_qpsk(self, snr_db=20, freq_offset=0, phase_offset=0, carrier_freq=0):
        """Generate QPSK modulated signal"""
        tb = gr.top_block()

        num_symbols = self.num_samples // self.samples_per_symbol + 100
        data_bits = self.generate_random_bits(num_symbols)

        data_source = blocks.vector_source_b(data_bits, False)

        # QPSK modulator
        qpsk_mod = digital.psk_mod(
            constellation_points=4,
            mod_code="gray",
            differential=False,
            samples_per_symbol=self.samples_per_symbol,
            excess_bw=0.35,
            verbose=False,
            log=False
        )

        # Channel impairments (same as BPSK)
        noise_source = analog.noise_source_c(analog.GR_GAUSSIAN,
                                           10**(-snr_db/20.0), 0)
        adder = blocks.add_cc()
        freq_shift = blocks.multiply_cc()
        freq_source = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE,
                                        freq_offset, 1, 0)
        phase_shift = blocks.multiply_const_cc(np.exp(1j * phase_offset))
        output_sink = blocks.vector_sink_c()

        tb.connect(data_source, qpsk_mod)
        tb.connect(qpsk_mod, (adder, 0))
        tb.connect(noise_source, (adder, 1))
        tb.connect(adder, freq_shift)
        tb.connect(freq_source, (freq_shift, 1))
        tb.connect(freq_shift, phase_shift)
        tb.connect(phase_shift, output_sink)

        tb.run()
        tb.wait()

        samples = np.array(output_sink.data())
        return samples[:self.num_samples]

    def generate_16qam(self, snr_db=20, freq_offset=0, phase_offset=0, carrier_freq=0):
        """Generate 16-QAM modulated signal"""
        tb = gr.top_block()

        num_symbols = self.num_samples // self.samples_per_symbol + 100
        data_bits = self.generate_random_bits(num_symbols * 4)  # 4 bits per symbol

        data_source = blocks.vector_source_b(data_bits, False)

        # 16-QAM modulator
        qam_constellation = digital.constellation_16qam()
        qam_mod = digital.generic_mod(
            constellation=qam_constellation,
            differential=False,
            samples_per_symbol=self.samples_per_symbol,
            pre_diff_code=True,
            excess_bw=0.35,
            verbose=False,
            log=False
        )

        # Channel impairments
        noise_source = analog.noise_source_c(analog.GR_GAUSSIAN,
                                           10**(-snr_db/20.0), 0)
        adder = blocks.add_cc()
        freq_shift = blocks.multiply_cc()
        freq_source = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE,
                                        freq_offset, 1, 0)
        phase_shift = blocks.multiply_const_cc(np.exp(1j * phase_offset))
        output_sink = blocks.vector_sink_c()

        tb.connect(data_source, qam_mod)
        tb.connect(qam_mod, (adder, 0))
        tb.connect(noise_source, (adder, 1))
        tb.connect(adder, freq_shift)
        tb.connect(freq_source, (freq_shift, 1))
        tb.connect(freq_shift, phase_shift)
        tb.connect(phase_shift, output_sink)

        tb.run()
        tb.wait()

        samples = np.array(output_sink.data())
        return samples[:self.num_samples]

    def generate_8psk(self, snr_db=20, freq_offset=0, phase_offset=0, carrier_freq=0):
        """Generate 8-PSK modulated signal"""
        tb = gr.top_block()

        num_symbols = self.num_samples // self.samples_per_symbol + 100
        data_bits = self.generate_random_bits(num_symbols * 3)  # 3 bits per symbol

        data_source = blocks.vector_source_b(data_bits, False)

        # 8-PSK modulator
        psk8_mod = digital.psk_mod(
            constellation_points=8,
            mod_code="gray",
            differential=False,
            samples_per_symbol=self.samples_per_symbol,
            excess_bw=0.35,
            verbose=False,
            log=False
        )

        # Carrier frequency modulation
        carrier_mod = blocks.multiply_cc()
        carrier_source = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE,
                                            carrier_freq, 1, 0)

        # Multiple noise sources for realism
        awgn_noise = analog.noise_source_c(analog.GR_GAUSSIAN,
                                         10**(-snr_db/20.0), 0)
        thermal_noise = analog.noise_source_c(analog.GR_GAUSSIAN,
                                            10**(-snr_db/20.0) * random.uniform(0.1, 0.2), 1)
        phase_noise = analog.noise_source_c(analog.GR_GAUSSIAN,
                                          0.05 * random.uniform(0.5, 1.5), 2)

        # Combine noise sources
        noise_adder1 = blocks.add_cc()
        noise_adder2 = blocks.add_cc()
        signal_adder = blocks.add_cc()

        # Frequency offset
        freq_shift = blocks.multiply_cc()
        freq_source = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE,
                                        freq_offset, 1, 0)

        # Phase offset
        phase_shift = blocks.multiply_const_cc(np.exp(1j * phase_offset))
        output_sink = blocks.vector_sink_c()

        # Connect blocks
        tb.connect(data_source, psk8_mod)
        tb.connect(psk8_mod, (carrier_mod, 0))
        tb.connect(carrier_source, (carrier_mod, 1))
        tb.connect(awgn_noise, (noise_adder1, 0))
        tb.connect(thermal_noise, (noise_adder1, 1))
        tb.connect(noise_adder1, (noise_adder2, 0))
        tb.connect(phase_noise, (noise_adder2, 1))
        tb.connect(carrier_mod, (signal_adder, 0))
        tb.connect(noise_adder2, (signal_adder, 1))
        tb.connect(signal_adder, (freq_shift, 0))
        tb.connect(freq_source, (freq_shift, 1))
        tb.connect(freq_shift, phase_shift)
        tb.connect(phase_shift, output_sink)

        tb.run()
        tb.wait()

        samples = np.array(output_sink.data())
        return samples[:self.num_samples]

    def generate_16psk(self, snr_db=20, freq_offset=0, phase_offset=0, carrier_freq=0):
        """Generate 16-PSK modulated signal"""
        tb = gr.top_block()

        num_symbols = self.num_samples // self.samples_per_symbol + 100
        data_bits = self.generate_random_bits(num_symbols * 4)  # 4 bits per symbol

        data_source = blocks.vector_source_b(data_bits, False)

        # 16-PSK constellation (very close points!)
        constellation_points = []
        for i in range(16):
            angle = 2 * np.pi * i / 16
            constellation_points.append(complex(np.cos(angle), np.sin(angle)))

        constellation = digital.constellation_calcdist(constellation_points, [], 4, 1).base()
        psk16_mod = digital.generic_mod(
            constellation=constellation,
            differential=False,
            samples_per_symbol=self.samples_per_symbol,
            pre_diff_code=True,
            excess_bw=0.35,
            verbose=False,
            log=False
        )

        # Carrier frequency modulation
        carrier_mod = blocks.multiply_cc()
        carrier_source = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE,
                                            carrier_freq, 1, 0)

        # Multiple noise sources
        awgn_noise = analog.noise_source_c(analog.GR_GAUSSIAN,
                                         10**(-snr_db/20.0), 0)
        thermal_noise = analog.noise_source_c(analog.GR_GAUSSIAN,
                                            10**(-snr_db/20.0) * random.uniform(0.1, 0.2), 1)
        phase_noise = analog.noise_source_c(analog.GR_GAUSSIAN,
                                          0.05 * random.uniform(0.5, 1.5), 2)

        # Combine noise sources
        noise_adder1 = blocks.add_cc()
        noise_adder2 = blocks.add_cc()
        signal_adder = blocks.add_cc()

        # Frequency offset
        freq_shift = blocks.multiply_cc()
        freq_source = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE,
                                        freq_offset, 1, 0)

        # Phase offset
        phase_shift = blocks.multiply_const_cc(np.exp(1j * phase_offset))
        output_sink = blocks.vector_sink_c()

        # Connect blocks
        tb.connect(data_source, psk16_mod)
        tb.connect(psk16_mod, (carrier_mod, 0))
        tb.connect(carrier_source, (carrier_mod, 1))
        tb.connect(awgn_noise, (noise_adder1, 0))
        tb.connect(thermal_noise, (noise_adder1, 1))
        tb.connect(noise_adder1, (noise_adder2, 0))
        tb.connect(phase_noise, (noise_adder2, 1))
        tb.connect(carrier_mod, (signal_adder, 0))
        tb.connect(noise_adder2, (signal_adder, 1))
        tb.connect(signal_adder, (freq_shift, 0))
        tb.connect(freq_source, (freq_shift, 1))
        tb.connect(freq_shift, phase_shift)
        tb.connect(phase_shift, output_sink)

        tb.run()
        tb.wait()

        samples = np.array(output_sink.data())
        return samples[:self.num_samples]

    def generate_ask(self, snr_db=20, freq_offset=0, phase_offset=0, carrier_freq=0):
        """Generate ASK (OOK) modulated signal"""
        tb = gr.top_block()

        num_symbols = self.num_samples // self.samples_per_symbol + 100
        data_bits = self.generate_random_bits(num_symbols)

        data_source = blocks.vector_source_b(data_bits, False)

        # Convert bits to symbols (0 -> 0, 1 -> 1)
        unpack = blocks.unpack_k_bits_bb(1)
        b2f = blocks.char_to_float()

        # Pulse shaping filter
        taps = filter.firdes.root_raised_cosine(1.0, self.sample_rate,
                                              self.symbol_rate, 0.35, 101)
        pulse_shape = filter.interp_fir_filter_fff(self.samples_per_symbol, taps)

        # Convert to complex and add carrier
        f2c = blocks.float_to_complex()
        carrier = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE,
                                    self.sample_rate/8, 1, 0)  # Carrier frequency
        mixer = blocks.multiply_cc()

        # Channel impairments
        noise_source = analog.noise_source_c(analog.GR_GAUSSIAN,
                                           10**(-snr_db/20.0), 0)
        adder = blocks.add_cc()
        freq_shift = blocks.multiply_cc()
        freq_source = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE,
                                        freq_offset, 1, 0)
        phase_shift = blocks.multiply_const_cc(np.exp(1j * phase_offset))
        output_sink = blocks.vector_sink_c()

        tb.connect(data_source, unpack)
        tb.connect(unpack, b2f)
        tb.connect(b2f, pulse_shape)
        tb.connect(pulse_shape, f2c)
        tb.connect(f2c, (mixer, 0))
        tb.connect(carrier, (mixer, 1))
        tb.connect(mixer, (adder, 0))
        tb.connect(noise_source, (adder, 1))
        tb.connect(adder, freq_shift)
        tb.connect(freq_source, (freq_shift, 1))
        tb.connect(freq_shift, phase_shift)
        tb.connect(phase_shift, output_sink)

        tb.run()
        tb.wait()

        samples = np.array(output_sink.data())
        return samples[:self.num_samples]

def apply_channel_impairments(signal, timing_offset, carrier_drift, amplitude_variation, sample_rate):
    """Apply realistic channel impairments to make dataset extremely challenging"""

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

def generate_dataset(output_prefix, train_samples=10000, test_samples=1000,
                    num_samples=1024, sample_rate=1e6, samples_per_symbol=8):
    """Generate complete dataset with all modulation types"""

    print(f"Generating radio modulation dataset...")
    print(f"Train samples per class: {train_samples}")
    print(f"Test samples per class: {test_samples}")
    print(f"Samples per signal: {num_samples}")
    print(f"Total training samples: {train_samples * 4}")
    print(f"Total test samples: {test_samples * 4}")

    generator = ModulationGenerator(sample_rate, samples_per_symbol, num_samples)

    # Similar PSK modulation types (much harder to distinguish!)
    modulations = ['BPSK', 'QPSK', '8PSK', '16PSK']
    mod_functions = [
        generator.generate_bpsk,
        generator.generate_qpsk,
        generator.generate_8psk,
        generator.generate_16psk
    ]

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

            # Extremely challenging and realistic channel conditions
            snr_db = random.uniform(-15, 5)  # Much harder SNR range!
            freq_offset = random.uniform(-15000, 15000)  # Large frequency offset
            phase_offset = random.uniform(0, 2*np.pi)  # Phase offset
            carrier_freq = random.uniform(1e6, 40e6)  # Carrier frequency 1-40 MHz

            # Additional realistic impairments
            timing_offset = random.uniform(-0.5, 0.5)  # Symbol timing offset
            carrier_drift = random.uniform(-100, 100)  # Hz/sec carrier drift
            amplitude_variation = random.uniform(0.7, 1.3)  # Amplitude fading

            signal = mod_func(snr_db, freq_offset, phase_offset, carrier_freq)

            # Apply additional realistic impairments
            signal = apply_channel_impairments(signal, timing_offset, carrier_drift,
                                             amplitude_variation, sample_rate)

            # Flatten I/Q data: each row is [I0, Q0, I1, Q1, ..., I1023, Q1023]
            iq_flattened = np.column_stack([signal.real, signal.imag]).flatten()

            train_data.append(iq_flattened)
            train_labels.append(mod_idx)

        # Generate test samples
        print(f"  Test samples: {test_samples}")
        for i in range(test_samples):
            if i % 200 == 0:
                print(f"    Generated {i}/{test_samples} test samples")

            # Extremely challenging test conditions
            snr_db = random.uniform(-10, 0)  # Very low SNR for test
            freq_offset = random.uniform(-12000, 12000)  # Large frequency offset
            phase_offset = random.uniform(0, 2*np.pi)  # Phase offset
            carrier_freq = random.uniform(1e6, 40e6)  # Carrier frequency 1-40 MHz

            # Additional realistic impairments for test
            timing_offset = random.uniform(-0.4, 0.4)  # Symbol timing offset
            carrier_drift = random.uniform(-80, 80)  # Hz/sec carrier drift
            amplitude_variation = random.uniform(0.8, 1.2)  # Amplitude fading

            signal = mod_func(snr_db, freq_offset, phase_offset, carrier_freq)

            # Apply additional realistic impairments
            signal = apply_channel_impairments(signal, timing_offset, carrier_drift,
                                             amplitude_variation, sample_rate)
            iq_flattened = np.column_stack([signal.real, signal.imag]).flatten()

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

    # Normalize data to [-1, 1] range
    train_data = train_data / np.max(np.abs(train_data))
    test_data = test_data / np.max(np.abs(test_data))

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

    # Save metadata as info file
    info_file = f"{output_prefix}_info.txt"
    with open(info_file, 'w') as f:
        f.write("Radio Modulation Dataset Information\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Modulation classes: {', '.join(modulations)}\n")
        f.write(f"Class mapping: 0=BPSK, 1=QPSK, 2=8PSK, 3=16PSK\n\n")
        f.write(f"Sample rate: {sample_rate} Hz\n")
        f.write(f"Samples per symbol: {samples_per_symbol}\n")
        f.write(f"Samples per signal: {num_samples}\n")
        f.write(f"Features per sample: {num_samples * 2} (I/Q interleaved)\n\n")
        f.write(f"Training samples per class: {train_samples}\n")
        f.write(f"Test samples per class: {test_samples}\n")
        f.write(f"Total training samples: {len(train_data)}\n")
        f.write(f"Total test samples: {len(test_data)}\n\n")
        f.write(f"Training data shape: {train_data.shape}\n")
        f.write(f"Test data shape: {test_data.shape}\n\n")
        f.write("Data format: Each row contains I/Q samples interleaved\n")
        f.write("  [I0, Q0, I1, Q1, I2, Q2, ..., I1023, Q1023]\n")
        f.write("  where I=real part, Q=imaginary part\n\n")
        f.write("Files generated:\n")
        f.write(f"  {train_data_file} - Training data (features)\n")
        f.write(f"  {train_labels_file} - Training labels\n")
        f.write(f"  {test_data_file} - Test data (features)\n")
        f.write(f"  {test_labels_file} - Test labels\n")

    print(f"Dataset saved successfully!")
    print(f"Files created:")
    print(f"  📊 {train_data_file}")
    print(f"  🏷️  {train_labels_file}")
    print(f"  📊 {test_data_file}")
    print(f"  🏷️  {test_labels_file}")
    print(f"  📋 {info_file}")
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Features per sample: {train_data.shape[1]}")
    print(f"Modulation classes: {modulations}")

def create_cpp_usage_example():
    """Create example C++ usage text"""
    usage_example = '''C++ Usage Example (after generating dataset):

#include "Utils/Helpers.h"
using namespace NEURAL_NETWORK;

// Load radio modulation dataset
Eigen::MatrixXd train_data, test_data;
Eigen::VectorXi train_labels, test_labels;

Helpers::ReadCSVMatrix("radio_dataset_train_data.csv", train_data);
Helpers::ReadCSVLabels("radio_dataset_train_labels.csv", train_labels);
Helpers::ReadCSVMatrix("radio_dataset_test_data.csv", test_data);
Helpers::ReadCSVLabels("radio_dataset_test_labels.csv", test_labels);

// Class mapping: 0=BPSK, 1=QPSK, 2=16QAM, 3=ASK
// Feature format: I/Q interleaved [I0,Q0,I1,Q1,...,I1023,Q1023]
// Features per sample: 2048 (1024 time steps × 2 channels)

// Create 1D CNN model for radio signal classification
auto model = std::make_shared<Model>();

// Input layer
model->AddLayer(std::make_shared<LayerInput>(2048));

// 1D Convolution layers for temporal feature extraction
model->AddLayer(std::make_shared<Convolution1D>(32, 7, 1024, 2, 1, 1));
model->AddLayer(std::make_shared<ActivationReLU>());
model->AddLayer(std::make_shared<MaxPooling1D>(32, 2, 1018, 32, 2));

model->AddLayer(std::make_shared<Convolution1D>(64, 5, 509, 32, 1, 1));
model->AddLayer(std::make_shared<ActivationReLU>());
model->AddLayer(std::make_shared<MaxPooling1D>(64, 2, 505, 64, 2));

// Dense layers for classification
model->AddLayer(std::make_shared<LayerDense>(252*64, 128));
model->AddLayer(std::make_shared<ActivationReLU>());
model->AddLayer(std::make_shared<LayerDense>(128, 4));
model->AddLayer(std::make_shared<ActivationSoftmax>());

// Train the model with your radio dataset
'''

    with open('radio_dataset_usage.txt', 'w') as f:
        f.write(usage_example)

    print("Created C++ usage example: radio_dataset_usage.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate radio modulation dataset in CSV format")
    parser.add_argument("--output", "-o", default="radio_dataset",
                       help="Output file prefix (will create _train_data.csv, _train_labels.csv, etc.)")
    parser.add_argument("--train-samples", "-t", type=int, default=10000,
                       help="Training samples per class")
    parser.add_argument("--test-samples", "-s", type=int, default=1000,
                       help="Test samples per class")
    parser.add_argument("--num-samples", "-n", type=int, default=1024,
                       help="Samples per signal")
    parser.add_argument("--sample-rate", "-r", type=float, default=1e6,
                       help="Sample rate (Hz)")
    parser.add_argument("--samples-per-symbol", "-p", type=int, default=8,
                       help="Samples per symbol")

    args = parser.parse_args()

    # Generate dataset
    generate_dataset(
        args.output,
        args.train_samples,
        args.test_samples,
        args.num_samples,
        args.sample_rate,
        args.samples_per_symbol
    )

    # Create C++ usage example
    create_cpp_usage_example()

    print(f"\n🎉 Dataset generation complete!")
    print(f"📂 Dataset files created with prefix: {args.output}")
    print(f"📄 C++ usage example: radio_dataset_usage.txt")
    print(f"🚀 Ready for neural network training!")
    print(f"\n💡 Load in C++: NEURAL_NETWORK::Helpers::ReadCSVMatrix(\"{args.output}_train_data.csv\", matrix)")