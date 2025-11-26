#!/usr/bin/env python3
"""
EXTREME Radio Modulation Dataset Generator - 16 Modulation Types
Generates synthetic I/Q samples for 16 modulation types covering PSK, QAM, FSK, and analog modulations.
EXTREME DIFFICULTY: Signals buried in noise for ultimate deep learning research challenge.
Training SNR: -15 to +5 dB, Test SNR: -10 to 0 dB (brutal conditions).
Perfect for testing state-of-the-art 1D CNN architectures on radio signals.

Modulation Types: BPSK, QPSK, 8PSK, 16PSK, 64PSK, 16QAM, 32QAM, 64QAM,
ASK, 2FSK, 4FSK, 8FSK, GMSK, MSK, AM, FM
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

    def generate_64psk(self, snr_db=20, freq_offset=0, phase_offset=0, carrier_freq=0):
        """Generate 64-PSK modulated signal"""
        tb = gr.top_block()

        num_symbols = self.num_samples // self.samples_per_symbol + 100
        data_bits = self.generate_random_bits(num_symbols * 6)  # 6 bits per symbol

        data_source = blocks.vector_source_b(data_bits, False)

        # 64-PSK constellation (very close points!)
        constellation_points = []
        for i in range(64):
            angle = 2 * np.pi * i / 64
            constellation_points.append(complex(np.cos(angle), np.sin(angle)))

        constellation = digital.constellation_calcdist(constellation_points, [], 6, 1).base()
        psk64_mod = digital.generic_mod(
            constellation=constellation,
            differential=False,
            samples_per_symbol=self.samples_per_symbol,
            pre_diff_code=True,
            excess_bw=0.35,
            verbose=False,
            log=False
        )

        # Channel impairments (same pattern as other PSK)
        noise_source = analog.noise_source_c(analog.GR_GAUSSIAN,
                                           10**(-snr_db/20.0), 0)
        adder = blocks.add_cc()
        freq_shift = blocks.multiply_cc()
        freq_source = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE,
                                        freq_offset, 1, 0)
        phase_shift = blocks.multiply_const_cc(np.exp(1j * phase_offset))
        output_sink = blocks.vector_sink_c()

        tb.connect(data_source, psk64_mod)
        tb.connect(psk64_mod, (adder, 0))
        tb.connect(noise_source, (adder, 1))
        tb.connect(adder, freq_shift)
        tb.connect(freq_source, (freq_shift, 1))
        tb.connect(freq_shift, phase_shift)
        tb.connect(phase_shift, output_sink)

        tb.run()
        tb.wait()

        samples = np.array(output_sink.data())
        return samples[:self.num_samples]

    def generate_32qam(self, snr_db=20, freq_offset=0, phase_offset=0, carrier_freq=0):
        """Generate 32-QAM modulated signal"""
        tb = gr.top_block()

        num_symbols = self.num_samples // self.samples_per_symbol + 100
        data_bits = self.generate_random_bits(num_symbols * 5)  # 5 bits per symbol

        data_source = blocks.vector_source_b(data_bits, False)

        # Create standard 32-QAM constellation (cross-shaped)
        constellation_points = []
        # Inner 4x4 square (16 points)
        levels = [-3, -1, 1, 3]
        for i in levels:
            for q in levels:
                constellation_points.append(complex(i, q))

        # Outer points to reach 32 total (16 additional points)
        outer_points = [
            complex(5, 1), complex(5, -1), complex(-5, 1), complex(-5, -1),
            complex(1, 5), complex(-1, 5), complex(1, -5), complex(-1, -5),
            complex(3, 5), complex(-3, 5), complex(3, -5), complex(-3, -5),
            complex(5, 3), complex(-5, 3), complex(5, -3), complex(-5, -3)
        ]
        constellation_points.extend(outer_points)

        # Normalize constellation to unit average power
        avg_power = sum(abs(c)**2 for c in constellation_points) / len(constellation_points)
        constellation_points = [c / np.sqrt(avg_power) for c in constellation_points]

        constellation = digital.constellation_calcdist(constellation_points, [], 5, 1).base()
        qam32_mod = digital.generic_mod(
            constellation=constellation,
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

        tb.connect(data_source, qam32_mod)
        tb.connect(qam32_mod, (adder, 0))
        tb.connect(noise_source, (adder, 1))
        tb.connect(adder, freq_shift)
        tb.connect(freq_source, (freq_shift, 1))
        tb.connect(freq_shift, phase_shift)
        tb.connect(phase_shift, output_sink)

        tb.run()
        tb.wait()

        samples = np.array(output_sink.data())
        return samples[:self.num_samples]

    def generate_64qam(self, snr_db=20, freq_offset=0, phase_offset=0, carrier_freq=0):
        """Generate 64-QAM modulated signal"""
        tb = gr.top_block()

        num_symbols = self.num_samples // self.samples_per_symbol + 100
        data_bits = self.generate_random_bits(num_symbols * 6)  # 6 bits per symbol

        data_source = blocks.vector_source_b(data_bits, False)

        # Create custom 64-QAM constellation (8x8 grid)
        constellation_points = []
        levels = [-7, -5, -3, -1, 1, 3, 5, 7]  # 8 amplitude levels
        for i in levels:
            for q in levels:
                constellation_points.append(complex(i, q))

        # Normalize constellation to unit average power
        avg_power = sum(abs(c)**2 for c in constellation_points) / len(constellation_points)
        constellation_points = [c / np.sqrt(avg_power) for c in constellation_points]

        constellation = digital.constellation_calcdist(constellation_points, [], 6, 1).base()
        qam64_mod = digital.generic_mod(
            constellation=constellation,
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

        tb.connect(data_source, qam64_mod)
        tb.connect(qam64_mod, (adder, 0))
        tb.connect(noise_source, (adder, 1))
        tb.connect(adder, freq_shift)
        tb.connect(freq_source, (freq_shift, 1))
        tb.connect(freq_shift, phase_shift)
        tb.connect(phase_shift, output_sink)

        tb.run()
        tb.wait()

        samples = np.array(output_sink.data())
        return samples[:self.num_samples]

    def generate_2fsk(self, snr_db=20, freq_offset=0, phase_offset=0, carrier_freq=0):
        """Generate 2-FSK (Binary Frequency Shift Keying) modulated signal"""
        tb = gr.top_block()

        num_symbols = self.num_samples // self.samples_per_symbol + 100
        data_bits = self.generate_random_bits(num_symbols)

        data_source = blocks.vector_source_b(data_bits, False)

        # Convert bits to frequency deviations
        unpack = blocks.unpack_k_bits_bb(1)
        b2f = blocks.char_to_float()
        add_const = blocks.add_const_ff(-0.5)  # Center around 0: 0->-0.5, 1->+0.5
        multiply_const = blocks.multiply_const_ff(20000)  # ±10 kHz deviation

        # Use frequency modulator for FSK
        freq_mod = analog.frequency_modulator_fc(2*np.pi / self.sample_rate)

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
        tb.connect(b2f, add_const)
        tb.connect(add_const, multiply_const)
        tb.connect(multiply_const, freq_mod)
        tb.connect(freq_mod, (adder, 0))
        tb.connect(noise_source, (adder, 1))
        tb.connect(adder, freq_shift)
        tb.connect(freq_source, (freq_shift, 1))
        tb.connect(freq_shift, phase_shift)
        tb.connect(phase_shift, output_sink)

        tb.run()
        tb.wait()

        samples = np.array(output_sink.data())
        return samples[:self.num_samples]

    def generate_4fsk(self, snr_db=20, freq_offset=0, phase_offset=0, carrier_freq=0):
        """Generate 4-FSK (4-ary Frequency Shift Keying) modulated signal"""
        tb = gr.top_block()

        num_symbols = self.num_samples // self.samples_per_symbol + 100
        data_bits = self.generate_random_bits(num_symbols * 2)  # 2 bits per symbol

        data_source = blocks.vector_source_b(data_bits, False)

        # Pack bits and convert to symbols (0,1,2,3)
        pack_bits = blocks.pack_k_bits_bb(2)
        b2f = blocks.char_to_float()
        add_const = blocks.add_const_ff(-1.5)  # Center around 0: -1.5, -0.5, 0.5, 1.5
        multiply_const = blocks.multiply_const_ff(15000)  # ±22.5 kHz total deviation

        # Use frequency modulator for FSK
        freq_mod = analog.frequency_modulator_fc(2*np.pi / self.sample_rate)

        # Channel impairments
        noise_source = analog.noise_source_c(analog.GR_GAUSSIAN,
                                           10**(-snr_db/20.0), 0)
        adder = blocks.add_cc()
        freq_shift = blocks.multiply_cc()
        freq_source = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE,
                                        freq_offset, 1, 0)
        phase_shift = blocks.multiply_const_cc(np.exp(1j * phase_offset))
        output_sink = blocks.vector_sink_c()

        tb.connect(data_source, pack_bits)
        tb.connect(pack_bits, b2f)
        tb.connect(b2f, add_const)
        tb.connect(add_const, multiply_const)
        tb.connect(multiply_const, freq_mod)
        tb.connect(freq_mod, (adder, 0))
        tb.connect(noise_source, (adder, 1))
        tb.connect(adder, freq_shift)
        tb.connect(freq_source, (freq_shift, 1))
        tb.connect(freq_shift, phase_shift)
        tb.connect(phase_shift, output_sink)

        tb.run()
        tb.wait()

        samples = np.array(output_sink.data())
        return samples[:self.num_samples]

    def generate_8fsk(self, snr_db=20, freq_offset=0, phase_offset=0, carrier_freq=0):
        """Generate 8-FSK (8-ary Frequency Shift Keying) modulated signal"""
        tb = gr.top_block()

        num_symbols = self.num_samples // self.samples_per_symbol + 100
        data_bits = self.generate_random_bits(num_symbols * 3)  # 3 bits per symbol

        data_source = blocks.vector_source_b(data_bits, False)

        # Pack bits and convert to symbols (0,1,2,3,4,5,6,7)
        pack_bits = blocks.pack_k_bits_bb(3)
        b2f = blocks.char_to_float()
        add_const = blocks.add_const_ff(-3.5)  # Center around 0: -3.5 to 3.5
        multiply_const = blocks.multiply_const_ff(12000)  # ±42 kHz total deviation

        # Use frequency modulator for FSK
        freq_mod = analog.frequency_modulator_fc(2*np.pi / self.sample_rate)

        # Channel impairments
        noise_source = analog.noise_source_c(analog.GR_GAUSSIAN,
                                           10**(-snr_db/20.0), 0)
        adder = blocks.add_cc()
        freq_shift = blocks.multiply_cc()
        freq_source = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE,
                                        freq_offset, 1, 0)
        phase_shift = blocks.multiply_const_cc(np.exp(1j * phase_offset))
        output_sink = blocks.vector_sink_c()

        tb.connect(data_source, pack_bits)
        tb.connect(pack_bits, b2f)
        tb.connect(b2f, add_const)
        tb.connect(add_const, multiply_const)
        tb.connect(multiply_const, freq_mod)
        tb.connect(freq_mod, (adder, 0))
        tb.connect(noise_source, (adder, 1))
        tb.connect(adder, freq_shift)
        tb.connect(freq_source, (freq_shift, 1))
        tb.connect(freq_shift, phase_shift)
        tb.connect(phase_shift, output_sink)

        tb.run()
        tb.wait()

        samples = np.array(output_sink.data())
        return samples[:self.num_samples]

    def generate_gmsk(self, snr_db=20, freq_offset=0, phase_offset=0, carrier_freq=0):
        """Generate GMSK (Gaussian Minimum Shift Keying) modulated signal"""
        tb = gr.top_block()

        num_symbols = self.num_samples // self.samples_per_symbol + 100
        data_bits = self.generate_random_bits(num_symbols)

        data_source = blocks.vector_source_b(data_bits, False)

        # GMSK modulator with BT=0.3 (common for GSM)
        gmsk_mod = digital.gmsk_mod(
            samples_per_symbol=self.samples_per_symbol,
            bt=0.3,
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

        tb.connect(data_source, gmsk_mod)
        tb.connect(gmsk_mod, (adder, 0))
        tb.connect(noise_source, (adder, 1))
        tb.connect(adder, freq_shift)
        tb.connect(freq_source, (freq_shift, 1))
        tb.connect(freq_shift, phase_shift)
        tb.connect(phase_shift, output_sink)

        tb.run()
        tb.wait()

        samples = np.array(output_sink.data())
        return samples[:self.num_samples]

    def generate_msk(self, snr_db=20, freq_offset=0, phase_offset=0, carrier_freq=0):
        """Generate MSK (Minimum Shift Keying) modulated signal"""
        tb = gr.top_block()

        num_symbols = self.num_samples // self.samples_per_symbol + 100
        data_bits = self.generate_random_bits(num_symbols)

        data_source = blocks.vector_source_b(data_bits, False)

        # Convert bits to symbols
        unpack = blocks.unpack_k_bits_bb(1)
        b2f = blocks.char_to_float()
        add_const = blocks.add_const_ff(-0.5)  # Map 0,1 to -0.5,0.5

        # MSK is essentially FSK with deviation = symbol_rate/4
        msk_deviation = self.symbol_rate / 4
        multiply_const = blocks.multiply_const_ff(msk_deviation * 2 * np.pi)

        # Use frequency modulator for MSK (simplified approach)
        freq_mod = analog.frequency_modulator_fc(1.0 / self.sample_rate)

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
        tb.connect(b2f, add_const)
        tb.connect(add_const, multiply_const)
        tb.connect(multiply_const, freq_mod)
        tb.connect(freq_mod, (adder, 0))
        tb.connect(noise_source, (adder, 1))
        tb.connect(adder, freq_shift)
        tb.connect(freq_source, (freq_shift, 1))
        tb.connect(freq_shift, phase_shift)
        tb.connect(phase_shift, output_sink)

        tb.run()
        tb.wait()

        samples = np.array(output_sink.data())
        return samples[:self.num_samples]

    def generate_am(self, snr_db=20, freq_offset=0, phase_offset=0, carrier_freq=0):
        """Generate AM (Amplitude Modulation) modulated signal"""
        tb = gr.top_block()

        # Generate random audio-like signal (much slower than symbol rate)
        audio_length = self.num_samples + 1000
        t = np.linspace(0, audio_length/self.sample_rate, audio_length)

        # Create audio-like modulating signal (mix of tones)
        audio_signal = 0.3 * np.sin(2*np.pi*1000*t) + 0.2 * np.sin(2*np.pi*2000*t) + 0.1 * np.sin(2*np.pi*3000*t)
        audio_signal = audio_signal.astype(np.float32)

        audio_source = blocks.vector_source_f(audio_signal, False)

        # AM modulator: (1 + m*audio) * carrier where m = modulation index
        modulation_index = 0.8
        add_const = blocks.add_const_ff(1.0)  # DC offset
        multiply_const = blocks.multiply_const_ff(modulation_index)

        # Carrier
        carrier = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE,
                                    carrier_freq or self.sample_rate/8, 1, 0)

        # Float to complex conversion
        f2c = blocks.float_to_complex()
        multiply_cc = blocks.multiply_cc()

        # Channel impairments
        noise_source = analog.noise_source_c(analog.GR_GAUSSIAN,
                                           10**(-snr_db/20.0), 0)
        adder = blocks.add_cc()
        freq_shift = blocks.multiply_cc()
        freq_source = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE,
                                        freq_offset, 1, 0)
        phase_shift = blocks.multiply_const_cc(np.exp(1j * phase_offset))
        output_sink = blocks.vector_sink_c()

        tb.connect(audio_source, multiply_const)
        tb.connect(multiply_const, add_const)
        tb.connect(add_const, f2c)
        tb.connect(f2c, (multiply_cc, 0))
        tb.connect(carrier, (multiply_cc, 1))
        tb.connect(multiply_cc, (adder, 0))
        tb.connect(noise_source, (adder, 1))
        tb.connect(adder, freq_shift)
        tb.connect(freq_source, (freq_shift, 1))
        tb.connect(freq_shift, phase_shift)
        tb.connect(phase_shift, output_sink)

        tb.run()
        tb.wait()

        samples = np.array(output_sink.data())
        return samples[:self.num_samples]

    def generate_fm(self, snr_db=20, freq_offset=0, phase_offset=0, carrier_freq=0):
        """Generate FM (Frequency Modulation) modulated signal"""
        tb = gr.top_block()

        # Generate random audio-like signal
        audio_length = self.num_samples + 1000
        t = np.linspace(0, audio_length/self.sample_rate, audio_length)

        # Create audio-like modulating signal
        audio_signal = 0.4 * np.sin(2*np.pi*1500*t) + 0.3 * np.sin(2*np.pi*2500*t) + 0.15 * np.sin(2*np.pi*4000*t)
        audio_signal = audio_signal.astype(np.float32)

        audio_source = blocks.vector_source_f(audio_signal, False)

        # FM modulator using frequency modulator
        # Frequency deviation = 75 kHz (standard for FM radio)
        freq_dev = 75000
        multiply_const = blocks.multiply_const_ff(freq_dev)
        freq_mod = analog.frequency_modulator_fc(2*np.pi / self.sample_rate)

        # Channel impairments
        noise_source = analog.noise_source_c(analog.GR_GAUSSIAN,
                                           10**(-snr_db/20.0), 0)
        adder = blocks.add_cc()
        freq_shift = blocks.multiply_cc()
        freq_source = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE,
                                        freq_offset, 1, 0)
        phase_shift = blocks.multiply_const_cc(np.exp(1j * phase_offset))
        output_sink = blocks.vector_sink_c()

        tb.connect(audio_source, multiply_const)
        tb.connect(multiply_const, freq_mod)
        tb.connect(freq_mod, (adder, 0))
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

def generate_dataset(output_prefix, train_samples=20000, test_samples=2000,
                    num_samples=1024, sample_rate=1e6, samples_per_symbol=8):
    """Generate comprehensive dataset with 16 modulation types"""

    print(f"🚀 Generating COMPREHENSIVE radio modulation dataset...")
    print(f"📊 Train samples per class: {train_samples:,}")
    print(f"🧪 Test samples per class: {test_samples:,}")
    print(f"📡 Samples per signal: {num_samples}")
    print(f"📈 Total training samples: {train_samples * 16:,}")
    print(f"📉 Total test samples: {test_samples * 16:,}")
    print(f"💾 Dataset size: ~{(train_samples * 16 + test_samples * 16) * num_samples * 2 * 4 / 1e9:.1f} GB")

    generator = ModulationGenerator(sample_rate, samples_per_symbol, num_samples)

    # COMPREHENSIVE 16 MODULATION TYPES for advanced deep learning research
    modulations = [
        'BPSK',    'QPSK',     '8PSK',    '16PSK',   '64PSK',     # PSK Family (5)
        '16QAM',   '32QAM',    '64QAM',                            # QAM Family (3)
        'ASK',     '2FSK',     '4FSK',    '8FSK',                 # Digital (4)
        'GMSK',    'MSK',      'AM',      'FM'                    # Advanced (4)
    ]

    mod_functions = [
        # PSK Family - Phase Shift Keying variants
        generator.generate_bpsk,      # Class 0:  Binary PSK (most robust)
        generator.generate_qpsk,      # Class 1:  Quadrature PSK
        generator.generate_8psk,      # Class 2:  8-ary PSK
        generator.generate_16psk,     # Class 3:  16-ary PSK (challenging)
        generator.generate_64psk,     # Class 4:  64-ary PSK (very challenging)

        # QAM Family - Quadrature Amplitude Modulation
        generator.generate_16qam,     # Class 5:  16-QAM (amplitude + phase)
        generator.generate_32qam,     # Class 6:  32-QAM
        generator.generate_64qam,     # Class 7:  64-QAM (high data rate)

        # Digital Modulations - Amplitude/Frequency based
        generator.generate_ask,       # Class 8:  Amplitude Shift Keying
        generator.generate_2fsk,      # Class 9:  Binary FSK
        generator.generate_4fsk,      # Class 10: 4-ary FSK
        generator.generate_8fsk,      # Class 11: 8-ary FSK

        # Advanced Digital - Specialized techniques
        generator.generate_gmsk,      # Class 12: Gaussian MSK (used in GSM)
        generator.generate_msk,       # Class 13: Minimum Shift Keying

        # Analog Modulations - Classic radio
        generator.generate_am,        # Class 14: Amplitude Modulation
        generator.generate_fm         # Class 15: Frequency Modulation
    ]

    print(f"\n📋 MODULATION CLASSES (16 total):")
    print(f"   PSK Family (0-4):  {modulations[0:5]}")
    print(f"   QAM Family (5-7):  {modulations[5:8]}")
    print(f"   Digital (8-11):    {modulations[8:12]}")
    print(f"   Advanced (12-15):  {modulations[12:16]}")
    print(f"   Random Baseline:   6.25% accuracy (1/16 classes)")

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

            # EXTREME difficulty channel conditions - true research challenge
            snr_db = random.uniform(-15, 5)  # EXTREME SNR: -15 to +5 dB (signals buried in noise)
            freq_offset = random.uniform(-15000, 15000)  # ±15 kHz frequency offset (severe)
            phase_offset = random.uniform(0, 2*np.pi)  # Full phase rotation range
            carrier_freq = random.uniform(1e6, 40e6)  # Wide carrier range: 1-40 MHz

            # EXTREME realistic impairments for maximum challenge
            timing_offset = random.uniform(-0.5, 0.5)  # ±0.5 symbol timing error
            carrier_drift = random.uniform(-100, 100)  # ±100 Hz/s carrier drift
            amplitude_variation = random.uniform(0.7, 1.3)  # 30% amplitude fading

            signal = mod_func(snr_db, freq_offset, phase_offset, carrier_freq)

            # Apply additional realistic impairments
            signal = apply_channel_impairments(signal, timing_offset, carrier_drift,
                                             amplitude_variation, sample_rate)

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

            # EXTREME test conditions - even harder than training!
            snr_db = random.uniform(-10, 0)  # BRUTAL test SNR: -10 to 0 dB (worse than training)
            freq_offset = random.uniform(-12000, 12000)  # ±12 kHz frequency offset
            phase_offset = random.uniform(0, 2*np.pi)  # Full phase rotation range
            carrier_freq = random.uniform(1e6, 40e6)  # Wide carrier range: 1-40 MHz

            # SEVERE impairments for ultimate test challenge
            timing_offset = random.uniform(-0.4, 0.4)  # ±0.4 symbol timing error
            carrier_drift = random.uniform(-80, 80)  # ±80 Hz/s carrier drift
            amplitude_variation = random.uniform(0.8, 1.2)  # 20% amplitude fading

            signal = mod_func(snr_db, freq_offset, phase_offset, carrier_freq)

            # Apply additional realistic impairments
            signal = apply_channel_impairments(signal, timing_offset, carrier_drift,
                                             amplitude_variation, sample_rate)

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

    # Save comprehensive metadata as info file
    info_file = f"{output_prefix}_info.txt"
    with open(info_file, 'w') as f:
        f.write("COMPREHENSIVE RADIO MODULATION DATASET - 16 CLASSES\n")
        f.write("=" * 60 + "\n\n")

        f.write("📋 MODULATION CLASSES (16 total):\n")
        for i, mod in enumerate(modulations):
            f.write(f"  Class {i:2d}: {mod}\n")
        f.write("\n")

        f.write("🏷️ CLASS FAMILIES:\n")
        f.write(f"  PSK Family (0-4):   {', '.join(modulations[0:5])}\n")
        f.write(f"  QAM Family (5-7):   {', '.join(modulations[5:8])}\n")
        f.write(f"  Digital (8-11):     {', '.join(modulations[8:12])}\n")
        f.write(f"  Advanced (12-15):   {', '.join(modulations[12:16])}\n\n")

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

        f.write("🛠️ CHANNEL CONDITIONS (EXTREME DIFFICULTY):\n")
        f.write("  Training SNR: -15 to +5 dB (signals buried in noise)\n")
        f.write("  Test SNR: -10 to 0 dB (even more brutal)\n")
        f.write("  Frequency offset: ±12-15 kHz (severe)\n")
        f.write("  Phase offset: 0 to 2π\n")
        f.write("  Carrier frequency: 1-40 MHz (wide range)\n")
        f.write("  Channel impairments: Extreme timing, carrier drift, amplitude fading\n\n")

        f.write("💾 DATA FORMAT:\n")
        f.write("  Each row contains I/Q samples interleaved\n")
        f.write("  [I0, Q0, I1, Q1, I2, Q2, ..., I1023, Q1023]\n")
        f.write("  where I=real part, Q=imaginary part\n")
        f.write("  Normalized to [-1, +1] range\n\n")

        f.write("📁 FILES GENERATED:\n")
        f.write(f"  📊 {train_data_file}\n")
        f.write(f"  🏷️  {train_labels_file}\n")
        f.write(f"  📊 {test_data_file}\n")
        f.write(f"  🏷️  {test_labels_file}\n")
        f.write(f"  📋 {info_file}\n\n")

        f.write("🧠 NEURAL NETWORK GUIDELINES (EXTREME CHALLENGE):\n")
        f.write(f"  Random baseline accuracy: {100/len(modulations):.2f}%\n")
        f.write("  Recommended architecture: Very deep 1D CNN with attention\n")
        f.write("  Expected performance: 25-45% (EXTREME difficulty)\n")
        f.write("  Key challenges: Signals buried in noise + similar modulations\n")
        f.write("  Strategy: Massive regularization, data augmentation, ensemble methods\n")
        f.write("  Research goal: Pushing limits of radio signal intelligence\n")

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

def create_cpp_usage_example():
    """Create comprehensive C++ usage example for 16-class system"""
    usage_example = '''C++ USAGE EXAMPLE - 16-CLASS RADIO MODULATION DATASET

#include "Utils/Helpers.h"
using namespace NEURAL_NETWORK;

int main() {
    // Load comprehensive 16-class radio modulation dataset
    Eigen::MatrixXd train_data, test_data;
    Eigen::VectorXi train_labels, test_labels;

    Helpers::ReadCSVMatrix("radio_dataset_comprehensive_train_data.csv", train_data);
    Helpers::ReadCSVLabels("radio_dataset_comprehensive_train_labels.csv", train_labels);
    Helpers::ReadCSVMatrix("radio_dataset_comprehensive_test_data.csv", test_data);
    Helpers::ReadCSVLabels("radio_dataset_comprehensive_test_labels.csv", test_labels);

    // 16 MODULATION CLASSES:
    // PSK Family (0-4):   BPSK, QPSK, 8PSK, 16PSK, 64PSK
    // QAM Family (5-7):   16QAM, 32QAM, 64QAM
    // Digital (8-11):     ASK, 2FSK, 4FSK, 8FSK
    // Advanced (12-15):   GMSK, MSK, AM, FM

    // Feature format: I/Q interleaved [I0,Q0,I1,Q1,...,I1023,Q1023]
    // Features per sample: 2048 (1024 time steps × 2 channels)
    // Random baseline: 6.25% accuracy (1/16 classes)

    std::cout << "Dataset loaded:" << std::endl;
    std::cout << "  Training samples: " << train_data.rows() << std::endl;
    std::cout << "  Test samples: " << test_data.rows() << std::endl;
    std::cout << "  Features: " << train_data.cols() << std::endl;
    std::cout << "  Classes: 16" << std::endl;

    // RECOMMENDED ADVANCED 1D CNN ARCHITECTURE FOR 16 CLASSES
    Model model;
    model.Add(std::make_shared<LayerInput>());

    // First Conv1D block - extract low-level temporal features
    model.Add(std::make_shared<Convolution1D>(
        64,        // More filters for 16 classes
        9,         // Larger kernels for complex patterns
        1024, 2,   // Input: 1024 time steps, 2 channels (I/Q)
        1, 1,      // Stride 1, padding 1
        0.0, 1e-4  // L2 regularization
    ));
    model.Add(std::make_shared<ActivationReLU>());
    model.Add(std::make_shared<MaxPooling1D>(64, 2, 1024, 64, 2));  // Downsample

    // Second Conv1D block - extract mid-level features
    model.Add(std::make_shared<Convolution1D>(
        128,       // Increase filters
        7,         // Kernel size 7
        512, 64,   // From pooling: 512 time steps, 64 channels
        1, 1,      // Stride 1, padding 1
        0.0, 1e-4  // L2 regularization
    ));
    model.Add(std::make_shared<ActivationReLU>());
    model.Add(std::make_shared<MaxPooling1D>(128, 2, 512, 128, 2));  // Downsample

    // Third Conv1D block - extract high-level features
    model.Add(std::make_shared<Convolution1D>(
        256,       // High-level features
        5,         // Kernel size 5
        256, 128,  // From pooling: 256 time steps, 128 channels
        1, 1,      // Stride 1, padding 1
        0.0, 2e-4  // Stronger regularization
    ));
    model.Add(std::make_shared<ActivationReLU>());
    model.Add(std::make_shared<MaxPooling1D>(256, 2, 256, 256, 2));  // Final downsample

    // Regularization
    model.Add(std::make_shared<LayerDropout>(0.3));

    // Dense layers for 16-class classification
    int flattened_size = 128 * 256;  // After final pooling
    model.Add(std::make_shared<LayerDense>(
        flattened_size, 512,  // Large dense layer
        0.0, 3e-4             // Strong L2 regularization
    ));
    model.Add(std::make_shared<ActivationReLU>());
    model.Add(std::make_shared<LayerDropout>(0.5));  // Heavy dropout

    model.Add(std::make_shared<LayerDense>(
        512, 256,    // Secondary dense layer
        0.0, 3e-4    // Strong L2 regularization
    ));
    model.Add(std::make_shared<ActivationReLU>());
    model.Add(std::make_shared<LayerDropout>(0.5));

    // Output layer for 16 classes
    model.Add(std::make_shared<LayerDense>(
        256, 16,     // 16 modulation classes
        0.0, 3e-4    // L2 regularization
    ));
    model.Add(std::make_shared<ActivationSoftmax>());

    // Configure training
    model.Set(
        std::make_unique<LossCategoricalCrossEntropy>(),
        std::make_unique<AccuracyCategorical>(),
        std::make_unique<Adam>(0.0001, 1e-4)  // Low learning rate for 16 classes
    );

    model.Finalize();

    // Train with strong regularization for 16-class problem
    std::cout << "Training 16-class radio modulation classifier..." << std::endl;
    model.Train(train_data, y_train, 64, 50, 500, test_data, y_test);

    // Save trained model
    model.SaveModel("radio_16class_model.bin");

    return 0;
}

PERFORMANCE EXPECTATIONS:
- Random baseline: 6.25% (1/16 classes)
- Good model: 35-50% accuracy
- Excellent model: 50-65% accuracy
- State-of-art: 65%+ accuracy

TRAINING TIPS:
1. Use strong regularization (L2 + dropout)
2. Lower learning rates (1e-4 to 1e-5)
3. More epochs (50-100)
4. Data augmentation (phase/amplitude variations)
5. Learning rate scheduling
6. Early stopping with patience
'''

    with open('radio_dataset_16class_usage.txt', 'w') as f:
        f.write(usage_example)

    print(f"📋 Created C++ usage example: radio_dataset_16class_usage.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comprehensive 16-class radio modulation dataset in CSV format")
    parser.add_argument("--output", "-o", default="radio_dataset_comprehensive",
                       help="Output file prefix (will create _train_data.csv, _train_labels.csv, etc.)")
    parser.add_argument("--train-samples", "-t", type=int, default=20000,
                       help="Training samples per class (default: 20,000)")
    parser.add_argument("--test-samples", "-s", type=int, default=2000,
                       help="Test samples per class (default: 2,000)")
    parser.add_argument("--num-samples", "-n", type=int, default=1024,
                       help="Samples per signal (default: 1024)")
    parser.add_argument("--sample-rate", "-r", type=float, default=1e6,
                       help="Sample rate in Hz (default: 1 MHz)")
    parser.add_argument("--samples-per-symbol", "-p", type=int, default=8,
                       help="Samples per symbol (default: 8)")

    args = parser.parse_args()

    print("🌟" + "="*70 + "🌟")
    print("🎯 COMPREHENSIVE RADIO MODULATION DATASET GENERATOR")
    print("🎯 16 Modulation Types - Deep Learning Research Dataset")
    print("🌟" + "="*70 + "🌟")

    # Display configuration
    total_train = args.train_samples * 16
    total_test = args.test_samples * 16
    estimated_size = (total_train + total_test) * args.num_samples * 2 * 4 / 1e9

    print(f"\n📋 CONFIGURATION:")
    print(f"   📊 Classes: 16 modulations")
    print(f"   🎓 Training samples: {args.train_samples:,} per class ({total_train:,} total)")
    print(f"   🧪 Test samples: {args.test_samples:,} per class ({total_test:,} total)")
    print(f"   📡 Signal length: {args.num_samples} samples")
    print(f"   📈 Features: {args.num_samples * 2:,} per sample (I/Q interleaved)")
    print(f"   💾 Estimated size: ~{estimated_size:.1f} GB")
    print(f"   🎯 Random baseline: {100/16:.2f}% accuracy")

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

    # Create C++ usage example
    create_cpp_usage_example()

    print(f"\n🌟" + "="*70 + "🌟")
    print(f"🎉 DATASET GENERATION COMPLETE!")
    print(f"📂 Dataset files created with prefix: {args.output}")
    print(f"📄 C++ usage example: radio_dataset_16class_usage.txt")
    print(f"🧠 Ready for advanced deep learning research!")
    print(f"🌟" + "="*70 + "🌟")

    print(f"\n💡 QUICK START:")
    print(f"   Load training data: NEURAL_NETWORK::Helpers::ReadCSVMatrix(\"{args.output}_train_data.csv\", matrix)")
    print(f"   Load training labels: NEURAL_NETWORK::Helpers::ReadCSVLabels(\"{args.output}_train_labels.csv\", labels)")
    print(f"   Expected architecture: Deep 1D CNN with 3 conv layers + 2 dense layers")
    print(f"   Target accuracy: 35-65% (vs 6.25% random baseline)")

    print(f"\n📚 RESEARCH POTENTIAL:")
    print(f"   🔬 16-class modulation recognition")
    print(f"   🧬 Transfer learning across modulation families")
    print(f"   🔍 Attention mechanisms for signal vs noise")
    print(f"   📊 Confusion matrix analysis (PSK vs QAM vs FSK)")
    print(f"   🎛️ Hyperparameter optimization for radio ML")


"""
Overview

  The signal generator in /src/Utils/generate_radio_dataset.py is a sophisticated radio frequency (RF) modulation dataset generator built on GNU Radio. It creates synthetic I/Q
  (In-phase/Quadrature) samples for machine learning training on radio signal classification.

  Signal Variables Analysis

  1. Core Signal Parameters

  | Parameter          | Default Value | Range/Options | Purpose                          |
  |--------------------|---------------|---------------|----------------------------------|
  | sample_rate        | 1 MHz         | Fixed         | Controls temporal resolution     |
  | samples_per_symbol | 8             | Fixed         | Oversampling factor              |
  | num_samples        | 1024          | Fixed         | Signal length (time steps)       |
  | symbol_rate        | 125 kHz       | Derived       | sample_rate / samples_per_symbol |

  2. Modulation Types Generated

  Current Active Modulations (lines 449-455):

  1. BPSK (Binary Phase Shift Keying)
    - 2 constellation points: {1, -1}
    - 1 bit per symbol
    - Most robust against noise
  2. QPSK (Quadrature Phase Shift Keying)
    - 4 constellation points: {1+1j, 1-1j, -1+1j, -1-1j}
    - 2 bits per symbol
    - Good noise performance
  3. 8PSK (8-ary Phase Shift Keying)
    - 8 constellation points (45° spacing)
    - 3 bits per symbol
    - More susceptible to phase noise
  4. 16PSK (16-ary Phase Shift Keying)
    - 16 constellation points (22.5° spacing)
    - 4 bits per symbol
    - Very challenging - close constellation points

  Alternative Modulations (implemented but unused):

  - 16QAM (Quadrature Amplitude Modulation) - lines 161-204
  - ASK (Amplitude Shift Keying) - lines 348-399

  3. Channel Impairment Variables

  Training Dataset Conditions (lines 468-493):

  | Impairment          | Range           | Impact                             |
  |---------------------|-----------------|------------------------------------|
  | SNR                 | -15 dB to +5 dB | Extremely challenging noise levels |
  | Frequency Offset    | ±15 kHz         | Large frequency errors             |
  | Phase Offset        | 0 to 2π         | Random phase rotation              |
  | Carrier Frequency   | 1-40 MHz        | Wide RF spectrum                   |
  | Timing Offset       | ±0.5 symbols    | Symbol synchronization errors      |
  | Carrier Drift       | ±100 Hz/sec     | Oscillator instability             |
  | Amplitude Variation | 0.7 to 1.3      | Fading channel effects             |

  Test Dataset Conditions (lines 496-520):

  | Impairment       | Range          | Difficulty Level      |
  |------------------|----------------|-----------------------|
  | SNR              | -10 dB to 0 dB | Even more challenging |
  | Frequency Offset | ±12 kHz        | Slightly reduced      |
  | Timing Offset    | ±0.4 symbols   | Slightly reduced      |
  | Carrier Drift    | ±80 Hz/sec     | Slightly reduced      |

  4. Advanced Realistic Impairments (lines 401-433)

  Multipath Fading:

  - 3-tap fading model with complex coefficients
  - Direct path weight: 1.0 (strongest)
  - Multipath taps: 0.1 * random_normal()

  Frequency-Selective Fading:

  - 1 kHz sinusoidal variation with 10% depth
  - Simulates frequency-dependent amplitude changes

  Multiple Noise Sources:

  1. AWGN (Primary): 10^(-SNR/20)
  2. Thermal Noise: 10-20% of AWGN power
  3. Phase Noise: 5% random Gaussian with 0.5-1.5x variation

  5. Data Processing Pipeline

  Signal Processing Chain:

  Random Bits → Modulator → Carrier Mixing → Noise Addition →
  Impairments → Frequency Shift → Phase Shift → I/Q Output

  Output Format:

  - Shape: [samples, 2048]
  - Encoding: I/Q interleaved [I₀, Q₀, I₁, Q₁, ..., I₁₀₂₃, Q₁₀₂₃]
  - Normalization: [-1, +1] range
  - Features: 2048 per signal (1024 complex samples × 2)

  Dataset Difficulty Assessment

  Extreme Challenge Factors:

  1. Very Low SNR: Training at -15 dB means signal is 30x weaker than noise
  2. Similar Modulations: BPSK→QPSK→8PSK→16PSK creates close constellation patterns
  3. Large Frequency Offsets: ±15 kHz offset on 125 kHz symbol rate is 12% error
  4. Realistic Channel: Multiple fading, noise, and timing effects combined
  5. Test Degradation: Test SNR (-10 to 0 dB) is worse than training

  Classification Difficulty Ranking:

  1. BPSK vs QPSK: Moderate (2-point vs 4-point)
  2. QPSK vs 8PSK: Hard (constellation density increases)
  3. 8PSK vs 16PSK: Extremely Hard (22.5° vs 45° phase spacing)
  4. All at low SNR: Nearly impossible without deep learning

  Recommendations for Neural Network

  Model Requirements:

  - Deep 1D CNN for temporal feature extraction
  - Strong regularization due to small inter-class differences
  - Attention mechanisms for focusing on signal vs noise
  - Data augmentation with additional rotations/scales

  Expected Performance:

  - Best case: 60-70% accuracy (given SNR conditions)
  - Realistic: 45-60% accuracy
  - Random baseline: 25% (4 classes)

  This dataset represents one of the most challenging RF classification problems, mimicking real-world conditions where signals are heavily corrupted and modulation types are
  very similar.

"""