#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Variable Modulations
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from PyQt5.QtCore import QObject, pyqtSlot
from gnuradio import analog
from gnuradio import blocks
from gnuradio import digital
from gnuradio import filter
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import iio
from gnuradio import zeromq
import configparser
import math
import numpy as np
import sip
import threading
import time



class VariableModulation(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Variable Modulations", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Variable Modulations")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "VariableModulation")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.sps = sps = 8
        self.samp_rate = samp_rate = 10**6
        self.qam32 = qam32 = digital.constellation_calcdist([-5-3j, -3-3j, -1-3j, 1-3j, 3-3j, 5-3j, -5-1j, -3-1j, -1-1j, 1-1j, 3-1j, 5-1j, -5+1j, -3+1j, -1+1j, 1+1j, 3+1j, 5+1j, -5+3j, -3+3j, -1+3j, 1+3j, 3+3j, 5+3j, -3-5j, -1-5j, 1-5j, 3-5j, -3+5j, -1+5j, 1+5j, 3+5j], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        4, 1, digital.constellation.AMPLITUDE_NORMALIZATION).base()
        self.qam32.set_npwr(1)
        self.qam16 = qam16 = digital.constellation_calcdist([-3-3j, -1-3j, 1-3j, 3-3j, -3-1j, -1-1j, 1-1j, 3-1j, -3+1j, -1+1j, 1+1j, 3+1j, -3+3j, -1+3j, 1+3j, 3+3j], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        4, 1, digital.constellation.AMPLITUDE_NORMALIZATION).base()
        self.qam16.set_npwr(1)
        self.nfilts = nfilts = 32
        self.alpha = alpha = 0.5
        self.tx_attenuation = tx_attenuation = 10
        self.rx_gain = rx_gain = 15
        self.rrc_taps = rrc_taps = firdes.root_raised_cosine(nfilts, nfilts * samp_rate,samp_rate/sps, alpha, (15 * sps * nfilts))
        self.rnd_mod_fnc = rnd_mod_fnc = 0
        self.qpsk = qpsk = digital.constellation_calcdist([-1 - 1j, +1 - 1j, +1+1j, -1 + 1j], [0, 1, 2, 3],
        4, 1, digital.constellation.AMPLITUDE_NORMALIZATION).base()
        self.qpsk.set_npwr(1)
        self._pluto_ip_tx_config = configparser.ConfigParser()
        self._pluto_ip_tx_config.read('default')
        try: pluto_ip_tx = self._pluto_ip_tx_config.get('main', 'key')
        except: pluto_ip_tx = 'ip:192.168.3.1'
        self.pluto_ip_tx = pluto_ip_tx
        self._pluto_ip_rx_config = configparser.ConfigParser()
        self._pluto_ip_rx_config.read('default')
        try: pluto_ip_rx = self._pluto_ip_rx_config.get('main', 'key')
        except: pluto_ip_rx = 'ip:192.168.2.1'
        self.pluto_ip_rx = pluto_ip_rx
        self.phase_shift_after_costas_loop = phase_shift_after_costas_loop = 0
        self.out_sps = out_sps = 1
        self.offset_tx = offset_tx = 100000
        self.offset_rx = offset_rx = -10 **  5
        self.lo_p = lo_p = 0
        self.lo_freq = lo_freq = 2200000000
        self.lms_32 = lms_32 = digital.adaptive_algorithm_lms( qam32, 0.6).base()
        self.lms = lms = digital.adaptive_algorithm_lms( qam16, 0.3).base()
        self.costas_loop_bw = costas_loop_bw = 0.02
        self.const_chooser = const_chooser = 0
        self.buff_size = buff_size = 2**15
        self.bpsk = bpsk = digital.constellation_calcdist([-1+0j, 1 + 0j], [0, 1],
        4, 1, digital.constellation.AMPLITUDE_NORMALIZATION).base()
        self.bpsk.set_npwr(1)

        ##################################################
        # Blocks
        ##################################################

        self.rnd_mod = blocks.probe_signal_b()
        self._tx_attenuation_range = qtgui.Range(0, 89, 1, 10, 200)
        self._tx_attenuation_win = qtgui.RangeWidget(self._tx_attenuation_range, self.set_tx_attenuation, "'tx_attenuation'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._tx_attenuation_win, 2, 0, 1, 1)
        for r in range(2, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._rx_gain_range = qtgui.Range(0, 60, 1, 15, 200)
        self._rx_gain_win = qtgui.RangeWidget(self._rx_gain_range, self.set_rx_gain, "'rx_gain'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._rx_gain_win, 2, 1, 1, 1)
        for r in range(2, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        def _rnd_mod_fnc_probe():
          self.flowgraph_started.wait()
          while True:

            val = self.rnd_mod.level()
            try:
              try:
                self.doc.add_next_tick_callback(functools.partial(self.set_rnd_mod_fnc,val))
              except AttributeError:
                self.set_rnd_mod_fnc(val)
            except AttributeError:
              pass
            time.sleep(1.0 / (0.2))
        _rnd_mod_fnc_thread = threading.Thread(target=_rnd_mod_fnc_probe)
        _rnd_mod_fnc_thread.daemon = True
        _rnd_mod_fnc_thread.start()
        self._phase_shift_after_costas_loop_range = qtgui.Range(0, 2*math.pi, 0.01, 0, 200)
        self._phase_shift_after_costas_loop_win = qtgui.RangeWidget(self._phase_shift_after_costas_loop_range, self.set_phase_shift_after_costas_loop, "Ph Shift (after CL) [rad]", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._phase_shift_after_costas_loop_win, 4, 0, 1, 1)
        for r in range(4, 5):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._offset_rx_range = qtgui.Range(-3 * 10 ** 5, 2.5 * 10 **5, 1, -10 **  5, 200)
        self._offset_rx_win = qtgui.RangeWidget(self._offset_rx_range, self.set_offset_rx, "'offset_rx'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._offset_rx_win, 6, 0, 1, 1)
        for r in range(6, 7):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._lo_p_range = qtgui.Range(-25000, 25000, 10, 0, 200)
        self._lo_p_win = qtgui.RangeWidget(self._lo_p_range, self.set_lo_p, "LO Diff", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._lo_p_win, 6, 1, 1, 1)
        for r in range(6, 7):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._costas_loop_bw_range = qtgui.Range(0.001, 0.2, 0.001, 0.02, 200)
        self._costas_loop_bw_win = qtgui.RangeWidget(self._costas_loop_bw_range, self.set_costas_loop_bw, "Costas Loop Bw [cycles/sample]", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._costas_loop_bw_win, 4, 1, 1, 1)
        for r in range(4, 5):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.zeromq_pub_sink_0 = zeromq.pub_sink(gr.sizeof_gr_complex, 2048, 'tcp://127.0.0.1:5555', 100, False, 1000, '', True, True)
        self.qtgui_time_sink_x_1 = qtgui.time_sink_c(
            200, #size
            samp_rate, #samp_rate
            "Received Samples", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_1.set_update_time(0.1)
        self.qtgui_time_sink_x_1.set_y_axis(-1.5, 1.5)

        self.qtgui_time_sink_x_1.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_1.enable_tags(True)
        self.qtgui_time_sink_x_1.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_1.enable_autoscale(False)
        self.qtgui_time_sink_x_1.enable_grid(False)
        self.qtgui_time_sink_x_1.enable_axis_labels(True)
        self.qtgui_time_sink_x_1.enable_control_panel(False)
        self.qtgui_time_sink_x_1.enable_stem_plot(False)


        labels = ['I', 'Q', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_x_1.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_1.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_1.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_1.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_1.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_1.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_1.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_1.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_1_win = sip.wrapinstance(self.qtgui_time_sink_x_1.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_time_sink_x_1_win, 1, 1, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_time_sink_x_0 = qtgui.time_sink_c(
            (200 * sps), #size
            samp_rate, #samp_rate
            'Transmitted Samples', #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0.set_update_time(0.1)
        self.qtgui_time_sink_x_0.set_y_axis(-1.5, 1.5)

        self.qtgui_time_sink_x_0.set_y_label("Amplitude", "")

        self.qtgui_time_sink_x_0.enable_tags(True)
        self.qtgui_time_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0.enable_grid(False)
        self.qtgui_time_sink_x_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0.enable_stem_plot(False)


        labels = ['I', 'Q', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_x_0.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_0.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_time_sink_x_0_win, 1, 0, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_freq_sink_x_0_0 = qtgui.freq_sink_c(
            8192, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            (samp_rate / 2), #bw
            "Frequencies", #name
            3,
            None # parent
        )
        self.qtgui_freq_sink_x_0_0.set_update_time(0.1)
        self.qtgui_freq_sink_x_0_0.set_y_axis((-145), 10)
        self.qtgui_freq_sink_x_0_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0_0.enable_grid(False)
        self.qtgui_freq_sink_x_0_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0_0.set_fft_window_normalized(False)



        labels = ['TX', 'RX Raw', 'RX After Filter', 'RX After Filter', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(3):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_freq_sink_x_0_0_win, 5, 0, 1, 2)
        for r in range(5, 6):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_eye_sink_x_0 = qtgui.eye_sink_c(
            100, #size
            samp_rate, #samp_rate
            1, #number of inputs
            None
        )
        self.qtgui_eye_sink_x_0.set_update_time(0.1)
        self.qtgui_eye_sink_x_0.set_samp_per_symbol(1)
        self.qtgui_eye_sink_x_0.set_y_axis(-1, 1)

        self.qtgui_eye_sink_x_0.set_y_label('Amplitude', '')

        self.qtgui_eye_sink_x_0.enable_tags(True)
        self.qtgui_eye_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_eye_sink_x_0.enable_autoscale(False)
        self.qtgui_eye_sink_x_0.enable_grid(False)
        self.qtgui_eye_sink_x_0.enable_axis_labels(True)
        self.qtgui_eye_sink_x_0.enable_control_panel(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'blue', 'blue', 'blue', 'blue',
            'blue', 'blue', 'blue', 'blue', 'blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_eye_sink_x_0.set_line_label(i, "Eye [Re{{Data {0}}}]".format(round(i/2)))
                else:
                    self.qtgui_eye_sink_x_0.set_line_label(i, "Eye [Im{{Data {0}}}]".format(round((i-1)/2)))
            else:
                self.qtgui_eye_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_eye_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_eye_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_eye_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_eye_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_eye_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_eye_sink_x_0_win = sip.wrapinstance(self.qtgui_eye_sink_x_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_eye_sink_x_0_win, 7, 0, 1, 2)
        for r in range(7, 8):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_const_sink_x_1 = qtgui.const_sink_c(
            1024, #size
            "Constellation Before Sync", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_1.set_update_time(0.1)
        self.qtgui_const_sink_x_1.set_y_axis((-2), 2)
        self.qtgui_const_sink_x_1.set_x_axis((-2), 2)
        self.qtgui_const_sink_x_1.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_1.enable_autoscale(False)
        self.qtgui_const_sink_x_1.enable_grid(False)
        self.qtgui_const_sink_x_1.enable_axis_labels(True)


        labels = ['Before Sync', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_1.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_1.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_1.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_1.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_1.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_1.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_1.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_1_win = sip.wrapinstance(self.qtgui_const_sink_x_1.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_const_sink_x_1_win, 3, 0, 1, 1)
        for r in range(3, 4):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(
            (1024 * 2), #size
            'Constellation After Sync', #name
            2, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0.set_update_time(0.1)
        self.qtgui_const_sink_x_0.set_y_axis((-2), 2)
        self.qtgui_const_sink_x_0.set_x_axis((-2), 2)
        self.qtgui_const_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0.enable_grid(True)
        self.qtgui_const_sink_x_0.enable_axis_labels(True)


        labels = ['Before fine-tuning', 'After fine-tuning', 'After fine-tuning', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(2):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_const_sink_x_0_win, 3, 1, 1, 1)
        for r in range(3, 4):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.iio_pluto_source_0 = iio.fmcomms2_source_fc32(pluto_ip_rx if pluto_ip_rx else iio.get_pluto_uri(), [True, True], buff_size)
        self.iio_pluto_source_0.set_len_tag_key('packet_len')
        self.iio_pluto_source_0.set_frequency(lo_freq)
        self.iio_pluto_source_0.set_samplerate(samp_rate)
        self.iio_pluto_source_0.set_gain_mode(0, 'manual')
        self.iio_pluto_source_0.set_gain(0, rx_gain)
        self.iio_pluto_source_0.set_quadrature(True)
        self.iio_pluto_source_0.set_rfdc(True)
        self.iio_pluto_source_0.set_bbdc(True)
        self.iio_pluto_source_0.set_filter_params('Auto', '', 0, 0)
        self.iio_pluto_sink_0 = iio.fmcomms2_sink_fc32(pluto_ip_tx if pluto_ip_tx else iio.get_pluto_uri(), [True, True], buff_size, False)
        self.iio_pluto_sink_0.set_len_tag_key('')
        self.iio_pluto_sink_0.set_bandwidth((samp_rate * 5))
        self.iio_pluto_sink_0.set_frequency((lo_freq + lo_p))
        self.iio_pluto_sink_0.set_samplerate(samp_rate)
        self.iio_pluto_sink_0.set_attenuation(0, tx_attenuation)
        self.iio_pluto_sink_0.set_filter_params('Auto', '', 0, 0)
        self.digital_symbol_sync_xx_0 = digital.symbol_sync_cc(
            digital.TED_GARDNER,
            sps,
            0.015,
            0.707,
            2,
            1.5,
            out_sps,
            None,
            digital.IR_PFB_MF,
            nfilts,
            rrc_taps)
        self.digital_linear_equalizer_0_0 = digital.linear_equalizer(1, out_sps, lms_32, True, [ ], "")
        self.digital_linear_equalizer_0 = digital.linear_equalizer(1, out_sps, lms, True, [ ], "")
        self.digital_fll_band_edge_cc_0 = digital.fll_band_edge_cc(sps, alpha, (sps*2+1), (2*math.pi/sps/100))
        self.digital_costas_loop_cc_0_0 = digital.costas_loop_cc((2*math.pi*costas_loop_bw), qpsk.arity(), False)
        self.digital_costas_loop_cc_0 = digital.costas_loop_cc((2*math.pi*costas_loop_bw), bpsk.arity(), False)
        self.digital_constellation_modulator_0_0_0_0 = digital.generic_mod(
            constellation=qam32,
            differential=True,
            samples_per_symbol=sps,
            pre_diff_code=True,
            excess_bw=alpha,
            verbose=False,
            log=False,
            truncate=False)
        self.digital_constellation_modulator_0_0_0 = digital.generic_mod(
            constellation=qam16,
            differential=True,
            samples_per_symbol=sps,
            pre_diff_code=True,
            excess_bw=alpha,
            verbose=False,
            log=False,
            truncate=False)
        self.digital_constellation_modulator_0_0 = digital.generic_mod(
            constellation=qpsk,
            differential=True,
            samples_per_symbol=sps,
            pre_diff_code=True,
            excess_bw=alpha,
            verbose=False,
            log=False,
            truncate=False)
        self.digital_constellation_modulator_0 = digital.generic_mod(
            constellation=bpsk,
            differential=True,
            samples_per_symbol=sps,
            pre_diff_code=True,
            excess_bw=alpha,
            verbose=False,
            log=False,
            truncate=False)
        # Create the options list
        self._const_chooser_options = [0, 1, 2, 3]
        # Create the labels list
        self._const_chooser_labels = ['bpsk', 'qpsk', 'qam16', 'qam32']
        # Create the combo box
        # Create the radio buttons
        self._const_chooser_group_box = Qt.QGroupBox("'const_chooser'" + ": ")
        self._const_chooser_box = Qt.QHBoxLayout()
        class variable_chooser_button_group(Qt.QButtonGroup):
            def __init__(self, parent=None):
                Qt.QButtonGroup.__init__(self, parent)
            @pyqtSlot(int)
            def updateButtonChecked(self, button_id):
                self.button(button_id).setChecked(True)
        self._const_chooser_button_group = variable_chooser_button_group()
        self._const_chooser_group_box.setLayout(self._const_chooser_box)
        for i, _label in enumerate(self._const_chooser_labels):
            radio_button = Qt.QRadioButton(_label)
            self._const_chooser_box.addWidget(radio_button)
            self._const_chooser_button_group.addButton(radio_button, i)
        self._const_chooser_callback = lambda i: Qt.QMetaObject.invokeMethod(self._const_chooser_button_group, "updateButtonChecked", Qt.Q_ARG("int", self._const_chooser_options.index(i)))
        self._const_chooser_callback(self.const_chooser)
        self._const_chooser_button_group.buttonClicked[int].connect(
            lambda i: self.set_const_chooser(self._const_chooser_options[i]))
        self.top_grid_layout.addWidget(self._const_chooser_group_box, 0, 0, 1, 2)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 2048)
        self.blocks_skiphead_0 = blocks.skiphead(gr.sizeof_gr_complex*1, samp_rate)
        self.blocks_selector_0_1 = blocks.selector(gr.sizeof_gr_complex*1,0,rnd_mod_fnc)
        self.blocks_selector_0_1.set_enabled(True)
        self.blocks_selector_0_0_0 = blocks.selector(gr.sizeof_gr_complex*1,rnd_mod_fnc,0)
        self.blocks_selector_0_0_0.set_enabled(True)
        self.blocks_selector_0_0 = blocks.selector(gr.sizeof_gr_complex*1,rnd_mod_fnc,0)
        self.blocks_selector_0_0.set_enabled(True)
        self.blocks_phase_shift_0 = blocks.phase_shift(phase_shift_after_costas_loop, True)
        self.blocks_packed_to_unpacked_xx_0_1_0 = blocks.packed_to_unpacked_bb(5, gr.GR_MSB_FIRST)
        self.blocks_packed_to_unpacked_xx_0_1 = blocks.packed_to_unpacked_bb(4, gr.GR_MSB_FIRST)
        self.blocks_packed_to_unpacked_xx_0_0 = blocks.packed_to_unpacked_bb(2, gr.GR_MSB_FIRST)
        self.blocks_packed_to_unpacked_xx_0 = blocks.packed_to_unpacked_bb(1, gr.GR_MSB_FIRST)
        self.blocks_multiply_xx_0_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(0.6)
        self.blocks_keep_one_in_n_0 = blocks.keep_one_in_n(gr.sizeof_char*1, 1000)
        self.analog_sig_source_x_0_0 = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, offset_rx, 1, 0, 0)
        self.analog_sig_source_x_0 = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, offset_tx, 1, 0, 0)
        self.analog_random_uniform_source_x_3_0 = analog.random_uniform_source_b(0, 4, 0)
        self.analog_random_uniform_source_x_3 = analog.random_uniform_source_b(0, 256, 0)
        self.analog_random_uniform_source_x_2 = analog.random_uniform_source_b(0, 256, 0)
        self.analog_random_uniform_source_x_1 = analog.random_uniform_source_b(0, 256, 0)
        self.analog_random_uniform_source_x_0 = analog.random_uniform_source_b(0, 256, 0)
        self.analog_agc_xx_0 = analog.agc_cc((1e-4), 1.0, 1.0, 65536)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_agc_xx_0, 0), (self.digital_fll_band_edge_cc_0, 0))
        self.connect((self.analog_random_uniform_source_x_0, 0), (self.blocks_packed_to_unpacked_xx_0, 0))
        self.connect((self.analog_random_uniform_source_x_1, 0), (self.blocks_packed_to_unpacked_xx_0_0, 0))
        self.connect((self.analog_random_uniform_source_x_2, 0), (self.blocks_packed_to_unpacked_xx_0_1, 0))
        self.connect((self.analog_random_uniform_source_x_3, 0), (self.blocks_packed_to_unpacked_xx_0_1_0, 0))
        self.connect((self.analog_random_uniform_source_x_3_0, 0), (self.blocks_keep_one_in_n_0, 0))
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.analog_sig_source_x_0_0, 0), (self.blocks_multiply_xx_0_0, 1))
        self.connect((self.blocks_keep_one_in_n_0, 0), (self.rnd_mod, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.qtgui_freq_sink_x_0_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.qtgui_time_sink_x_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.iio_pluto_sink_0, 0))
        self.connect((self.blocks_multiply_xx_0_0, 0), (self.analog_agc_xx_0, 0))
        self.connect((self.blocks_multiply_xx_0_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.blocks_multiply_xx_0_0, 0), (self.qtgui_freq_sink_x_0_0, 1))
        self.connect((self.blocks_packed_to_unpacked_xx_0, 0), (self.digital_constellation_modulator_0, 0))
        self.connect((self.blocks_packed_to_unpacked_xx_0_0, 0), (self.digital_constellation_modulator_0_0, 0))
        self.connect((self.blocks_packed_to_unpacked_xx_0_1, 0), (self.digital_constellation_modulator_0_0_0, 0))
        self.connect((self.blocks_packed_to_unpacked_xx_0_1_0, 0), (self.digital_constellation_modulator_0_0_0_0, 0))
        self.connect((self.blocks_phase_shift_0, 0), (self.qtgui_const_sink_x_0, 1))
        self.connect((self.blocks_phase_shift_0, 0), (self.qtgui_eye_sink_x_0, 0))
        self.connect((self.blocks_phase_shift_0, 0), (self.qtgui_time_sink_x_1, 0))
        self.connect((self.blocks_selector_0_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.blocks_selector_0_0_0, 0), (self.blocks_phase_shift_0, 0))
        self.connect((self.blocks_selector_0_1, 0), (self.digital_costas_loop_cc_0, 0))
        self.connect((self.blocks_selector_0_1, 1), (self.digital_costas_loop_cc_0_0, 0))
        self.connect((self.blocks_selector_0_1, 2), (self.digital_linear_equalizer_0, 0))
        self.connect((self.blocks_selector_0_1, 3), (self.digital_linear_equalizer_0_0, 0))
        self.connect((self.blocks_skiphead_0, 0), (self.digital_symbol_sync_xx_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.zeromq_pub_sink_0, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.blocks_selector_0_0, 0))
        self.connect((self.digital_constellation_modulator_0_0, 0), (self.blocks_selector_0_0, 1))
        self.connect((self.digital_constellation_modulator_0_0_0, 0), (self.blocks_selector_0_0, 2))
        self.connect((self.digital_constellation_modulator_0_0_0_0, 0), (self.blocks_selector_0_0, 3))
        self.connect((self.digital_costas_loop_cc_0, 0), (self.blocks_selector_0_0_0, 0))
        self.connect((self.digital_costas_loop_cc_0_0, 0), (self.blocks_selector_0_0_0, 1))
        self.connect((self.digital_fll_band_edge_cc_0, 0), (self.blocks_skiphead_0, 0))
        self.connect((self.digital_fll_band_edge_cc_0, 0), (self.qtgui_const_sink_x_1, 0))
        self.connect((self.digital_fll_band_edge_cc_0, 0), (self.qtgui_freq_sink_x_0_0, 2))
        self.connect((self.digital_linear_equalizer_0, 0), (self.blocks_selector_0_0_0, 2))
        self.connect((self.digital_linear_equalizer_0_0, 0), (self.blocks_selector_0_0_0, 3))
        self.connect((self.digital_symbol_sync_xx_0, 0), (self.blocks_selector_0_1, 0))
        self.connect((self.digital_symbol_sync_xx_0, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.iio_pluto_source_0, 0), (self.blocks_multiply_xx_0_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "VariableModulation")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.set_rrc_taps(firdes.root_raised_cosine(self.nfilts, self.nfilts * self.samp_rate, self.samp_rate/self.sps, self.alpha, (15 * self.sps * self.nfilts)))
        self.digital_fll_band_edge_cc_0.set_loop_bandwidth((2*math.pi/self.sps/100))
        self.digital_symbol_sync_xx_0.set_sps(self.sps)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_rrc_taps(firdes.root_raised_cosine(self.nfilts, self.nfilts * self.samp_rate, self.samp_rate/self.sps, self.alpha, (15 * self.sps * self.nfilts)))
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)
        self.analog_sig_source_x_0_0.set_sampling_freq(self.samp_rate)
        self.iio_pluto_sink_0.set_bandwidth((self.samp_rate * 5))
        self.iio_pluto_sink_0.set_samplerate(self.samp_rate)
        self.iio_pluto_source_0.set_samplerate(self.samp_rate)
        self.qtgui_eye_sink_x_0.set_samp_rate(self.samp_rate)
        self.qtgui_freq_sink_x_0_0.set_frequency_range(0, (self.samp_rate / 2))
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)
        self.qtgui_time_sink_x_1.set_samp_rate(self.samp_rate)

    def get_qam32(self):
        return self.qam32

    def set_qam32(self, qam32):
        self.qam32 = qam32

    def get_qam16(self):
        return self.qam16

    def set_qam16(self, qam16):
        self.qam16 = qam16

    def get_nfilts(self):
        return self.nfilts

    def set_nfilts(self, nfilts):
        self.nfilts = nfilts
        self.set_rrc_taps(firdes.root_raised_cosine(self.nfilts, self.nfilts * self.samp_rate, self.samp_rate/self.sps, self.alpha, (15 * self.sps * self.nfilts)))

    def get_alpha(self):
        return self.alpha

    def set_alpha(self, alpha):
        self.alpha = alpha
        self.set_rrc_taps(firdes.root_raised_cosine(self.nfilts, self.nfilts * self.samp_rate, self.samp_rate/self.sps, self.alpha, (15 * self.sps * self.nfilts)))

    def get_tx_attenuation(self):
        return self.tx_attenuation

    def set_tx_attenuation(self, tx_attenuation):
        self.tx_attenuation = tx_attenuation
        self.iio_pluto_sink_0.set_attenuation(0,self.tx_attenuation)

    def get_rx_gain(self):
        return self.rx_gain

    def set_rx_gain(self, rx_gain):
        self.rx_gain = rx_gain
        self.iio_pluto_source_0.set_gain(0, self.rx_gain)

    def get_rrc_taps(self):
        return self.rrc_taps

    def set_rrc_taps(self, rrc_taps):
        self.rrc_taps = rrc_taps

    def get_rnd_mod_fnc(self):
        return self.rnd_mod_fnc

    def set_rnd_mod_fnc(self, rnd_mod_fnc):
        self.rnd_mod_fnc = rnd_mod_fnc
        self.blocks_selector_0_0.set_input_index(self.rnd_mod_fnc)
        self.blocks_selector_0_0_0.set_input_index(self.rnd_mod_fnc)
        self.blocks_selector_0_1.set_output_index(self.rnd_mod_fnc)

    def get_qpsk(self):
        return self.qpsk

    def set_qpsk(self, qpsk):
        self.qpsk = qpsk

    def get_pluto_ip_tx(self):
        return self.pluto_ip_tx

    def set_pluto_ip_tx(self, pluto_ip_tx):
        self.pluto_ip_tx = pluto_ip_tx

    def get_pluto_ip_rx(self):
        return self.pluto_ip_rx

    def set_pluto_ip_rx(self, pluto_ip_rx):
        self.pluto_ip_rx = pluto_ip_rx

    def get_phase_shift_after_costas_loop(self):
        return self.phase_shift_after_costas_loop

    def set_phase_shift_after_costas_loop(self, phase_shift_after_costas_loop):
        self.phase_shift_after_costas_loop = phase_shift_after_costas_loop
        self.blocks_phase_shift_0.set_shift(self.phase_shift_after_costas_loop)

    def get_out_sps(self):
        return self.out_sps

    def set_out_sps(self, out_sps):
        self.out_sps = out_sps

    def get_offset_tx(self):
        return self.offset_tx

    def set_offset_tx(self, offset_tx):
        self.offset_tx = offset_tx
        self.analog_sig_source_x_0.set_frequency(self.offset_tx)

    def get_offset_rx(self):
        return self.offset_rx

    def set_offset_rx(self, offset_rx):
        self.offset_rx = offset_rx
        self.analog_sig_source_x_0_0.set_frequency(self.offset_rx)

    def get_lo_p(self):
        return self.lo_p

    def set_lo_p(self, lo_p):
        self.lo_p = lo_p
        self.iio_pluto_sink_0.set_frequency((self.lo_freq + self.lo_p))

    def get_lo_freq(self):
        return self.lo_freq

    def set_lo_freq(self, lo_freq):
        self.lo_freq = lo_freq
        self.iio_pluto_sink_0.set_frequency((self.lo_freq + self.lo_p))
        self.iio_pluto_source_0.set_frequency(self.lo_freq)

    def get_lms_32(self):
        return self.lms_32

    def set_lms_32(self, lms_32):
        self.lms_32 = lms_32

    def get_lms(self):
        return self.lms

    def set_lms(self, lms):
        self.lms = lms

    def get_costas_loop_bw(self):
        return self.costas_loop_bw

    def set_costas_loop_bw(self, costas_loop_bw):
        self.costas_loop_bw = costas_loop_bw
        self.digital_costas_loop_cc_0.set_loop_bandwidth((2*math.pi*self.costas_loop_bw))
        self.digital_costas_loop_cc_0_0.set_loop_bandwidth((2*math.pi*self.costas_loop_bw))

    def get_const_chooser(self):
        return self.const_chooser

    def set_const_chooser(self, const_chooser):
        self.const_chooser = const_chooser
        self._const_chooser_callback(self.const_chooser)

    def get_buff_size(self):
        return self.buff_size

    def set_buff_size(self, buff_size):
        self.buff_size = buff_size

    def get_bpsk(self):
        return self.bpsk

    def set_bpsk(self, bpsk):
        self.bpsk = bpsk




def main(top_block_cls=VariableModulation, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
