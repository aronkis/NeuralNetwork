import numpy as np
from gnuradio import gr

class labeled_iq_block(gr.sync_block):
    """Appends modulation label to I/Q vector"""
    
    def __init__(self, label_var=0):
        gr.sync_block.__init__(
            self,
            name='Labeled IQ',
            in_sig=[(np.complex64, 2048)],  # Input: 2048 complex samples
            out_sig=[(np.float32, 4097)]    # Output: 4096 floats + 1 label
        )
        self.label_var = label_var
    
    def set_label_var(self, label_var):
        self.label_var = label_var
    
    def work(self, input_items, output_items):
        # Get label - handle both callable and direct value
        if callable(self.label_var):
            label = float(self.label_var())
        else:
            label = float(self.label_var)
        
        for i in range(len(input_items[0])):
            iq = input_items[0][i]
            # Interleave I/Q: [I0, Q0, I1, Q1, ...]
            output_items[0][i][:4096:2] = iq.real
            output_items[0][i][1:4096:2] = iq.imag
            # Append label as last element
            output_items[0][i][4096] = label
        return len(input_items[0])
