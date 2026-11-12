"""TCQ code rate for the whole pipeline (user 2026-11-11): TCQ_BITS = bits per weight of every quantized Linear (2 = the main rows, default;
1 = the 1-bit rows). Only the trellis rate K changes (bitshift trellis L = 16, V = 2: each step emits two weights and shifts in K * V new
state bits, so K = bits per weight); the 2^9 x 2 FP16 table, 16 x 16 tiles, two-sided rotation, row scales, damping ladder, fine-tune and
evaluation settings are untouched. A rate other than 2 gets its own run / model trees (suffix _w<bits>), so rows of different rates never mix."""
import os

BITS = int(os.environ.get('TCQ_BITS', '2'))
assert BITS in (1, 2, 3, 4), BITS
TAG = '' if BITS == 2 else f'_w{BITS}'
