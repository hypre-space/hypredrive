#!/usr/bin/env python3
"""
Post-processing and validation script for lid-driven cavity simulation.

This script provides multiple analysis and visualization capabilities:
- Validation: Extracts centerline velocity profiles from VTK output and compares
  with reference data from validation.pdf (Marchi et al., 2021).
- Visualization: Generates comparison plots and streamline visualizations.

Usage:
    python3 postprocess.py <pvd_file> [--Re <Re>] [--plot] [--save <output>] [-c] [-s]

Options:
    --Re <Re>        Reynolds number for reference data lookup (default: 100)
    --plot           Show comparison plots interactively
    --save <output>  Save plots to file (PNG or PDF)
    -c, --compact    Create a compact single plot with both u and v centerlines
    -s, --streamline Plot streamlines from steady-state result (last time step)
"""

import sys
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import struct

# Reference data from:
# Marchi et al., 2021. "Lid-Driven Square Cavity Flow: A Benchmark Solution With an 8192×8192 Grid"
# Journal of Verification, Validation and Uncertainty Quantification
#
# 1. The paper provides data for Re = 1, 10, 100, 400, 1000, 3200, 5000, 7500, 10000
# 2. u-velocity along vertical centerline (x=0.5) extracted at 28 y-points
# 3. v-velocity along horizontal centerline (y=0.5) extracted at 28 x-points
#
# Format: {Re: {'u_centerline': {'y': [...], 'u': [...]}, 'v_centerline': {'x': [...], 'v': [...]}}}
REFERENCE_DATA = {
    1: {
        'u_centerline': {
            'y': np.array([0.0546875, 0.0625000, 0.0703125, 0.1015625, 0.1250000, 0.1718750, 0.1875000, 0.2500000, 0.2812500, 0.3125000, 0.3750000, 0.4375000, 0.4531250, 0.5000000, 0.5625000, 0.6171875, 0.6250000, 0.6875000, 0.7343750, 0.7500000, 0.8125000, 0.8515625, 0.8750000, 0.9375000, 0.9531250, 0.9609375, 0.9687500, 0.9765625]),
            'u': np.array([-3.4218782900e-02, -3.8527691200e-02, -4.2722831900e-02, -5.8535357400e-02, -6.9584813800e-02, -9.0288520500e-02, -9.6907481300e-02, -1.2259680590e-01, -1.3512985380e-01, -1.4746345200e-01, -1.7106903100e-01, -1.9153729400e-01, -1.9577933400e-01, -2.0519139500e-01, -2.0608614400e-01, -1.8967541500e-01, -1.8557455200e-01, -1.3220097400e-01, -6.2493180000e-02, -3.2438028000e-02, 1.2705351000e-01, 2.6138995560e-01, 3.5522029650e-01, 6.5116984800e-01, 7.3432887600e-01, 7.7705743000e-01, 8.2048300900e-01, 8.6454993500e-01])
        },
        'v_centerline': {
            'x': np.array([0.0625000, 0.0703125, 0.0781250, 0.0937500, 0.1250000, 0.1562500, 0.1875000, 0.2265625, 0.2343750, 0.2500000, 0.3125000, 0.3750000, 0.4375000, 0.5000000, 0.5625000, 0.6250000, 0.6875000, 0.7500000, 0.8046875, 0.8125000, 0.8593750, 0.8750000, 0.9062500, 0.9375000, 0.9453125, 0.9531250, 0.9609375, 0.9687500]),
            'v': np.array([9.4403358600e-02, 1.0396983370e-01, 1.1299865210e-01, 1.2941753470e-01, 1.5562317800e-01, 1.7308471900e-01, 1.8223634300e-01, 1.8307957700e-01, 1.8197068700e-01, 1.7859266600e-01, 1.5180658000e-01, 1.0923251300e-01, 5.7162281000e-02, 6.3676365000e-04, -5.6054827000e-02, -1.0857887500e-01, -1.5176339200e-01, -1.7911489100e-01, -1.8411084200e-01, -1.8306071200e-01, -1.6622156300e-01, -1.5635874900e-01, -1.2997913040e-01, -9.4748076900e-02, -8.4597751870e-02, -7.3930964190e-02, -6.2761535100e-02, -5.1105321300e-02])
        }
    },
    10: {
        'u_centerline': {
            'y': np.array([0.0546875, 0.0625000, 0.0703125, 0.1015625, 0.1250000, 0.1718750, 0.1875000, 0.2500000, 0.2812500, 0.3125000, 0.3750000, 0.4375000, 0.4531250, 0.5000000, 0.5625000, 0.6171875, 0.6250000, 0.6875000, 0.7343750, 0.7500000, 0.8125000, 0.8515625, 0.8750000, 0.9375000, 0.9531250, 0.9609375, 0.9687500, 0.9765625]),
            'u': np.array([-3.4231439100e-02, -3.8542581400e-02, -4.2740107100e-02, -5.8563899400e-02, -6.9623858500e-02, -9.0354373900e-02, -9.6983964000e-02, -1.2272197740e-01, -1.3528027320e-01, -1.4763619130e-01, -1.7126074100e-01, -1.9167701700e-01, -1.9588949100e-01, -2.0516469900e-01, -2.0577014800e-01, -1.8906654900e-01, -1.8492806000e-01, -1.3138918200e-01, -6.1822984000e-02, -3.1879270000e-02, 1.2691210700e-01, 2.6078424437e-01, 3.5443035260e-01, 6.5052927400e-01, 7.3385916000e-01, 7.7668135200e-01, 8.2020081500e-01, 8.6435732500e-01])
        },
        'v_centerline': {
            'x': np.array([0.0625000, 0.0703125, 0.0781250, 0.0937500, 0.1250000, 0.1562500, 0.1875000, 0.2265625, 0.2343750, 0.2500000, 0.3125000, 0.3750000, 0.4375000, 0.5000000, 0.5625000, 0.6250000, 0.6875000, 0.7500000, 0.8046875, 0.8125000, 0.8593750, 0.8750000, 0.9062500, 0.9375000, 0.9453125, 0.9531250, 0.9609375, 0.9687500]),
            'v': np.array([9.2970114500e-02, 1.0230359260e-01, 1.1110074710e-01, 1.2707442900e-01, 1.5254782500e-01, 1.6961114600e-01, 1.7878142700e-01, 1.8024673200e-01, 1.7933522100e-01, 1.7641506700e-01, 1.5205579000e-01, 1.1214774000e-01, 6.2104805000e-02, 6.3603660000e-03, -5.1041711000e-02, -1.0561569700e-01, -1.5162206500e-01, -1.8163352700e-01, -1.8798789500e-01, -1.8702162700e-01, -1.7004160100e-01, -1.5989817770e-01, -1.3268351670e-01, -9.6409944500e-02, -8.5994798300e-02, -7.5070186000e-02, -6.3654529100e-02, -5.1768367000e-02])
        }
    },
    100: {
        'u_centerline': {
            'y': np.array([0.0546875, 0.0625000, 0.0703125, 0.1015625, 0.1250000, 0.1718750, 0.1875000, 0.2500000, 0.2812500, 0.3125000, 0.3750000, 0.4375000, 0.4531250, 0.5000000, 0.5625000, 0.6171875, 0.6250000, 0.6875000, 0.7343750, 0.7500000, 0.8125000, 0.8515625, 0.8750000, 0.9375000, 0.9531250, 0.9609375, 0.9687500, 0.9765625]),
            'u': np.array([-3.7219941200e-02, -4.1975000330e-02, -4.6627217840e-02, -6.4410655000e-02, -7.7125410600e-02, -1.0172901400e-01, -1.0981623266e-01, -1.4193004900e-01, -1.5764857000e-01, -1.7271234900e-01, -1.9847079000e-01, -2.1296230000e-01, -2.1397790000e-01, -2.0914904000e-01, -1.8208050900e-01, -1.3880941200e-01, -1.3125624700e-01, -6.0245577800e-02, 4.1499630000e-03, 2.7874431000e-02, 1.4042528700e-01, 2.3644446200e-01, 3.1055704800e-01, 5.9746664100e-01, 6.9118278600e-01, 7.4071014600e-01, 7.9160880100e-01, 8.4348161600e-01])
        },
        'v_centerline': {
            'x': np.array([0.0625000, 0.0703125, 0.0781250, 0.0937500, 0.1250000, 0.1562500, 0.1875000, 0.2265625, 0.2343750, 0.2500000, 0.3125000, 0.3750000, 0.4375000, 0.5000000, 0.5625000, 0.6250000, 0.6875000, 0.7500000, 0.8046875, 0.8125000, 0.8593750, 0.8750000, 0.9062500, 0.9375000, 0.9453125, 0.9531250, 0.9609375, 0.9687500]),
            'v': np.array([9.4807577000e-02, 1.0359882200e-01, 1.1177707000e-01, 1.2638468400e-01, 1.4924294400e-01, 1.6480432900e-01, 1.7434287000e-01, 1.7935504100e-01, 1.7955928700e-01, 1.7924326800e-01, 1.6913201300e-01, 1.4573016600e-01, 1.0877585000e-01, 5.7536573300e-02, -7.7484560000e-03, -8.4066630000e-02, -1.6301003000e-01, -2.2782720000e-01, -2.5354143000e-01, -2.5376853300e-01, -2.3371248300e-01, -2.1869084640e-01, -1.7715781200e-01, -1.2331822000e-01, -1.0850916200e-01, -9.3339087000e-02, -7.7902060000e-02, -6.2292972000e-02])
        }
    },
    400: {
        'u_centerline': {
            'y': np.array([0.0546875, 0.0625000, 0.1015625, 0.1250000, 0.1718750, 0.1875000, 0.2500000, 0.2812500, 0.3125000, 0.3750000, 0.4375000, 0.4531250, 0.5000000, 0.5625000, 0.6171875, 0.6250000, 0.6875000, 0.7343750, 0.7500000, 0.8125000, 0.8515625, 0.8750000, 0.9375000, 0.9531250, 0.9609375, 0.9687500, 0.9765625]),
            'u': np.array([-8.1804740000e-02, -9.2599240000e-02, -1.4613915000e-01, -1.7874786000e-01, -2.4374610000e-01, -2.6391673000e-01, -3.2122844000e-01, -3.2871715000e-01, -3.2025057000e-01, -2.6630607000e-01, -1.9073044000e-01, -1.7145187800e-01, -1.1505358500e-01, -4.2568959000e-02, 3.0242930000e-02, 2.1019144000e-02, 1.0545582000e-01, 1.6253696000e-01, 1.8130654000e-01, 2.5220340000e-01, 2.9199508000e-01, 3.1682921000e-01, 4.6957996000e-01, 5.6173911000e-01, 6.2021499000e-01, 7.6030384000e-01, 8.2048301800e-01])
        },
        'v_centerline': {
            'x': np.array([0.0625000, 0.0703125, 0.0781250, 0.0937500, 0.1250000, 0.1562500, 0.1875000, 0.2265625, 0.2343750, 0.2500000, 0.3125000, 0.3750000, 0.4375000, 0.5000000, 0.5625000, 0.6250000, 0.6875000, 0.7500000, 0.8046875, 0.8125000, 0.8593750, 0.8750000, 0.9062500, 0.9375000, 0.9453125, 0.9531250, 0.9609375, 0.9687500]),
            'v': np.array([1.8513195000e-01, 1.9880845000e-01, 2.1100116000e-01, 2.3166137000e-01, 2.6225080000e-01, 2.8351460000e-01, 2.9747871000e-01, 3.0382475000e-01, 3.0344817000e-01, 3.0095954000e-01, 2.6831061000e-01, 2.0657123000e-01, 1.3057167610e-01, 5.2058152000e-02, -2.4714380000e-02, -1.0088399000e-01, -1.8210903000e-01, -2.8098993000e-01, -3.8563188000e-01, -4.0004172000e-01, -4.5383122000e-01, -4.4901119000e-01, -3.8983372000e-01, -2.7035478000e-01, -2.3466422000e-01, -1.9809185000e-01, -1.6141253000e-01, -1.2537388000e-01])
        }
    },
    1000: {
        'u_centerline': {
            'y': np.array([0.0546875, 0.0625000, 0.0703125, 0.1015625, 0.1250000, 0.1718750, 0.1875000, 0.2500000, 0.2812500, 0.3125000, 0.3750000, 0.4375000, 0.4531250, 0.5000000, 0.5625000, 0.6171875, 0.6250000, 0.6875000, 0.7343750, 0.7500000, 0.8125000, 0.8515625, 0.8750000, 0.9375000, 0.9531250, 0.9609375, 0.9687500, 0.9765625]),
            'u': np.array([-1.8125321000e-01, -2.0232910000e-01, -2.2292710000e-01, -3.0036880000e-01, -3.4784310000e-01, -3.8856760000e-01, -3.8440780000e-01, -3.1894525000e-01, -2.8042785000e-01, -2.4569306000e-01, -1.8373161000e-01, -1.2341016000e-01, -1.0817529000e-01, -6.2056050000e-02, 5.6167000000e-04, 5.7004360000e-02, 6.5248390000e-02, 1.3357199000e-01, 1.8864358000e-01, 2.0791378000e-01, 2.8844120000e-01, 3.3717630000e-01, 3.6254518050e-01, 4.2293070000e-01, 4.7244910000e-01, 5.1718300000e-01, 5.8036550000e-01, 6.6397203000e-01])
        },
        'v_centerline': {
            'x': np.array([0.0625000, 0.0703125, 0.0781250, 0.0937500, 0.1250000, 0.1562500, 0.1875000, 0.2265625, 0.2343750, 0.2500000, 0.3125000, 0.3750000, 0.4375000, 0.5000000, 0.5625000, 0.6250000, 0.6875000, 0.7500000, 0.8046875, 0.8125000, 0.8593750, 0.8750000, 0.9062500, 0.9375000, 0.9453125, 0.9531250, 0.9609375, 0.9687500]),
            'v': np.array([2.8070420000e-01, 2.9629220000e-01, 3.0994930000e-01, 3.3297680000e-01, 3.6504000000e-01, 3.7691570000e-01, 3.6785120000e-01, 3.3403190000e-01, 3.2538660000e-01, 3.0710340000e-01, 2.3126780000e-01, 1.6056381000e-01, 9.2969110000e-02, 2.5799473000e-02, -4.1840460000e-02, -1.1079783000e-01, -1.8167907000e-01, -2.5338060000e-01, -3.2019540000e-01, -3.3156580600e-01, -4.2638920000e-01, -4.6777410000e-01, -5.2641550000e-01, -4.5615050000e-01, -4.1029230000e-01, -3.5513120000e-01, -2.9337860000e-01, -2.2834070000e-01])
        }
    },
    3200: {
        'u_centerline': {
            'y': np.array([0.0546875, 0.0625000, 0.0703125, 0.1015625, 0.1250000, 0.1718750, 0.1875000, 0.2500000, 0.2812500, 0.3125000, 0.3750000, 0.4375000, 0.4531250, 0.5000000, 0.5625000, 0.6171875, 0.6250000, 0.6875000, 0.7343750, 0.7500000, 0.8125000, 0.8515625, 0.8750000, 0.9375000, 0.9531250, 0.9609375, 0.9687500, 0.9765625]),
            'u': np.array([-3.5649980000e-01, -3.8541700000e-01, -4.0830950000e-01, -4.3291490000e-01, -4.0507600000e-01, -3.4555930000e-01, -3.3035700000e-01, -2.7213020000e-01, -2.4261590000e-01, -2.1317020000e-01, -1.5452860000e-01, -9.5911600000e-02, -8.1215760000e-02, -3.6897630000e-02, 2.3142090000e-02, 7.7207200000e-02, 8.5092500000e-02, 1.5012510000e-01, 2.0181400000e-01, 2.1974200000e-01, 2.9579230000e-01, 3.4818810000e-01, 3.8200910000e-01, 4.5739420000e-01, 4.6179860000e-01, 4.6541440000e-01, 4.8098780000e-01, 5.2771960000e-01])
        },
        'v_centerline': {
            'x': np.array([0.0625000, 0.0703125, 0.0781250, 0.0937500, 0.1250000, 0.1562500, 0.1875000, 0.2265625, 0.2343750, 0.2500000, 0.3125000, 0.3750000, 0.4375000, 0.5000000, 0.5625000, 0.6250000, 0.6875000, 0.7500000, 0.8046875, 0.8593750, 0.8750000, 0.9062500, 0.9375000, 0.9453125, 0.9531250, 0.9609375, 0.9687500]),
            'v': np.array([3.9614210000e-01, 4.1128500000e-01, 4.2245270000e-01, 4.3274360000e-01, 4.1552450000e-01, 3.7724190000e-01, 3.3951980000e-01, 2.9653480000e-01, 2.8815110000e-01, 2.7145680000e-01, 2.0551420000e-01, 1.4097670000e-01, 7.7418240000e-02, 1.4261810000e-02, -4.9180580000e-02, -1.1370930000e-01, -1.8020650000e-01, -2.4960530000e-01, -3.1367290000e-01, -3.7875020000e-01, -3.9627010000e-01, -4.4350440000e-01, -5.4960990000e-01, -5.6722940000e-01, -5.6100500000e-01, -5.1977700000e-01, -4.3973060000e-01])
        }
    },
    5000: {
        'u_centerline': {
            'y': np.array([0.0546875, 0.0625000, 0.0703125, 0.1015625, 0.1250000, 0.1718750, 0.1875000, 0.2500000, 0.2812500, 0.3125000, 0.3750000, 0.4375000, 0.4531250, 0.5000000, 0.5625000, 0.6171875, 0.6250000, 0.6875000, 0.7343750, 0.7500000, 0.8125000, 0.8515625, 0.8750000, 0.9375000, 0.9531250, 0.9609375, 0.9687500, 0.9765625]),
            'u': np.array([-4.1684086700e-01, -4.3682673820e-01, -4.4619700000e-01, -4.1708030000e-01, -3.8395600000e-01, -3.3835443400e-01, -3.2391976500e-01, -2.6534848900e-01, -2.3618188300e-01, -2.0707752800e-01, -1.4894519750e-01, -9.0734500000e-02, -7.6133600100e-02, -3.2088410000e-02, 2.7571311700e-02, 8.1244532600e-02, 8.9064600000e-02, 1.5349480000e-01, 2.0459200000e-01, 2.2230067105e-01, 2.9733270000e-01, 3.4842118040e-01, 3.8127736600e-01, 4.6999944000e-01, 4.7851900000e-01, 4.7792665700e-01, 4.7825767600e-01, 4.9683530000e-01])
        },
        'v_centerline': {
            'x': np.array([0.0625000, 0.0703125, 0.0781250, 0.0937500, 0.1250000, 0.1562500, 0.1875000, 0.2265625, 0.2343750, 0.2500000, 0.3125000, 0.3750000, 0.4375000, 0.5000000, 0.5625000, 0.6250000, 0.6875000, 0.7500000, 0.8046875, 0.8125000, 0.8593750, 0.8750000, 0.9062500, 0.9375000, 0.9453125, 0.9531250, 0.9609375, 0.9687500]),
            'v': np.array([4.3434600000e-01, 4.4378000000e-01, 4.4740300000e-01, 4.4091000000e-01, 4.0334390000e-01, 3.6540760000e-01, 3.3117360000e-01, 2.8941810000e-01, 2.8114620000e-01, 2.6469220000e-01, 1.9995260000e-01, 1.3652600000e-01, 7.3956200000e-02, 1.1732330000e-02, -5.0775480000e-02, -1.1431120000e-01, -1.7970700000e-01, -2.4784770000e-01, -3.1042300000e-01, -3.1964020000e-01, -3.7651830000e-01, -3.9534880000e-01, -4.3025850000e-01, -5.0553530000e-01, -5.4102900000e-01, -5.7033700000e-01, -5.7156300000e-01, -5.2133200000e-01])
        }
    },
    7500: {
        'u_centerline': {
            'y': np.array([0.0546875, 0.0625000, 0.0703125, 0.1015625, 0.1250000, 0.1718750, 0.1875000, 0.2500000, 0.2812500, 0.3125000, 0.3750000, 0.4375000, 0.4531250, 0.5000000, 0.5625000, 0.6171875, 0.6250000, 0.6875000, 0.7343750, 0.7500000, 0.8125000, 0.8515625, 0.8750000, 0.9375000, 0.9531250, 0.9609375, 0.9687500, 0.9765625]),
            'u': np.array([-4.5128700000e-01, -4.5475900000e-01, -4.4806800000e-01, -4.0013600000e-01, -3.7637900000e-01, -3.3344500000e-01, -3.1887900000e-01, -2.6093140000e-01, -2.3202690000e-01, -2.0313330000e-01, -1.4533130000e-01, -8.7368800000e-02, -7.2820600000e-02, -2.8930910000e-02, 3.0509500000e-02, 8.3939500000e-02, 9.1720300000e-02, 1.5574060000e-01, 2.0640950000e-01, 2.2394270000e-01, 2.9815900000e-01, 3.4865100000e-01, 3.8080600000e-01, 4.7499400000e-01, 4.9076700000e-01, 4.9250900000e-01, 4.8918100000e-01, 4.8856600000e-01])
        },
        'v_centerline': {
            'x': np.array([0.0625000, 0.0703125, 0.0781250, 0.1250000, 0.1562500, 0.1875000, 0.2265625, 0.2343750, 0.2500000, 0.3125000, 0.3750000, 0.4375000, 0.5000000, 0.5625000, 0.6250000, 0.6875000, 0.7500000, 0.8046875, 0.8125000, 0.8593750, 0.8750000, 0.9062500, 0.9375000, 0.9453125, 0.9531250, 0.9609375, 0.9687500]),
            'v': np.array([4.5739100000e-01, 4.5765600000e-01, 4.5243700000e-01, 3.9357600000e-01, 3.5893800000e-01, 3.2530400000e-01, 2.8408710000e-01, 2.7594780000e-01, 2.5975670000e-01, 1.9595300000e-01, 1.3332130000e-01, 7.1452300000e-02, 9.8851600000e-03, -5.1958100000e-02, -1.1477380000e-01, -1.7935150000e-01, -2.4653530000e-01, -3.0811700000e-01, -3.1715800000e-01, -3.7293800000e-01, -3.9219900000e-01, -4.2939700000e-01, -4.7193500000e-01, -4.9914800000e-01, -5.3826200000e-01, -5.7439300000e-01, -5.7066500000e-01])
        }
    },
    10000: {
        'u_centerline': {
            'y': np.array([0.0546875, 0.0625000, 0.0703125, 0.1015625, 0.1250000, 0.1718750, 0.1875000, 0.2500000, 0.2812500, 0.3125000, 0.3750000, 0.4375000, 0.4531250, 0.5000000, 0.5625000, 0.6171875, 0.6250000, 0.6875000, 0.7343750, 0.7500000, 0.8125000, 0.8515625, 0.8750000, 0.9375000, 0.9531250, 0.9609375, 0.9687500, 0.9765625]),
            'u': np.array([-4.5865000000e-01, -4.5041600000e-01, -4.3689600000e-01, -3.9466700000e-01, -3.7366100000e-01, -3.3045300000e-01, -3.1605200000e-01, -2.5855600000e-01, -2.2979600000e-01, -2.0101840000e-01, -1.4338550000e-01, -8.5540600000e-02, -7.1017300000e-02, -2.7199050000e-02, 3.2135100000e-02, 8.5435600000e-02, 9.3193500000e-02, 1.5697300000e-01, 2.0737600000e-01, 2.2480100000e-01, 2.9847500000e-01, 3.4856500000e-01, 3.8047500000e-01, 4.7527900000e-01, 4.9603000000e-01, 5.0088400000e-01, 4.9926100000e-01, 4.9246700000e-01])
        },
        'v_centerline': {
            'x': np.array([0.0625000, 0.0703125, 0.0781250, 0.0937500, 0.1250000, 0.1562500, 0.1875000, 0.2265625, 0.2343750, 0.2500000, 0.3125000, 0.3750000, 0.4375000, 0.5000000, 0.5625000, 0.6250000, 0.6875000, 0.7500000, 0.8046875, 0.8125000, 0.8593750, 0.8750000, 0.9062500, 0.9375000, 0.9453125, 0.9531250, 0.9609375, 0.9687500]),
            'v': np.array([4.6395400000e-01, 4.5761800000e-01, 4.4786000000e-01, 4.2626700000e-01, 3.8916500000e-01, 3.5506400000e-01, 3.2168100000e-01, 2.8089700000e-01, 2.7283800000e-01, 2.5680100000e-01, 1.9355340000e-01, 1.3139240000e-01, 6.9935700000e-02, 8.7566000000e-03, -5.2688700000e-02, -1.1506580000e-01, -1.7913770000e-01, -2.4572200000e-01, -3.0667600000e-01, -3.1561900000e-01, -3.7062700000e-01, -3.8957000000e-01, -4.2801200000e-01, -4.6206700000e-01, -4.7794400000e-01, -5.0839000000e-01, -5.5311500000e-01, -5.8215600000e-01])
        }
    }
}

def read_vtk_file_regex(raw_content, xml_content):
    """Fallback: Extract VTK data using regex when XML parsing fails."""
    coords = {}
    is_binary = False
    binary_data = None

    # Check if binary format
    if '<AppendedData' in xml_content:
        is_binary = True
        appended_start = raw_content.find(b'<AppendedData')
        if appended_start != -1:
            underscore_pos = raw_content.find(b'_', appended_start)
            if underscore_pos != -1:
                binary_data = raw_content[underscore_pos + 1:]

    # Extract coordinate arrays using regex
    for name in ['x', 'y', 'z']:
        # Look for DataArray with Name="x" (or y, z)
        pattern = rf'<DataArray[^>]*Name="{name}"[^>]*>(.*?)</DataArray>'
        match = re.search(pattern, xml_content, re.DOTALL)
        if match:
            format_attr_match = re.search(rf'format="([^"]*)"', match.group(0))
            format_attr = format_attr_match.group(1) if format_attr_match else 'ascii'

            if format_attr == 'appended' and is_binary:
                # Extract offset
                offset_match = re.search(r'offset="(\d+)"', match.group(0))
                if offset_match:
                    offset = int(offset_match.group(1))
                    type_match = re.search(r'type="([^"]*)"', match.group(0))
                    data_type = type_match.group(1) if type_match else 'Float64'

                    # Create a minimal data_array-like object
                    class FakeDataArray:
                        def __init__(self, dtype, ncomp):
                            self.dtype = dtype
                            self.ncomp = ncomp
                        def get(self, key, default):
                            if key == 'type': return self.dtype
                            if key == 'NumberOfComponents': return str(self.ncomp)
                            return default

                    fake_array = FakeDataArray(data_type, 1)
                    arr, _ = read_binary_data_array(fake_array, binary_data, offset)
                    coords[name] = arr
            else:
                # ASCII format
                text = match.group(1).strip()
                if text:
                    values = re.split(r'\s+', text)
                    coords[name] = np.array([float(v) for v in values if v])

    # Extract velocity
    pattern = r'<DataArray[^>]*Name="velocity"[^>]*>(.*?)</DataArray>'
    match = re.search(pattern, xml_content, re.DOTALL)
    if not match:
        # Try with format="appended"
        pattern = r'<DataArray[^>]*Name="velocity"[^>]*offset="(\d+)"[^>]*>'
        match = re.search(pattern, xml_content)
        if match and is_binary:
            offset = int(match.group(1))
            type_match = re.search(r'type="([^"]*)"', match.group(0))
            data_type = type_match.group(1) if type_match else 'Float64'

            class FakeDataArray:
                def __init__(self, dtype, ncomp):
                    self.dtype = dtype
                    self.ncomp = ncomp
                def get(self, key, default):
                    if key == 'type': return self.dtype
                    if key == 'NumberOfComponents': return str(self.ncomp)
                    return default

            fake_array = FakeDataArray(data_type, 3)
            velocity_data, _ = read_binary_data_array(fake_array, binary_data, offset)
        else:
            raise ValueError("Could not find velocity data")
    else:
        # ASCII format
        format_attr_match = re.search(r'format="([^"]*)"', match.group(0))
        format_attr = format_attr_match.group(1) if format_attr_match else 'ascii'
        if format_attr == 'appended':
            raise ValueError("Binary format detected but offset not found")
        text = match.group(1).strip()
        if text:
            values = re.split(r'\s+', text)
            float_values = [float(v) for v in values if v]
            npts = len(float_values) // 3
            velocity_data = np.array(float_values).reshape(npts, 3)

    if 'x' not in coords or 'y' not in coords:
        raise ValueError("Could not find coordinate data")

    nx = len(coords['x'])
    ny = len(coords['y'])

    u = velocity_data[:, 0].reshape(ny, nx)
    v = velocity_data[:, 1].reshape(ny, nx)

    return coords, u, v


def read_binary_data_array(data_array, binary_data, offset):
    """Read a binary data array from the appended data section."""

    # Get number of components and type
    n_components = int(data_array.get('NumberOfComponents', '1'))
    data_type = data_array.get('type', 'Float64')

    # Map VTK types to struct format and numpy dtype
    type_map = {
        'Float64': ('d', np.float64, 8),
        'Float32': ('f', np.float32, 4),
        'Int32': ('i', np.int32, 4),
        'Int64': ('q', np.int64, 8),
    }

    if data_type not in type_map:
        raise ValueError(f"Unsupported data type: {data_type}")

    struct_fmt, np_dtype, type_size = type_map[data_type]

    # Read size (4 bytes, int32)
    if offset + 4 > len(binary_data):
        raise ValueError("Binary data too short")
    size_bytes = binary_data[offset:offset+4]
    size = struct.unpack('i', size_bytes)[0]
    offset += 4

    # Read actual data
    n_elements = size // (type_size * n_components)
    if offset + size > len(binary_data):
        raise ValueError("Binary data too short for array")

    data_bytes = binary_data[offset:offset+size]
    offset += size

    # Convert to numpy array
    if n_components == 1:
        arr = np.frombuffer(data_bytes, dtype=np_dtype, count=n_elements)
    else:
        arr = np.frombuffer(data_bytes, dtype=np_dtype, count=n_elements * n_components)
        arr = arr.reshape(n_elements, n_components)

    return arr, offset


def read_vtk_file(vtk_path):
    """Read VTK RectilinearGrid file and extract velocity data (supports both ASCII and binary)."""
    # Read entire file as binary
    with open(vtk_path, 'rb') as f:
        raw_content = f.read()

    # Find XML section - files may not have closing tags if binary data follows
    # Look for </VTKFile> first (complete XML)
    xml_end = raw_content.find(b'</VTKFile>')
    if xml_end != -1:
        # Include the closing tag
        xml_end = raw_content.find(b'>', xml_end) + 1
        xml_content = raw_content[:xml_end].decode('utf-8', errors='ignore')
    else:
        # No closing tag - find start of binary data and reconstruct XML
        appended_start = raw_content.find(b'<AppendedData')
        if appended_start != -1:
            # Find the pattern: <AppendedData encoding="raw">\n   _
            # The underscore marks the start of binary data
            tag_end = raw_content.find(b'>', appended_start) + 1
            # Look for the underscore after the tag (with whitespace before it)
            underscore_pos = raw_content.find(b'_', tag_end)
            if underscore_pos != -1:
                # XML ends at the tag end, binary starts after underscore
                xml_content = raw_content[:tag_end].decode('utf-8', errors='ignore')
                # Add closing tags to make valid XML
                xml_content += '\n  </AppendedData>\n</VTKFile>'
                xml_end = underscore_pos  # For binary data location
            else:
                # Fallback
                xml_end = tag_end
                xml_content = raw_content[:xml_end].decode('utf-8', errors='ignore')
                xml_content += '\n  </AppendedData>\n</VTKFile>'
        else:
            # Pure ASCII file
            xml_end = len(raw_content)
            xml_content = raw_content[:xml_end].decode('utf-8', errors='ignore')

    # Clean XML content - remove any binary contamination
    # Keep only printable ASCII and common XML characters, but be more aggressive
    xml_content_clean = ''
    for c in xml_content:
        if ord(c) < 128:
            if c.isprintable() or c in '\n\r\t ':
                xml_content_clean += c
            elif c == '<' or c == '>':
                xml_content_clean += c

    # Parse XML - try multiple strategies
    root = None
    try:
        root = ET.fromstring(xml_content_clean)
    except ET.ParseError:
        # Strategy 1: Extract up to AppendedData and add closing tags
        xml_match = re.search(r'<VTKFile.*?<AppendedData[^>]*>', xml_content_clean, re.DOTALL)
        if xml_match:
            try:
                xml_part = xml_match.group(0) + '\n  </AppendedData>\n</VTKFile>'
                root = ET.fromstring(xml_part)
            except ET.ParseError:
                pass

        # Strategy 2: Try to find complete RectilinearGrid section
        if root is None:
            xml_match = re.search(r'<RectilinearGrid.*?</RectilinearGrid>', xml_content_clean, re.DOTALL)
            if xml_match:
                try:
                    # Wrap in minimal VTKFile structure
                    xml_part = '<VTKFile type="RectilinearGrid">' + xml_match.group(0) + '</VTKFile>'
                    root = ET.fromstring(xml_part)
                except ET.ParseError:
                    pass

        # Strategy 3: Use regex to extract metadata directly (fallback)
        if root is None:
            # Extract data array info using regex
            return read_vtk_file_regex(raw_content, xml_content_clean)

    if root is None:
        raise ValueError(f"Could not parse XML from {vtk_path}")

    # Check if binary format is used
    is_binary = False
    binary_data = None

    appended_data = root.find('.//AppendedData')
    if appended_data is not None:
        is_binary = True
        # Find start of binary data (after "_" marker)
        appended_start = raw_content.find(b'<AppendedData')
        if appended_start != -1:
            # Find the "_" marker that precedes binary data
            underscore_pos = raw_content.find(b'_', appended_start)
            if underscore_pos != -1:
                # Binary data starts right after the underscore
                binary_start = underscore_pos + 1
                binary_data = raw_content[binary_start:]

    # Extract coordinates
    coords = {}
    for name in ['x', 'y', 'z']:
        data_array = root.find(f'.//DataArray[@Name="{name}"]')
        if data_array is not None:
            format_attr = data_array.get('format', 'ascii')

            if format_attr == 'appended' and is_binary:
                # Read from binary data
                offset_str = data_array.get('offset', '0')
                try:
                    offset = int(offset_str)
                    arr, _ = read_binary_data_array(data_array, binary_data, offset)
                    coords[name] = arr
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Could not read binary coordinate data for {name}: {e}")
            else:
                # Read from ASCII text
                text = data_array.text
                if text:
                    values = re.split(r'\s+', text.strip())
                    coords[name] = np.array([float(v) for v in values if v])

    # Extract velocity data
    velocity_data = None
    vel_array = root.find('.//DataArray[@Name="velocity"]')
    if vel_array is not None:
        format_attr = vel_array.get('format', 'ascii')

        if format_attr == 'appended' and is_binary:
            # Read from binary data
            offset_str = vel_array.get('offset', '0')
            try:
                offset = int(offset_str)
                velocity_data, _ = read_binary_data_array(vel_array, binary_data, offset)
            except (ValueError, IndexError) as e:
                raise ValueError(f"Could not read binary velocity data: {e}")
        else:
            # Read from ASCII text
            text = vel_array.text
            if text:
                values = re.split(r'\s+', text.strip())
                float_values = [float(v) for v in values if v]
                npts = len(float_values) // 3
                velocity_data = np.array(float_values).reshape(npts, 3)

    if velocity_data is None:
        raise ValueError(f"Could not find velocity data in {vtk_path}")

    if 'x' not in coords or 'y' not in coords:
        raise ValueError(f"Could not find coordinate data in {vtk_path}")

    # Create grid
    nx = len(coords['x'])
    ny = len(coords['y'])
    npts_expected = nx * ny

    # Verify we have the right number of points
    if velocity_data.shape[0] != npts_expected:
        raise ValueError(f"Mismatch: expected {npts_expected} points (nx={nx}, ny={ny}), "
                        f"but got {velocity_data.shape[0]} points")

    # Reshape velocity to match grid
    # VTK RectilinearGrid ordering: x varies fastest, then y
    # Point k corresponds to (x[k % nx], y[k // nx])
    # So we reshape to (ny, nx) where u[j, i] is u at (x[i], y[j])
    if velocity_data.ndim == 1:
        # Single component - shouldn't happen for velocity, but handle it
        npts = len(velocity_data)
        velocity_data = velocity_data.reshape(npts, 1)

    if velocity_data.shape[1] >= 2:
        # Extract u and v components and reshape to (ny, nx)
        # velocity_data is (npts, 3) where npts = nx * ny
        u_flat = velocity_data[:, 0]  # Shape: (npts,)
        v_flat = velocity_data[:, 1]  # Shape: (npts,)

        # Reshape: (ny, nx) where u[j, i] is u at (x[i], y[j])
        # This means: u[0, :] is u for all x at y[0] (bottom)
        #            u[ny-1, :] is u for all x at y[ny-1] (top)
        u = u_flat.reshape(ny, nx)
        v = v_flat.reshape(ny, nx)
    else:
        raise ValueError(f"Velocity data has insufficient components: {velocity_data.shape}")

    return coords, u, v


def read_pvd_collection(pvd_path):
    """Read PVD collection file and return list of VTK files."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(pvd_path)
    root = tree.getroot()

    pvd_dir = Path(pvd_path).parent
    vtk_files = []

    for dataset in root.findall('.//DataSet'):
        file_path = dataset.get('file')
        if file_path:
            full_path = pvd_dir / file_path
            if full_path.exists():
                vtk_files.append(str(full_path))

    return sorted(vtk_files)


def combine_rank_data(rank_files):
    """Combine VTK data from multiple ranks into a single global grid.

    Each rank file contains a portion of the domain. We combine them to get
    the full domain data.

    Returns:
        global_x: Global x coordinates (sorted, unique)
        global_y: Global y coordinates (sorted, unique)
        global_u: Global u-velocity array (ny, nx)
        global_v: Global v-velocity array (ny, nx)
    """
    # Read all rank files and collect their data
    rank_data = []
    for rank, vtk_file in rank_files:
        try:
            coords, u, v = read_vtk_file(vtk_file)
            rank_data.append({
                'x': coords['x'],
                'y': coords['y'],
                'u': u,
                'v': v,
                'file': vtk_file
            })
        except (ValueError, IndexError, struct.error) as e:
            print(f"Warning: Could not read {vtk_file}: {e}")
            continue

    if not rank_data:
        return None, None, None, None

    # If only one rank, return its data directly
    if len(rank_data) == 1:
        d = rank_data[0]
        return d['x'], d['y'], d['u'], d['v']

    # Collect all unique coordinates from all ranks
    all_x = set()
    all_y = set()
    for d in rank_data:
        all_x.update(d['x'].tolist())
        all_y.update(d['y'].tolist())

    global_x = np.array(sorted(all_x))
    global_y = np.array(sorted(all_y))
    nx_global = len(global_x)
    ny_global = len(global_y)

    # Create global arrays (initialized to NaN to detect missing data)
    global_u = np.full((ny_global, nx_global), np.nan)
    global_v = np.full((ny_global, nx_global), np.nan)

    # Create coordinate to index mappings
    x_to_idx = {x: i for i, x in enumerate(global_x)}
    y_to_idx = {y: j for j, y in enumerate(global_y)}

    # Fill in data from each rank
    for d in rank_data:
        local_x = d['x']
        local_y = d['y']
        local_u = d['u']
        local_v = d['v']

        for j_local, y_val in enumerate(local_y):
            j_global = y_to_idx.get(y_val)
            if j_global is None:
                # Find closest y
                j_global = np.argmin(np.abs(global_y - y_val))

            for i_local, x_val in enumerate(local_x):
                i_global = x_to_idx.get(x_val)
                if i_global is None:
                    # Find closest x
                    i_global = np.argmin(np.abs(global_x - x_val))

                # If not NaN, we have overlap - average or overwrite
                if np.isnan(global_u[j_global, i_global]):
                    global_u[j_global, i_global] = local_u[j_local, i_local]
                    global_v[j_global, i_global] = local_v[j_local, i_local]
                else:
                    # Average overlapping values (ghost overlap regions)
                    global_u[j_global, i_global] = 0.5 * (global_u[j_global, i_global] + local_u[j_local, i_local])
                    global_v[j_global, i_global] = 0.5 * (global_v[j_global, i_global] + local_v[j_local, i_local])

    # Check for missing data
    nan_count = np.sum(np.isnan(global_u))
    if nan_count > 0:
        print(f"Warning: {nan_count} points have missing data after combining ranks")

    return global_x, global_y, global_u, global_v


def extract_centerline_velocities(vtk_files):
    """Extract u-velocity at x=0.5 and v-velocity at y=0.5 from all VTK files.

    For parallel runs, multiple VTK files exist per timestep (one per rank).
    We combine data from all ranks to get the complete domain.
    """
    all_u_centerline = []
    all_v_centerline = []

    # Group files by timestep (assuming filename pattern: step_XXXXX_RANK.vtr)
    timestep_files = {}
    for vtk_file in vtk_files:
        match = re.search(r'step_(\d+)_(\d+)\.vtr', vtk_file)
        if match:
            timestep = int(match.group(1))
            rank = int(match.group(2))
            if timestep not in timestep_files:
                timestep_files[timestep] = []
            timestep_files[timestep].append((rank, vtk_file))

    # Process each timestep, combining data from all ranks
    for timestep in sorted(timestep_files.keys()):
        rank_files = sorted(timestep_files[timestep])
        if not rank_files:
            continue

        # Combine data from all ranks
        x_coords, y_coords, u, v = combine_rank_data(rank_files)
        if x_coords is None:
            continue

        # Ensure coordinates are in increasing order (should be after combine_rank_data)
        if len(x_coords) > 1 and x_coords[0] > x_coords[-1]:
            x_coords = x_coords[::-1]
            u = u[:, ::-1]
            v = v[:, ::-1]
        if len(y_coords) > 1 and y_coords[0] > y_coords[-1]:
            y_coords = y_coords[::-1]
            u = u[::-1, :]
            v = v[::-1, :]

        # Find index closest to x=0.5
        x_center_idx = np.argmin(np.abs(x_coords - 0.5))
        # Find index closest to y=0.5
        y_center_idx = np.argmin(np.abs(y_coords - 0.5))

        # Extract u-velocity along vertical centerline (x=0.5)
        # u is shape (ny, nx) where u[j, i] is u at (x[i], y[j])
        # u[:, x_center_idx] gives u for all y values at x=0.5
        u_centerline = u[:, x_center_idx].copy()
        # Extract v-velocity along horizontal centerline (y=0.5)
        # v[y_center_idx, :] gives v for all x values at y=0.5
        v_centerline = v[y_center_idx, :].copy()

        all_u_centerline.append((y_coords.copy(), u_centerline))
        all_v_centerline.append((x_coords.copy(), v_centerline))

    if len(all_u_centerline) == 0:
        raise ValueError("Could not extract velocity data from any VTK files")

    # Use the last timestep (steady-state)
    return all_u_centerline[-1], all_v_centerline[-1]


def compare_with_reference(sim_y, sim_u, sim_x, sim_v, ref_data, Re):
    """Compare simulation results with reference data."""
    if Re not in ref_data:
        print(f"Warning: No reference data available for Re={Re}")
        return None, None

    ref = ref_data[Re]

    # Interpolate simulation data to reference grid points
    try:
        from scipy.interpolate import interp1d
    except ImportError:
        print("Error: scipy is required for interpolation. Install with: pip install scipy")
        return None, None

    # u-velocity comparison
    u_interp = interp1d(sim_y, sim_u, kind='linear', bounds_error=False, fill_value='extrapolate')
    u_sim_at_ref = u_interp(ref['u_centerline']['y'])

    # v-velocity comparison
    v_interp = interp1d(sim_x, sim_v, kind='linear', bounds_error=False, fill_value='extrapolate')
    v_sim_at_ref = v_interp(ref['v_centerline']['x'])

    # Compute errors
    u_error = np.abs(u_sim_at_ref - ref['u_centerline']['u'])
    v_error = np.abs(v_sim_at_ref - ref['v_centerline']['v'])

    u_max_error = np.max(u_error)
    v_max_error = np.max(v_error)
    u_rmse = np.sqrt(np.mean(u_error**2))
    v_rmse = np.sqrt(np.mean(v_error**2))

    return {
        'u': {
            'y_ref': ref['u_centerline']['y'],
            'u_ref': ref['u_centerline']['u'],
            'u_sim': u_sim_at_ref,
            'error': u_error,
            'max_error': u_max_error,
            'rmse': u_rmse
        },
        'v': {
            'x_ref': ref['v_centerline']['x'],
            'v_ref': ref['v_centerline']['v'],
            'v_sim': v_sim_at_ref,
            'error': v_error,
            'max_error': v_max_error,
            'rmse': v_rmse
        }
    }


def extract_full_velocity_field(vtk_files):
    """Extract full velocity field from the last timestep for streamline plotting.

    Returns:
        x_coords: Global x coordinates (1D array)
        y_coords: Global y coordinates (1D array)
        u_field: u-velocity field (ny, nx)
        v_field: v-velocity field (ny, nx)
    """
    # Group files by timestep (assuming filename pattern: step_XXXXX_RANK.vtr)
    timestep_files = {}
    for vtk_file in vtk_files:
        match = re.search(r'step_(\d+)_(\d+)\.vtr', vtk_file)
        if match:
            timestep = int(match.group(1))
            rank = int(match.group(2))
            if timestep not in timestep_files:
                timestep_files[timestep] = []
            timestep_files[timestep].append((rank, vtk_file))

    if not timestep_files:
        return None, None, None, None

    # Get the last timestep (steady-state)
    last_timestep = max(timestep_files.keys())
    rank_files = sorted(timestep_files[last_timestep])

    # Combine data from all ranks
    x_coords, y_coords, u_field, v_field = combine_rank_data(rank_files)

    return x_coords, y_coords, u_field, v_field


def plot_streamlines(vtk_files, Re, save_path=None):
    """Plot streamlines from the steady-state velocity field.

    Args:
        vtk_files: List of VTK file paths
        Re: Reynolds number (for title)
        save_path: Optional path to save the plot (will append '_streamlines' if other plots also saved)
    """
    print("Extracting full velocity field for streamline plot...")
    x_coords, y_coords, u_field, v_field = extract_full_velocity_field(vtk_files)

    if x_coords is None:
        print("Error: Could not extract velocity field")
        return

    # Create meshgrid for streamplot
    X, Y = np.meshgrid(x_coords, y_coords)

    # Create figure - adjust size for horizontal colorbar at bottom
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Compute velocity magnitude for colormap
    speed = np.sqrt(u_field**2 + v_field**2)

    # Plot streamlines with uniform density for even spatial distribution
    # The density parameter controls spacing between streamlines throughout the domain
    # This ensures streamlines maintain even spacing as they curve, avoiding clustering
    # Lower density = more spacing = fewer but more spread out streamlines
    stream = ax.streamplot(X, Y, u_field, v_field,
                          density=1.1, color=speed, cmap='plasma',
                          linewidth=0.8, arrowsize=1.2, arrowstyle='->',
                          minlength=0.05, maxlength=10.0,
                          broken_streamlines=False)

    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add subtle grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # Add boundary box to emphasize domain
    from matplotlib.patches import Rectangle
    rect = Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black',
                    facecolor='none', zorder=10)
    ax.add_patch(rect)

    # Remove x and y labels, keep only title and ticks
    ax.tick_params(axis='both', labelsize=14)
    ax.set_title(f'Streamlines - Lid-Driven Cavity (Re={Re})',
                fontsize=17, fontweight='bold', pad=15)

    # Add horizontal colorbar at the bottom - smaller than plot area
    # Use shrink to make it narrower than the plot, and reduce pad to minimize spacing
    cbar = plt.colorbar(stream.lines, ax=ax, orientation='horizontal',
                        pad=0.08, aspect=25, shrink=0.55, location='bottom')
    cbar.set_label('Velocity Magnitude', fontsize=16, fontweight='bold', labelpad=8)
    cbar.ax.tick_params(labelsize=14)

    # Adjust layout to remove bottom whitespace - tight layout after colorbar
    plt.tight_layout(rect=[0, 0, 1, 1])

    if save_path:
        # If save_path is provided, modify it for streamline plot
        if save_path.endswith('.png'):
            stream_path = save_path.replace('.png', f'_Re{Re}_streamlines.png')
        elif save_path.endswith('.pdf'):
            stream_path = save_path.replace('.pdf', f'_Re{Re}_streamlines.pdf')
        else:
            stream_path = save_path + f'_Re{Re}_streamlines.png'
        plt.savefig(stream_path, dpi=200, bbox_inches='tight')
        print(f"Streamline plot saved to {stream_path}")
    else:
        plt.show()


def plot_comparison(comparison, Re, save_path=None, compact=False):
    """Plot comparison between simulation and reference data.

    Args:
        comparison: Dictionary with comparison data
        Re: Reynolds number
        save_path: Optional path to save the plot
        compact: If True, create a single compact plot with both centerlines
    """
    if compact:
        # Compact mode: single overlaid plot with dual x and y axes
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))

        # u-velocity profile: u vs y
        # Uses bottom x-axis (u-velocity) and left y-axis (y)
        line1 = ax.plot(comparison['u']['u_ref'], comparison['u']['y_ref'], 'o-',
                       label='u (Ref.)', markersize=5, linewidth=1.5, alpha=0.7, color='black')
        line2 = ax.plot(comparison['u']['u_sim'], comparison['u']['y_ref'], '--',
                       label='u (Current)', linewidth=2, color='red')

        # Set up left y-axis and bottom x-axis for u-velocity
        ax.set_xlabel('u-velocity', color='black', fontweight='bold', fontsize=12)
        ax.set_ylabel('y', color='black', fontweight='bold', fontsize=12)
        ax.tick_params(axis='x', labelcolor='black', labelsize=11)
        ax.tick_params(axis='y', labelcolor='black', labelsize=11)

        # v-velocity profile: v vs x
        # Create twin axes: right y-axis for v-velocity, top x-axis for x coordinate
        ax2 = ax.twinx()  # Right y-axis for v-velocity
        ax3 = ax2.twiny()  # Top x-axis for x coordinate (shares right y-axis)

        # Plot v-velocity on ax3 (uses top x-axis and right y-axis)
        line3 = ax3.plot(comparison['v']['x_ref'], comparison['v']['v_ref'], 's-',
                        label='v (Ref.)', markersize=5, linewidth=1.5, alpha=0.7, color='black')
        line4 = ax3.plot(comparison['v']['x_ref'], comparison['v']['v_sim'], '--',
                        label='v (Current)', linewidth=2, color='red')

        # Set up right y-axis and top x-axis for v-velocity
        ax2.set_ylabel('v-velocity', color='black', fontweight='bold', fontsize=12)
        ax3.set_xlabel('x', color='black', fontweight='bold', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='black', labelsize=11)
        ax3.tick_params(axis='x', labelcolor='black', labelsize=11)

        # Title with error metrics
        title = (f'Centerline Velocity Profiles (Re={Re})\n'
                 f'u: Max err={comparison["u"]["max_error"]:.1e}, RMSE={comparison["u"]["rmse"]:.1e} | '
                 f'v: Max err={comparison["v"]["max_error"]:.1e}, RMSE={comparison["v"]["rmse"]:.1e}')
        ax.set_title(title, fontsize=14, pad=15)

        # Combine legends from all axes
        # Place legend in upper left corner to avoid crossing data lines
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', fontsize=11, framealpha=0.95,
                 fancybox=True, shadow=True, bbox_to_anchor=(1.0, 0.9))

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
    else:
        # Full mode: 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # u-velocity profile
        ax = axes[0, 0]
        ax.plot(comparison['u']['u_ref'], comparison['u']['y_ref'], 'ko-',
                label='Reference', markersize=4, linewidth=1.5)
        ax.plot(comparison['u']['u_sim'], comparison['u']['y_ref'], 'r--',
                label='Simulation', linewidth=2)
        ax.set_xlabel('u-velocity')
        ax.set_ylabel('y')
        ax.set_title(f'u-velocity at x=0.5 (Re={Re})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # u-velocity error
        ax = axes[0, 1]
        ax.plot(comparison['u']['y_ref'], comparison['u']['error'], 'r-', linewidth=2)
        ax.set_xlabel('y')
        ax.set_ylabel('Absolute Error')
        ax.set_title(f'u-velocity Error (Re={Re})\nMax: {comparison["u"]["max_error"]:.6f}, RMSE: {comparison["u"]["rmse"]:.6f}')
        ax.grid(True, alpha=0.3)

        # v-velocity profile
        ax = axes[1, 0]
        ax.plot(comparison['v']['x_ref'], comparison['v']['v_ref'], 'ko-',
                label='Reference', markersize=4, linewidth=1.5)
        ax.plot(comparison['v']['x_ref'], comparison['v']['v_sim'], 'r--',
                label='Simulation', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('v-velocity')
        ax.set_title(f'v-velocity at y=0.5 (Re={Re})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # v-velocity error
        ax = axes[1, 1]
        ax.plot(comparison['v']['x_ref'], comparison['v']['error'], 'r-', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('Absolute Error')
        ax.set_title(f'v-velocity Error (Re={Re})\nMax: {comparison["v"]["max_error"]:.6f}, RMSE: {comparison["v"]["rmse"]:.6f}')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

    if save_path:
        # If save_path is provided, modify it for streamline plot
        if save_path.endswith('.png'):
            save_path = save_path.replace('.png', f'_Re{Re}_centerlines.png')
        elif save_path.endswith('.pdf'):
            save_path = save_path.replace('.pdf', f'_Re{Re}_centerlines.pdf')
        else:
            save_path = save_path + f'_Re{Re}_centerlines.png'

        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Post-process and validate lid-driven cavity simulation results')
    parser.add_argument('pvd_file', help='Path to PVD collection file')
    parser.add_argument('--Re', type=float, required=True, help='Reynolds number')
    parser.add_argument('--plot', action='store_true', help='Show comparison plots')
    parser.add_argument('--save', help='Save plot to file')
    parser.add_argument('-c', '--compact', action='store_true',
                       help='Create a compact single plot with both u and v centerlines')
    parser.add_argument('-s', '--streamline', action='store_true',
                       help='Plot streamlines from steady-state result (last time step)')

    args = parser.parse_args()

    # Read simulation data
    print(f"Reading PVD collection: {args.pvd_file}")
    vtk_files = read_pvd_collection(args.pvd_file)
    print(f"Found {len(vtk_files)} VTK files")

    # Handle streamline plot (can be done independently)
    if args.streamline:
        Re_int = int(args.Re)
        plot_streamlines(vtk_files, Re_int, args.save)
        # If only streamline was requested, return early
        if not args.plot and not args.save:
            return 0

    print("Extracting centerline velocities...")
    (y_coords, u_centerline), (x_coords, v_centerline) = extract_centerline_velocities(vtk_files)

    # Compare with reference
    Re_int = int(args.Re)
    print(f"\nComparing with reference data for Re={Re_int}...")
    comparison = compare_with_reference(y_coords, u_centerline, x_coords, v_centerline,
                                       REFERENCE_DATA, Re_int)

    if comparison:
        print("\n=== Validation Results ===")
        print(f"u-velocity (x=0.5):")
        print(f"  Max error: {comparison['u']['max_error']:.6e}")
        print(f"  RMSE:      {comparison['u']['rmse']:.6e}")
        print(f"v-velocity (y=0.5):")
        print(f"  Max error: {comparison['v']['max_error']:.6e}")
        print(f"  RMSE:      {comparison['v']['rmse']:.6e}")

        if args.plot or args.save:
            plot_comparison(comparison, Re_int, args.save, compact=args.compact)
    else:
        print("No reference data available for comparison")

    return 0


if __name__ == '__main__':
    sys.exit(main())
