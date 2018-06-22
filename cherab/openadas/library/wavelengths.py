
# Copyright 2016-2018 Euratom
# Copyright 2016-2018 United Kingdom Atomic Energy Authority
# Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.


from cherab.core.atomic.elements import *
from cherab.core.utility.recursivedict import RecursiveDict


# TODO: there are concerns about the accuracy of this data
# Data from NIST Atomic Spectra Database - http://www.nist.gov/pml/data/asd.cfm
# line emission natural wavelength (in nanometers):
# config[ion][ionisation][(initial_level, final_level)] = wavelength


wavelength_database = RecursiveDict()


wavelength_database[hydrogen][0] = {
    (2, 1): 121.52,
    (3, 1): 102.53,
    (3, 2): 656.19,
    (4, 1): 97.21,
    (4, 2): 486.06,
    (4, 3): 1874.82,
    (5, 1): 94.93,
    (5, 2): 433.99,
    (5, 3): 1281.61,
    (5, 4): 4050.53,
    (6, 1): 93.74,
    (6, 2): 410.12,
    (6, 3): 1093.64,
    (6, 4): 2624.75,
    (6, 5): 7456.66,
    (7, 1): 93.04,
    (7, 2): 396.95,
    (7, 3): 1004.79,
    (7, 4): 2165.19,
    (7, 5): 4651.78,
    (7, 6): 12366.59,
    (8, 1): 92.58,
    (8, 2): 388.85,
    (8, 3): 954.45,
    (8, 4): 1944.26,
    (8, 5): 3738.95,
    (8, 6): 7499.27,
    (8, 7): 19053.71,
    (9, 1): 92.28,
    (9, 2): 383.49,
    (9, 3): 922.76,
    (9, 4): 1817.13,
    (9, 5): 3295.58,
    (9, 6): 5905.68,
    (9, 7): 11303.84,
    (9, 8): 27791.42,
    (10, 1): 92.06,
    (10, 2): 379.74,
    (10, 3): 901.35,
    (10, 4): 1735.94,
    (10, 5): 3037.9,
    (10, 6): 5126.46,
    (10, 7): 8756.3,
    (10, 8): 16202.13,
    (10, 9): 38853.14,
    (11, 1): 91.9,
    (11, 2): 377.01,
    (11, 3): 886.14,
    (11, 4): 1680.39,
    (11, 5): 2871.76,
    (11, 6): 4670.5,
    (11, 7): 7504.88,
    (11, 8): 12381.84,
    (11, 9): 22330.84,
    (11, 10): 52512.27,
    (12, 1): 91.77,
    (12, 2): 374.96,
    (12, 3): 874.92,
    (12, 4): 1640.47,
    (12, 5): 2757.09,
    (12, 6): 4374.58,
    (12, 7): 6769.08,
    (12, 8): 10498.98,
    (12, 9): 16873.36,
    (12, 10): 29826.65,
    (12, 11): 69042.22,
}


wavelength_database[helium][1] = {
    (2, 1): 30.378,     # 2p -> 1s
    (3, 1): 25.632,     # 3p -> 1s
    (3, 2): 164.04,     # 3d -> 2p
    (4, 2): 121.51,     # 4d -> 2p
    (4, 3): 468.71,     # 4f -> 3d
    (5, 3): 320.28,     # 5f -> 3d
    (5, 4): 1012.65,    # 5g -> 4f
    (6, 4): 656.20,     # 6g -> 4f
    (6, 5): 1864.20,    # 6h -> 5g
    (7, 5): 1162.53,    # from ADAS comment, unknown source
    (7, 6): 3090.55     # from ADAS comment, unknown source
}


# Be3+
wavelength_database[beryllium][3] = {
    (3, 1): 6.4065,     # 3p -> 1s
    (3, 2): 41.002,     # 3d -> 2p
    (4, 2): 30.373,     # 4d -> 2p
    (4, 3): 117.16,     # 4f -> 3d
    (5, 3): 80.092,     # 5f -> 3d
    (5, 4): 253.14,     # 5g -> 4f
    (6, 4): 164.03,     # 6g -> 4f
    (6, 5): 466.01,     # 6h -> 5g
    (7, 5): 290.62,     # from ADAS comment, unknown source
    (7, 6): 772.62,     # from ADAS comment, unknown source
    (8, 6): 468.53,     # from ADAS comment, unknown source
    (8, 7): 1190.42     # from ADAS comment, unknown source
}

# B4+
wavelength_database[boron][4] = {
    (3, 1): 4.0996,     # 3p -> 1s
    (3, 2): 26.238,     # 3d -> 2p
    (4, 2): 19.437,     # 4d -> 2p
    (4, 3): 74.980,     # 4f -> 3d
    (5, 3): 51.257,     # 5f -> 3d
    (5, 4): 162.00,     # 5g -> 4f
    (6, 4): 104.98,     # 6g -> 4f
    (6, 5): 298.24,     # 6h -> 5g
    (7, 5): 186.05,     # 7h -> 5g
    (7, 6): 494.48,     # 7i -> 6h
    (8, 6): 299.86,     # 8i -> 6h
    (8, 7): 761.87,     # 8k -> 7i
    (9, 7): 451.99,     # 9k -> 7i
    (9, 8): 1111.25     # from ADAS comment, unknown source
}

# C5+
wavelength_database[carbon][5] = {
    (4, 2): 13.496,     # 4d -> 2p
    (4, 3): 52.067,     # 4f -> 3d
    (5, 3): 35.594,     # 5f -> 3d
    (5, 4): 112.50,     # 5g -> 4f
    (6, 4): 72.900,     # 6g -> 4f
    (6, 5): 207.11,     # 6h -> 5g
    (7, 5): 129.20,     # from ADAS comment, unknown source
    (7, 6): 343.38,     # from ADAS comment, unknown source
    (8, 6): 208.23,     # from ADAS comment, unknown source
    (8, 7): 529.07,     # from ADAS comment, unknown source
    (9, 7): 313.87,     # from ADAS comment, unknown source
    (9, 8): 771.69,     # from ADAS comment, unknown source
    (10, 8): 449.89,    # from ADAS comment, unknown source
    (10, 9): 1078.86    # from ADAS comment, unknown source
}

# Ne9+
wavelength_database[neon][9] = {
    (6, 5): 74.54,      # from ADAS comment, unknown source
    (7, 6): 123.64,     # from ADAS comment, unknown source
    (8, 7): 190.50,     # from ADAS comment, unknown source
    (9, 8): 277.79,     # from ADAS comment, unknown source
    (10, 9): 388.37,    # from ADAS comment, unknown source
    (11, 10): 524.92,   # from ADAS comment, unknown source
    (12, 11): 690.16,   # from ADAS comment, unknown source
    (13, 12): 886.83,   # from ADAS comment, unknown source
    (6, 4): 26.24,      # from ADAS comment, unknown source
    (7, 5): 46.51,      # from ADAS comment, unknown source
    (8, 6): 74.98,      # from ADAS comment, unknown source
    (9, 7): 113.02,     # from ADAS comment, unknown source
    (10, 8): 162.00,    # from ADAS comment, unknown source
    (11, 9): 223.22,    # from ADAS comment, unknown source
    (12, 10): 298.15,   # from ADAS comment, unknown source
    (13, 11): 388.12    # from ADAS comment, unknown source
}


wavelength_database = wavelength_database.freeze()
