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

import os
import json
import numpy as np
from cherab.core.atomic import Element
from ..utility import DEFAULT_REPOSITORY_PATH, valid_ionisation

"""
Utilities for managing the local rate repository - beam stopping section.
"""


def add_beam_stopping_rate(beam_species, target_ion, target_ionisation, rate, repository_path=None):
    """
    Adds a single beam stopping/excitation rate to the repository.

    :param beam_species:
    :param target_ion:
    :param target_ionisation:
    :param rate:
    :return:
    """

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH

    # sanitise and validate arguments
    if not isinstance(beam_species, Element):
        raise TypeError('The beam_species must be an Element object.')

    if not isinstance(target_ion, Element):
        raise TypeError('The beam_species must be an Element object.')

    if not valid_ionisation(target_ion, target_ionisation):
        raise ValueError('Ionisation level is larger than the number of protons in the target ion.')

    # sanitise and validate rate data
    e = np.array(rate['e'], np.float64)
    n = np.array(rate['n'], np.float64)
    t = np.array(rate['t'], np.float64)
    sen = np.array(rate['sen'], np.float64)
    st = np.array(rate['st'], np.float64)

    if e.ndim != 1:
        raise ValueError('Beam energy array must be a 1D array.')

    if n.ndim != 1:
        raise ValueError('Density array must be a 1D array.')

    if t.ndim != 1:
        raise ValueError('Temperature array must be a 1D array.')

    if (e.shape[0], n.shape[0]) != sen.shape:
        raise ValueError('Beam energy, density and combined rate data arrays have inconsistent sizes.')

    if t.shape != st.shape:
        raise ValueError('Temperature and temperature rate data arrays have inconsistent sizes.')

    # convert numpy arrays to lists for json export
    rate['e'] = e.tolist()
    rate['n'] = n.tolist()
    rate['t'] = t.tolist()
    rate['sen'] = sen.tolist()
    rate['st'] = st.tolist()

    rate['eref'] = float(rate['eref'])
    rate['nref'] = float(rate['nref'])
    rate['tref'] = float(rate['tref'])
    rate['sref'] = float(rate['sref'])

    path = os.path.join(repository_path, 'beam/stopping/{}/{}/{}.json'.format(beam_species.symbol.lower(), target_ion.symbol.lower(), target_ionisation))

    # create directory structure if missing
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # write new data (simply overwrite any existing rate)
    with open(path, 'w') as f:
        json.dump(rate, f, indent=2, sort_keys=True)


def update_beam_stopping_rates(rates, repository_path=None):
    """
    Beam stopping rate file structure

    /beam/stopping/<beam species>/<target ion>/<target_ionisation>.json

    Each json file contains a single rate, so it can simply be replaced.
    """

    for beam_species, target_ions in rates.items():
        for target_ion, target_ionisations in target_ions.items():
            for target_ionisation, rate in target_ionisations.items():
                add_beam_stopping_rate(beam_species, target_ion, target_ionisation, rate, repository_path)


def get_beam_stopping_rate(beam_species, target_ion, target_ionisation, repository_path=None):

    repository_path = repository_path or DEFAULT_REPOSITORY_PATH
    path = os.path.join(repository_path, 'beam/stopping/{}/{}/{}.json'.format(beam_species.symbol.lower(), target_ion.symbol.lower(), target_ionisation))
    try:
        with open(path, 'r') as f:
            rate = json.load(f)
    except FileNotFoundError:
        raise RuntimeError('Requested beam stopping rate (beam species={}, target ion={}, target ionisation={})'
                           ' is not available.'.format(beam_species.symbol, target_ion.symbol, target_ionisation))

    # convert lists to numpy arrays
    rate['e'] = np.array(rate['e'], np.float64)
    rate['n'] = np.array(rate['n'], np.float64)
    rate['t'] = np.array(rate['t'], np.float64)
    rate['sen'] = np.array(rate['sen'], np.float64)
    rate['st'] = np.array(rate['st'], np.float64)

    return rate
