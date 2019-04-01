from cherab.openadas import OpenADAS
from cherab.openadas.parse import parse_adf11
from cherab.core.utility import Cm3ToM3
from cherab.openadas.repository.rates_adf11_lib import rates_lib
from cherab.openadas.install import _locate_adas_file

import numpy as np
import unittest

ERROR_ABSOLUTE = 0
ERROR_RELATIVE = 1e-6


class TestRateInterpolation(unittest.TestCase):

    def test_recombination(self):
        adas = OpenADAS(permit_extrapolation=True)

        # pick the ADF11 acd file type to test the recombination rate interpolation
        for data_type in rates_lib:
            if data_type.lower() in "adf11acd":
                data_info = rates_lib[data_type]

        for adf11_file in data_info:
            elem = adf11_file[0]  # element
            file_path = _locate_adas_file(adf11_file[1], download=True)  # get the file path and download if missing

            rates_original = parse_adf11(elem, file_path)  # parse the original rates for testing
            # load cherab rates
            for elemcharge in range(1, elem.atomic_number + 1):
                rate_cherab = adas.recombination_rate(elem, elemcharge)

                # get original Te data and transform them into linear scale
                te_test = np.power(10, rates_original[elem][elemcharge]["te"])

                # get original ne data and transform them into linear scale and m%**-3
                ne_test = np.power(10, rates_original[elem][elemcharge]["ne"] + 6)

                for n_ne, knot_ne in enumerate(ne_test):
                    for n_te, knot_te in enumerate(te_test):
                        value_adas = rates_original[elem][elemcharge]["rates"][n_ne, n_te]
                        value_cherab = np.log10(Cm3ToM3.inv(rate_cherab.evaluate(knot_ne, knot_te)))

                        self.assertTrue(np.isclose(value_cherab, value_adas, atol=ERROR_ABSOLUTE, rtol=ERROR_RELATIVE))

    def test_ionisation(self):
        adas = OpenADAS(permit_extrapolation=True)

        # pick the ADF11 acd file type to test the recombination rate interpolation
        for data_type in rates_lib:
            if data_type.lower() in "adf11scd":
                data_info = rates_lib[data_type]

        for adf11_file in data_info:
            elem = adf11_file[0]  # element
            file_path = _locate_adas_file(adf11_file[1], download=True)  # get the file path and download if missing

            rates_original = parse_adf11(elem, file_path)  # parse the original rates for testing
            # load cherab rates
            for elemcharge in range(1, elem.atomic_number + 1):
                rate_cherab = adas.ionisation_rate(elem, elemcharge - 1)

                # get original Te data and transform them into linear scale
                te_test = np.power(10, rates_original[elem][elemcharge]["te"])

                # get original ne data and transform them into linear scale and m%**-3
                ne_test = np.power(10, rates_original[elem][elemcharge]["ne"] + 6)

                for n_ne, knot_ne in enumerate(ne_test):
                    for n_te, knot_te in enumerate(te_test):
                        value_adas = rates_original[elem][elemcharge]["rates"][n_ne, n_te]
                        value_cherab = np.log10(Cm3ToM3.inv(rate_cherab.evaluate(knot_ne, knot_te)))

                        self.assertTrue(np.isclose(value_cherab, value_adas, atol=ERROR_ABSOLUTE, rtol=ERROR_RELATIVE))

    def test_thermal_cx(self):
        adas = OpenADAS(permit_extrapolation=True)

        # pick the ADF11 acd file type to test the recombination rate interpolation
        for data_type in rates_lib:
            if data_type.lower() in "adf11ccd":
                data_info = rates_lib[data_type]

        for adf11_file in data_info:
            donor_element = adf11_file[0]
            donor_charge = adf11_file[1]
            receiver_element = adf11_file[2]  # element
            file_path = _locate_adas_file(adf11_file[3], download=True)  # get the file path and download if missing

            rates_original = parse_adf11(receiver_element, file_path)  # parse the original rates for testing
            # load cherab rates
            for elemcharge in range(1, receiver_element.atomic_number + 1):
                rate_cherab = adas.thermal_cx_rate(donor_element=donor_element, donor_charge=donor_charge,
                                                   receiver_element=receiver_element, receiver_charge=elemcharge)

                # get original Te data and transform them into linear scale
                te_test = np.power(10, rates_original[receiver_element][elemcharge]["te"])

                # get original ne data and transform them into linear scale and m%**-3
                ne_test = np.power(10, rates_original[receiver_element][elemcharge]["ne"] + 6)

                for n_ne, knot_ne in enumerate(ne_test):
                    for n_te, knot_te in enumerate(te_test):
                        value_adas = rates_original[receiver_element][elemcharge]["rates"][n_ne, n_te]
                        value_cherab = np.log10(Cm3ToM3.inv(rate_cherab.evaluate(knot_ne, knot_te)))

                        self.assertTrue(np.isclose(value_cherab, value_adas, atol=ERROR_ABSOLUTE, rtol=ERROR_RELATIVE))


if __name__ == "__main__":
    unittest.main()
