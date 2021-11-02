import datetime
import unittest

import numpy as np

from machine_learning.data_prep import DataPrep

class TestDataPrep(unittest.TestCase):
    def test_prepare_data_matrix(self):
        """test if data generation works as expected
        """
        prices = np.array([10., 15., 20., 300., 200., 150., 2500., 5000., 7500.]).reshape(-1, 1)
        dates = []
        date_one = datetime.datetime(2010, 1, 1)
        for i in range(9):
            dates.append(date_one + datetime.timedelta(days=i))
        dates = np.array(dates).reshape(-1, 1)
        volumes = np.array([2., 1.2, 4., 10., 20., 5., 7., 8., 9.]).reshape(-1, 1)
        price_matrix,_ ,volume_matrix, price_matrix_scaled, volume_matrix_scaled = DataPrep.prepare_data_matrix(prices, dates, volumes, seq_len=3)
        self.assertEqual(len(price_matrix), 7)
        self.assertEqual(len(volume_matrix), 7)
        self.assertEqual(len(price_matrix_scaled), 7)
        self.assertEqual(len(volume_matrix_scaled), 7)

        self.assertAlmostEqual(price_matrix[0][1][0], 15.)
        self.assertAlmostEqual(price_matrix_scaled[0][1][0], 0.3009039)

        self.assertAlmostEqual(price_matrix[-1][-1][0], 7500.)
        self.assertAlmostEqual(price_matrix_scaled[-1][0][0], 0.)
        self.assertAlmostEqual(price_matrix_scaled[-1][1][0], 0.6018078)
        self.assertAlmostEqual(price_matrix_scaled[-1][2][0], 1.0)

        self.assertAlmostEqual(volume_matrix[-2][1][0], 7.)
        self.assertAlmostEqual(volume_matrix_scaled[-2][0][0], 0.)
        self.assertAlmostEqual(volume_matrix_scaled[-2][1][0], 0.0199999999)
        self.assertAlmostEqual(volume_matrix_scaled[-2][2][0], 0.0300000000)

    def test_combine_data(self):
        """test that combining of data works
        """
        prices = np.array([10., 15., 20., 300., 200., 150., 2500., 5000., 7500.]).reshape(-1, 1)
        dates = []
        date_one = datetime.datetime(2010, 1, 1)
        for i in range(9):
            dates.append(date_one + datetime.timedelta(days=i))
        dates = np.array(dates).reshape(-1, 1)
        volumes = np.array([2., 1.2, 4., 10., 20., 5., 7., 8., 9.]).reshape(-1, 1)
        price_matrix,_ ,volume_matrix, price_matrix_scaled, volume_matrix_scaled = DataPrep.prepare_data_matrix(prices, dates, volumes, seq_len=3)        
        combined_data = DataPrep.combine_data([price_matrix, volume_matrix, price_matrix_scaled, volume_matrix_scaled])

        self.assertAlmostEqual(combined_data[0][1][0], 15.)
        self.assertAlmostEqual(combined_data[-1][-1][0], 7500.)

        self.assertAlmostEqual(combined_data[-2][1][1], 7.)

        print(combined_data[0][1][2])
        self.assertAlmostEqual(combined_data[0][1][2][0], 0.300903915)
        self.assertAlmostEqual(combined_data[-1][0][2][0], .0)
        self.assertAlmostEqual(combined_data[-1][1][2][0], 0.60180783)
        self.assertAlmostEqual(combined_data[-1][2][2][0], 1.)

        self.assertAlmostEqual(combined_data[-2][0][3][0], .0)
        self.assertAlmostEqual(combined_data[-2][1][3][0], 0.0199999999)
        self.assertAlmostEqual(combined_data[-2][2][3][0], .030000000)

    def test_normalize_window_advanced(self):
        test_list_one = np.array([1., 2., 3., 4.]).reshape(-1, 1) 
        normalized_list_one = DataPrep.normalize_window_advanced(test_list_one)
        normalized_final_one = np.array(normalized_list_one) / 3.
        self.assertAlmostEqual(normalized_final_one[0], 0.)
        self.assertAlmostEqual(normalized_final_one[1], 0.33333333)
        self.assertAlmostEqual(normalized_final_one[2], 0.66666667)
        self.assertAlmostEqual(normalized_final_one[3], 1.)
        
        self.assertAlmostEqual(DataPrep.normalize_window_advanced_rewind(normalized_final_one[0], test_list_one, 3.), test_list_one[0])
        self.assertAlmostEqual(DataPrep.normalize_window_advanced_rewind(normalized_final_one[1], test_list_one, 3.), test_list_one[1])
        self.assertAlmostEqual(DataPrep.normalize_window_advanced_rewind(normalized_final_one[2], test_list_one, 3.), test_list_one[2])
        self.assertAlmostEqual(DataPrep.normalize_window_advanced_rewind(normalized_final_one[3], test_list_one, 3.), test_list_one[3])

        test_list_two = np.array([1000, 1., 1999]).reshape(-1, 1)
        normalized_list_two = DataPrep.normalize_window_advanced(test_list_two)
        normalized_final_two = np.array(normalized_list_two) / 2.
        self.assertAlmostEqual(normalized_final_two[0], 0.)
        self.assertAlmostEqual(normalized_final_two[1], -0.4995)
        self.assertAlmostEqual(normalized_final_two[2], .4995)

        self.assertAlmostEqual(DataPrep.normalize_window_advanced_rewind(normalized_final_two[0], test_list_two, 2.), test_list_two[0])
        self.assertAlmostEqual(DataPrep.normalize_window_advanced_rewind(normalized_final_two[1], test_list_two, 2.), test_list_two[1])
        self.assertAlmostEqual(DataPrep.normalize_window_advanced_rewind(normalized_final_two[2], test_list_two, 2.), test_list_two[2])


