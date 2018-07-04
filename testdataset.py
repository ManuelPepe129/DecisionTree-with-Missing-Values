"""
This module contains three different dataset taken by the UCI Machine Learning Repository. The list of examples are
contained in the respective .txt file. Attributes names and values have been manually inserted in this module.

Uncomment any test_and_plot function to run it for the proper dataset, otherwise uncomment the two lines below
to print a non-pruned tree with the display() function.

The m_range for every dataset was chosen according to the major possible error. This gives the chance to the
decision learning algorithm to return a tree with just a single node.
The nursery data set has m_range = 1000 instead of 8640 for computational reasons.
"""
from dataset import DataSet
from utils import parser
from test import test

tictactoe_values = {0: ['x', 'o', 'b'],
                    1: ['x', 'o', 'b'],
                    2: ['x', 'o', 'b'],
                    3: ['x', 'o', 'b'],
                    4: ['x', 'o', 'b'],
                    5: ['x', 'o', 'b'],
                    6: ['x', 'o', 'b'],
                    7: ['x', 'o', 'b'],
                    8: ['x', 'o', 'b'],
                    9: ['positive', 'negative']}

tictactoe = parser("tictactoe.csv")
tictactoe_dataset = DataSet(name="tictactoe", examples=tictactoe, inputs=range(len(tictactoe[0])),
                            values=tictactoe_values, target=9)

test(tictactoe_dataset)

balance_scale_values = {0: ['L', 'B', 'R'],
                        1: ['1', '2', '3', '4', '5'],
                        2: ['1', '2', '3', '4', '5'],
                        3: ['1', '2', '3', '4', '5'],
                        4: ['1', '2', '3', '4', '5']}

balance_scale = parser("balance-scale.txt")

balance_scale_dataset = DataSet(name="balance-scale", examples=balance_scale, inputs=range(len(balance_scale[0])),
                                values=balance_scale_values, target=0)

test(balance_scale_dataset)
