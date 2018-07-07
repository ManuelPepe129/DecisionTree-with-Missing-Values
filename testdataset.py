"""
This module contains three different dataset taken by the UCI Machine Learning Repository. The list of examples are
contained in the respective file. Attributes values have been manually inserted in this module.
"""
from dataset import DataSet
from utils import parser
from test import test
import random
from datetime import datetime

random.seed(datetime.now())

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
tictactoe_dataset = DataSet(name="Tic Tac Toe", examples=tictactoe, inputs=range(len(tictactoe_values)),
                            values=tictactoe_values, target=9)

balance_scale_values = {0: ['L', 'B', 'R'],
                        1: ['1', '2', '3', '4', '5'],
                        2: ['1', '2', '3', '4', '5'],
                        3: ['1', '2', '3', '4', '5'],
                        4: ['1', '2', '3', '4', '5']}

balance_scale = parser("balance-scale.txt")
balance_scale_dataset = DataSet(name="Balance Scale", examples=balance_scale, inputs=range(len(balance_scale_values)),
                                values=balance_scale_values, target=0)

car_values = {0: ['vhigh', 'high', 'med', 'low'],
              1: ['vhigh', 'high', 'med', 'low'],
              2: ['2', '3', '4', '5more'],
              3: ['2', '4', 'more'],
              4: ['small', 'med', 'big'],
              5: ['low', 'med', 'high'],
              6: ['unacc', 'acc', 'good', 'vgood']}

car = parser("car.txt")
car_dataset = DataSet(name="Car", examples=car, inputs=range(len(car_values)), values=car_values, target=6)


test(tictactoe_dataset)

test(balance_scale_dataset)

test(car_dataset)
