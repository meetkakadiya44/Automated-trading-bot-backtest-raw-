# -*- coding: utf-8 -*-
"""
Created on Sun May 15 18:01:47 2020

@author: MEET KAKADIYA
email= meet.17u122@viit.ac.in
"""
'''
STRATEGY:
SIMPLE SMA CROSSOVER strategy (slow=20 period, fast=5 period), with trailing stop loss
using Daily data


    
'''

#IMPLEMENTING STRATEGY NO 1
#---------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pandas_datareader.data as web
#from datetime import datetime
from collections import defaultdict
import pprint
import matplotlib.pyplot as plt


#---------------------------------------------------------------------------------------------------------------------------------------------
class Strategy:
    def __init__(self, ticker_symbol, capital, capital_allocation, cost, slow_sma, fast_sma, max_stop_loss):
        self.ticker_symbol=ticker_symbol
        self.capital=capital
        self.capital_allocation=capital_allocation
        self.cost=cost
        self.slow_sma=slow_sma
        self.fast_sma=fast_sma
        self.max_stop_loss=max_stop_loss

    def data_manupalation(self):

        # dataframe

        self.start = '2014'
        self.end = '2017'

        self.df = web.DataReader(self.ticker_symbol, 'yahoo', start=self.start, end=self.end)
        self.df = pd.DataFrame(self.df)

        self.df['date'] = np.array(self.df.index, dtype='datetime64[D]')  # [D] day level, ns means nano sec level
        self.df['row'] = np.arange(len(self.df))
        self.df.drop(columns=['Adj Close'], inplace=True)

        self.df.set_index('row', inplace=True)

        # dataaframe object passed to instance


        # SMA VALUES FOR 5 AND 20
        self.sma5 = self.df['Close'].rolling(self.fast_sma).mean()
        self.sma20 = self.df['Close'].rolling(self.slow_sma).mean()




        lx = []
        ly = []
        #inner fucntion for calc
        def condition_for_state(sma5, sma20):

            if len(lx) < 20:
                lx.append(sma5)
                ly.append(sma20)
                return (0)
            else:
                lx.append(sma5)
                ly.append(sma20)

                if (lx[-1] > ly[-1]) and (lx[-2] < ly[-2]):
                    return (1)  # crossOVER
                elif (lx[-1] < ly[-1]) and (lx[-2] > ly[-2]):
                    return (-1)  # crossUNDER
                else:
                    return (0)

        self.state = np.vectorize(pyfunc=condition_for_state, otypes=[int])(self.sma5, self.sma20)
        print(self.state)

    def new_long(self, indexx):
        # print('im in the func')
        # pprint.pprint(positions)
        indexx = int(indexx)

        # INCREASING THE self.position_counter TO CREATE A DICT KEY FOR NEW POSTION
        self.position_counter[-1] = self.position_counter[-1] + 1
        # this fucntion will create a dict value, which means it will create a ner positon
        # 1st parameter is date.......@nd is close_price, third is flag: 1 means long, last is the very initial SL value

        cp = float(self.df[self.df.index == indexx]['Close'])
        date_val = (str(self.df[self.df.index == indexx]['date'].values))[2:12]

        lp1 = float(self.df[self.df.index == indexx - 1]['Low'])
        lp2 = float(self.df[self.df.index == indexx - 2]['Low'])
        lp3 = float(self.df[self.df.index == indexx - 3]['Low'])
        lp4 = float(self.df[self.df.index == indexx - 4]['Low'])

        # 1st STOP LOSS CALC
        sl1 = min(lp1, lp2, lp3, lp4)  # creating STOP LOSS 1, which lowest low of last for candles

        self.positions[self.position_counter[-1]].append(date_val)
        self.positions[self.position_counter[-1]].append(cp)
        self.positions[self.position_counter[-1]].append(1)
        self.positions[self.position_counter[-1]].append(sl1)
        self.positions[self.position_counter[-1]].append(sl1)  # this will be later changed as Sl changes
        # positions[self.position_counter[-1]][0][1] # returns buy price

    def exit_long(self, indexx):
        cp = float(self.df[self.df.index == indexx]['Close'])
        self.positions[self.position_counter[-1]].append(cp)
        date_val = (str(self.df[self.df.index == indexx]['date'].values))[2:12]
        self.positions[self.position_counter[-1]].append(date_val)
        self.positions[self.position_counter[-1]].append(2)  # 2 flag is for indicating  exiting long position
        pnll = self.positions[self.position_counter[-1]][5] - self.positions[self.position_counter[-1]][1]
        self.positions[self.position_counter[-1]].append(pnll)

    def exit_short(self, indexx):
        cp = float(self.df[self.df.index == indexx]['Close'])
        self.positions[self.position_counter[-1]].append(cp)
        date_val = (str(self.df[self.df.index == indexx]['date'].values))[2:12]
        self.positions[self.position_counter[-1]].append(date_val)
        self.positions[self.position_counter[-1]].append(4)  # 4 flag is for indicating  exiting short position
        pnll = self.positions[self.position_counter[-1]][1] - self.positions[self.position_counter[-1]][5]
        self.positions[self.position_counter[-1]].append(pnll)

    def check_sl_long(self, indexx):

        # SL check
        cp = float(self.df[self.df.index == indexx]['Close'])
        bp = self.positions[self.position_counter[-1]][1]

        lp1 = float(self.df[self.df.index == indexx - 1]['Low'])
        lp2 = float(self.df[self.df.index == indexx - 2]['Low'])
        lp3 = float(self.df[self.df.index == indexx - 3]['Low'])
        sl = self.positions[self.position_counter[-1]][4]

        pointer2 = bp * 1.03  # the current price must be 3% more than buy point
        pointer3 = bp * 1.05  # the current price must be 5% more than buy point

        # calc sl
        pointer4 = bp * (1 - self.max_stop_loss)
        if cp <= pointer4:
            self.exit_long(indexx)
            return (3)

        if (cp >= pointer2 and cp < pointer3):
            secondsl = min(lp1, lp2, lp3)  # we're updating the SL constantly here
            self.positions[self.position_counter[-1]][4] = secondsl
            return (2)


        elif (cp >= pointer3):
            third_sl = min(lp1, lp2)  # we're updating the SL constantly here
            self.positions[self.position_counter[-1]][4] = third_sl
            return (2)


        elif (cp <= sl):
            # in this case we will append exit price ,exit date and exit flag
            self.positions[self.position_counter[-1]][4] = sl
            self.exit_long(indexx)
            return (3)

        elif (cp < pointer2 and cp > sl):
            return (2)

    def new_short(self,  indexx):
        indexx = int(indexx)

        # INCREASING THE self.position_counter TO CREATE A DICT KEY FOR NEW POSTION
        self.position_counter[-1] = self.position_counter[-1] + 1
        cp = float(self.df[self.df.index == indexx]['Close'])
        date_val = (str(self.df[self.df.index == indexx]['date'].values))[2:12]

        hp1 = float(self.df[self.df.index == indexx - 1]['High'])
        hp2 = float(self.df[self.df.index == indexx - 2]['High'])
        hp3 = float(self.df[self.df.index == indexx - 3]['High'])
        hp4 = float(self.df[self.df.index == indexx - 4]['High'])

        sl1 = max(hp1, hp2, hp3, hp4)  # creating STOP LOSS 1, which highest high of last for candles

        self.positions[self.position_counter[-1]].append(date_val)
        self.positions[self.position_counter[-1]].append(cp)
        self.positions[self.position_counter[-1]].append(3)  # new short pointer
        self.positions[self.position_counter[-1]].append(sl1)
        self.positions[self.position_counter[-1]].append(sl1)  # this will be later changed as Sl changes

    def check_sl_short(self, indexx):
        # SL check
        cp = float(self.df[self.df.index == indexx]['Close'])
        sp = self.positions[self.position_counter[-1]][1]

        hp1 = float(self.df[self.df.index == indexx - 1]['High'])
        hp2 = float(self.df[self.df.index == indexx - 2]['High'])
        hp3 = float(self.df[self.df.index == indexx - 3]['High'])
        sl = self.positions[self.position_counter[-1]][4]

        pointer2 = sp * 0.97  # the current price must be 3% less than sell point
        pointer3 = sp * 0.95  # the current price must be 5% less than sell point

        # calc sl
        pointer4 = sp * (1 + self.max_stop_loss)
        if cp >= pointer4:
            self.exit_long(indexx)
            return (9)

        if (cp <= pointer2 and cp > pointer3):
            secondsl = max(hp1, hp2, hp3)  # we're updating the SL constantly here
            self.positions[self.position_counter[-1]][4] = secondsl
            return (8)


        elif (cp <= pointer3):
            third_sl = max(hp1, hp2)  # we're updating the SL constantly here
            self.positions[self.position_counter[-1]][4] = third_sl
            return (8)


        elif (cp >= sl):
            # in this case we will append exit price ,exit date and exit flag
            self.positions[self.position_counter[-1]][4] = sl
            self.exit_short(indexx)
            return (9)

        elif (cp > pointer2 and cp < sl):
            return (8)

    def strategy_logic(self):
    #putting it all together
        self.data_manupalation()
        #calculates self.state param
        self.no_position_counter=0
        self.position_counter = [0]  # this will be the postion number
        self.positions = defaultdict(list)  # list backed multidict OUR MAIN DICT OF self.positions
        temp1 = [0]


        def strategy_conditions(state, indexx):
            # print(indexx)
            # ----------------most used cases----------------
            # do nothing
            if state == 0 and temp1[-1] == 0:
                self.no_position_counter=self.no_position_counter+1
                temp1.append(0)

                return (0)
            # check sl for longs
            elif state == 0 and (temp1[-1] == 1 or temp1[-1] == 2):
                y = self.check_sl_long(indexx)  # we will check sl here and update if necessary and return the value 2/3 depending the returned value
                temp1.append(y)
                return (y)
            # check sl for shorts
            elif state == 0 and (temp1[-1] == 7 or temp1[-1] == 8):
                y = self.check_sl_short(indexx)  # we will check sl here and update if necessary and return the value 8/9depending the returned value
                temp1.append(y)
                return (y)



            # --------------moderately used case---------------------
            # newbuy condition
            elif state == 1 and (temp1[-1] == 0 or temp1[-1] == 9):
                temp1.append(1)
                self.new_long(indexx)
                return (1)
            # new short
            elif (state == -1 and (temp1[-1] == 0 or temp1[-1] == 3)):  # long gets exited before crossunder
                self.new_short(indexx)  # absolutely new short
                temp1.append(7)
                return (7)  # new  short here too

            # we need to clear last long position
            elif state == -1 and temp1[-1] == 2:

                self.exit_long(indexx)
                # the last position was closed and updated
                # we need to create a new short pos
                self.new_short(indexx)
                temp1.append(7)
                return (7)
            # crossover exit for short position and needs to clear short and enter a new long
            elif (state == 1 and temp1[-1] == 8):
                self.exit_short(indexx)
                self.new_long(indexx)
                temp1.append(1)
                return (1)

            # we exited long but no self.positions now
            elif state == 0 and temp1[-1] == 3:
                temp1.append(0)
                self.no_position_counter=self.no_position_counter+1
                return (0)
            # we exited short but no self.positions now
            elif state == 0 and temp1[-1] == 9:
                temp1.append(0)
                self.no_position_counter=self.no_position_counter+1
                return (0)


            # ----------------least used cases-------------------------

            # bought last and selling on next moment
            elif state == -1 and temp1[-1] == 1:
                self.exit_long(indexx)
                temp1.append(3)
                return (3)

            # new long when last position closed was also long
            # very rare case
            elif (state == 1 and temp1[-1] == 3):
                self.new_long(indexx)
                temp1.append(1)
                return (1)
            # new short when last position closed was alse short
            # very rare case
            elif (state == -1 and temp1[-1] == 9):
                self.new_short(indexx)
                temp1.append(7)
                return (7)

            else:   #resolving any left case
                if temp1[-1]==1 or temp1[-1]==2:
                    return (2)
                    temp1.append(2)
                if temp1[-1] == 7 or temp1[-1] == 8:
                    return(8)
                    temp1.append(8)

        self.positionarray = np.vectorize(pyfunc=strategy_conditions)(self.state, self.df.index)
        # positionarray=np.array(positionarray)
        print(self.positionarray)

        self.strategy_results()

    def strategy_results(self):
        y = np.arange(1, len(self.positions), 1)
        pchange=0
        self.daily_returns=[]
        capitall=[self.capital_allocation *self.capital]
        for y in range(1,len(self.positions)):
            #alloting only specified amount of capital
            capital1= capitall[-1]
            if self.positions[y][2] == 1:
                pchange = (self.positions[y][5] - self.positions[y][1]) / self.positions[y][1]
            elif (self.positions[y][2] == 3):
                pchange = (self.positions[y][1] - self.positions[y][5]) / self.positions[y][1]
            self.daily_returns.append(pchange)
            pchange = (1 + pchange)
            capitall.append(capital1 * pchange)

        #print(capitall)
        
        self.initial_capital_alloted=capitall[0]
        self.final_capital=capitall[-1]

        print('---------------------RESULTS-------------------------')
        print(f"start date:  {self.start} to end date: {self.end}" )

        print(f"FINAL AMOUNT: {self.final_capital}")

        self.percent_change = ((self.final_capital - self.capital) / self.capital) * 100
        print("ROI: " + str(self.percent_change))

        #calc sharp ratio
        plt.plot(self.daily_returns)
        plt.show()
        self.sharp_ratio=np.sqrt(252) *(np.mean(self.daily_returns))/(np.std(self.daily_returns))

        print(f"annualized Sharp ratio: {self.sharp_ratio}")
        print(f"number of days with no positions open: {self.no_position_counter}")
        plt.plot(self.daily_returns)
if __name__== "__main__":
    # ticker_symbol, capital, capital_allocation, cost, slow_sma, fast_sma, max_stop_loss

    s1 = Strategy('AMZN',                 #ticker_symbol/ AMAZON
                  1000000,              #inital capital
                  1,                    # capital allocation
                  0.75,                 # cost
                  20,                   # slow sma value
                  5,                    # fast sma value
                  0.05                  # max_stop_loss
                  )

    s1.strategy_logic()

