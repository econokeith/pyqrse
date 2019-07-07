import numpy as np
# import Quandl as qd
import pandas as pd
import scipy as sp
import numpy.random as npr
import scipy.stats as sps
from numpy import array
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from scipy import stats
import datetime
from scipy.integrate import quad
from py3qrse.helpers2 import date_to_datetime, datetime_to_date
from py3qrse.model import QRSE
import seaborn as sns; sns.set()

try:
    import happybase
    connection = happybase.Connection()
except:
    print("Turn On HBASE")


## HBase requires utf-8 encoding.
## -String output need to be str(output, 'utf-8') or output.decode()
## -String inputs need to be converted via input.encode()

class HBaseStockGetter:

    def __init__(self, table, field='data:d1a'):

        self.field = field
        if isinstance(field, str):
            self.field = self.field.encode()
        self.table = table
        self.data = None
        self.min_volume = 10
        self.max_dev = 6
        self.out = None
        self.clen = 0
        self.quantile = 10
        self.last_date = None


    def get(self, date, field=None, tickers=None, silent=True):

        if isinstance(date, datetime.datetime) or isinstance(date, datetime.date):
            the_date = datetime_to_date(date)
        else:
            the_date = date

        if field is None:
            the_field = self.field
        else:
            the_field = field

        if isinstance(the_date, str):
            the_date = the_date.encode()

        if isinstance(the_field, str):
            the_field = the_field.encode()

        self.last_date = the_date.decode()

        print(the_field)

        data = []
        names = []
        self.bad_points = []


        for _, _data in self.table.scan(row_prefix=the_date):
            dlist = [_data[v].decode() for v in [the_field, b'data:adj_volume', b'data:ticker']]

            try:
                dlist[0] = float(dlist[0])
                dlist[1] = float(dlist[1])
                if dlist[0] != 0.:
                    data.append(dlist[:2])
                    names.append(dlist[2])
                else:
                    self.bad_points.append(dlist[2])
            except:
                self.bad_points.append(dlist[2])

        data = np.asarray(data)

        n_data = data.shape[0]
        self.data = data

        #Filter By Volume
        min_volume = np.percentile(data[:,1], self.quantile)

        low_volume = data[:,1]>min_volume

        data = data[low_volume]
        names = [names[i] for i in low_volume]


        ##Filter By Tickers
        if tickers is not None:
            in_indices = np.isin(names, tickers)
            data = data[in_indices]
            if silent is False:
                print('kept {} of {} names'.format(data.shape[0], n_data))

        data = data[:, 0]

        #Filter Outliers by Max Std Deviations

        mean = data.mean()
        std = data.std()

        [min_v, max_v]= [ mean - self.max_dev*std,mean + self.max_dev*std ]

        n_data = data.shape[0]
        inbounds_data_index = (data>min_v)&(data<max_v)
        n_out_of_bounds = n_data-inbounds_data_index.shape[0]
        if silent is False:
            print('{} of {} names more than {} stds'.format(n_out_of_bounds, n_data, self.max_dev))
        self.data = data[inbounds_data_index]

    def plot(self, **kwargs):
        date = datetime.datetime.strptime(self.last_date, "%Y%m%d").strftime("%m/%d/%Y")
        title = "{} : {}".format(date, self.field)
        sns.distplot(self.data, **kwargs)
        plt.title(title);





# def date_to_datetime(d):
#     return datetime.date(int(d[:4]),int(d[4:6]), int(d[6:]) )
#
# def datetime_to_date(d):
#     day = d.day
#     month = d.month
#     day = '0'+str(day) if day < 10 else str(day)
#     month = '0'+str(month) if month < 10 else str(month)
#     return '{}{}{}'.format(d.year, month, day)

