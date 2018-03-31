#PULLING HISTORICAL GDAX DATA
#This uses a third party library from https://github.com/danpaquin/gdax-python
import time
import sys
import datetime
import gdax
import numpy as np
import csv
import tzlocal

def main(argv1,argv2,argv3):
    period = 0.35
    file_handler = open('Historical_eth-usd_data.txt', 'w')
    file_handler.write("time,low,high,open,close,volume\n")
    points = 200
    history_step = (argv3*points)
    count = 0
    for i in range(argv1,argv2,history_step):
        public_client = gdax.PublicClient()
        #order_book = public_client.get_product_order_book('ETH-USD', level=1)
        primera = int(i)
        ultima = int(i+history_step)
        tz = tzlocal.get_localzone()
        comienzo = datetime.datetime.fromtimestamp(primera).astimezone(tz)
        final = datetime.datetime.fromtimestamp(ultima).astimezone(tz)
        history = public_client.get_product_historic_rates(product_id='ETH-USD', start=comienzo, end=final, granularity=argv3)
        history_array = np.asarray(history)
        to_write_into = csv.writer(file_handler, delimiter=",")
        to_write_into.writerows(history)
        time.sleep(period)
        print (count)
        count = count+1
        #print ("{:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))
    file_handler.close()

if __name__=="__main__":
    main(1514352720,1514500346,300)
#12/15/2017 - 3pm - GTC-5 - Timestamp:1513368000 - iso8601:2017-12-15T15:00:00-05:00
#12/15/2017 - 4pm - GTC-5 - Timestamp:1513371600 - iso8601:2017-12-15T16:00:00-05:00
# if dates are fed in the incorrect order, this script will not get any historical data
