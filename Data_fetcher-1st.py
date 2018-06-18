#PULLING HISTORICAL GDAX DATA
import time
import datetime
import gdax
import csv
import tzlocal
import numpy as np

def main(argv1,argv2,argv3):
    period = 1.5 #helps with time gap between calls 3 in one second (seconds)
    #create a a new file where pulled data will be saved
    file_handler = open('Historical_eth-usd_data_hourly_step.txt', 'w')
    #writes a specific header in which data is pulled from the web
    file_handler.write("time,low,high,open,close,volume\n")
    #We throttle public endpoints by IP: 3 requests per second
    #Up to 6 requests per second in bursts
    #A maximum of 300 candel sticks are permitted to be pulled per request
    points = 50
    history_step = (argv3*points)
    count = 0
    for i in range(argv1,argv2,history_step):
        #Create a web connection
        public_client = gdax.PublicClient()
        #First step
        primera = int(i)
        #Second step
        ultima = int(i+history_step)
        #Get local time zone
        tz = tzlocal.get_localzone()
        #strat time
        comienzo = datetime.datetime.fromtimestamp(primera)#.astimezone(tz)
        
        #final time
        final = datetime.datetime.fromtimestamp(ultima)#.astimezone(tz)
        
        #https://docs.gdax.com/#get-historic-rates
        ###
        ###Change product_id to the crypto and its value representation to pull
        history = public_client.get_product_historic_rates(product_id='ETH-USD', start=comienzo, end=final, granularity=argv3)
        #history=np.flip(history,0)
        print (i)
        print (np.array(history).shape)
        #Set file and parameter to be saved
        to_write_into = csv.writer(file_handler, delimiter=",")
        #Write into csv file
        to_write_into.writerows(np.flip(history,0))
        #Sleep to no exceed the number of requests (3) per second
        time.sleep(period)
        #debugginf purposes
        print (count)
        count = count+1
    #close memory for file    
    file_handler.close()

if __name__=="__main__":
    #arguments are {iso8601 start time, iso8601 end time, granularity}
    #granularity values permitted {60, 300, 900, 3600, 21600, 86400} in seconds
    main(1465331600,1528478640,900)
#1463531600   1528478640
#We get 16715 points
#Saturday, April 21, 2018 11:37:06 PM GMT-04:00 DST - Timestamp:1524368226
#Tuesday, May 17, 2016 8:33:20 PM GMT-04:00 DST - Timestamp:1463531600
# if dates are fed in the incorrect order, this script will not get any historical data