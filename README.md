# Crypto-market

Script will pull historical data from gdax's API and put save into a CSV file locally to facilitate the exploration of this data. The API limits the amount of data that one can pull per second.

Exploring spikes in the change of price:

Insert_name_of_script_file... script calculates the absolute change of the crypto price per time stamp ( |High price - Low price| ) for a give time data series, this is given the name "range". After this it calculates the moving average of the range for a given time window. The window is 20 time steps of the time series data time step which for this example is 1 hour. We calculate this for every single point in time.

Following this we analyze point by point comparing if it deviates significantly from the mean of the previous specified time window. For this a sigmoind function of the form 1/(1 + exp((Intensity * Mean_of_the_previous_time_window) - x) is used, where x is the point being analyzed. In this example the varaible Intensity serves as a way to determine how significant the deviation should be. All points that make this function have a value higher than 0.5 are recorded for an Intensity equal to 4.
Below you can see an example of these calculation done on a small data set:

![alt text](https://github.com/kryogenica/Crypto-market/master/Rare_change_of_value_event_detection.png)
