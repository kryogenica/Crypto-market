# Crypto-market
<img src="ETH_values_examples.png" width="500">

Script will pull historical data from gdax's API and save into a CSV file locally to facilitate the exploration of this data. The API limits the amount of data that one can pull per second.

Exploring spikes in the change of price:

The script titled Plotting_moving_averge.py calculates the absolute change of the crypto price per time stamp ( |High price - Low price| ) for a give time data series, this is given the name "range". After this it calculates the moving average of the range for a given time window. The window is 20 time steps of the time series data time step which for this example is 1 hour. We calculate this for every single point in time.

Following this we analyze point by point comparing if it deviates significantly from the mean of the previous specified time window. For this a sigmoind function of the form 1/(1 + exp((Intensity * Mean_of_the_previous_time_window) - x) is used, where x is the point being analyzed. In this example the varaible Intensity serves as a way to determine how significant the deviation should be. All points that make this function have a value higher than 0.5 are recorded for an Intensity equal to 4.

<img src="Spike_changes_in_range_values.png" width="700">
In the image above the first row shows an extract of Ethereum high and low price values, the second row shows the range (diference between high and low) of the first row, finally the third row shows the deviation of each point to the moving avergae of 20 time steps. The y axis of the third row is the value of the sigmoind function described above. From the image above one can see that there is only one point with a value higher than 0.5 which would be recored for further analysis described in the following paragrahps.

Below are matrixes comparing the set of the n previous points of the (95) points that have met the criteria above. These matrixes compare one set of points to another set to see if there exists any correlations among them, therefore exploring if there are any patterns that may indicate when a spike might occure.

3 points before spike:
<img src="Correlations_with_3_span.png" width="500">

5 points before spike:
<img src="Correlations_with_5_span.png" width="500">

7 points before spike:
<img src="Correlations_with_7_span.png" width="500">

10 points before spike:
<img src="Correlations_with_10_span.png" width="500">
