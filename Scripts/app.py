import streamlit as st
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mplfinance.original_flavor import candlestick_ohlc

# Load data (this will need to point to the correct file)
#data = np.genfromtxt('Historical_eth-usd_data_hourly_step.txt', delimiter=',')

# Create a sidebar for file upload
st.sidebar.title("File Uploader")

# File uploader widget
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    try:
        data = np.genfromtxt(uploaded_file, delimiter=',')
        # Display the message and set it to disappear after 5 seconds
        message_placeholder = st.empty()
        message_placeholder.write("Data loaded successfully!")
        time.sleep(1)  # Wait for 5 seconds
        message_placeholder.empty()  # Clear the message

        # Define initial dataframes
        open_eth_data = pd.DataFrame({'Open': data[1:, 3]})
        low_eth_data = pd.DataFrame({'Low': data[1:, 1]})
        high_eth_data = pd.DataFrame({'High': data[1:, 2]})
        close_eth_data = pd.DataFrame({'Close': data[1:, 2]})
        rang_eth_data = high_eth_data['High'] - low_eth_data['Low']

        # HEIKIN-ASHI DATA
        # Calculate the Heikin-Ashi 'Open' price using the previous 'Open' and 'Close' prices
        open_heikin = (open_eth_data['Open'][0:-2] + close_eth_data['Close'][0:-2]) / 2
        # Calculate the Heikin-Ashi 'Close' price as the average of the current 'Open', 'Close', 'Low', and 'High' prices
        close_heikin = (open_eth_data['Open'][1:-1] + close_eth_data['Close'][1:-1] + low_eth_data['Low'][1:-1] + high_eth_data['High'][1:-1]) / 4
        # Calculate the Heikin-Ashi 'High' price as the maximum of 'Open', 'Close', and 'High'
        high_heikin = np.maximum.reduce([open_eth_data['Open'][1:-1], close_eth_data['Close'][1:-1], high_eth_data['High'][1:-1]])
        # Calculate the Heikin-Ashi 'Low' price as the minimum of 'Open', 'Close', and 'Low'
        low_heikin = np.minimum.reduce([low_eth_data['Low'][1:-1], close_eth_data['Close'][1:-1], open_eth_data['Open'][1:-1]])

        # Set initial values if not already set
        if 'upper' not in st.session_state:
            st.session_state['upper'] = 1000
        if 'diff' not in st.session_state:
            st.session_state['diff'] = 200
        if 'window' not in st.session_state:
            st.session_state['window'] = 20
        if 'sigma' not in st.session_state:
            st.session_state['sigma'] = 1.0

        # Function to reset session state
        def reset_session_state():
            st.session_state['upper'] = 1000
            st.session_state['diff'] = 200
            st.session_state['window'] = 20
            st.session_state['sigma'] = 1.0

        # Button to reset values
        if st.sidebar.button("Reset to initial values"):
            reset_session_state()

        # Create sliders
        upper = st.sidebar.slider("Upper", min_value=120, max_value=data.shape[0], key='upper')
        diff = st.sidebar.slider("Lower", min_value=20, max_value=1000, key='diff')
        lower = upper - diff
        window = st.sidebar.slider("Window", min_value=20, max_value=200, key='window')
        sigma = st.sidebar.slider("Sigma", min_value=0.1, max_value=5.0, key='sigma')

        # Explanation box below the sliders
        st.sidebar.markdown("""
        ### Explanation:
        - **Upper**: Sets the upper limit of the data range for analysis.
        - **Lower**: Controls the difference between the upper and lower limits, adjusting the range of data analyzed.
        - **Window**: Defines the size of the moving average window, affecting the sensitivity to short-term or long-term trends.
        - **Sigma**: Adjusts the sensitivity for detecting significant deviations in price, with lower values detecting smaller spikes.
        """)

        # DEFINE A FUNCTION TO BE USED FOR CURVE FITTING
        # This function models the data with an exponential decay term and a polynomial of degree 3
        def fun(x,a,b,c,d):
            return ((d*((np.exp(-1*b*x))*(np.cos(c*x))))+a)


        # SMOOTH THE DATA TO THE WINDOW SIZE
        # Apply a rolling mean to smooth the 'Open' prices over the specified window
        rm_eth_data = open_eth_data.rolling(window).mean()

        # Apply a rolling mean to smooth the 'Low' prices over the specified window
        rm_low_eth_data = low_eth_data.rolling(window).mean()

        # Apply a rolling mean to smooth the 'High' prices over the specified window
        rm_high_eth_data = high_eth_data.rolling(window).mean()

        # Apply a rolling mean to smooth the 'Range' (High - Low) over the specified window
        rm_rang_eth_data = rang_eth_data.rolling(window).mean()

        # SMOOTH THE HEIKIN-ASHI 'OPEN' DATA TO THE WINDOW SIZE
        # Apply a rolling mean to smooth the Heikin-Ashi 'Open' prices
        heikin_rm_eth_data = open_heikin.rolling(window).mean()

        # CALCULATE THE STANDARD DEVIATION FOR THE ORIGINAL DATAFRAMES WITH THE WINDOW SIZE
        # Compute the rolling standard deviation for the 'Open' prices
        std_open_eth_data = open_eth_data.rolling(window).std()

        # Compute the rolling standard deviation for the 'Low' prices
        std_low_eth_data = low_eth_data.rolling(window).std()

        # Compute the rolling standard deviation for the 'High' prices
        std_high_eth_data = high_eth_data.rolling(window).std()

        # CREATING THE UPPER AND LOWER BOLLINGER BANDS (R)
        # Calculate the upper Bollinger Band by adding the standard deviation multiplied by sigma to the rolling mean
        upper_bollinger_bands_r = rm_eth_data + sigma * std_open_eth_data

        # Calculate the lower Bollinger Band by subtracting the standard deviation multiplied by sigma from the rolling mean
        lower_bollinger_bands_r = rm_eth_data - sigma * std_open_eth_data

        # CALCULATE SIGMOID VALUES
        sigmoids = np.array([])
        for i in range(window, rang_eth_data.shape[0]):
            # Calculate the sigmoid function based on the range and smoothed range data
            if (1 / (1 + np.exp(-rang_eth_data[i] + 4 * rm_rang_eth_data[i]))) > 0.45:
                # Append the index if the sigmoid value exceeds 0.45
                sigmoids = np.append(sigmoids, i)
        print(sigmoids.shape[0])  # Print the number of significant sigmoids found

        # COMPUTE POLYNOMIAL FIT ERRORS
        Errors = np.array([])
        for i in range(1, 11):
            # Fit a polynomial of degree `i` to the 'Open' data within the specified range
            z = np.polyfit(range(0, upper - lower), open_eth_data['Open'][lower - 1:upper - 1], i)
            p = np.poly1d(z)
            # Calculate the absolute sum of errors between the actual and fitted values
            square_error = np.abs(open_eth_data['Open'][lower - 1:upper - 1] - p(range(0, upper - lower))).sum()
            # Append the calculated error
            Errors = np.append(Errors, square_error)

        # Fit a 10th degree polynomial to the 'Open' data
        z = np.polyfit(range(0, upper - lower), open_eth_data['Open'][lower - 1:upper - 1], 10)
        p = np.poly1d(z)



        # Summary Explanation in a Collapsible Section
        with st.expander("Explanation of Charts"):
            st.markdown("""
            - **Heikin-Ashi Chart**: A smoothed candlestick chart that filters market noise, making trends easier to spot.
            - **10-Degree Polynomial Fit**: Displays a polynomial curve fitted to the price data, highlighting trends and patterns that might not be immediately visible.
            - **Error of Polynomial Fit**: Shows the error between the actual data and the fitted polynomial, providing insight into the accuracy and reliability of the fit.
            - **Moving Average Deviation**: Displays price deviations from the moving average, highlighting significant market movements.
            """)


        # FIGURE 1: Heikin-Ashi Candlestick Chart
        fig1, ax_heikin = plt.subplots()

        # candlestick2_ohlc(ax_heikin, open_eth_data['Open'][lower:upper],
        #                   high_eth_data['High'][lower:upper],
        #                   low_eth_data['Low'][lower:upper],
        #                   close_eth_data['Close'][lower:upper], width=0.6)

        # Prepare the data in the format required by candlestick_ohlc
        ohlc_data = []
        for j, i in enumerate(range(lower, upper)):
            ohlc_data.append([j, 
                            open_eth_data['Open'][i], 
                            high_eth_data['High'][i], 
                            low_eth_data['Low'][i], 
                            close_eth_data['Close'][i]])

        # Now plot using candlestick_ohlc
        candlestick_ohlc(ax_heikin, ohlc_data, width=0.6, colorup='g', colordown='r')

        # Plot the smoothed Heikin-Ashi 'Open' prices on the same chart
        ax_heikin.plot(range(0, upper - lower), heikin_rm_eth_data[lower - 1:upper - 1])
        # Plot the lower Bollinger Band on the same chart
        ax_heikin.plot(range(0, upper - lower), lower_bollinger_bands_r['Open'][lower - 1:upper - 1])
        # Plot the upper Bollinger Band on the same chart
        ax_heikin.plot(range(0, upper - lower), upper_bollinger_bands_r['Open'][lower - 1:upper - 1])
        ax_heikin.set_title("Heikin-Ashi Candlestick Chart with Bollinger Bands")
        # Calculate 10 evenly spaced positions between 0 and upper-lower
        num_ticks = 10
        tick_positions = np.linspace(0, upper - lower - 1, num_ticks, dtype=int)  # Positions of the ticks

        # Corresponding labels for these positions
        tick_labels = np.linspace(lower, upper - 1, num_ticks, dtype=int)  # Evenly spaced labels

        ax_heikin.set_xticks(tick_positions)  # Set the positions of the ticks
        ax_heikin.set_xticklabels(tick_labels, rotation=45, ha='right')  # Set the labels of the ticks
        ax_heikin.set_xlabel("Timestamp")
        ax_heikin.set_ylabel("Price")
        st.pyplot(fig1)

        # FIGURE 2: Polynomial Fit Plot
        fig2, ax = plt.subplots()
        ax.plot(range(0, upper - lower), p(range(0, upper - lower)))
        ax.set_xticks(tick_positions)  # Set the positions of the ticks
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')  # Set the labels of the ticks
        ax.set_xlabel("Timestamp")
        ax_heikin.set_ylabel("Price")
        ax.set_title("Fitted Polynomial")
        st.pyplot(fig2)

        # FIGURE 3: Polynomial Fit Errors Plot
        fig3, ax_poly_errors = plt.subplots()
        ax_poly_errors.plot(range(1, 11), Errors)
        ax_poly_errors.set_title("Polynomial Fit Errors")
        ax_poly_errors.set_xlabel("Polynomial degree")
        ax_poly_errors.set_ylabel("Squared error")
        st.pyplot(fig3)




    except Exception as e:
        st.error(f"Error loading the file: {e}")
else:
    st.info('''
            Please upload a file to load the data.

            **Data Format:**  
            The uploaded file should contain historical data in a CSV format,
            where each row represents a specific timestamp with the following columns:
            1. Timestamp (Unix time format)
            2. Low price
            3. High price
            4. Open price
            5. Close price
            6. Volume
            
            You can download an example file [here](https://github.com/kryogenica/Crypto-market/blob/master/Scripts/Historical_eth-usd_data_hourly_step.txt).''')