# testgit
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from statsmodels.tsa.seasonal import seasonal_decompose  
from statsmodels.tsa.arima_model import ARIMA  
from sklearn.metrics import mean_squared_error  
  
  # Load the dataset  
  df = pd.read_csv('demand_data.csv', index_col='date', parse_dates=['date'])  
    
    # Convert the index to datetime format  
    df.index = pd.to_datetime(df.index)  
      
      # Plot the original time series  
      plt.plot(df)  
      plt.title('Original Time Series')  
      plt.xlabel('Date')  
      plt.ylabel('Demand')  
      plt.show()  
        
        # Decompose the time series into trend, seasonality, and residuals  
        decomposition = seasonal_decompose(df, model='additive')  
        trend = decomposition.trend  
        seasonal = decomposition.seasonal  
        residual = decomposition.resid  
          
          # Plot the decomposed components  
          plt.subplot(411)  
          plt.plot(df, label='Original')  
          plt.legend(loc='best')  
          plt.subplot(412)  
          plt.plot(trend, label='Trend')  
          plt.legend(loc='best')  
          plt.subplot(413)  
          plt.plot(seasonal,label='Seasonality')  
          plt.legend(loc='best')  
          plt.subplot(414)  
          plt.plot(residual, label='Residuals')  
          plt.legend(loc='best')  
          plt.tight_layout()  
          plt.show()  
            
            # Fit an ARIMA model to the residuals  
            model = ARIMA(residual, order=(1,1,1))  
            model_fit = model.fit(disp=0)  
              
              # Forecast the next 30 days  
              forecast, stderr, conf_int = model_fit.forecast(steps=30)  
                
                # Plot the forecast  
                plt.plot(df)  
                plt.plot(np.arange(len(df), len(df)+30), forecast, label='Forecast')  
                plt.fill_between(np.arange(len(df), len(df)+30), conf_int[:,0], conf_int[:,1], alpha=0.2, label='Confidence Interval')  
                plt.title('Forecast')  
                plt.xlabel('Date')  
                plt.ylabel('Demand')  
                plt.legend(loc='best')  
                plt.show()  
                  
                  # Evaluate the model using mean squared error  
                  mse = mean_squared_error(df, model_fit.fittedvalues)  
                  print('Mean Squared Error:', mse)  
                  