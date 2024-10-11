import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
# -*- coding: utf-8 -*-


st.title("Delivery Time Prediction App")
"""
 This app uses the Amazon_delivery dataset to predict delivery time using linear regression modelling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('amazon_delivery.csv')
st.sidebar.title("Operations on the Dataset")

#st.subheader("Checkbox")
w1 = st.sidebar.checkbox("show table", False)
traff = st.sidebar.checkbox("Traffic Vs Delivery Time", False)
plothist= st.sidebar.checkbox("Correlation Heatmap", False)
trainmodel= st.sidebar.checkbox("Train model", False)


if w1:
    st.dataframe(df,width=2000,height=500)



df.isnull().sum()



df.dropna(subset=['Agent_Age', 'Agent_Rating', 'Store_Latitude',
       'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude', 'Order_Date',
       'Order_Time', 'Pickup_Time', 'Weather', 'Traffic', 'Vehicle', 'Area',
       'Delivery_Time', 'Category'], inplace=True)







df = df.drop_duplicates()





df = df.drop(['Order_ID','Order_Date'], axis=1)

"""## Exploratory Data Analysis


"""

features  = list(df)[:-1]


numerical_features = ['Agent_Age', 'Agent_Rating', 'Store_Latitude', 'Store_Longitude',
                      'Drop_Latitude', 'Drop_Longitude', 'Delivery_Time']

features  = list(numerical_features)


plt.figure(figsize=(20, 7))
for i in range(0, len(features)):
    plt.subplot(1, 9, i+1)
    sns.boxplot(y=features[i],data=df,color='green')
    plt.tight_layout()

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Daftar fitur yang ingin Anda periksa untuk outlier
features_with_outliers = ['Agent_Rating',  'Store_Latitude',
 'Store_Longitude',
 'Drop_Latitude',
 'Drop_Longitude', 'Delivery_Time']

# Menghapus outlier untuk setiap fitur
for feature in features_with_outliers:
    df = remove_outliers_iqr(df, feature)

plt.figure(figsize=(15, 7))
for i in range(0, len(features)):
    plt.subplot(1, 9, i+1)
    sns.boxplot(y=features[i],data=df,color='green')
    plt.tight_layout()

"""### Data Visualization"""



df = pd.DataFrame(df)
if traff:
# Create a scatter plot
    fig = px.scatter(
        df,
        x="Delivery_Time",
        y="Traffic",
        size="Delivery_Time",
        color="Traffic",
        hover_name="Traffic",
        size_max=60,
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)




"""Delivery Time based on Weather Distribution"""
df = pd.DataFrame(df)

# Group by weather and calculate average delivery time
weather_avg = df.groupby('Weather')['Delivery_Time'].mean().reset_index()

# Create an area chart
st.area_chart(weather_avg.set_index('Weather'))

plt.figure(figsize=(10, 6))
sns.histplot(df['Delivery_Time'], bins=30, kde=True)
plt.title('Distribution of Delivery Time')
plt.xlabel('Delivery Time')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Weather', data=df, palette='pastel')  # Menggunakan palet warna 'pastel'
plt.title('Distribution of Weather')
plt.xlabel('Weather')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Agent_Rating', y='Delivery_Time', data=df, color='orange', alpha=0.7)
plt.title('Correlation Between Agent Rating and Delivery Time')
plt.xlabel('Agent Rating')
plt.ylabel('Delivery Time')
plt.grid(True)
plt.show()

numerical_features = df[['Agent_Age', 'Agent_Rating', 'Store_Latitude', 'Store_Longitude',
                         'Drop_Latitude', 'Drop_Longitude', 'Delivery_Time']]

plt.figure(figsize=(12, 6))
sns.boxplot(x='Traffic', y='Delivery_Time', data=df, palette='muted')
plt.title('Delivery Time Based on Traffic', fontsize=16)
plt.xlabel('Traffic', fontsize=14)
plt.ylabel('Delivery Time (minutes)', fontsize=14)
plt.grid(True)
plt.show()

df['Delivery_Time'].describe()



    


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


df['Distance_km'] = df.apply(lambda row: haversine(row['Store_Latitude'], row['Store_Longitude'],
                                                   row['Drop_Latitude'], row['Drop_Longitude']), axis=1)

df['Distance_km'] = df['Distance_km'].round(2)




from sklearn.preprocessing import OrdinalEncoder


traffic_order = ['Low', 'Medium', 'High', 'Jam']
encoder = OrdinalEncoder(categories=[traffic_order])

df['Traffic'] = df['Traffic'].str.strip()

df['Traffic_encoded'] = encoder.fit_transform(df[['Traffic']])

df['Traffic'].value_counts()



df['Pickup_Time'] = pd.to_timedelta(df['Pickup_Time'])
df['Order_Time'] = pd.to_timedelta(df['Order_Time'])

df['pickup_time'] = df['Pickup_Time']-df['Order_Time']

unwanted_cols = ['Store_Latitude','Store_Longitude','Drop_Latitude','Drop_Longitude','Traffic']
df.drop(unwanted_cols, inplace=True, axis=1)



df['pickup_time'] = df.pickup_time.astype(str).str.replace('0 days ', '')





df['checking'] = df['pickup_time'].str.contains('days', na=False)



df = df[~df['checking']]



pt = df['pickup_time']



a= pt.str.split(':')
a=pt.str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))

f = a.rename("pickup_time(min)")



df_new = pd.concat([df, f], axis=1)


df = df_new





df['Vehicle'].value_counts()

df["Vehicle"] = df["Vehicle"].astype('category')


df["Vehicle"] = df["Vehicle"].cat.codes




df["Area"] = df["Area"].astype('category')
df["Weather"] = df["Weather"].astype('category')


df['Weather'].value_counts()

df["Area"] = df["Area"].cat.codes
df["Weather"] = df["Weather"].cat.codes








df['Category'].value_counts()



from sklearn.preprocessing import LabelEncoder
# Mengelompokkan data ke dalam kategori
def categorize_category(category):
    if category in ['Apparel', 'Clothing', 'Shoes', 'Jewelry']:
        return 'Fashion dan Aksesoris'
    elif category in ['Skincare', 'Cosmetics', 'Grocery', 'Snacks', 'Pet Supplies']:
        return 'Kebutuhan Sehari-hari dan Perawatan Diri'
    elif category in ['Electronics', 'Books', 'Toys', 'Sports', 'Outdoors']:
        return 'Elektronik, Buku, dan Hobi'
    elif category in ['Home', 'Kitchen']:
        return 'Rumah dan Dapur'
    else:
        return 'Lainnya'

# Menerapkan fungsi kategorisasi
df['Category Group'] = df['Category'].apply(categorize_category)

# Label Encoding pada kolom 'Category Group'
le = LabelEncoder()
df['Category Group Encoded'] = le.fit_transform(df['Category Group'])

# Menampilkan hasil




unwanted_cols = ['Category','pickup_time','checking','Category Group','Order_Time','Pickup_Time']
df.drop(unwanted_cols, inplace=True, axis=1)
df.reset_index(drop=True, inplace=True)





from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
df[['Agent_Age', 'Agent_Rating', 'Weather', 'Vehicle',
    'Area', 'Distance_km', 'Traffic_encoded', 'pickup_time(min)']] = scaler.fit_transform(
    df[['Agent_Age', 'Agent_Rating', 'Weather', 'Vehicle',
        'Area', 'Distance_km', 'Traffic_encoded', 'pickup_time(min)']]
)


# Menampilkan hasil








df_new = df

df_new.reset_index()

# split train test
from sklearn.model_selection import train_test_split

feature = df_new.drop(columns='Delivery_Time')
target = df_new[['Delivery_Time']]

feature_df_new_train, feature_df_new_test, target_df_new_train, target_df_new_test = train_test_split(feature, target, test_size=0.20, random_state=42)

train_shape = feature_df_new_train.shape
test_shape = feature_df_new_test.shape

# Menampilkan hasil
print(f"Jumlah baris dan kolom pada data train: {train_shape}")
print(f"Jumlah baris dan kolom pada data test: {test_shape}")

print(feature_df_new_train.dtypes)







import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Mengasumsikan df_new adalah DataFrame Anda
X = df_new.drop(columns=['Delivery_Time'])  # Menghapus kolom target dari fitur
y = df_new['Delivery_Time']  # Target (waktu pengiriman)

# Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun model Ridge
ridge_model = Ridge(alpha=1.0)  # Anda bisa mencoba beberapa nilai alpha
ridge_model.fit(X_train, y_train)

# Prediksi dan evaluasi model
y_pred = ridge_model.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Ridge): {mse_ridge}")

# Menampilkan koefisien fitur
coefficients = pd.Series(ridge_model.coef_, index=X.columns)
print("Koefisien Fitur:")
print(coefficients)

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Mengasumsikan df_new adalah DataFrame Anda
X = df_new.drop(columns=['Delivery_Time'])  # Menghapus kolom target dari fitur
y = df_new['Delivery_Time']  # Target (waktu pengiriman)

# Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun model Ridge
ridge_model = Ridge(alpha=1.0)  # Anda bisa mencoba beberapa nilai alpha
ridge_model.fit(X_train, y_train)

# Prediksi dan evaluasi model
y_pred = ridge_model.predict(X_test)

# Menghitung metrik kesalahan
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)

# Mean Absolute Percentage Error (MAPE)
mape = (abs((y_test - y_pred) / y_test)).mean() * 100

# Mean Error (ME)
mean_error = (y_test - y_pred).mean()

# Mean Percentage Error (MPE)
mpe = (y_test - y_pred).mean() / y_test.mean() * 100

# Mean Absolute Scaled Error (MASE)
naive_forecast = y_test.shift(1)  # Naive forecast (lagged values)
naive_forecast = naive_forecast[1:]  # Remove the first NaN value
actual_values = y_test.values[1:]  # Align actual values
mae_naive = mean_absolute_error(actual_values, naive_forecast)
mase = mae / mae_naive

# Menampilkan hasil
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
print(f"Mean Error (ME): {mean_error}")
print(f"Mean Percentage Error (MPE): {mpe}%")
print(f"Mean Absolute Scaled Error (MASE): {mase}")


from sklearn.linear_model import Lasso

# Membangun model Lasso
lasso_model = Lasso(alpha=1.0)  # Anda bisa mencoba beberapa nilai alpha
lasso_model.fit(X_train, y_train)

# Prediksi dan evaluasi model
y_pred_lasso = lasso_model.predict(X_test)

# Menghitung metrik kesalahan untuk Lasso
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = mse_lasso ** 0.5
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
mape_lasso = (abs((y_test - y_pred_lasso) / y_test)).mean() * 100
mean_error_lasso = (y_test - y_pred_lasso).mean()
mpe_lasso = (y_test - y_pred_lasso).mean() / y_test.mean() * 100
mae_naive_lasso = mean_absolute_error(actual_values, naive_forecast)
mase_lasso = mae_lasso / mae_naive_lasso

# Menampilkan hasil untuk Lasso
print(f"Root Mean Squared Error (Lasso): {rmse_lasso}")
print(f"Mean Absolute Error (Lasso): {mae_lasso}")
print(f"Mean Absolute Percentage Error (Lasso): {mape_lasso}%")
print(f"Mean Error (Lasso): {mean_error_lasso}")
print(f"Mean Percentage Error (Lasso): {mpe_lasso}%")
print(f"Mean Absolute Scaled Error (Lasso): {mase_lasso}")

"""### Linear Regression"""
if trainmodel:
    from sklearn.linear_model import LinearRegression


    multi_reg = LinearRegression()


    X_df_new_train = feature_df_new_train.to_numpy()
    y_df_new_train = target_df_new_train.to_numpy()
    y_df_new_train = y_df_new_train.reshape(len(y_df_new_train,))

    multi_reg.fit(X_df_new_train, y_df_new_train)

    from sklearn.linear_model import LinearRegression

    # Membangun model Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Prediksi dan evaluasi model
    y_pred_linear = linear_model.predict(X_test)

    correlation_matrix = feature_df_new_train.corr()
    import plotly.figure_factory as ff
    # Create a heatmap using Plotly
    if plothist:
        fig = ff.create_annotated_heatmap(
            z=correlation_matrix.values,
            x=list(correlation_matrix.columns),
            y=list(correlation_matrix.index),
            annotation_text=correlation_matrix.round(2).values,
            colorscale='Viridis'
        )

        # Update layout for better readability
        fig.update_layout(
            title='Feature Correlation Heatmap',
            xaxis_title='Features',
            yaxis_title='Features',
            xaxis=dict(tickangle=-45)
        )

        # Display the heatmap in Streamlit
        st.plotly_chart(fig, use_container_width=True)


    # Menghitung metrik kesalahan untuk Linear Regression
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    rmse_linear = mse_linear ** 0.5
    mae_linear = mean_absolute_error(y_test, y_pred_linear)
    mape_linear = (abs((y_test - y_pred_linear) / y_test)).mean() * 100
    mean_error_linear = (y_test - y_pred_linear).mean()
    mpe_linear = (y_test - y_pred_linear).mean() / y_test.mean() * 100
    mae_naive_linear = mean_absolute_error(actual_values, naive_forecast)
    mase_linear = mae_linear / mae_naive_linear

    # Menampilkan hasil untuk Linear Regression
    print(f"Root Mean Squared Error (Linear Regression): {rmse_linear}")
    print(f"Mean Absolute Error (Linear Regression): {mae_linear}")
    print(f"Mean Absolute Percentage Error (Linear Regression): {mape_linear}%")
    print(f"Mean Error (Linear Regression): {mean_error_linear}")
    print(f"Mean Percentage Error (Linear Regression): {mpe_linear}%")
    print(f"Mean Absolute Scaled Error (Linear Regression): {mase_linear}")

    from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
    from statsmodels.tools.tools import add_constant


    """Multicollinearity"""
    X = add_constant(feature_df_new_train)

    vif_df = pd.DataFrame([vif(X.values, i)
                for i in range(X.shape[1])],
                index=X.columns).reset_index()
    vif_df.columns = ['feature','vif_score']
    vif_df = vif_df.loc[vif_df.feature!='const']
    vif_df

    # heatmap correlation
    df_new_train = pd.concat([feature_df_new_train, target_df_new_train], axis=1)
    corr = df_new_train.corr()

    plt.figure(figsize=(10,7))
    sns.heatmap(corr, annot=True, fmt='.2f')
    plt.show()

    # retrieve the coefficients
    # show as a nice dataframe

    data = feature_df_new_train
    model = multi_reg

    coef_df = pd.DataFrame({
        'feature':['Intercept'] + data.columns.tolist(),
        'coefficient':[model.intercept_] + list(model.coef_)
    })

    coef_df

    y_predict_train = multi_reg.predict(X_df_new_train)
    residual = y_df_new_train - y_predict_train

    # prepare dataframe
    # >1 predictor --> predicted value VS residual
    df_resid = pd.DataFrame({
        'x_axis': y_predict_train,
        'residual': residual
    })

    # residual plot
    sns.scatterplot(data=df_resid, x="x_axis", y="residual")
    plt.axhline(0)
    plt.show()

    # QQplot
    from sklearn.preprocessing import StandardScaler

    std_resid = StandardScaler().fit_transform(residual.reshape(-1,1))
    std_resid = np.array([value for nested_array in std_resid for value in nested_array])

    import statsmodels.api as sm
    sm.qqplot(std_resid, line='45')
    plt.show()

    # calculate residuals
    y_predict_train = multi_reg.predict(X_df_new_train)
    residual = y_df_new_train - y_predict_train

    # prepare dataframe
    # >1 predictor --> predicted value VS residual
    df_resid = pd.DataFrame({
        'x_axis': y_predict_train,
        'residual': residual
    })

    # residual plot
    sns.scatterplot(data=df_resid, x="x_axis", y="residual")
    plt.axhline(0)
    plt.show()



    from sklearn.ensemble import RandomForestRegressor

    # Membangun model Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # Anda bisa mencoba beberapa nilai n_estimators
    rf_model.fit(X_train, y_train)

    # Prediksi dan evaluasi model
    y_pred_rf = rf_model.predict(X_test)

    # Menghitung metrik kesalahan untuk Random Forest
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = mse_rf ** 0.5
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mape_rf = (abs((y_test - y_pred_rf) / y_test)).mean() * 100
    mean_error_rf = (y_test - y_pred_rf).mean()
    mpe_rf = (y_test - y_pred_rf).mean() / y_test.mean() * 100
    mae_naive_rf = mean_absolute_error(actual_values, naive_forecast)
    mase_rf = mae_rf / mae_naive_rf

    # Menampilkan hasil untuk Random Forest
    print(f"Root Mean Squared Error (Random Forest): {rmse_rf}")
    print(f"Mean Absolute Error (Random Forest): {mae_rf}")
    print(f"Mean Absolute Percentage Error (Random Forest): {mape_rf}%")
    print(f"Mean Error (Random Forest): {mean_error_rf}")
    print(f"Mean Percentage Error (Random Forest): {mpe_rf}%")
    print(f"Mean Absolute Scaled Error (Random Forest): {mase_rf}")


    from sklearn.ensemble import GradientBoostingRegressor

    # Membangun model Gradient Boosting
    gbr_model = GradientBoostingRegressor(n_estimators=100, random_state=42)  # Anda bisa mencoba beberapa nilai n_estimators
    gbr_model.fit(X_train, y_train)

    # Prediksi dan evaluasi model
    y_pred_gbr = gbr_model.predict(X_test)

    # Menghitung metrik kesalahan untuk Gradient Boosting
    mse_gbr = mean_squared_error(y_test, y_pred_gbr)
    rmse_gbr = mse_gbr ** 0.5
    mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
    mape_gbr = (abs((y_test - y_pred_gbr) / y_test)).mean() * 100
    mean_error_gbr = (y_test - y_pred_gbr).mean()
    mpe_gbr = (y_test - y_pred_gbr).mean() / y_test.mean() * 100
    mae_naive_gbr = mean_absolute_error(actual_values, naive_forecast)
    mase_gbr = mae_gbr / mae_naive_gbr

    # Menampilkan hasil untuk Gradient Boosting
    print(f"Root Mean Squared Error (Gradient Boosting): {rmse_gbr}")
    print(f"Mean Absolute Error (Gradient Boosting): {mae_gbr}")
    print(f"Mean Absolute Percentage Error (Gradient Boosting): {mape_gbr}%")
    print(f"Mean Error (Gradient Boosting): {mean_error_gbr}")
    print(f"Mean Percentage Error (Gradient Boosting): {mpe_gbr}%")
    print(f"Mean Absolute Scaled Error (Gradient Boosting): {mase_gbr}")



    def calculate_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_true, y_pred)
        mape = (abs((y_true - y_pred) / y_true)).mean() * 100
        mean_error = (y_true - y_pred).mean()
        mpe = (y_true - y_pred).mean() / y_true.mean() * 100
        return rmse, mae, mape, mean_error, mpe

    # Menyimpan hasil metrik dalam dictionary
    results = {}

    # Model Ridge


    # Model Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    results['Linear'] = calculate_metrics(y_test, y_pred_linear)

    # Model Random Forest

    # Membuat DataFrame untuk hasil
    metrics_df = pd.DataFrame.from_dict(results, orient='index', columns=['RMSE', 'MAE', 'MAPE (%)', 'Mean Error', 'Mean Percentage Error (%)'])

    # Menampilkan tabel
    print(metrics_df)

    metrics_df

    """RMSE from the Linear Regression model is 40 minutes"""

    y_pred = model.predict(X_test)

    # Calculate residuals
    residuals = y_test - y_pred

    # Create a DataFrame for plotting
    residuals_df = pd.DataFrame({
        'Predicted': y_pred,
        'Residuals': residuals
    })

    # Create a scatter plot of residuals
    fig = px.scatter(residuals_df, x='Predicted', y='Residuals', labels={'Predicted': 'Predicted Values', 'Residuals': 'Residuals'},
                    title='Residuals vs Predicted Values')

    # Add a horizontal line at y=0 for reference
    fig.add_shape(type='line', x0=min(y_pred), x1=max(y_pred), y0=0, y1=0, line=dict(color='Red', dash='dash'))

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    quantiles = np.percentile(data, np.linspace(0, 100, len(data)))
    theoretical_quantiles = np.percentile(np.random.normal(loc=0, scale=1, size=1000), np.linspace(0, 100, len(data)))

    # Create a DataFrame for plotting
    qq_df = pd.DataFrame({
        'Theoretical Quantiles': theoretical_quantiles,
        'Sample Quantiles': quantiles
    })

    # Create a Q-Q plot
    fig = px.scatter(qq_df, x='Theoretical Quantiles', y='Sample Quantiles', title='Q-Q Plot',
                    labels={'Theoretical Quantiles': 'Theoretical Quantiles', 'Sample Quantiles': 'Sample Quantiles'})

    # Add a 45-degree reference line
    fig.add_shape(type='line', x0=min(theoretical_quantiles), x1=max(theoretical_quantiles),
                y0=min(theoretical_quantiles), y1=max(theoretical_quantiles),
                line=dict(color='red', dash='dash'))

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    from sklearn.model_selection import RandomizedSearchCV

    # Menentukan parameter yang akan dicari
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Menggunakan RandomizedSearchCV untuk mencari kombinasi terbaik
    random_search = RandomizedSearchCV(estimator=gbr_model, param_distributions=param_dist,
                                    scoring='neg_mean_squared_error', n_iter=10, cv=3, verbose=1, n_jobs=-1)

    # Melatih model dengan RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Menampilkan hasil terbaik
    print("Best parameters found: ", random_search.best_params_)
    print("Best cross-validation score: ", -random_search.best_score_)

    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    from sklearn.metrics import mean_squared_error

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE: {rmse}")



    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})


    plt.figure(figsize=(10, 6))


    sns.set(style="whitegrid")


    sns.scatterplot(x='Actual', y='Predicted', data=results_df, color='blue', s=100, label='Predicted Values', alpha=0.7)


    min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
    max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')


    plt.title('Actual vs Predicted Values', fontsize=16)
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid(True)


    plt.show()
