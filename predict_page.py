import streamlit as st
import pickle as pkl
import numpy as np
import pandas as pd
import shap
import seaborn as sns
import matplotlib.pyplot as plt

def load_model():
        with open('assets/wal_model.pkl', 'rb') as f:
            wal_model = pkl.load(f)
        return wal_model

def load_x_train():
    with open('assets/x_train_file.pkl', 'rb') as f:
        X_train = pkl.load(f)
    return X_train

def show_predict_page():
    
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0B59D5;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.image('assets/buzzbuy-logo.png', width=200)
    st.title('BuzzBuy Prediction')

    X_train = load_x_train()
    wal_model = load_model()

    subcategories = ['Dairy', 
                     'Beverages',
                     'Rice & Rice Products', 
                     'DMart Grocery', 
                     'Grocery', 
                     'Fruits & Vegetables', 
                     'Dry Fruits',
                     'Dals', 
                     'Pulses',
                     'Masala & Spices',
                     'Salt / Sugar / Jaggery', 
                     'Cooking Oil', 
                     'Flours & Grains', ]

    price = st.number_input('Price: ', min_value=0.00, step=0.01)
    subcategory = st.selectbox('Subcategory: ', subcategories)
    shelf_life = st.number_input('Current Shelf Life: ', min_value=0)
    max_shelf_life = st.number_input('Max Shelf Life: ', min_value=0)
    inventory_count = st.number_input('Inventory Count: ', min_value=0)
    current_sales = st.number_input('Current Sales: ', min_value=0)

    submitted = st.button('Predict')

    if submitted:
        to_predict = pd.DataFrame({
            'Price': [price],
            'SubCategory': [subcategory],
            'shelf_life': [shelf_life],
            'max_shelf_life': [max_shelf_life],
            'inventory_count': [inventory_count],
            'current_sales': [current_sales]
        })

        discounted_price = wal_model.predict(to_predict)

        st.subheader(f'The discounted price is Rs.{discounted_price[0]}')

        # Initialize SHAP JavaScript visualization
        shap.initjs()

        # Convert the sparse matrix to a dense format
        X_train_dense = wal_model.named_steps['preprocessor'].transform(X_train).toarray()

        # Create the SHAP explainer using the dense matrix
        explainer = shap.Explainer(wal_model.named_steps['model'], X_train_dense)

        # Example of making a prediction with new data
        # new_data = pd.DataFrame({
        #     'Price': [1000],
        #     'SubCategory': ['Grocery/Dry Fruits'],
        #     'shelf_life': [364],
        #     'max_shelf_life': [365],
        #     'inventory_count': [500],
        #     'current_sales': [50]
        # })

        # Predict the discounted price
        predicted_discounted_price = wal_model.predict(to_predict)
        print(f'Predicted Discounted Price: {predicted_discounted_price[0]}')
        transformed_new_data_dense = wal_model.named_steps['preprocessor'].transform(to_predict).toarray()

        # Calculate SHAP values for the new data
        shap_values = explainer(transformed_new_data_dense)

        # Get feature names
        feature_names = wal_model.named_steps['preprocessor'].get_feature_names_out()

        mean_shap_values = np.mean(shap_values.values, axis=0)

        # Create a DataFrame for filtering negative impacts
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean SHAP Value': mean_shap_values
        })

        # Filter for negative impacts only
        negative_shap_df = shap_df[shap_df['Mean SHAP Value'] < 0].sort_values(by='Mean SHAP Value')

        # Plot the SHAP values for features contributing to lower the price
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Mean SHAP Value', y='Feature', data=negative_shap_df, palette='Blues_r')
        plt.title('Factors Contributing to Lower Discounted Price')
        plt.xlabel('Mean SHAP Value')
        plt.ylabel('Feature')

        st.pyplot(plt)
