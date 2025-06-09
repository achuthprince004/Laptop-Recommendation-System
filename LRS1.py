import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset (same directory as script)
csv_path = os.path.join(os.path.dirname(__file__), "laptops.csv")
laptop_data = pd.read_csv(csv_path)

# Create Price Range
laptop_data['Price Range'] = laptop_data['price(in Rs.)'].apply(
    lambda x: 'Low' if x < 60000 else ('Medium' if x < 100000 else 'High')
)

# Prepare collaborative data
def prepare_collaborative_data(data):
    ratings_data = data.dropna(subset=['name', 'rating'])
    ratings_data = ratings_data.groupby('name')['rating'].mean().reset_index()
    user_item_matrix = ratings_data.set_index('name')
    return user_item_matrix

# Calculate similarity
def calculate_cosine_similarity(user_item_matrix):
    user_item_matrix = user_item_matrix.fillna(0)
    cosine_sim = cosine_similarity(user_item_matrix)
    return pd.DataFrame(cosine_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

# Get recommendations
def get_collaborative_recommendations(laptop_name, cosine_sim_df, data, num=5):
    if laptop_name not in cosine_sim_df.index:
        return pd.DataFrame()
    scores = cosine_sim_df[laptop_name].sort_values(ascending=False).drop(laptop_name, errors='ignore')
    top_names = scores.head(num).index.tolist()
    return data[data['name'].isin(top_names)].drop_duplicates(subset=['name'])[['name', 'processor', 'price(in Rs.)', 'img_link']]

# Streamlit UI
st.title("Laptop Recommendation System")
st.header("Find the Best Laptop for You")

# Dynamic dropdowns from data
processor = st.selectbox("Select Processor", options=["Any"] + sorted(laptop_data['processor'].dropna().unique()))
ram = st.selectbox("Select RAM", options=["Any"] + sorted(laptop_data['ram'].dropna().unique()))
os = st.selectbox("Select Operating System", options=["Any"] + sorted(laptop_data['os'].dropna().unique()))
storage = st.selectbox("Select Storage Type", options=["Any"] + sorted(laptop_data['storage'].dropna().unique()))
price_min, price_max = st.slider("Select Price Range", 0, 300000, (0, 300000))

# Filter logic
filtered_laptops = laptop_data[
    ((laptop_data['processor'] == processor) if processor != "Any" else True) &
    ((laptop_data['ram'] == ram) if ram != "Any" else True) &
    ((laptop_data['os'] == os) if os != "Any" else True) &
    ((laptop_data['storage'] == storage) if storage != "Any" else True) &
    (laptop_data['price(in Rs.)'].between(price_min, price_max))
]

st.subheader("Filtered Laptops")
st.write(filtered_laptops[['name', 'processor', 'price(in Rs.)', 'img_link']])

if not filtered_laptops.empty:
    selected_laptop = filtered_laptops.iloc[0]['name']
    st.subheader(f"Recommended Laptops Similar to: {selected_laptop}")
    user_matrix = prepare_collaborative_data(laptop_data)
    cosine_df = calculate_cosine_similarity(user_matrix)
    recommendations = get_collaborative_recommendations(selected_laptop, cosine_df, laptop_data)
    st.write(recommendations)
else:
    st.warning("No matching laptops found. Try adjusting filters.")