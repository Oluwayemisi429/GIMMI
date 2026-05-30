
import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('GIMMI_light.pkl')
scaler = joblib.load('GIMMI_scaler_light.pkl')

st.title("GIMMIAI 🚀")
st.subheader("Post Performance Predictor for Content Marketing")
st.write("Fill in your post details below to predict performance:")

follower_count = st.number_input("Follower Count", min_value=0)
post_hour = st.slider("Post Hour (0-23)", 0, 23, 12)
caption_length = st.number_input("Caption Length (characters)", min_value=0)
hashtags_count = st.number_input("Number of Hashtags", min_value=0)
has_call_to_action = st.selectbox("Has Call to Action?", [0, 1])
engagement_rate = st.number_input("Expected Engagement Rate (%)", min_value=0.0)
likes = st.number_input("Likes", min_value=0)
comments = st.number_input("Comments", min_value=0)
shares = st.number_input("Shares", min_value=0)
saves = st.number_input("Saves", min_value=0)
reach = st.number_input("Reach", min_value=0)
impressions = st.number_input("Impressions", min_value=0)
followers_gained = st.number_input("Followers Gained", min_value=0)

if st.button("Predict Performance"):
    input_data = pd.DataFrame([[
        follower_count, has_call_to_action, post_hour,
        likes, comments, shares, saves, reach, impressions,
        engagement_rate, followers_gained, caption_length,
        hashtags_count] + [0]*23], columns=range(36))

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"GIMMI predicts this post will perform: {str(prediction[0]).upper()}")
