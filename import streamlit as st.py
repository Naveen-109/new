import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('model/calories_model.pkl')

# Set the title of the app
st.title("AI-powered Personal Health Assistant")

st.write("""
## Enter Your Daily Health Data
""")

# User inputs
steps = st.number_input('Steps per Day', min_value=0, max_value=50000, value=5000, step=100)
sleep = st.slider('Hours of Sleep', min_value=0.0, max_value=12.0, value=7.0, step=0.1)
heart_rate = st.number_input('Heart Rate (BPM)', min_value=30, max_value=200, value=70, step=1)

if st.button('Get Recommendations'):
    # Create a DataFrame for the input
    user_data = pd.DataFrame({
        'steps_per_day': [steps],
        'hours_of_sleep': [sleep],
        'heart_rate': [heart_rate]
    })
    
    # Predict calories burned
    predicted_calories = model.predict(user_data)[0]
    st.success(f"*Predicted Calories Burned:* {predicted_calories:.0f} kcal")
    
    # Personalized Recommendations
    recommendations = []
    
    # Assuming average calories burned is 2500 kcal
    if predicted_calories < 2000:
        recommendations.append("Your predicted calorie burn is below average. Consider increasing your activity levels.")
    elif predicted_calories > 3000:
        recommendations.append("Your predicted calorie burn is above average. Ensure you're maintaining a balanced diet.")
    else:
        recommendations.append("Your calorie burn is within the average range. Keep up the good work!")
    
    # Sleep recommendations
    if sleep < 7:
        recommendations.append("Aim for at least 7 hours of sleep for better recovery.")
    elif sleep > 9:
        recommendations.append("Ensure you're not oversleeping, which can affect your daily energy levels.")
    
    # Steps recommendations
    if steps < 5000:
        recommendations.append("Try to increase your daily steps to at least 7,500 for improved health.")
    elif steps > 15000:
        recommendations.append("Great job on maintaining a high activity level! Ensure adequate rest.")
    
    # Heart rate recommendations
    if heart_rate < 60:
        recommendations.append("Your heart rate is lower than average. Consult a healthcare provider if you experience symptoms.")
    elif heart_rate > 100:
        recommendations.append("Your heart rate is higher than average. Consider stress-relief activities or consult a healthcare provider.")
    
    st.write("## Personalized Recommendations:")
    for rec in recommendations:
        st.write(f"- {rec}")
    
    # Optional: Visualize the input data
    st.write("## Your Health Data vs. Average")
    avg_data = {
        'steps_per_day': 7500,
        'hours_of_sleep': 7,
        'heart_rate': 75
    }
    user_data_flat = user_data.iloc[0].to_dict()
    categories = ['Steps per Day', 'Hours of Sleep', 'Heart Rate (BPM)']
    user_values = [user_data_flat['steps_per_day'], user_data_flat['hours_of_sleep'], user_data_flat['heart_rate']]
    avg_values = [avg_data['steps_per_day'], avg_data['hours_of_sleep'], avg_data['heart_rate']]
    
    x = np.arange(len(categories))  # label locations
    width = 0.35  # bar width
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, user_values, width, label='You')
    rects2 = ax.bar(x + width/2, avg_values, width, label='Average')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Values')
    ax.set_title('Your Health Data vs. Average')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Attach a text label above each bar in rects, displaying its height.
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(int(height)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    st.pyplot(fig)