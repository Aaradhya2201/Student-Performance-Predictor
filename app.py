# app.py
import streamlit as st
from model import load_or_train_model, predict_performance
from langchain_utils import setup_langchain, get_explanation, get_chat_response


st.set_page_config(page_title="Student Performance Predictor", layout="wide", initial_sidebar_state="expanded")


st.markdown("""
    <style>
    .stTabs {
        background: transparent;
    }
    .stTabs > div > button {
        background-color: #3498db;
        color: white;
        border-radius: 5px 5px 0 0;
        margin-right: 5px;
        padding: 10px;
    }
    .stTabs > div > button:hover {
        background-color: #2980b9;
    }
    .tab-content {
        padding: 20px;
        color: #2c3e50; /* Dark text for contrast with default light background */
    }
    .sidebar .sidebar-content {
        background: #2c3e50;
        color: #ecf0f1; /* Light text for dark sidebar */
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #27ae60;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #219653;
    }
    h1, h2, h3 {
        color: #ffffff; /* Slightly darker for better readability */
        font-family: 'Arial', sans-serif;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    .stExpander {
        background: #ffffff;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }
    .stExpander > div > div {
        color: #2c3e50;
    }
    .footer {
        text-align: center;
        color: #000000;
        padding: 10px;
        font-size: 12px;
        background: #2c3e50;
        border-radius: 5px;
        box-shadow: 0 -2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

def main():
    model, encoders = load_or_train_model()
    if model is None or encoders is None:
        return
    
    langchain_chain = setup_langchain()
    st.session_state['explanation'] = "no context"
    
  
    tab1, tab2, tab3 = st.tabs(["📊Performance Prediction", "🤖Chatbot", "ℹ️About"])
    
   
    with tab1:
        st.title("🎓Student Performance Predictor")
        st.markdown("**Predict your academic performance with AI-powered insights.**", unsafe_allow_html=True)
        
        with st.sidebar:
            st.markdown("### Input Your Details")
            with st.expander("Personal Info", expanded=True):
                gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
                age = st.number_input("Age", min_value=16, max_value=30, value=20, key="age")
                department = st.selectbox("Department", ["Engineering", "Business", "CS", "Mathematics"], key="dept")
            
            with st.expander("Academic Metrics"):
                attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, key="attendance")
                midterm_score = st.number_input("Midterm Score", min_value=0.0, max_value=100.0, value=50.0, key="midterm")
                final_score = st.number_input("Final Score", min_value=0.0, max_value=100.0, value=50.0, key="final")
                assignments_avg = st.number_input("Assignments Avg", min_value=0.0, max_value=100.0, value=50.0, key="assign")
                quizzes_avg = st.number_input("Quizzes Avg", min_value=0.0, max_value=100.0, value=50.0, key="quizzes")
                participation_score = st.number_input("Participation Score", min_value=0.0, max_value=10.0, value=5.0, key="part")
                projects_score = st.number_input("Projects Score", min_value=0.0, max_value=100.0, value=50.0, key="proj")
                total_score = st.number_input("Total Score", (midterm_score + final_score + assignments_avg + quizzes_avg + participation_score*10 + projects_score) / 6, key="total")
            
            with st.expander("Lifestyle Factors"):
                study_hours = st.number_input("Study Hours per Week", min_value=0.0, max_value=50.0, value=10.0, key="study")
                extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"], key="extra")
                internet_access = st.selectbox("Internet Access at Home", ["Yes", "No"], key="internet")
                parent_edu = st.selectbox("Parent Education Level", [ "High School", "Bachelor's", "Master's", "PhD"], key="parent")
                family_income = st.selectbox("Family Income Level", ["Low", "Medium", "High"], key="income")
                stress_level = st.number_input("Stress Level", min_value=0, max_value=10, value=5, key="stress")
                sleep_hours = st.number_input("Sleep Hours per Night", min_value=0.0, max_value=12.0, value=7.0, key="sleep")
            
            submit_button = st.button("🎯Predict Performance")
        
        if submit_button:
            input_data = {
                'Gender': gender, 'Age': age, 'Department': department, 'Attendance (%)': attendance,
                'Midterm_Score': midterm_score, 'Final_Score': final_score, 'Assignments_Avg': assignments_avg,
                'Quizzes_Avg': quizzes_avg, 'Participation_Score': participation_score, 'Projects_Score': projects_score,
                'Total_Score': total_score, 'Study_Hours_per_Week': study_hours, 'Extracurricular_Activities': extracurricular,
                'Internet_Access_at_Home': internet_access, 'Parent_Education_Level': parent_edu,
                'Family_Income_Level': family_income, 'Stress_Level': stress_level, 'Sleep_Hours_per_Night': sleep_hours
            }
            
            try:
                predicted_grade = predict_performance(model, encoders, input_data)
                performance_map = {'A': "Excellent", 'B': "Good", 'C': "Average", 'D': "Needs Improvement", 'F': "Poor"}
                performance = performance_map.get(predicted_grade, "Unknown")
                
                features_str = ", ".join([f"{k}: {v}" for k, v in input_data.items()])
                explanation = get_explanation(langchain_chain, predicted_grade, features_str)
                
                st.subheader("📈Prediction Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"Predicted Grade: {predicted_grade}", icon="✅")
                    
                with col2:
                    
                    st.info(f"Performance: {performance}", icon="📊")
                
                st.session_state['context'] = "grade : " + predicted_grade + " " +  explanation
                st.markdown("**Explanation:**")
                st.write(explanation)
            except ValueError as e:
                st.error(f"Prediction failed: {str(e)}. Check your input data against the training dataset.")
    

    with tab2:
        st.title("🤖Student Chatbot")
        st.markdown("**Ask our AI assistant for personalized advice!**", unsafe_allow_html=True)
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        user_query = st.text_input("Your question:", key="chat_input", placeholder="e.g., How can I improve my grades?")
        if st.button("Send"):
            if user_query:
                print(st.session_state["context"])
                response = get_chat_response(langchain_chain, user_query,st.session_state["context"])
                st.session_state.chat_history.append({"user": user_query, "bot": response})
        
        st.markdown("### Chat History")
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**Assistant:** {chat['bot']}")
            st.markdown("---")
 
    with tab3:
        st.title("ℹ️About Us")
        st.markdown("""
                    
        ### 👨‍🎓 Developed by Students at Symbiosis Institute of Technology, Nagpur  
        This app is a smart tool to predict student academic performance using machine learning and generative AI. 
        ### Our Mission
        Empowering students to understand and improve their academic performance with cutting-edge AI technology.
        
        ### Product Details
        The **Student Performance Predictor** uses a Random Forest model trained on real student data, combined with Google's Gemini AI, to:
        - Predict your grade based on academic and lifestyle factors.
        - Provide friendly, actionable explanations.
        - Offer a chatbot for personalized study advice.
        
        ### Features
        - **Accurate Predictions**: Leverage machine learning for precise grade forecasts.
        - **AI Explanations**: Get insights tailored to your input.
        - **Interactive Chatbot**: Ask questions and get real-time responses.
        
        ### Our Team
        - **Aaradhya Bali**
        - **Gaurav Katre**
        - **Kaushik Tamgadge**
        
        ### Get Involved
        Have feedback or ideas? Contact us at aaradhyabali2004@gmail.com!
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="footer">© 2025 Student Performance Predictor | Powered by Students Of Symbiosis Institute Of Technology, Nagpur</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()