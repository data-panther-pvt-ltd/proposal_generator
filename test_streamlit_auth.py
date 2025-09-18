import streamlit as st
from utils.auth import authenticate, show_user_info

# Require login
auth_status, authenticator = authenticate()

# Show user info in sidebar
show_user_info(authenticator)

# Main app content only runs if authenticated\
if auth_status:
    st.title("My Secure App 🚀")
    st.write("streamlit-authenticator==0.2.3")
    st.write("This content is only visible to logged-in users.")
