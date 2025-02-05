import streamlit as st
import pandas as pd
import joblib
import re
import string

model = joblib.load('job.pkl')


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def contains_contact_info(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'\b\d{10}\b'  # Matches a 10-digit phone number
    
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))


def predict_job_real_or_fake(descriptions):
    preprocessed_descriptions = [preprocess_text(desc) for desc in descriptions]
    predictions = model.predict(preprocessed_descriptions)
    return predictions


def main():
    st.title('Job Posting Authenticity Checker')
    st.sidebar.title('About')
    st.sidebar.info('This app helps predict whether a job posting is real or fake.')

    st.subheader('Enter Job Information')
    name = st.text_input('Name')
    company = st.text_input('Company')
    job_title = st.text_input('Job Title')

    st.subheader('Enter Job Description(s)')
    descriptions = st.text_area('Description(s)', height=200)
    descriptions = descriptions.split('\n')  

    if st.button('Predict'):
        if name and company and job_title and descriptions:
            results = []
            for desc in descriptions:
                if not contains_contact_info(desc):
                    st.error(f'Job description: {desc}\nThis is a fake job posting (No contact info found).')
                    results.append((desc, 1))  # Fake job (1)
                else:
                    prediction = predict_job_real_or_fake([desc])[0]
                    results.append((desc, prediction))

                    if prediction == 1:
                        st.error(f'Job description: {desc}\nThis is a fake job posting.')
                    else:
                        st.success('Your job has been posted successfully.')

                df = pd.DataFrame({'Name': [name], 'Company': [company], 'Job Title': [job_title], 'Description': [desc], 'Prediction': [results[-1][1]]})
                df.to_csv('your_postings.csv', mode='a', index=False, header=False)

        else:
            st.warning('Please enter all the required information.')

    st.sidebar.title('Your Postings')
    try:
        st.sidebar.dataframe(pd.read_csv('your_postings.csv'))
    except FileNotFoundError:
        st.sidebar.warning('Your postings will appear here once you make predictions.')


if __name__ == '__main__':
    main()
