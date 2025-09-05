import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("🚫 GROQ_API_KEY not set in environment!")
    st.stop()

# Embeddings & Vector Store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    folder_path="faiss_index_insights2",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Prompt Template
prompt_template = """
You are a helpful and concise interview assistant for IIT Kanpur students.

Rules:
- Answer ONLY using the context below.
- Say "Sorry, I don't have enough information" if context is insufficient.
- Do NOT generalize. Stick to the selected company.
- Use bullet points if there are multiple facts.

🎯 Company: {company}

📌 Context:
{context}

❓ Question: {question}

💡 Answer:
"""

prompt = PromptTemplate(
    input_variables=["company", "context", "question"],
    template=prompt_template
)

# Load LLM and Chain
llm = ChatGroq(
    temperature=0.1,
    model_name="llama3-8b-8192",
    groq_api_key="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"
)
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Define function to prepare input
def prepare_input(query, company):
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"company": company, "question": query, "context": context}

# --- Streamlit UI ---
st.set_page_config(page_title="IITK Student-insights")
st.title("🎓 IIT Kanpur Interview Insights Assistant (Created by Abheet Sonker)")

# Company lists
placement_company_list = ['Alan Harshan Jaguar Land Rover India Limited',
 'Applied Intelligence',
 'Bajaj Auto Limited  Chetak Technology Limited',
 'Bajaj Electricals',
 'Balvantbhai Sap Labs',
 'Barclays',
 'Battery Smart',
 'Caterpillar',
 'Chaudhuri United Airlines',
 'Cohesity',
 'Das Kivi Capital',
 'Databricks',
 'Deutsche India Private Limited',
 'Electric',
 'Finmechanics',
 'Flipkart',
 'Ford Business Solutions',
 'Glean',
 'Godrej  Boyce',
 'Google',
 'Greyorange',
 'Group India',
 'Gupta Raspa Pharma Private Limited',
 'Hilabs',
 'Hilti Technology',
 'Hitachi Energy',
 'Hpcl',
 'Hyper Bots System',
 'Icici Bank',
 'Idfc First Bank',
 'Jadhav Flipkart',
 'Jaguar Land Rover India Limited',
 'Javis',
 'Karthik Ormae',
 'Kumar Goud Sap Labs',
 'Kunsoth Merilytics',
 'Larsen And Toubro Limited',
 'Mahindra Susten Pvt Ltd',
 'Master Card',
 'Medianetdirecti',
 'Mekala Western Digital',
 'Menon Publicis Sapient',
 'Microsoft India',
 'Na',
 'Navi',
 'Neterwala Group Aon',
 'Niva Bupa Health Insurance',
 'Nmtronics India Private Limited',
 'Novo Nordisk',
 'Npci',
 'Ola',
 'Palo Alto Networks',
 'Petronet Lng Limited',
 'Pharmaace',
 'Pine Labs',
 'Qualcomm',
 'Quantbox Research',
 'Rosy Blue India Pvt Ltd',
 'Singh Nvidia',
 'Smarttrak Ai',
 'Solutions Pvt Ltd',
 'Sprinklr',
 'Steel',
 'Stripe',
 'Taiwan Semiconductor Manufacturing Company',
 'Tata Projects Ltd',
 'Technology',
 'Thakur Navi',
 'Uniorbit Technologies',
 'Uniorbit Technologies Private Ltd',
 'Varghese Mrf Limited',
 'World Quant',
 'Zomato']

intern_company_list = ['Amazon',
    'American Express',
    'Anuj Rubrik',
    'Atlassian',
    'Axxela',
    'Bain  Company',
    'Bain And Company',
    'Barclays',
    'Bcg',
    'Bny Mellon',
    'Bosch',
    'Boston Consulting Group',
    'Cisco',
    'Citi',
    'Citi Bank',
    'Cohesity',
    'Databricks Sde',
    'Deshpande Alphagrep Securities Private Limited',
    'Discovery',
    'Dr Reddys Laboratories',
    'Dr Reddys Laboratories  Core Engineering',
    'Edelweiss Financial Services Limited',
    'Express',
    'Finmechanics',
    'Flipkart',
    'Geetika Uber',
    'Glean',
    'Goldman Sachs',
    'Google',
    'Google India',
    'Graviton',
    'Hindustan Unilever Limited',
    'Hul',
    'India',
    'Indxx',
    'Itc Limited',
    'Itc Ltd',
    'J P Morgan  Chase',
    'Jaguar Land Rover India Limited',
    'Jindal Medianet',
    'Jiwane Oracle',
    'Jlr',
    'Jp Morgan Chase',
    'Jsw',
    'Kane American Express',
    'Kumar Samsung Noida',
    'Mckinsey  Company',
    'Medianet',
    'Microsoft India',
    'Mondelez International',
    'Morgan Stanley',
    'N Bny Mellon',
    'National Payments Corporation Of India Npci',
    'Nestle',
    'Nk Security',
    'Nobroker',
    'Nobroker Technologies',
    'Nvidia',
    'Optiver',
    'Optiverquant Role',
    'Quadeye',
    'Quantbox',
    'R Qualcomm',
    'Raj Databricks',
    'Rubrik',
    'Sachan Ibm',
    'Sachs',
    'Saluja Mastercard',
    'Samplytics Technologies Private Limited',
    'Samsung Rd Banglore',
    'Samsung Research Bangalore',
    'Samsung South Korea',
    'Sprinklr',
    'Standard Chartered',
    'Tata Steel',
    'Texas Instruments',
    'Tiwari De Shaw',
    'Tomar Jlr',
    'Tower Research Capital',
    'Trexquant Investment Llp',
    'Uber',
    'Vedanta Resources Limited',
    'Winzo Games']


query_types = ["Sample Interview Questions", "Interview Process", "Resources", "Advice"]

# UI Inputs
purpose = st.selectbox("🎯 Select Purpose", ["Placement", "Intern"])
company_list = placement_company_list if purpose == "Placement" else intern_company_list
company = st.selectbox("🏢 Select Company", company_list)

query_type_selection = st.multiselect("🔎 Choose Query Type(s)", query_types)
custom_question = st.text_area("💬 Or Ask Custom Question", placeholder="e.g. Resources for Google")

submit = st.button("🧠 Ask")

# --- Processing ---
if submit:
    if not company:
        st.warning("❗ Please select a company.")
        st.stop()

    queries = []
    if query_type_selection:
        queries = [f"Give me {qt.lower()} only for {company}." for qt in query_type_selection]
    elif custom_question.strip():
        queries = [custom_question.strip()]
    else:
        st.warning("❗ Please select a query type or enter a question.")
        st.stop()

    for q in queries:
        try:
            input_data = prepare_input(q, company)
            result = llm_chain.invoke(input_data)
            answer = result["text"].strip()

            st.markdown(f"### ❓ {q}")
            st.markdown(f"**💡 Answer:**\n\n{answer}")
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")