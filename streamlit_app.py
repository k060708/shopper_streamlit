import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu


st.set_page_config(page_title="Shopper Spectrum",layout="wide",initial_sidebar_state="expanded")

# CUSTOM CSS 
st.markdown("""<style>.main-title {
        font-size: 42px;
        font-weight: 800;
        color: #1f1f1f;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
        color: white;
    }

    /*PRODUCT CARDS (force black text)*/
    .product-card {
        background-color: #f8f9fa !important;
        color: #000000 !important;
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #ff4b4b;
        font-weight: 600;
        font-size: 16px;
        display: block;
    }
    .product-card * {
        color: #000000 !important;
    }

    /*SEGMENT BOX (force black text)*/
    .segment-box {
        background-color: #f0f2f6 !important;
        color: #000000 !important;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    .segment-box h3,
    .segment-box p,
    .segment-box span {
        color: #000000 !important;
        margin: 0;
    }
    .segment-box .highlight {
        color: #ff4b4b !important;
        font-weight: 700;
    }

    /*CLUSTER LABE */
    .cluster-label {
        color: #ffffff !important;   /* white so it's visible on dark bg */
        font-size: 22px;
        font-weight: 600;
        margin-top: 15px;
    }
    </style> """, unsafe_allow_html=True)

# LOAD MODELS AND DATA (cached for performance)
@st.cache_resource
def load_clustering_model():
    """Load the KMeans model saved from your clustering script."""
    with open("best_kmeans_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    return model_data


@st.cache_data
def load_retail_data():
    """Load cleaned retail data for product recommendation."""
    df = pd.read_csv("cleaned_online_retail.csv")
    df["Description"] = df["Description"].astype(str).str.strip().str.upper()
    return df


@st.cache_data
def build_product_similarity(df):
    """
    Build item-item similarity using collaborative filtering.
    Creates a Customer × Product matrix and computes cosine similarity
    between products based on which customers bought them.
    """
    # Create customer-product purchase matrix
    basket = (df.groupby(["CustomerID", "Description"])["Quantity"].sum().unstack().fillna(0))

    # Binary: bought (1) or not (0)
    basket = (basket > 0).astype(int)

    # Compute cosine similarity between products (columns)
    # Transpose so rows are products
    product_matrix = basket.T
    similarity     = cosine_similarity(product_matrix)

    similarity_df = pd.DataFrame(similarity,index=product_matrix.index, columns=product_matrix.index)
    return similarity_df


def get_recommendations(product_name, similarity_df, top_n=5): 
    """Return top N most similar products to the given product."""
    product_name = product_name.strip().upper()

    if product_name not in similarity_df.index:
        return None

    # Get similarity scores for the product, sorted descending
    similar_scores = similarity_df[product_name].sort_values(ascending=False)

    # Exclude the product itself (first result)
    similar_products = similar_scores.iloc[1:top_n + 1].index.tolist()

    return similar_products


def predict_customer_segment(recency, frequency, monetary, model_data):
    """Predict which segment a customer belongs to."""
    kmeans      = model_data["kmeans_model"]
    scaler      = model_data["scaler"]
    segment_map = model_data["segment_labels"]

    new_customer         = np.array([[recency, frequency, monetary]])
    new_customer_scaled  = scaler.transform(new_customer)
    cluster              = kmeans.predict(new_customer_scaled)[0]
    segment              = segment_map.get(cluster, "Unknown")

    return cluster, segment


# SIDEBAR NAVIGATION
with st.sidebar:
    st.markdown("## Shopper Spectrum")
    st.markdown("---")

    selected = option_menu(
        menu_title=None,
        options=["Home", "Clustering", "Recommendation"],
        icons=["house", "diagram-3", "bag-heart"],
        default_index=0,
        styles={
            "container":       {"padding": "5px", "background-color": "#f8f9fa"},
            "icon":            {"color": "#333", "font-size": "18px"},
            "nav-link":        {"font-size": "16px", "text-align": "left", "margin": "5px 0", "border-radius": "8px","color": "#333"},
            "nav-link-selected": {"background-color": "#267dbf", "color": "white"},
        } )

# HOME PAGE
if selected == "Home":
    st.markdown('<div class="main-title"> Shopper Spectrum</div>', unsafe_allow_html=True)
    st.markdown("### Customer Segmentation and Product Recommendation System")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### Product Recommendation
        Get **5 similar product recommendations** based on
        collaborative filtering from past customer purchases.

        - Enter any product name
        - Discover related products
        - Powered by item-item cosine similarity
        """)

    with col2:
        st.markdown("""
        #### Customer Segmentation
        Predict which customer segment a shopper belongs to
        based on their **RFM** (Recency, Frequency, Monetary) values.

        - **High-Value**: Recent, frequent, big spenders
        - **Regular**: Steady purchasers
        - **Occasional**: Rare buyers
        - **At-Risk**: Haven't purchased in a long time
        """)

    st.markdown("---")
    st.info("Use the sidebar to navigate to **Recommendation** or **Clustering**.")


# CLUSTERING (CUSTOMER SEGMENTATION) PAGE
elif selected == "Clustering":
    st.markdown('<div class="main-title">Customer Segmentation</div>',
                unsafe_allow_html=True)
    st.markdown("Predict the customer segment based on RFM values.")
    st.markdown("---")

    # Load model
    try:
        model_data = load_clustering_model()
    except FileNotFoundError:
        st.error(" Could not find `best_kmeans_model.pkl`. "
                 "Please run the clustering script first.")
        st.stop()

    # INPUT FIELDS
    recency = st.number_input(
        "Recency (days since last purchase)",
        min_value=0, max_value=1000,
        value=325, step=1,
        help="How many days ago did the customer last make a purchase?")

    frequency = st.number_input(
        "Frequency (number of purchases)",
        min_value=0, max_value=500,
        value=1, step=1,
        help="Total number of unique transactions by the customer")

    monetary = st.number_input(
        "Monetary (total spend)",
        min_value=0.0, max_value=1_000_000.0,
        value=765322.00, step=10.0, format="%.2f",
        help="Total amount the customer has spent (£)")

    # PREDICT BUTTON 
    if st.button("Predict Segment"):
        cluster, segment = predict_customer_segment(recency, frequency, monetary, model_data)

        # Display result
        st.markdown(f"### {cluster}")
        st.markdown(
            f"<div class='segment-box'>"
            f"<h3>This customer belongs to: "
            f"<span style='color:#000000;'>{segment} Shopper</span></h3>"
            f"</div>",
            unsafe_allow_html=True)

        # Show cluster profile info
        with st.expander("View Cluster Details"):
            profiles = model_data["cluster_profiles"]
            st.dataframe(profiles, use_container_width=True)

            st.markdown("**Segment Meanings:**")
            st.markdown("""
            - **High-Value**: Recent, frequent, and high-spending customers
            - **Regular**: Steady, consistent buyers
            - **Occasional**: Infrequent, low-spend buyers
            - **At-Risk**: Haven't purchased in a long time
            """)

# RECOMMENDATION PAGE
elif selected == "Recommendation":
    st.markdown('<div class="main-title">Product Recommender</div>', unsafe_allow_html=True)
    st.markdown("Get 5 similar products based on past customer behaviour.")
    st.markdown("---")

    # Load data and similarity
    try:
        df = load_retail_data()
    except FileNotFoundError:
        st.error("Could not find `cleaned_online_retail.csv`.")
        st.stop()

    with st.spinner("Building product similarity matrix (first run only)..."):
        similarity_df = build_product_similarity(df)

    #INPUT
    product_name = st.text_input( "Enter Product Name", value="GREEN VINTAGE SPOT BEAKER", help="Type the exact product name (case-insensitive)")

    # dropdown of available products for user convenience
    with st.expander(" Browse Available Products"):
        all_products = sorted(similarity_df.index.tolist())
        selected_product = st.selectbox(
            "Or pick from the list:",
            options=[""] + all_products)
        if selected_product:
            product_name = selected_product

    # RECOMMEND BUTTON
    if st.button("Recommend"):
        recommendations = get_recommendations(product_name, similarity_df, top_n=5)
        if recommendations is None:
            st.error(f" Product **'{product_name}'** not found in the database. "
                     f"Please check the spelling or pick from the list above.")
        else:
            st.markdown("### Recommended Products:")

            for i, product in enumerate(recommendations, start=1):
                st.markdown(
                    f"<div class='product-card'>{i}. {product}</div>",
                    unsafe_allow_html=True)
            st.success(f" Showing top 5 products similar to **{product_name}**")


# FOOTER
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align:center; color:gray; font-size:12px;'>"
    "Made using StreamLlit"
    "</div>",
    unsafe_allow_html=True )