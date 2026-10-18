import pandas as pd
import numpy as np
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Shopping Assistant", layout="wide", page_icon="🛒")

st.title("🛒 Shopping Assistant")
st.write("Developed by UMR")


uploaded_file = st.file_uploader("Upload your transaction data (CSV format)", type=["csv"]) 

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")

    st.dataframe(df.head(11), use_container_width=True)

    required_columns = ['TransactionID', 'Product']

    if not all (col in df.columns for col in required_columns):
        st.error(f"The uploaded CSV must contain the following columns:  TransactionID and Product")
        st.stop()

    df=df.dropna(subset=['TransactionID', 'Product'])   

    
    
    df["Product"] = df["Product"].astype(str).str.strip()

    total_transactions = df['TransactionID'].nunique() 
    total_items = df['Product'].nunique()
    total_records = len(df)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", total_transactions)
    col2.metric("Total Items", total_items) 
    col3.metric("Total Records", total_records)

    basket=pd.crosstab(df['TransactionID'], df['Product'])
    basket = (basket > 0).astype(int)

    st.sidebar.header("Apriori Section ")
    min_support = st.sidebar.slider("Minimum Support", 0.5, 1.5, 0.1)
    min_confidence = st.sidebar.slider("Minimum Confidence", 0.5, 1.0, 0.5)

    if st.button("Discover Shopping Patterns",type="primary"):
        with st.spinner("Analysing shopping patterns..."):
            frequent_itemsets = apriori(basket, min_support=min_support )
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

            if rules.empty:
                st.warning("No association rules found with the given parameters.")
            else:
                st.subheader("Association Rules")
                st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']], use_container_width=True)


                st.session_state['rules'] = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                st.session_state['frequent_itemsets'] = frequent_itemsets

                st.success("Shopping patterns discovered successfully!")

                st.divider()
                st.subheader("Shoping Recommendations")
                all_products = sorted(df['Product'].unique())
                selected_product = st.selectbox("Select a product to get recommendations", all_products)    

                if st.button("Get Recommendations"):

                    matching_rules = rules[rules['antecedents'].apply(lambda x: selected_product in x)].copy()

                    matching_rules= matching_rules.sort_values(by=['confidence', 'lift'], ascending=False)

                    st.divider()
                    st.subheader(
                        f"Recommendations for customers who bought"
                        f"** {selected_product}** aslo prucheased:"


                    )
                    if matching_rules.empty:
                        st.warning("No strong recommendations found for this product")
                    else:
                        recommendations = []   #list of dictionaries
                        for product in rules["consequents"]:
                            recommendations.append({
                                "Product": product,
                                "Confidence": rules["confidence"],
                                "support": rules["support"],
                                "Lift": rules["lift"],

                            })

                        recommendations_df = pd.DataFrame(recommendations)

                        reccommendations_df = recommendations_df.drop_duplicates(subset=["Product"]).head(10)

                        for i, row in reccommendations_df.iterrows():
                            product = row["Product"]
                            confidence = row["Confidence"]  
                            support = row["support"]
                            lift = row["Lift"]

                            st.markdown(
                                f"""
                                **{product}**
                                 Confidence: {confidence:.2f}
                                 Support: {support:.2f}
                                 Lift: {lift:.2f}


                                 """
                            )
                            st.divider()

    if "rules" in st.session_state:
        with st.expander("View All Association Rules"):
            
# Convert antecedents and consequents into readable product names
         rules["if Customer buys"] = rules["antecedents"].apply(
         lambda x: ', '.join(list(x))
        )

    rules["then Customer also buys"] = rules["consequents"].apply(
    lambda x: ', '.join(list(x))
)

# Select and display the recommendation rules
st.subheader("Product Recommendation Rules")

st.dataframe(
    rules[["if Customer buys", "then Customer also buys", "support", "confidence", "lift"]],
    use_container_width=True

)

