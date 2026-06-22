import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def create_customer_product_matrix(df_clean):
    """
    Pivots the cleaned transactional DataFrame to create a Customer-Product interaction matrix.
    Rows are CustomerIDs, columns are product Descriptions, values are sum of Quantities.
    """
    customer_product_matrix = pd.pivot_table(
        df_clean,
        index='CustomerID',
        columns='Description',
        values='Quantity',
        aggfunc='sum',
        fill_value=0
    )
    return customer_product_matrix

def calculate_cosine_similarity(customer_product_matrix):
    """
    Computes cosine similarity between products.
    Takes the Customer-Product matrix, transposes it to Product-Customer matrix,
    calculates cosine similarity, and returns a DataFrame with product descriptions as index/columns.
    """
    # Product-Customer Matrix
    product_customer_matrix = customer_product_matrix.T
    
    # Calculate similarity
    similarity_array = cosine_similarity(product_customer_matrix)
    
    # Create similarity DataFrame
    similarity_df = pd.DataFrame(
        similarity_array,
        index=product_customer_matrix.index,
        columns=product_customer_matrix.index
    )
    
    return similarity_df
