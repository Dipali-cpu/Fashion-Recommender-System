1. This is machine learning application in which user can search their products by showing image and get 5 similar products according to the type of a product.
2. For showing similar 5 different products I used deep learning CNN RESNET-50 model for feature extraction and k-nearest neighbours for similarity matching.
3. Major Tools and Libraries -  python, tensorflow / keras, numpy, pandas, streamlit, pickle.

How it works:

1. **Image Feature Extraction:** Extracts embeddings from fashion product images using a pre-trained CNN model.
2. **Similarity Matching:** Uses Nearest Neighbors to find visually similar products
3. **User Interface:** Streamlit app displays the results in a clean, responsive layout.
