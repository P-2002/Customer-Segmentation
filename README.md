#  Customer Segmentation   

### Project Overview  
Welcome to the **Customer Segmentation Project**, a group effort by six members during our internship at **AIVariant**.  
We focused on **Exploratory Data Analysis (EDA)**, **feature engineering**, and **machine learning** to understand customer behavior. Using **PCA** for dimensionality reduction and **KMeans** for clustering, we identified clear customer segments that can support smarter business strategies.  

---

###  Technologies Used  
- Python 3  
- NumPy  
- Pandas  
- Matplotlib & Seaborn  
- SciPy  
- Scikit-learn
- joblib  
- Streamlit (for live app)

- ###  Project Structure  
├── GROUP5PCA.ipynb # Main Jupyter Notebook (analysis + models)
├── app.py # Streamlit app for live demo
├── requirements.txt # Project dependencies
└── README.md # Project documentation

###  Installation  
1. Clone this repository:  
   git clone https://github.com/PrasannaChouhan/P-2002.git
cd P-2002
   
Create a virtual environment:
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\\Scripts\\activate      # On Windows

Install the dependencies:
pip install -r requirements.txt

### Usage
Run the Notebook
jupyter notebook GROUP5PCA.ipynb

This will allow you to:

Perform EDA and visualize customer behavior

Apply feature engineering

Reduce dimensions with PCA

Segment customers using KMeans clustering

Evaluate results with silhouette score

### Run the Streamlit App
streamlit run app.py
This launches an interactive web app where you can:

Upload your own CSV dataset

Select numeric features

Apply PCA and KMeans clustering

Visualize clusters in 2D space

Check silhouette score

Download clustered results as CSV

### Live Deployment
You can deploy the app for free using Streamlit Community Cloud:

Push this repo to GitHub.

Go to Streamlit Cloud
.

Connect your GitHub, select the repo, and deploy.

Your app will be live and shareable with a link.

### Results

Dimensionality reduction improved clarity in visualization

Customers segmented into distinct, meaningful clusters

Clusters evaluated using silhouette score

Insights generated for targeted business strategies

### Contributors
This project was developed as a Group Project of six members during our internship at AIVariant. It reflects the combined effort, dedication, and teamwork of the entire group.

### Closing Note

This project was a collaborative effort involving countless hours of hard work, brainstorming, and analysis.
The teamwork ensured the successful completion of a comprehensive customer segmentation system.

 Thank you for exploring our project!
Feel free to check out the live website and share your feedback.
