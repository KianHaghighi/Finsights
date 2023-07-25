{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bb3c6c54",
   "metadata": {},
   "outputs": [],
   "source": [
    "#scikit-learn and nltk together to perform NLP and \n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import nltk\n",
    "nltk.download('punkt')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "3ced34d6",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#load the data\n",
    "df_tweets = pd.read_csv(r\"C:\\Users\\kianh\\finsights\\finsights\\archive\\stock_tweets.csv\")\n",
    "df_yf = pd.read_csv(r\"C:\\Users\\kianh\\finsights\\finsights\\archive\\stock_yfinance_data.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "45a3682a",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['Date', 'Tweet', 'Stock Name', 'Company Name'], dtype='object')\n"
     ]
    }
   ],
   "source": [
    "columns = df_tweets.columns\n",
    "print(columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "87329c1f",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#tokenize the data\n",
    "from nltk.tokenize import word_tokenize\n",
    "df_tweets['tokens'] = df_tweets['Tweet'].apply(word_tokenize)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "0164efac",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\kianh\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "#remove the stop words\n",
    "nltk.download('stopwords')\n",
    "from nltk.corpus import stopwords\n",
    "stop_words = stopwords.words('english')\n",
    "df_tweets['text_without_stopwords'] = df_tweets['Tweet'].apply(lambda x: [' '.join([word for word in sentence.split() if word not in stop_words]) for sentence in x])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "572be0d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "#vectorization - the process of converting data into numerical arrays\n",
    "#or vectors that can be used as input for a model\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "vectorizer = CountVectorizer()\n",
    "x = vectorizer.fit_transform(df_tweets['Tweet'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "265c2575",
   "metadata": {},
   "outputs": [],
   "source": [
    "#modeling\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.model_selection import train_test_split\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "0ce35271",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "\n",
    "# Create an instance of the TfidfVectorizer\n",
    "#reflects how important a word is o a document in a collection\n",
    "tfidf = TfidfVectorizer()\n",
    "\n",
    "# Fit and transform the data to create the term-document matrix\n",
    "tdm = tfidf.fit_transform(df_tweets['Tweet'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "9d6a59b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create a term-document matrix\n",
    "from sklearn.decomposition import TruncatedSVD\n",
    "import scipy.sparse\n",
    "#create the sparse matrix after the TfidVectorizer is applied\n",
    "sparse_matrix = scipy.sparse.csr_matrix(tdm)\n",
    "\n",
    "#initiliaze TruncatedSVD instance\n",
    "lsa = TruncatedSVD(n_components=50)\n",
    "\n",
    "#perform LSA\n",
    "lsa.fit(sparse_matrix)\n",
    "dtm_lsa = lsa.transform(sparse_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "ca6a5e69",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[42.60978993 21.43464643 19.13859675 18.84822117 17.20834655 16.53406846\n",
      " 15.99595888 15.02081124 14.71434353 14.52845552 14.22799408 13.85756344\n",
      " 13.56144277 13.45163533 13.3798755  13.22057477 12.97109279 12.95360208\n",
      " 12.783109   12.51745999 12.47947642 12.29312774 12.19049259 12.05294675\n",
      " 12.03269724 11.87193261 11.7512058  11.61204717 11.51465084 11.37597844\n",
      " 11.24898654 11.16456943 11.13885803 11.08269459 10.93526045 10.85373531\n",
      " 10.77884292 10.70506207 10.6387075  10.55930922 10.50488656 10.44955663\n",
      " 10.40711501 10.29126231 10.26731111 10.20233291 10.14013299 10.02049312\n",
      "  9.97866799  9.94561596]\n"
     ]
    }
   ],
   "source": [
    "#inspect the results\n",
    "print(lsa.singular_values_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "5b90c328",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting annoy\n",
      "  Using cached annoy-1.17.1.tar.gz (647 kB)\n",
      "  Preparing metadata (setup.py): started\n",
      "  Preparing metadata (setup.py): finished with status 'done'\n",
      "Building wheels for collected packages: annoy\n",
      "  Building wheel for annoy (setup.py): started\n",
      "  Building wheel for annoy (setup.py): finished with status 'done'\n",
      "  Created wheel for annoy: filename=annoy-1.17.1-cp39-cp39-win_amd64.whl size=52737 sha256=25770bbe8066d3467d1bfa27f9d6bba356fb9e691ae5ca99a495ed7d0bbfcc40\n",
      "  Stored in directory: c:\\users\\kianh\\appdata\\local\\pip\\cache\\wheels\\bd\\31\\97\\98a495a4ac686cc6068ab4f52963e2b28d119cb1697b62cd19\n",
      "Successfully built annoy\n",
      "Installing collected packages: annoy\n",
      "Successfully installed annoy-1.17.1\n"
     ]
    }
   ],
   "source": [
    "!pip install annoy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b630bbd5",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5c12b8c0",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#I keep running into the error that \"MemoryError: Unable to allocate 48.6 GiB for an array with shape (80793, 80793) and data type float64\"\n",
    "#Solutions:\n",
    "#1.Use a sparse matrix to store the similarity matrix\n",
    "#this avoids loading the entire similarity matrix at once\n",
    "import annoy\n",
    "from scipy.sparse import csr_matrix\n",
    "from scipy.sparse import lil_matrix\n",
    "# Convert to sparse matrix\n",
    "dtm_lsa = csr_matrix(dtm_lsa)\n",
    "\n",
    "# Build the index\n",
    "index = annoy.AnnoyIndex(dtm_lsa.shape[1], 'angular')\n",
    "for i in range(dtm_lsa.shape[0]):\n",
    "    index.add_item(i, dtm_lsa[i].toarray()[0])\n",
    "index.build(10)\n",
    "\n",
    "# Compute the pairwise similarities\n",
    "similarity_matrix = np.zeros((dtm_lsa.shape[0], dtm_lsa.shape[0]))\n",
    "for i in range(dtm_lsa.shape[0]):\n",
    "    nn = index.get_nns_by_item(i, dtm_lsa.shape[0], include_distances=True)\n",
    "    for j in range(len(nn[0])):\n",
    "        similarity_matrix[i,nn[0][j]] = nn[1][j]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7c8f2b16",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "87630761",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
