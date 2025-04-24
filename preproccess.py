import pandas as pd
import nltk
import string
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Download resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# Initialize the WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Define stop words with nltk
stop_words = set(stopwords.words('english'))

# Function to map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(nltk_tag):
    tag_map = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}
    return tag_map.get(nltk_tag[0], wordnet.NOUN) # Default to noun if no better match

# Function to preprocess text
def preprocess(text):
    # 1. & 2. Remove extra blank spaces and convert to lowercase
    text = text.lower().strip()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Split the text into individual words via tokenization
    words = nltk.word_tokenize(text)

    # 4. Remove stop words
    words = [word for word in words if word not in stop_words]

    # Get POS tags for each word
    pos_tags = nltk.pos_tag(words)

    # 5. Lemmatization
    words = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    
    return words

# Load the CSV files
file_path1 = 'test_data.csv'
file_path2 = 'BBC_train_full.csv'
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# Apply the function to each row in the 'text' column
df1['words'] = df1['text'].apply(preprocess)
df2['words'] = df2['text'].apply(preprocess)

# Save the result to a new CSV file
output_file1 = 'cleaned_test.csv'
output_file2 = 'cleaned_bbc.csv'
df1.to_csv(output_file1, index=False)
df2.to_csv(output_file2, index=False)

print("Processing complete. Cleaned data saved to:", output_file1, output_file2)