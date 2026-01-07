from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__, template_folder='templates', static_folder='static')

# Loading Resources
print("Loading models...")
try:
    clf_model = joblib.load('model_class.pkl')
    reg_model = joblib.load('model_score.pkl')
    tfidf = joblib.load('tfidf.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    
    # NLTK Setup
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    print("Resources loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")


# constraint patterns
#extracting important keyword and constraint features from problem statement
import re
keywords = [
    "array", "tree","distance","minimum","maximum","largest","smallest","sliding window","points","square", "graph","vertices", "dp", "dynamic programming","pattern","backtracking","greedy","linked list",
    "path","subarray","subsequence","kth","closest","frequent",
    "binary","search", "segment tree", "bit", "fenwick", 
    "geometry", "matrix", "queue", 
    "dfs", "bfs", "shortest path", "prime", "gcd","line","input",
    "single","integer","output","characters","character","letters","contains",
    "words","message","grid","time","possible","connected","different","players","people",
    "strategy","scheduling","road","modulo","n","N","m","M","ith","range","bits","greater","smaller","substring",
    "subtree","diameter","level","count","frequency","hash","linear",
    "implementation","brute force","string","max","min","two sum","reverse","palindrome",
    "pointer","pointers","stack", "parentheses","valid","swap","next","merge","root","all","possible","distinct","without",
    "repeating","prefix","suffix","order","return","all","combinations","permutations","subsets","generate","partitioning",
    "median","rotated","sorted","equals","continuous","numbers","ways","cost","coin","change","increasing","word","break",
    "climbing","stairs","divide","conquer","query","game","theory","number","segment","bitmask","optimization"
] 
constraint_patterns = {
    # Upper bounds (Updated to handle '⋅', '.', and messy spacing)
    # Added [⋅\.] to capture "2⋅10^5" and "2.10^5"
    "constraint_1e5": r"(10\^5|1e5|100\s?000|2\s?[×x*⋅\.]\s?10\^5)",
    "constraint_1e6": r"(10\^6|1e6|1\s?000\s?000)",
    "constraint_1e9": r"(10\^9|1e9|1\s?000\s?000\s?000)",
    "constraint_1e18": r"(10\^18|1e18)",

    # Context Indicators (New!)
    # Matches "sum of n", "∑ n", "sum of m", etc.
    "sum_context": r"(?:sum of|∑|total)\s?[a-zA-Z0-9_]*",
    "math_xor": r"(\\oplus|⊕|xor)",
    "math_infinity": r"(\\infty|∞)",
    "math_fraction": r"(\\frac|/)",
    "math_equality": r"(==|!=|\\ne|\\equiv)",
    
    # Modulo arithmetic
    "modulo": r"(mod(ulo)?|1e9\s?\+\s?7|10\^9\s?\+\s?7|1000000007|998244353)",

    # Time constraints
    "time_limit": r"(time limit|seconds?|sec|ms|milliseconds?)",

    # Memory constraints
    "memory_limit": r"(memory limit|MB|megabytes?)"
}

def extract_features(text):
    text = text.lower()
    features = {}
    
    # Check keywords
    for word in keywords:
        features[f"count_{word.replace(' ', '_')}"] = text.count(word)
        
    # Check constraints
    for name, pattern in constraint_patterns.items():
        if re.search(pattern, text):
            features[name] = 1
        else:
            features[name] = 0
            
    return pd.Series(features)


#clean_text function
def clean_cp_text(text):
    if not isinstance(text, str):
        return ""
    
    # Fix LaTex number spacing (e.g., "10\, 000" -> "10000") in datasets
    text = re.sub(r'(\d)\\[, ]+(\d)', r'\1\2', text)
    
    # Normalize Exponents (e.g., "10^5" -> "10e5", "10^{18}" -> "10e18")
    # This keeps it intact as a single "word"
    text = re.sub(r'10\^\{?(\d+)\}?', r'10e\1', text)
    
    # Handle specific LaTeX symbols that indicate logic/constraints
    text = text.replace(r'\le', ' <= ')
    text = text.replace(r'\ge', ' >= ')
    text = text.replace(r'\times', ' * ')
    text = text.replace(r'\dots', ' ... ')
    
    # Handle Modulo (Crucial for Number Theory/DP problems)
    # Matches "10^9 + 7" or "10^9+7" common in hard problems
    text = re.sub(r'10e9\s*\+\s*7', 'mod_const', text)
    
    
    # Remove LaTeX delimiters ($) and backslashes
    text = text.replace('$', ' ')
    text = text.replace('\\', ' ')
    
    # Lowercase
    text = text.lower()
    
    # Tokenize
    words = nltk.word_tokenize(text)
    
    # SMART FILTERING
    clean_words = []
    for w in words:
        # Check if it's a number/math token we created (like "10e5" or "<=")
        is_math = (
            '10e' in w or 
            w in ['<=', '>=', '*', 'mod_const'] or 
            w.isdigit()
        )
        
        # Check if it's a significant variable (N, M, K, Q are common in CP)
        is_variable = (len(w) == 1 and w in ['n', 'm', 'k', 'q','a' ,'b', 'c'])
        
        # Check if it's a useful English word (alphabetic and not a stopword)
        is_valid_word = (w.isalpha() and w not in stop_words and len(w) > 1)
        
        if is_math or is_variable:
            clean_words.append(w)
        elif is_valid_word:
            clean_words.append(ps.stem(w))    
            
    return " ".join(clean_words)

# Routes 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        desc = data.get('description', '')
        in_desc = data.get('input_description', '')
        out_desc = data.get('output_description', '')
        
        full_text_raw = f"{desc} {in_desc} {out_desc}"
        
        # Cleaning Text
        cleaned_text = clean_cp_text(full_text_raw)
        
        # Calculate Features
        # Length features
        des_len = len(nltk.word_tokenize(desc)) if desc else 0
        in_len = len(nltk.word_tokenize(str(in_desc))) if in_desc else 0
        out_len = len(nltk.word_tokenize(str(out_desc))) if out_desc else 0
        
        len_features = pd.DataFrame([[des_len, in_len, out_len]], 
                                  columns=['des_words', 'input_words', 'output_words'])
        
        # Domain features
        domain_dict = extract_features(full_text_raw)
        domain_features = pd.DataFrame([domain_dict])
        
        # Combine & Scale
        add_feats = pd.concat([len_features, domain_features], axis=1)
        add_feats_scaled = scaler.transform(add_feats)
        
        # TF-IDF
        tfidf_vec = tfidf.transform([cleaned_text]).toarray()
        
        # Final Stack
        X_final = np.hstack((tfidf_vec, add_feats_scaled))
        
        # C. Predict
        pred_class_idx = clf_model.predict(X_final)[0]
        problem_class = encoder.inverse_transform([pred_class_idx])[0]
        problem_score = reg_model.predict(X_final)[0]
        
        return jsonify({
            'problem_class': problem_class,      # Returns "Easy", "Medium", or "Hard"
            'problem_score': round(float(problem_score), 2)
        })

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # port=0 tells the OS to pick a random free port
    app.run(debug=True, port=8000)