import pandas as pd
import numpy as np
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class DataPreprocessor:
    def __init__(self, data_path="../data/job_details_full.csv"):
        """
        Initialize the DataPreprocessor with the path to the dataset
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vietnamese_stopwords = set(stopwords.words('english'))  # Default to English stopwords
        self.lemmatizer = WordNetLemmatizer()
        self.label_encoders = {}
        self.standard_scaler = StandardScaler()
        
        # Add some Vietnamese stopwords
        vietnamese_words = ['và', 'hoặc', 'các', 'của', 'có', 'trong', 'là', 'cho', 'những', 'được', 'với', 'tốt', 'từ',
                           'năm', 'tuổi', 'trên', 'ngành', 'tại', 'công', 'việc', 'về', 'làm', 'trình', 'độ', 'thành', 'thạo']
        self.vietnamese_stopwords.update(vietnamese_words)

    def load_data(self):
        """
        Load the dataset
        """
        self.df = pd.read_csv(self.data_path)
        # Drop URL column as requested
        if 'URL' in self.df.columns:
            self.df = self.df.drop('URL', axis=1)
        print(f"Loaded dataset with shape: {self.df.shape}")
        return self.df

    def clean_salary(self):
        """
        Clean and convert the salary column to a numeric value
        """
        # Create a copy to avoid SettingWithCopyWarning
        self.df = self.df.copy()
        
        # Function to extract numeric salary values
        def extract_salary_value(salary_str):
            if pd.isna(salary_str) or salary_str == 'Thoả thuận':
                return np.nan
            
            # Extract numbers from strings like "9 - 12 triệu" or "Từ 15 triệu" etc.
            numbers = re.findall(r'\d+', salary_str)
            if len(numbers) == 0:
                return np.nan
            elif len(numbers) == 1:
                return float(numbers[0])
            else:
                # Use average of range
                return (float(numbers[0]) + float(numbers[1])) / 2
        
        self.df['Salary_Value'] = self.df['Salary'].apply(extract_salary_value)
        # Remove rows with NaN salary values
        self.df = self.df.dropna(subset=['Salary_Value'])
        print(f"Dataset after cleaning salary: {self.df.shape}")
        return self.df

    def preprocess_text(self, text):
        """
        Preprocess text data: lowercase, remove special characters, tokenize, remove stopwords, lemmatize
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.vietnamese_stopwords]
        
        return ' '.join(tokens)

    def extract_features_from_requirements(self):
        """
        Extract features from the job requirements text using NLP techniques
        """
        # Clean the requirements text
        self.df['Processed_Requirements'] = self.df['Job Requirements'].apply(self.preprocess_text)
        
        # Extract common skills from requirements
        skills_to_check = ['excel', 'word', 'misa', 'tiếng anh', 'english', 'tiếng trung', 'kế toán', 'kinh nghiệm', 
                          'đại học', 'cao đẳng', 'quản lý', 'giao tiếp', 'trung thực', 'nhanh nhẹn', 'windows',
                          'autocad', 'photoshop', 'revit', 'thiết kế']
        
        for skill in skills_to_check:
            self.df[f'has_{skill.replace(" ", "_")}'] = self.df['Job Requirements'].str.lower().str.contains(skill).astype(int)
        
        return self.df

    def encode_categorical_features(self):
        """
        Encode categorical features
        """
        categorical_cols = ['Field', 'Experience', 'Location', 'Company Size']
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
        
        return self.df

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets
        """
        # Drop rows with missing target values
        self.df = self.df.dropna(subset=['Salary_Value'])
        
        # Define features
        feature_cols = [col for col in self.df.columns if col.endswith('_encoded') or col.startswith('has_')]
        
        X = self.df[feature_cols]
        y = self.df['Salary_Value']
        
        # Scale the features
        X_scaled = self.standard_scaler.fit_transform(X)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {self.X_train.shape}, {self.y_train.shape}")
        print(f"Testing set: {self.X_test.shape}, {self.y_test.shape}")
        
        # Save feature names for later use
        self.feature_names = feature_cols
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def save_processed_data(self, output_dir='./'):
        """
        Save the processed data for later use
        """
        os.makedirs(output_dir, exist_ok=True)
        
        self.df.to_csv(os.path.join(output_dir, 'processed_data.csv'), index=False)
        np.save(os.path.join(output_dir, 'X_train.npy'), self.X_train)
        np.save(os.path.join(output_dir, 'X_test.npy'), self.X_test)
        np.save(os.path.join(output_dir, 'y_train.npy'), self.y_train)
        np.save(os.path.join(output_dir, 'y_test.npy'), self.y_test)
        
        # Save feature names
        pd.DataFrame({'feature_name': self.feature_names}).to_csv(
            os.path.join(output_dir, 'feature_names.csv'), index=False
        )
        
        print(f"Processed data saved to {output_dir}")

    def run_preprocessing(self):
        """
        Execute the entire preprocessing pipeline
        """
        self.load_data()
        self.clean_salary()
        self.extract_features_from_requirements()
        self.encode_categorical_features()
        self.split_data()
        self.save_processed_data()
        return self.X_train, self.X_test, self.y_train, self.y_test

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run_preprocessing() 