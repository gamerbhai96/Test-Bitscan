"""
BABD-13 Dataset Integration for BitScan
Bitcoin Address Behavior Dataset (BABD-13) from arxiv:2204.05746
Downloads, preprocesses, and loads the BABD-13 dataset for training fraud detection models
"""

import os
import pandas as pd
import numpy as np
import requests
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import zipfile
import urllib.parse
from tqdm import tqdm
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)

class BABDDatasetLoader:
    """
    Loader for BABD-13 dataset - Bitcoin Address Behavior Dataset with 544,462 labeled addresses
    Paper: BABD: A Bitcoin Address Behavior Dataset for Pattern Analysis (arxiv:2204.05746)
    """
    
    def __init__(self, data_dir: str = "data/babd_dataset"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # BABD-13 dataset info from the paper
        self.dataset_info = {
            'total_addresses': 544462,
            'address_types': 13,
            'feature_categories': 5,
            'total_features': 148,
            'collection_period': '2019-07-12 to 2021-05-26',
            'address_types_list': [
                'Exchange', 'Gambling', 'DarkNetMarket', 'Scam', 'Ransom',
                'Stealing', 'Mining', 'Service', 'Mixer', 'CoinJoin',
                'Address-Reuse', 'Sybil-Attack', 'Normal'
            ],
            'feature_categories_list': [
                'Basic Statistics', 'Flow Concentration', 'Temporal Patterns',
                'Network Topology', 'Behavioral Patterns'
            ]
        }
        
        self.babd_data = None
        self.feature_mappings = {}
        
    def download_dataset(self, google_drive_folder_id: str = "1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l"):
        """
        Download BABD-13 dataset from Google Drive
        Note: For now, we'll create instructions for manual download
        """
        download_instructions = f"""
        🚨 MANUAL DOWNLOAD REQUIRED - BABD-13 Dataset 🚨
        
        Paper: BABD: A Bitcoin Address Behavior Dataset for Pattern Analysis
        arxiv: https://arxiv.org/abs/2204.05746
        
        To use the BABD-13 dataset:
        
        1. Visit: https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l
        2. Download the dataset files (should include CSV files with features and labels)
        3. Extract all files to: {self.data_dir.absolute()}
        
        Expected files:
        - babd_features.csv (or similar naming)
        - babd_labels.csv (or similar naming)
        - babd_combined.csv (if available)
        
        Dataset contains:
        - 544,462 labeled Bitcoin addresses
        - 13 address types: {', '.join(self.dataset_info['address_types_list'])}
        - 148 behavioral features in 5 categories
        - Collection period: {self.dataset_info['collection_period']}
        
        After downloading, run: loader.load_babd_dataset()
        """
        
        print(download_instructions)
        
        # Check if files already exist
        possible_files = list(self.data_dir.glob("*.csv"))
        if possible_files:
            print(f"✅ Found {len(possible_files)} CSV files in BABD dataset directory!")
            return True
        else:
            print("❌ No BABD dataset files found. Please follow download instructions above.")
            return False
    
    def load_babd_dataset(self) -> Dict[str, Any]:
        """
        Load the BABD-13 dataset
        """
        try:
            print("Loading BABD-13 Bitcoin Address Behavior Dataset...")
            
            # Look for dataset files
            csv_files = list(self.data_dir.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
            
            print(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
            
            # Try to identify the main dataset file
            main_file = None
            for file in csv_files:
                if any(keyword in file.name.lower() for keyword in ['combined', 'full', 'babd', 'dataset']):
                    main_file = file
                    break
            
            if main_file is None:
                # Use the largest file
                main_file = max(csv_files, key=lambda f: f.stat().st_size)            
            print(f"Loading main dataset from: {main_file.name}")
            
            # Load the dataset
            dataset_df = pd.read_csv(main_file)
            print(f"📊 Loaded dataset shape: {dataset_df.shape}")
            print(f"📊 Columns: {list(dataset_df.columns[:10])}{'...' if len(dataset_df.columns) > 10 else ''}")
            
            # Identify label column
            label_columns = [col for col in dataset_df.columns if any(keyword in col.lower() for keyword in ['label', 'class', 'type', 'category'])]
            if not label_columns:
                # Look for columns with categorical values
                categorical_columns = [col for col in dataset_df.columns if dataset_df[col].dtype == 'object']
                if categorical_columns:
                    label_columns = categorical_columns[:1]
                else:
                    raise ValueError("Could not identify label column. Please check dataset format.")
            
            label_column = label_columns[0]
            print(f"📊 Using '{label_column}' as label column")
            
            # Identify feature columns (exclude non-feature columns)
            exclude_keywords = ['address', 'id', 'hash', 'time', 'date', 'label', 'class', 'type', 'category']
            feature_columns = [col for col in dataset_df.columns 
                             if col != label_column and 
                             not any(keyword in col.lower() for keyword in exclude_keywords)]
            
            print(f"📊 Identified {len(feature_columns)} feature columns")
            
            # Create class mapping
            unique_labels = dataset_df[label_column].unique()
            print(f"📊 Found {len(unique_labels)} unique address types: {unique_labels}")
            
            # Map to binary classification (fraud vs legitimate)
            fraud_types = ['scam', 'ransom', 'stealing', 'darknetmarket', 'mixer']
            legitimate_types = ['normal', 'exchange', 'service', 'mining']
            
            def map_to_binary(label):
                label_lower = str(label).lower()
                if any(fraud_type in label_lower for fraud_type in fraud_types):
                    return 1  # Fraud
                elif any(legit_type in label_lower for legit_type in legitimate_types):
                    return 0  # Legitimate
                else:
                    return 2  # Unknown/Other (gambling, coinjoin, etc.)
            
            dataset_df['binary_class'] = dataset_df[label_column].apply(map_to_binary)
            
            # Create detailed mapping
            class_mapping = {}
            for label in unique_labels:
                binary_class = map_to_binary(label)
                class_name = 'fraud' if binary_class == 1 else ('legitimate' if binary_class == 0 else 'other')
                class_mapping[label] = {'binary': binary_class, 'name': class_name}
            
            self.babd_data = {
                'data': dataset_df,
                'features': dataset_df[feature_columns],
                'labels': dataset_df[label_column],
                'binary_labels': dataset_df['binary_class'],
                'feature_columns': feature_columns,
                'label_column': label_column,
                'class_mapping': class_mapping,
                'original_labels': unique_labels,
                'dataset_info': self.dataset_info
            }
            
            print(f"✅ BABD-13 dataset loaded successfully!")
            print(f"📊 Dataset shape: {dataset_df.shape}")
            print(f"📊 Features: {len(feature_columns)}")
            
            # Class distribution
            print("📊 Original class distribution:")
            original_dist = dataset_df[label_column].value_counts()
            for class_name, count in original_dist.items():
                binary_class = class_mapping[class_name]['name']
                print(f"   {class_name} ({binary_class}): {count:,} ({count/len(dataset_df)*100:.1f}%)")
            
            print("📊 Binary class distribution:")
            binary_dist = dataset_df['binary_class'].value_counts()
            binary_names = {0: 'legitimate', 1: 'fraud', 2: 'other'}
            for class_id, count in binary_dist.items():
                class_name = binary_names.get(class_id, f'class_{class_id}')
                print(f"   {class_name}: {count:,} ({count/len(dataset_df)*100:.1f}%)")
            
            return self.babd_data
            
        except Exception as e:
            logger.error(f"Error loading BABD-13 dataset: {e}")
            raise
    
    def prepare_training_data(self, include_other: bool = False, 
                            balance_classes: bool = True,
                            feature_subset: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare BABD-13 training data for machine learning models
        
        Args:
            include_other: Whether to include 'other' class (gambling, coinjoin, etc.)
            balance_classes: Whether to balance the dataset
            feature_subset: Specific features to use (None for all)
            
        Returns:
            X: Feature matrix
            y: Target labels (0=legitimate, 1=fraud)
            feature_names: List of feature names
        """
        try:
            if self.babd_data is None:
                raise ValueError("BABD-13 dataset not loaded. Call load_babd_dataset() first.")
            
            data = self.babd_data['data']
            feature_columns = self.babd_data['feature_columns']
            
            # Use subset of features if specified
            if feature_subset:
                available_features = [f for f in feature_subset if f in feature_columns]
                if not available_features:
                    raise ValueError(f"None of the specified features found in dataset")
                feature_columns = available_features
                print(f"📊 Using {len(feature_columns)} specified features")
            
            # Filter data based on class
            if include_other:
                # Include all classes: 0=legitimate, 1=fraud, 2=other
                filtered_data = data.copy()
            else:
                # Only include fraud vs legitimate: 0=legitimate, 1=fraud
                filtered_data = data[data['binary_class'].isin([0, 1])].copy()
            
            print(f"📊 Filtered data shape: {filtered_data.shape}")
            print(f"📊 Class distribution after filtering:")
            print(filtered_data['binary_class'].value_counts())
            
            # Extract features and labels
            X = filtered_data[feature_columns].values
            y = filtered_data['binary_class'].values
            
            # Handle missing values
            print(f"📊 Handling missing values...")
            X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Remove any remaining infinite values
            finite_mask = np.isfinite(X).all(axis=1)
            X = X[finite_mask]
            y = y[finite_mask]
            
            print(f"📊 After cleaning: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Balance classes if requested
            if balance_classes and not include_other and len(np.unique(y)) == 2:
                from imblearn.under_sampling import RandomUnderSampler
                from imblearn.over_sampling import SMOTE
                
                print(f"📊 Balancing classes...")
                
                # First, undersample the majority class
                undersampler = RandomUnderSampler(random_state=42)
                X_balanced, y_balanced = undersampler.fit_resample(X, y)
                
                # Then oversample the minority class with SMOTE
                smote = SMOTE(random_state=42, k_neighbors=min(5, len(X_balanced[y_balanced==1])-1))
                X_final, y_final = smote.fit_resample(X_balanced, y_balanced)
                
                print(f"📊 Balanced data shape: {X_final.shape}")
                print(f"📊 Balanced class distribution: {np.bincount(y_final)}")
                
                return X_final, y_final, feature_columns
            
            return X, y, feature_columns
            
        except Exception as e:
            logger.error(f"Error preparing BABD-13 training data: {e}")
            raise
    
    def get_feature_importance_mapping(self) -> Dict[str, str]:
        """
        Get mappings for BABD-13 features to understand their meanings
        Based on the 5 feature categories from the paper
        """
        feature_categories = {
            'basic_stats': [
                'transaction_count', 'total_received', 'total_sent', 'balance',
                'avg_transaction_value', 'std_transaction_value'
            ],
            'flow_concentration': [
                'gini_coefficient', 'unique_inputs', 'unique_outputs',
                'fan_in_ratio', 'fan_out_ratio'
            ],
            'temporal_patterns': [
                'activity_span', 'avg_interval', 'burst_count',
                'periodicity_score', 'weekend_activity'
            ],
            'network_topology': [
                'degree_centrality', 'betweenness_centrality', 'clustering_coefficient',
                'pagerank_score', 'neighbor_diversity'
            ],
            'behavioral_patterns': [
                'round_amount_ratio', 'self_loop_count', 'repeated_amounts',
                'mixing_score', 'chain_depth'
            ]
        }
        
        return feature_categories
    
    def save_processed_data(self, save_dir: str = "data/processed"):
        """
        Save processed BABD-13 dataset for quick loading
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.babd_data is not None:
            babd_file = save_path / "babd_processed.joblib"
            joblib.dump(self.babd_data, babd_file)
            print(f"💾 Saved processed BABD-13 data to {babd_file}")
    
    def load_processed_data(self, save_dir: str = "data/processed"):
        """
        Load previously processed BABD-13 dataset
        """
        save_path = Path(save_dir)
        babd_file = save_path / "babd_processed.joblib"
        
        if babd_file.exists():
            self.babd_data = joblib.load(babd_file)
            print(f"📁 Loaded processed BABD-13 data from {babd_file}")
            return True
        return False

# Test and demonstration function
def test_babd_dataset():
    """
    Test the BABD-13 dataset loader
    """
    print("🧪 Testing BABD-13 Dataset Loader")
    print("=" * 50)
    
    loader = BABDDatasetLoader()
    
    # Check if dataset exists
    if not loader.download_dataset():
        print("⚠️  BABD-13 dataset not available for testing. Please download manually.")
        return
    
    try:
        # Load BABD dataset
        babd_data = loader.load_babd_dataset()
        print(f"✅ BABD-13 dataset loaded: {babd_data['data'].shape}")
        
        # Prepare training data
        X, y, feature_names = loader.prepare_training_data(include_other=False)
        print(f"✅ Training data prepared: X={X.shape}, y={y.shape}")
        
        # Show feature categories
        feature_categories = loader.get_feature_importance_mapping()
        print(f"✅ Feature categories: {list(feature_categories.keys())}")
        
        # Save processed data
        loader.save_processed_data()
        
        return loader
        
    except Exception as e:
        print(f"❌ Error testing BABD dataset loader: {e}")
        return None

if __name__ == "__main__":
    test_babd_dataset()