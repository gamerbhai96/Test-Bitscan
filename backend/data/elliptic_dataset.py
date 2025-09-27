"""
EllipticPlusPlus Dataset Integration for BitScan
Downloads, preprocesses, and loads the EllipticPlusPlus dataset for training fraud detection models
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

class EllipticDatasetLoader:
    """
    Loader for EllipticPlusPlus dataset - Real-world labeled Bitcoin fraud detection data
    """
    
    def __init__(self, data_dir: str = "data/elliptic_dataset"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs (these would need to be updated with actual download links)
        self.dataset_info = {
            'actors': {
                'features_file': 'wallets_features.csv',
                'classes_file': 'wallets_classes.csv',
                'addr_addr_edges': 'AddrAddr_edgelist.csv',
                'addr_tx_edges': 'AddrTx_edgelist.csv',
                'tx_addr_edges': 'TxAddr_edgelist.csv'
            },
            'transactions': {
                'features_file': 'txs_features.csv',
                'classes_file': 'txs_classes.csv',
                'edges_file': 'txs_edgelist.csv'
            }
        }
        
        self.actors_data = None
        self.transactions_data = None
        self.feature_mappings = {}
        
    def download_dataset(self, google_drive_folder_id: Optional[str] = None):
        """
        Download EllipticPlusPlus dataset from Google Drive
        Note: For now, we'll create instructions for manual download
        """
        download_instructions = """
        🚨 MANUAL DOWNLOAD REQUIRED 🚨
        
        To use the EllipticPlusPlus dataset:
        
        1. Visit: https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l
        2. Download the entire dataset folder
        3. Extract all CSV files to: {data_dir}
        
        Required files:
        - Actors Dataset/wallets_features.csv
        - Actors Dataset/wallets_classes.csv
        - Actors Dataset/AddrAddr_edgelist.csv
        - Actors Dataset/AddrTx_edgelist.csv
        - Actors Dataset/TxAddr_edgelist.csv
        - Transactions Dataset/txs_features.csv
        - Transactions Dataset/txs_classes.csv
        - Transactions Dataset/txs_edgelist.csv
        
        After downloading, run: loader.load_actors_dataset() and loader.load_transactions_dataset()
        """.format(data_dir=self.data_dir.absolute())
        
        print(download_instructions)
        
        # Check if files already exist
        actors_features = self.data_dir / self.dataset_info['actors']['features_file']
        if actors_features.exists():
            print("✅ EllipticPlusPlus dataset files found!")
            return True
        else:
            print("❌ Dataset files not found. Please follow download instructions above.")
            return False
    
    def load_actors_dataset(self) -> Dict[str, Any]:
        """
        Load the actors (wallet addresses) dataset
        """
        try:
            print("Loading EllipticPlusPlus Actors Dataset...")
            
            # Load features
            features_file = self.data_dir / self.dataset_info['actors']['features_file']
            if not features_file.exists():
                raise FileNotFoundError(f"Actors features file not found: {features_file}")
            
            print(f"Loading features from {features_file}")
            features_df = pd.read_csv(features_file)
            
            # Load classes
            classes_file = self.data_dir / self.dataset_info['actors']['classes_file']
            if not classes_file.exists():
                raise FileNotFoundError(f"Actors classes file not found: {classes_file}")
            
            print(f"Loading classes from {classes_file}")
            classes_df = pd.read_csv(classes_file)
            
            # Load edge lists (optional)
            edges = {}
            for edge_type in ['addr_addr_edges', 'addr_tx_edges', 'tx_addr_edges']:
                edge_file = self.data_dir / self.dataset_info['actors'][edge_type]
                if edge_file.exists():
                    print(f"Loading {edge_type} from {edge_file}")
                    edges[edge_type] = pd.read_csv(edge_file)
                else:
                    print(f"Warning: {edge_type} file not found: {edge_file}")
            
            # Merge features and classes
            if 'txId' in features_df.columns:
                # If using transaction ID as key
                actors_df = pd.merge(features_df, classes_df, on='txId', how='left')
            elif 'address' in features_df.columns and 'address' in classes_df.columns:
                # If using address as key
                actors_df = pd.merge(features_df, classes_df, on='address', how='left')
            else:
                # Try merging on index
                actors_df = pd.merge(features_df, classes_df, left_index=True, right_index=True, how='left')
            
            self.actors_data = {
                'data': actors_df,
                'features': features_df,
                'classes': classes_df,
                'edges': edges,
                'feature_columns': [col for col in features_df.columns if col not in ['txId', 'address']],
                'class_mapping': {
                    1: 'illicit',
                    2: 'licit', 
                    3: 'unknown'
                }
            }
            
            print(f"✅ Actors dataset loaded successfully!")
            print(f"📊 Dataset shape: {actors_df.shape}")
            print(f"📊 Features: {len(self.actors_data['feature_columns'])}")
            
            # Class distribution
            if 'class' in actors_df.columns:
                class_dist = actors_df['class'].value_counts()
                print("📊 Class distribution:")
                for class_id, count in class_dist.items():
                    class_name = self.actors_data['class_mapping'].get(class_id, f'class_{class_id}')
                    print(f"   {class_name}: {count:,} ({count/len(actors_df)*100:.1f}%)")
            
            return self.actors_data
            
        except Exception as e:
            logger.error(f"Error loading actors dataset: {e}")
            raise
    
    def load_transactions_dataset(self) -> Dict[str, Any]:
        """
        Load the transactions dataset
        """
        try:
            print("Loading EllipticPlusPlus Transactions Dataset...")
            
            # Load features
            features_file = self.data_dir / self.dataset_info['transactions']['features_file']
            if not features_file.exists():
                raise FileNotFoundError(f"Transactions features file not found: {features_file}")
            
            features_df = pd.read_csv(features_file)
            
            # Load classes
            classes_file = self.data_dir / self.dataset_info['transactions']['classes_file']
            if not classes_file.exists():
                raise FileNotFoundError(f"Transactions classes file not found: {classes_file}")
            
            classes_df = pd.read_csv(classes_file)
            
            # Load edges
            edges_file = self.data_dir / self.dataset_info['transactions']['edges_file']
            edges_df = None
            if edges_file.exists():
                edges_df = pd.read_csv(edges_file)
            
            # Merge features and classes
            if 'txId' in features_df.columns and 'txId' in classes_df.columns:
                transactions_df = pd.merge(features_df, classes_df, on='txId', how='left')
            else:
                transactions_df = pd.merge(features_df, classes_df, left_index=True, right_index=True, how='left')
            
            self.transactions_data = {
                'data': transactions_df,
                'features': features_df,
                'classes': classes_df,
                'edges': edges_df,
                'feature_columns': [col for col in features_df.columns if col not in ['txId']],
                'class_mapping': {
                    1: 'illicit',
                    2: 'licit',
                    3: 'unknown'
                }
            }
            
            print(f"✅ Transactions dataset loaded successfully!")
            print(f"📊 Dataset shape: {transactions_df.shape}")
            print(f"📊 Features: {len(self.transactions_data['feature_columns'])}")
            
            # Class distribution
            if 'class' in transactions_df.columns:
                class_dist = transactions_df['class'].value_counts()
                print("📊 Class distribution:")
                for class_id, count in class_dist.items():
                    class_name = self.transactions_data['class_mapping'].get(class_id, f'class_{class_id}')
                    print(f"   {class_name}: {count:,} ({count/len(transactions_df)*100:.1f}%)")
            
            return self.transactions_data
            
        except Exception as e:
            logger.error(f"Error loading transactions dataset: {e}")
            raise
    
    def prepare_training_data(self, dataset_type: str = 'actors', 
                            include_unknown: bool = False,
                            balance_classes: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data for machine learning models
        
        Args:
            dataset_type: 'actors' or 'transactions'
            include_unknown: Whether to include unknown class (3) in training
            balance_classes: Whether to balance the dataset
            
        Returns:
            X: Feature matrix
            y: Target labels (0=licit, 1=illicit)
            feature_names: List of feature names
        """
        try:
            if dataset_type == 'actors':
                if self.actors_data is None:
                    raise ValueError("Actors dataset not loaded. Call load_actors_dataset() first.")
                data = self.actors_data['data']
                feature_columns = self.actors_data['feature_columns']
            elif dataset_type == 'transactions':
                if self.transactions_data is None:
                    raise ValueError("Transactions dataset not loaded. Call load_transactions_dataset() first.")
                data = self.transactions_data['data']
                feature_columns = self.transactions_data['feature_columns']
            else:
                raise ValueError("dataset_type must be 'actors' or 'transactions'")
            
            # Filter data based on class
            if include_unknown:
                # Include all classes, map: 1->1 (illicit), 2->0 (licit), 3->2 (unknown)
                filtered_data = data[data['class'].isin([1, 2, 3])].copy()
                filtered_data['binary_class'] = filtered_data['class'].map({1: 1, 2: 0, 3: 2})
            else:
                # Only include labeled data: 1->1 (illicit), 2->0 (licit)
                filtered_data = data[data['class'].isin([1, 2])].copy()
                filtered_data['binary_class'] = filtered_data['class'].map({1: 1, 2: 0})
            
            print(f"📊 Filtered data shape: {filtered_data.shape}")
            print(f"📊 Class distribution after filtering:")
            print(filtered_data['binary_class'].value_counts())
            
            # Extract features and labels
            X = filtered_data[feature_columns].values
            y = filtered_data['binary_class'].values
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Balance classes if requested
            if balance_classes and not include_unknown:
                from imblearn.under_sampling import RandomUnderSampler
                from imblearn.over_sampling import SMOTE
                
                # First, undersample the majority class
                undersampler = RandomUnderSampler(random_state=42)
                X_balanced, y_balanced = undersampler.fit_resample(X, y)
                
                # Then oversample the minority class
                smote = SMOTE(random_state=42)
                X_final, y_final = smote.fit_resample(X_balanced, y_balanced)
                
                print(f"📊 Balanced data shape: {X_final.shape}")
                print(f"📊 Balanced class distribution: {np.bincount(y_final)}")
                
                return X_final, y_final, feature_columns
            
            return X, y, feature_columns
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
    
    def get_feature_mappings(self, dataset_type: str = 'actors') -> Dict[str, int]:
        """
        Get mappings between EllipticPlusPlus features and our BitScan features
        """
        if dataset_type == 'actors':
            # Map EllipticPlusPlus actor features to BitScan features
            # This is a simplified mapping - would need to be refined based on actual feature analysis
            elliptic_to_bitscan = {
                # Basic transaction metrics
                'local_feature_1': 'transaction_count',  # Placeholder - needs actual feature analysis
                'local_feature_2': 'total_received_btc',
                'local_feature_3': 'total_sent_btc',
                'local_feature_4': 'balance_btc',
                
                # Network features
                'local_feature_5': 'unique_input_addresses',
                'local_feature_6': 'unique_output_addresses',
                
                # Temporal features
                'local_feature_7': 'activity_span_days',
                'local_feature_8': 'average_interval_hours',
                
                # Add more mappings as we analyze the actual feature meanings
            }
            
            return elliptic_to_bitscan
        
        return {}
    
    def save_processed_data(self, save_dir: str = "data/processed"):
        """
        Save processed datasets for quick loading
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.actors_data is not None:
            actors_file = save_path / "elliptic_actors_processed.joblib"
            joblib.dump(self.actors_data, actors_file)
            print(f"💾 Saved processed actors data to {actors_file}")
        
        if self.transactions_data is not None:
            transactions_file = save_path / "elliptic_transactions_processed.joblib"
            joblib.dump(self.transactions_data, transactions_file)
            print(f"💾 Saved processed transactions data to {transactions_file}")
    
    def load_processed_data(self, save_dir: str = "data/processed"):
        """
        Load previously processed datasets
        """
        save_path = Path(save_dir)
        
        actors_file = save_path / "elliptic_actors_processed.joblib"
        if actors_file.exists():
            self.actors_data = joblib.load(actors_file)
            print(f"📁 Loaded processed actors data from {actors_file}")
        
        transactions_file = save_path / "elliptic_transactions_processed.joblib"
        if transactions_file.exists():
            self.transactions_data = joblib.load(transactions_file)
            print(f"📁 Loaded processed transactions data from {transactions_file}")

# Test and demonstration function
def test_elliptic_dataset():
    """
    Test the EllipticPlusPlus dataset loader
    """
    print("🧪 Testing EllipticPlusPlus Dataset Loader")
    print("=" * 50)
    
    loader = EllipticDatasetLoader()
    
    # Check if dataset exists
    if not loader.download_dataset():
        print("⚠️  Dataset not available for testing. Please download manually.")
        return
    
    try:
        # Load actors dataset
        actors_data = loader.load_actors_dataset()
        print(f"✅ Actors dataset loaded: {actors_data['data'].shape}")
        
        # Prepare training data
        X, y, feature_names = loader.prepare_training_data('actors', include_unknown=False)
        print(f"✅ Training data prepared: X={X.shape}, y={y.shape}")
        
        # Load transactions dataset
        try:
            transactions_data = loader.load_transactions_dataset()
            print(f"✅ Transactions dataset loaded: {transactions_data['data'].shape}")
        except Exception as e:
            print(f"⚠️  Could not load transactions dataset: {e}")
        
        # Save processed data
        loader.save_processed_data()
        
        return loader
        
    except Exception as e:
        print(f"❌ Error testing dataset loader: {e}")
        return None

if __name__ == "__main__":
    test_elliptic_dataset()