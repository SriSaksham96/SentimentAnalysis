import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

class DatasetMerger:
    def __init__(self, config_path: str = "configs/dataset_config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.target_cols = self.config['unified_schema']['target_columns']

    def _process_dataset(self, dataset_name: str, params: dict) -> pd.DataFrame:
        """Reads and standardizes a single dataset based on YAML rules."""
        raw_dir = Path(params['raw_dir'])
        df = pd.read_csv(raw_dir / params['annotation_file'])
        
        # Rename columns to unified schema
        df = df.rename(columns=params['column_map'])
        
        # Map sentiment strings to standard integers (0, 1, 2)
        df['sentiment'] = df['sentiment'].str.lower().map(params['sentiment_map'])
        
        # Map offensive labels if provided; default to -1 (masked) for datasets without it
        if 'offensive' in df.columns and 'offensive_map' in params:
            df['offensive'] = df['offensive'].astype(str).str.lower().map(params['offensive_map']).fillna(-1).astype(int)
        else:
            df['offensive'] = -1

        # Inject standard tags
        df['language'] = params['language_tag']
        df['dataset_source'] = dataset_name
        
        # Clean up image paths (ensuring absolute/relative consistency)
        df['image_path'] = df['image_path'].apply(lambda x: str(raw_dir / "images" / str(x)))
        
        # Drop rows with unmapped sentiments or missing text
        df = df.dropna(subset=['sentiment', 'ocr_text'])
        df['sentiment'] = df['sentiment'].astype(int)
        
        return df[self.target_cols]

    def merge_all(self) -> pd.DataFrame:
        """Iterates through all datasets in config and concatenates them."""
        dataframes = []
        for dataset_name, params in self.config['datasets'].items():
            raw_annotation = Path(params['raw_dir']) / params['annotation_file']
            if not raw_annotation.exists():
                print(f"Warning: '{dataset_name}' annotation file not found at {raw_annotation}. Skipping for now.")
                continue
            processed_df = self._process_dataset(dataset_name, params)
            dataframes.append(processed_df)
            
        if not dataframes:
            raise FileNotFoundError("No dataset annotation files were found to merge.")
            
        unified_df = pd.concat(dataframes, ignore_index=True)
        return unified_df

if __name__ == "__main__":
    merger = DatasetMerger(config_path="configs/dataset_config.yaml")
    final_dataset = merger.merge_all()
    print(f"Successfully merged {len(final_dataset)} total multimodal samples.")
    
    # Run image verification, OCR cleaning, deduplication & stratified splitting
    from src.data.preprocess_and_split import DataPipeline
    pipeline = DataPipeline(config_path="configs/dataset_config.yaml")
    train_df, val_df, test_df = pipeline.process(input_data=final_dataset)
