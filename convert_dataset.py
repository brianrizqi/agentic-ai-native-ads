import pandas as pd
import json
import os
import argparse

def convert_excel_to_json(input_file, output_file):
    """
    Converts native ads dataset from Excel to a refined JSON format.
    Supports both Indonesian (ID) and English (EN) examples.
    """
    print(f"Reading {input_file}...")
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Check for required columns
    required_cols = ['id', 'content', 'label', 'lang']
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Required column '{col}' not found. Available columns: {df.columns.tolist()}")
            # Attempt to use alternative names or defaults if common
            if col == 'content' and 'text' in df.columns:
                df['content'] = df['text']
            elif col == 'lang' and 'language' in df.columns:
                df['lang'] = df['language']

    json_data = []

    print(f"Processing {len(df)} rows...")
    for idx, row in df.iterrows():
        lang = str(row.get('lang', 'id')).lower()
        
        # Determine instruction based on language
        if lang == 'en':
            instruction = "Classify the following article as native ads or berita murni."
            label_map = {
                "berita murni": "berita murni",
                "native ads": "native ads"
            }
        else:
            instruction = "Klasifikasikan artikel berikut sebagai native ads atau berita murni."
            label_map = {
                "berita murni": "berita murni",
                "native ads": "native ads"
            }

        # Map label and create output structure
        raw_label = str(row.get('label', 'berita murni')).strip().lower()
        processed_label = label_map.get(raw_label, raw_label)
        
        # Standard output structure as seen in refined JSON
        output_obj = {
            "label": processed_label,
            "confidence": 0.85,  # Default confidence as seen in reference
            "reasoning": "Artikel bersifat netral dan objektif, menyajikan informasi tanpa promosi produk/brand." if processed_label in ["berita murni", "pure news"] else "Artikel mengandung unsur promosi atau persuasi terhadap produk/brand tertentu."
        }
        
        # Adjust reasoning for English
        if lang == 'en':
            if processed_label == "berita murni":
                output_obj["reasoning"] = "The article is neutral and objective, presenting information without product/brand promotion."
            else:
                output_obj["reasoning"] = "The article contains promotional or persuasive elements towards a specific product/brand."

        item = {
            "id": int(row.get('id', idx)),
            "instruction": instruction,
            "title": str(row.get('title', '')),
            "input": str(row.get('content', '')),
            "output": json.dumps(output_obj, ensure_ascii=False)
        }
        
        json_data.append(item)

    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Native Ads Excel dataset to Refined JSON.")
    parser.add_argument("--input", default="native_ads_dataset.xlsx", help="Path to input Excel file")
    parser.add_argument("--output", default="data/llm_dataset_12k_refined_full.json", help="Path to output JSON file")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    convert_excel_to_json(args.input, args.output)
