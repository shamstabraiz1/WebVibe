"""Create sample Urdu sentiment dataset for training."""

import pandas as pd
import random
from pathlib import Path

def create_sample_dataset():
    """Create a sample dataset with Urdu sentiment examples."""
    
    # Sample Urdu texts with labels (0=extremely negative, 1=negative, 2=neutral, 3=positive, 4=extremely positive)
    sample_data = [
        # Extremely Positive (4)
        ("یہ فلم بہت شاندار ہے", 4),
        ("واہ! کیا بات ہے", 4),
        ("سب سے بہترین تجربہ", 4),
        ("بہت خوبصورت اور دلچسپ", 4),
        ("انتہائی عمدہ کارکردگی", 4),
        
        # Positive (3)
        ("یہ اچھا ہے", 3),
        ("کافی بہتر ہے", 3),
        ("پسند آیا", 3),
        ("اچھی کوشش ہے", 3),
        ("ٹھیک ٹھاک ہے", 3),
        
        # Neutral (2)
        ("عام سی بات ہے", 2),
        ("کوئی خاص بات نہیں", 2),
        ("ٹھیک ہے", 2),
        ("معمولی ہے", 2),
        ("کچھ خاص نہیں", 2),
        
        # Negative (1)
        ("پسند نہیں آیا", 1),
        ("برا تجربہ تھا", 1),
        ("کمزور کارکردگی", 1),
        ("مایوس کن", 1),
        ("بہتری کی ضرورت", 1),
        
        # Extremely Negative (0)
        ("یہ بکواس ہے", 0),
        ("بہت برا ہے", 0),
        ("انتہائی مایوس کن", 0),
        ("وقت کی بربادی", 0),
        ("سب سے بدترین", 0),
    ]
    
    # Expand dataset by creating variations
    expanded_data = []
    
    for text, label in sample_data:
        # Add original
        expanded_data.append((text, label))
        
        # Add variations with different contexts
        contexts = [
            f"میرے خیال میں {text}",
            f"سچ کہوں تو {text}",
            f"یقیناً {text}",
            f"واقعی {text}",
            f"حقیقت میں {text}"
        ]
        
        for context in contexts[:2]:  # Add 2 variations per original
            expanded_data.append((context, label))
    
    # Add more diverse examples
    additional_examples = [
        # Movie reviews
        ("یہ فلم دیکھنے کے قابل ہے", 3),
        ("اداکاری بہت اچھی ہے", 4),
        ("کہانی کمزور ہے", 1),
        ("موسیقی عمدہ ہے", 3),
        ("ہدایت کاری شاندار ہے", 4),
        
        # Product reviews
        ("یہ پروڈکٹ اچھا ہے", 3),
        ("قیمت مناسب ہے", 3),
        ("کوالٹی خراب ہے", 1),
        ("بہت مہنگا ہے", 1),
        ("پیسے کی قدر ہے", 4),
        
        # Social media style
        ("آج کا دن اچھا گزرا", 3),
        ("موسم خوشگوار ہے", 3),
        ("ٹریفک بہت زیادہ ہے", 1),
        ("کھانا لذیذ تھا", 4),
        ("سروس اچھی نہیں تھی", 1),
        
        # Mixed script examples
        ("ye film bohat acha hai", 4),
        ("mujhe pasand nahi aya", 1),
        ("theek hai", 2),
        ("bahut kharab hai", 0),
        ("excellent performance", 4),
    ]
    
    expanded_data.extend(additional_examples)
    
    # Shuffle the data
    random.shuffle(expanded_data)
    
    # Create DataFrame
    df = pd.DataFrame(expanded_data, columns=['text', 'label'])
    
    # Add metadata
    df['source'] = 'synthetic'
    df['dialect'] = 'mixed'
    df['confidence'] = 1.0
    
    return df

def main():
    """Create and save the sample dataset."""
    print("Creating sample Urdu sentiment dataset...")
    
    # Create dataset
    df = create_sample_dataset()
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Save dataset
    output_path = data_dir / "urdu_sentiment_sample.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✅ Dataset created with {len(df)} samples")
    print(f"📁 Saved to: {output_path}")
    
    # Print statistics
    print("\n📊 Dataset Statistics:")
    print(df['label'].value_counts().sort_index())
    
    print("\n🔤 Sample texts:")
    for i, row in df.head(10).iterrows():
        print(f"  {row['text']} -> {row['label']}")

if __name__ == "__main__":
    main()