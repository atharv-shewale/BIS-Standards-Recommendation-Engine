import fitz
import json
import re
import os

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_metadata(text):
    # Rule-based metadata extraction
    materials = ["cement", "concrete", "steel", "aggregate", "masonry", "timber", "wood", "gypsum", "bitumen", "tar", "glass", "plastic", "asbestos", "ferrocement"]
    applications = ["structural", "marine", "reinforcement", "roofing", "cladding", "flooring", "water mains", "paving", "insulation", "precast"]
    
    extracted = {
        "material": next((m for m in materials if m in text.lower()), "general"),
        "application": next((a for a in applications if a in text.lower()), "general"),
        "keywords": [w for w in text.lower().split() if len(w) > 4 and w.isalnum()][:10]
    }
    return extracted

def ingest_pdf(pdf_path, output_path):
    print(f"Ingesting {pdf_path}...")
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        
        full_text = clean_text(full_text)
        
        # Broad Standard-aware chunking
        sections = re.split(r'(IS\s+\d+(?:\s*\(Part\s+\d+\))?\s*:\s*\d+)', full_text)
        
        chunk_data = []
        if len(sections) > 1:
            for i in range(1, len(sections), 2):
                std_id = sections[i]
                std_id_clean = re.sub(r'\s+', ' ', std_id).strip()
                
                content = (sections[i] + sections[i+1]) if i+1 < len(sections) else sections[i]
                
                metadata = extract_metadata(content)
                metadata["standard_id"] = std_id_clean
                
                chunk_data.append({
                    "id": len(chunk_data),
                    "text": content,
                    "metadata": metadata
                })
        else:
            print("Warning: Standard split failed.")
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, indent=2)
        print(f"Ingested {len(chunk_data)} structured chunks to {output_path}")
    except Exception as e:
        print(f"Error ingesting PDF: {e}")
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, indent=2)
        print(f"Ingested {len(chunk_data)} structured chunks to {output_path}")
    except Exception as e:
        print(f"Error ingesting PDF: {e}")

if __name__ == "__main__":
    ingest_pdf("dataset.pdf", "data/processed/chunks.json")
