import fitz
import json
import re
import os

def clean_text(text):
    # Remove multiple spaces and newlines that break words
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text) # Handle hyphenated words
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_metadata(text):
    # Rule-based metadata extraction
    materials = ["cement", "concrete", "steel", "aggregate", "masonry", "timber", "wood", "gypsum", "bitumen", "tar", "glass", "plastic", "asbestos", "ferrocement"]
    applications = ["structural", "marine", "reinforcement", "roofing", "cladding", "flooring", "water mains", "paving", "insulation", "precast"]
    
    text_lower = text.lower()
    extracted = {
        "material": next((m for m in materials if m in text_lower), "general"),
        "application": next((a for a in applications if a in text_lower), "general"),
        "keywords": [w for w in text_lower.split() if len(w) > 4 and w.isalnum()][:10]
    }
    return extracted

def ingest_pdf(pdf_path, output_path):
    print(f"Ingesting {pdf_path}...")
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            # Join lines but keep page breaks somewhat visible
            full_text += page.get_text("text") + "\n---PAGE---\n"
        
        # Pre-clean: join lines that are split by whitespace within a standard ID
        # e.g. "IS \n 269" -> "IS 269"
        full_text = re.sub(r'IS\s*\n\s*(\d+)', r'IS \1', full_text)
        full_text = re.sub(r'(\d+)\s*\n\s*:\s*(\d+)', r'\1 : \2', full_text)
        
        # Find all standards and their titles
        # Regex for IS numbers: IS [number] : [year] or IS [number] (Part [n]) : [year]
        std_pattern = r'(IS\s+\d+(?:\s*\(Part\s+\d+\))?\s*:\s*\d{4})'
        
        # Split text by standard pattern
        parts = re.split(std_pattern, full_text)
        
        chunk_data = []
        if len(parts) > 1:
            # parts[0] is header/garbage
            for i in range(1, len(parts), 2):
                std_id = clean_text(parts[i])
                content = parts[i+1] if i+1 < len(parts) else ""
                
                # Take the next few lines as title (usually after the IS number)
                # But since we split, content starts right after the match
                clean_content = clean_text(content)
                
                metadata = extract_metadata(clean_content)
                metadata["standard_id"] = std_id
                
                chunk_data.append({
                    "id": len(chunk_data),
                    "text": f"{std_id} {clean_content}",
                    "metadata": metadata
                })
        else:
            # Fallback to simple sliding window if split fails
            print("Warning: Standard-based split failed. Using sliding window.")
            words = clean_text(full_text).split()
            for i in range(0, len(words), 200):
                chunk_text = " ".join(words[i:i+250])
                chunk_data.append({
                    "id": len(chunk_data),
                    "text": chunk_text,
                    "metadata": extract_metadata(chunk_text)
                })
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, indent=2)
        print(f"Ingested {len(chunk_data)} structured chunks to {output_path}")
    except Exception as e:
        print(f"Error ingesting PDF: {e}")

if __name__ == "__main__":
    ingest_pdf("dataset.pdf", "data/processed/chunks.json")
