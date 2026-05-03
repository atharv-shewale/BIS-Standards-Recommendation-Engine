import re

class QueryParser:
    def __init__(self):
        self.materials = [
            "cement", "concrete", "steel", "aggregate", "masonry", "timber", "wood", "gypsum", 
            "bitumen", "tar", "glass", "plastic", "asbestos", "ferrocement", "pozzolana", "slag", 
            "clinker", "coir", "sand", "ppc", "opc", "psc"
        ]
        
        self.applications = [
            "structural", "marine", "reinforcement", "roofing", "cladding",
            "flooring", "water mains", "paving", "insulation",
            "foundation", "precast", "corrugated", "hollow", "solid", "architectural", "decorative",
            "blocks", "pipes", "sheets", "dams", "bridges"
        ]
        
        self.properties = [
            "strength", "durability", "grade", "compressive", "tensile",
            "chemical", "physical", "lightweight", "rapid hardening", "sulphate resisting",
            "white", "hydrophobic", "calcined clay", "corrugated"
        ]

    def parse(self, query):
        query_lower = query.lower()
        
        extracted = {
            "material": "general",
            "application": "general",
            "properties": [],
            "standard_ids": []
        }
        
        # 1. Extract Standard IDs (e.g., IS 456, IS:456, IS 12269)
        std_matches = re.findall(r'\bIS\s*:?\s*(\d+)\b', query.upper())
        for match in std_matches:
            extracted["standard_ids"].append(f"IS {match}")
            
        # 2. Extract Material
        for mat in self.materials:
            if re.search(r'\b' + re.escape(mat) + r'\b', query_lower):
                extracted["material"] = mat
                break
            
        # 3. Extract Application
        for app in self.applications:
            if re.search(r'\b' + re.escape(app) + r'\b', query_lower):
                extracted["application"] = app
                break
                
        # 4. Extract Properties
        for prop in self.properties:
            if re.search(r'\b' + re.escape(prop) + r'\b', query_lower):
                extracted["properties"].append(prop)
                
        return extracted
