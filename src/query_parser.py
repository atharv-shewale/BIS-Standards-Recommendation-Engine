import re

class QueryParser:
    def __init__(self):
        # Extremely fast rule-based dictionaries for compliance intelligence
        self.materials = [
            "cement", "concrete", "steel", "aggregate", "aggregates", 
            "masonry", "timber", "wood", "gypsum", "bitumen", "tar", 
            "glass", "plastic", "asbestos", "ferrocement", "pozzolana", "slag", "clinker", "coir", "sand"
        ]
        
        self.applications = [
            "structural", "marine", "reinforcement", "roofing", "cladding",
            "masonry", "flooring", "water mains", "paving", "insulation",
            "foundation", "precast", "corrugated", "hollow", "solid", "architectural", "decorative",
            "blocks", "pipes", "sheets"
        ]
        
        self.properties = [
            "strength", "durability", "grade", "compressive", "tensile",
            "chemical", "physical", "lightweight", "rapid hardening", "sulphate resisting",
            "white", "hydrophobic", "calcined clay", "corrugated"
        ]

    def parse(self, query):
        query_lower = query.lower()
        
        extracted = {
            "material": "",
            "application": "",
            "properties": []
        }
        
        # Extract Material (can match multiple and join, or just take first. Let's take all matched for robustness)
        matched_mats = []
        for mat in self.materials:
            if re.search(r'\b' + re.escape(mat) + r'\b', query_lower):
                matched_mats.append(mat)
        if matched_mats:
            extracted["material"] = matched_mats[0] # primary material
            
        # Extract Application
        matched_apps = []
        for app in self.applications:
            if re.search(r'\b' + re.escape(app) + r'\b', query_lower):
                matched_apps.append(app)
        if matched_apps:
            extracted["application"] = matched_apps[0]
                
        # Extract Properties
        for prop in self.properties:
            if re.search(r'\b' + re.escape(prop) + r'\b', query_lower):
                extracted["properties"].append(prop)
                
        return extracted
