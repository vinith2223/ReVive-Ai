"""
Demo NGO / recycler / composting-partner directory.

NOTE: This is placeholder demo data (carried over from the original prototype
notebook) so the app has something realistic to match against. The phone
numbers are NOT real. Before going live, replace this with a real partner
network (a database, an admin-managed table, or a live partner API).
"""

import pandas as pd

RECEIVERS = [
    {
        "waste_type": "Flowers",
        "receiver": "Temple Compost Center",
        "category": "Compost",
        "distance_km": 3,
        "contact": "+91-9876543212",
    },
    {
        "waste_type": "Flowers",
        "receiver": "Flower Recycling NGO",
        "category": "Compost",
        "distance_km": 5,
        "contact": "+91-9876543213",
    },
    {
        "waste_type": "Leftover Food",
        "receiver": "Robin Hood Army",
        "category": "Donation",
        "distance_km": 4,
        "contact": "+91-9876543210",
    },
    {
        "waste_type": "Plastic Bottles",
        "receiver": "Green Plastic Recyclers",
        "category": "Recycle",
        "distance_km": 5,
        "contact": "+91-9876543214",
    },
    {
        "waste_type": "Fabric",
        "receiver": "Textile Donation Trust",
        "category": "Reuse",
        "distance_km": 2,
        "contact": "+91-9876543216",
    },
    {
        "waste_type": "Wood",
        "receiver": "Furniture Reuse NGO",
        "category": "Reuse",
        "distance_km": 6,
        "contact": "+91-9876543218",
    },
    {
        "waste_type": "Paper Plates",
        "receiver": "Organic Compost Unit",
        "category": "Compost",
        "distance_km": 4,
        "contact": "+91-9876543220",
    },
    {
        "waste_type": "Banana Leaves",
        "receiver": "Green Composting Center",
        "category": "Compost",
        "distance_km": 3,
        "contact": "+91-9876543221",
    },
    {
        "waste_type": "Decorations",
        "receiver": "Decoration Rental Store",
        "category": "Reuse",
        "distance_km": 5,
        "contact": "+91-9876543222",
    },
]

receivers_df = pd.DataFrame(RECEIVERS)
