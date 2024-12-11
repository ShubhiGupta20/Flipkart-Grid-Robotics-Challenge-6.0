import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load the pretrained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# List of brand names
brand_labels = ["Nike", "Adidas", "Apple", "Samsung", "StarBucks", "Coca Cola", "Allen Solly", "Levis", "Lee", "Pepe Jeans", "Calvin Klein", "Diesel", "Wrangler", "Spykar",
                "Flying Machine", "7 For All Mankind", "DL1961", "Gap", "Mufti", "Numero Uno", "True Religion", "US Polo", "Van Heusen", "Everyday Denim", "FRAME", "Gas jeans",
                "Jack & Jones", "Jean brands", "M&S", "Res Denim", "Peter England", "Tommy Hilfiger", "Arrow", "Louis Philippe", "Raymond", "Nike", "United Colors of Benetton",
                "Blackberrys", "Park Avenue", "Fabindia", "Hugo Boss", "Indian Terrain", "Lacoste", "Nestle", "Amul", "Bisleri", "Banas Dairy", "Britannia", "DABUR",
                "Haldiram's", "Mother Dairy", "Parle", "Pantanjali", "Balaji Waffers", "Vadilal", "Havmor", "Joy", "Nutella", "Pringles", "Bingo", "Coca-Cola", "Tic-Tac",
                "Thumps Up", "Kissan", "PaperBoat", "Balaji", "Emami", "Mc Cain", "MTR", "MDH", "McDonald", "Everest", "Rooh Afza", "Burger King", "Kwality Walls",
                "Saffola", "Caswell-Massey", "Vimal Agro", "Fortune", "Tata Sampann", "Basmati Rice", "India Gate", "RedBull", "GUCCI", "Bellavita", "Olay", "WhiteTone",
                "Ponds", "Nivea", "JOY", "Himalaya", "Lakme", "Garnier", "BOROPLUS", "Biotique", "Mamaearth", "Vaseline", "Parachute Advansed", "Cetaphill", "Khadi Natural",
                "Lotus", "L’Oréal Paris", "VENUSIA", "FIXDERMA", "Fair & Lovely", "Dr.Sheth's", "Lba", "FoxTale", "TRESemme", "Dove", "Plum", "Park Avenue",
                "Head & Shoulders", "Herbal Essences", "Clinic Plus", "SunSilk", "Bare Anatomy", "Pantene", "WishCare", "Nyle", "Meera", "St.Botanica", "Indulekha", "WOW ",
                "Pilgrim", "Kesh King", "Ravel", "sesa", "Colgate", "Pesodent", "Sensodyne", "Patanjali", "Perfora", "BENTODENT", "Vantej", "VICCO", "DentoShine", "Meridol",
                "aquawhite", "Aquafresh", "All Out", "Cocomo", "ZANDU", "Cadbury", "Amul", "Ferrero", "KIT KAT", "Nestle", "Fabelle", "HERSHEY'S", "Sunfeast Dark Fantasy",
                "Snickers", "Unibic", "Bounty", "Mars", "Dukes", "Galaxy", "Parle", "Britannia", "MYFITNESS", "Loacker", "Lotte", "CookieMan", "Sundrop", "Candyman",
                "RAGE COFFEE", "Sugar Free D'lite", "Oreo", "Dukes", "McVities", "Biscoff", "MAGGI", "SITARA", "20-20", "50 50", "Hide & Seek", "Krackjack", "Milano",
                "PriyaGold", "Anmol Marie", "Sunfeast", "Cremica", "Unibic", "Haldiram", "Bella Vita Luxury", "Park Avenue", "Wild Stone", "BEARDO", "Yardley", "Engage",
                "The Man Company", "RAMSONS", "Skinn By Titan", "BLANKO", "Calvin Klein", "Plum", "Carlton London", "RENEE", "VILLAIN", "FOGG", "GUESS", "AJMAL", "Layer'r",
                "AdilQadri", "Dior", "Lattafa", "ARMAF", "ACO perfumes", "Aqualogica", "Riya", "MARYAJ", "HVNLY", "Tommy Hilfiger", "Victoria's Secret", "OSR", "Nike",
                "Tom Ford", "JAGUAR", "OSCAR", "Armani Beauty", "REVLON", "Dolce & Gabbana", "Just Herbs","RASASI","Apple","Sony","Dell"
,"HP","Lenovo","Asus","Boat","JBL","Mi","Puma","Adidas","Nike","Levi's","Woodland","Allen Solly","Peter England","BIBA","Van Heusen","bata","Skechers","Red Tape","Crocs","Reebok","LG","Whirlpool","Godrej","Voltas","Maybelline","OnePlus","Vivo","Realme","Motorola","Nokia","Canon","Fujifilm","Panasonic","Acer","U.S.Polo","H&M","Spykar","ONLY","Lancer","Relaxo","Mochi","Campus","Clarks","Liberty","Metro","Philips","Prestige","Bajaj","Havells","Kent","Garnier","Nykaa","Colgate","Vaseline","HRX","Wildcraft","Yonex","Cosco","Pigeon","Borosil","Cello","Wonderchef","Zebronics","Seagate","Belkin","Syska","Poco","Pantaloons","Max","Lee","AND","Khadim's","Paragon","Surya","Lotus","Gillete","Olay","Faces Canada","Swiss Beauty","Revlon","Fastrack","Sonata","V-Guard","Ubon","Roadster","Flying Machine","Jockey","Lux Cozi","Zivame","Jordan","Marc Loire","Asian Footwear","Plum","The Man Company","Bombay Shaving Company","Whisper","Stayfree","Sofy","Carefree","Nua","Paree","Bella","Kotex","Everteen","FabPad","Huda Beauty","Sephora Collection","Bobbi Brown","MAC","Colorbar","Sugar Cosmetics","MyGlamm","Blue Heaven","Elle 18","Insight Cosmetics","Dior","NARS","Just Herbs","Xiaomi","Oppo","Apple","Acer","Bosch","IFB","Fila","Prestige","Crompton","BoAt","Chicco","Nerf","Tecno","Lava","Havells","Mars",
"Kobo","MuscleBlaze","Avent","Myprotein","Fitbit","HealthKart","Sole Fitness","Decathlon"]

def detect_brand(image):
    """Detect the brand from an image."""
    image = image.resize((224, 224))
    inputs = processor(text=brand_labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    similarity = torch.cosine_similarity(image_embeds, text_embeds)
    best_match_index = similarity.argmax().item()
    return brand_labels[best_match_index]