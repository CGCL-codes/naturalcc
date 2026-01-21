import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("tablegpt")
model = AutoModelForCausalLM.from_pretrained("tablegpt", torch_dtype="auto", device_map="auto")

# 已有分类数据
wiki_categories = {
    "Culture and the arts": {
        "subcategories": {
            "Architecture": ["Ancient architecture", "Medieval architecture", "Modern architecture", "Islamic architecture", "Gothic architecture", "Baroque architecture", "Renaissance architecture", "Sustainable architecture", "Brutalist architecture", "Vernacular architecture"],
            "Art": ["Painting", "Sculpture", "Photography", "Performance art", "Printmaking", "Ceramics", "Textile arts", "Graffiti", "Digital art", "Illustration"],
            "Music": ["Classical music", "Jazz", "Rock", "Hip hop", "Folk music", "Opera", "Electronic music", "Pop music", "Blues", "Metal"],
            "Literature": ["Poetry", "Novels", "Drama", "Short stories", "Essays", "Biographies", "Fantasy", "Science fiction", "Mystery", "Historical fiction"],
            "Philosophy": ["Ethics", "Logic", "Metaphysics", "Epistemology", "Aesthetics", "Political philosophy", "Philosophy of science", "Philosophy of mind", "Phenomenology", "Existentialism"],
            "Religion": ["Christianity", "Islam", "Hinduism", "Buddhism", "Judaism", "Sikhism", "Shinto", "Taoism", "Paganism", "New religious movements"],
            "Sports": ["Ball games", "Athletic sports", "Water sports", "Winter sports", "Combat sports", "Extreme sports", "Olympic Games", "Motorsports", "Gymnastics", "Cycling"],
            "Games": ["Board games", "Card games", "Video games", "Role-playing games", "Miniature wargames", "Puzzle games", "Tabletop games", "Esports", "Party games", "Casual games"],
            "Theater": ["Tragedy", "Comedy", "Musicals", "Puppetry", "Mime", "Opera", "Improvisational theater", "Physical theater", "Street performance", "Stagecraft"],
            "Dance": ["Ballet", "Modern dance", "Hip hop dance", "Jazz dance", "Tap dance", "Folk dance", "Ballroom dance", "Contemporary dance", "Flamenco", "Salsa"]
        }
    },
    "Geography and places": {
        "subcategories": {
            "Continents": ["Africa", "Asia", "Europe", "North America", "South America", "Oceania", "Antarctica", "Arctic", "Eurasia", "Americas"],
            "Countries": ["United States", "China", "Germany", "Brazil", "India", "United Kingdom", "France", "Russia", "Japan", "Canada"],
            "Cities": ["New York City", "London", "Tokyo", "Paris", "Beijing", "Berlin", "Moscow", "Los Angeles", "Rome", "Mexico City"],
            "Landforms": ["Mountains", "Rivers", "Deserts", "Lakes", "Oceans", "Plains", "Volcanoes", "Canyons", "Islands", "Glaciers"],
            "Regions": ["Middle East", "Scandinavia", "Southeast Asia", "Latin America", "Caribbean", "Eastern Europe", "Sub-Saharan Africa", "North Africa", "Pacific Islands", "South Asia"],
            "Bodies of water": ["Atlantic Ocean", "Pacific Ocean", "Indian Ocean", "Arctic Ocean", "Mediterranean Sea", "Red Sea", "Caspian Sea", "Lake Victoria", "Amazon River", "Nile"],
            "Climate zones": ["Tropical", "Arid", "Temperate", "Continental", "Polar", "Mediterranean", "Monsoon", "Savanna", "Steppe", "Tundra"],
            "Political geography": ["Borders", "Capitals", "Provinces", "States", "Territories", "Colonies", "Dependencies", "Autonomous regions", "Federations", "City-states"],
            "Cultural geography": ["Urban areas", "Rural areas", "Cultural landscapes", "Sacred places", "Heritage sites", "Tourist destinations", "Megacities", "Suburbs", "Townships", "Villages"]
        }
    },
    "History and events": {
        "subcategories": {
            "By period": ["Prehistory", "Ancient history", "Classical antiquity", "Medieval history", "Renaissance", "Early modern period", "Industrial era", "Modern history", "Contemporary history", "Information age"],
            "By region": ["European history", "Asian history", "African history", "American history", "Middle Eastern history", "Oceanian history", "Central Asian history", "Caribbean history", "Arctic history", "Global history"],
            "Events": ["Wars", "Revolutions", "Epidemics", "Inventions", "Natural disasters", "Explorations", "Colonization", "Treaties", "Political assassinations", "Social movements"],
            "Wars": ["World War I", "World War II", "Cold War", "American Civil War", "Napoleonic Wars", "Vietnam War", "Korean War", "Iraq War", "Afghanistan War", "Crusades"],
            "Revolutions": ["French Revolution", "Russian Revolution", "American Revolution", "Industrial Revolution", "Scientific Revolution", "Digital Revolution", "Iranian Revolution", "Cuban Revolution", "Velvet Revolution", "Arab Spring"],
            "Inventions": ["Wheel", "Printing press", "Steam engine", "Electricity", "Telephone", "Airplane", "Computer", "Internet", "Smartphone", "Artificial intelligence"],
            "Exploration": ["Age of Discovery", "Space exploration", "Polar expeditions", "Ocean voyages", "Colonial expeditions", "Mars missions", "Moon landing", "Trade routes", "Cartography", "Navigation"],
            "Historical figures": ["Monarchs", "Presidents", "Revolutionaries", "Inventors", "Scientists", "Explorers", "Artists", "Military leaders", "Philosophers", "Religious leaders"],
            "Cultural eras": ["Stone Age", "Bronze Age", "Iron Age", "Middle Ages", "Renaissance", "Enlightenment", "Romantic era", "Industrial age", "Atomic age", "Digital age"]
        }
    },
    "Society": {
        "subcategories": {
            "Education": ["Primary education", "Secondary education", "Higher education", "Vocational training", "Online learning", "Homeschooling", "Special education", "Adult education", "Educational psychology", "Curriculum development"],
            "Politics": ["Democracy", "Monarchy", "Republics", "Dictatorships", "Political parties", "Elections", "Ideologies", "Parliaments", "Governance", "Public policy"],
            "Economics": ["Microeconomics", "Macroeconomics", "International trade", "Finance", "Banking", "Economic systems", "Development economics", "Labor economics", "Behavioral economics", "Economic history"],
            "Law": ["Constitutional law", "Criminal law", "Civil law", "International law", "Human rights law", "Property law", "Contract law", "Corporate law", "Environmental law", "Family law"],
            "Social issues": ["Poverty", "Inequality", "Racism", "Gender issues", "Disability rights", "Human rights", "Censorship", "Unemployment", "Crime", "Immigration"],
            "Family": ["Marriage", "Parenting", "Kinship", "Childhood", "Elder care", "Households", "Divorce", "Family law", "Domestic violence", "Adoption"],
            "Health": ["Public health", "Nutrition", "Mental health", "Diseases", "Vaccines", "Healthcare systems", "Epidemiology", "Medical research", "Fitness", "Occupational health"],
            "Communication": ["Language", "Media", "Mass communication", "Social media", "Interpersonal communication", "Telecommunication", "Journalism", "Broadcasting", "Advertising", "Publishing"],
            "Culture": ["Customs", "Traditions", "Festivals", "Cuisine", "Fashion", "Mythology", "Cultural identity", "Rituals", "Heritage", "Folklore"]
        }
    },
    "Science": {
        "subcategories": {
            "Physics": ["Mechanics", "Thermodynamics", "Electromagnetism", "Quantum physics", "Relativity", "Nuclear physics", "Optics", "Acoustics", "Astrophysics", "Condensed matter physics"],
            "Chemistry": ["Organic chemistry", "Inorganic chemistry", "Physical chemistry", "Biochemistry", "Analytical chemistry", "Theoretical chemistry", "Electrochemistry", "Materials chemistry", "Nuclear chemistry", "Environmental chemistry"],
            "Biology": ["Genetics", "Evolution", "Microbiology", "Botany", "Zoology", "Ecology", "Human biology", "Molecular biology", "Cell biology", "Marine biology"],
            "Earth sciences": ["Geology", "Meteorology", "Oceanography", "Paleontology", "Geography", "Climatology", "Seismology", "Volcanology", "Hydrology", "Environmental science"],
            "Astronomy": ["Planets", "Stars", "Galaxies", "Cosmology", "Exoplanets", "Astrobiology", "Space telescopes", "Observatories", "Astrophysics", "Dark matter"],
            "Mathematics": ["Algebra", "Geometry", "Calculus", "Statistics", "Probability", "Number theory", "Topology", "Applied mathematics", "Discrete mathematics", "Mathematical logic"],
            "Computer science": ["Algorithms", "Data structures", "Artificial intelligence", "Machine learning", "Databases", "Programming languages", "Operating systems", "Networks", "Cybersecurity", "Software engineering"],
            "Medicine": ["Anatomy", "Physiology", "Pathology", "Pharmacology", "Surgery", "Pediatrics", "Oncology", "Neurology", "Cardiology", "Psychiatry"],
            "Engineering": ["Civil engineering", "Mechanical engineering", "Electrical engineering", "Chemical engineering", "Aerospace engineering", "Biomedical engineering", "Computer engineering", "Industrial engineering", "Environmental engineering", "Materials engineering"]
        }
    },
    "Technology": {
        "subcategories": {
            "Information technology": ["Computing", "Software", "Hardware", "Networking", "Internet", "Cybersecurity", "Data science", "Cloud computing", "Mobile technology", "Artificial intelligence"],
            "Transportation": ["Automobiles", "Aviation", "Rail transport", "Maritime transport", "Spaceflight", "Public transport", "Cycling", "Motorcycles", "Logistics", "Infrastructure"],
            "Energy": ["Fossil fuels", "Nuclear energy", "Renewable energy", "Solar power", "Wind power", "Hydropower", "Geothermal energy", "Bioenergy", "Energy storage", "Energy efficiency"],
            "Communication": ["Telephone", "Radio", "Television", "Satellite communication", "Internet", "Mobile communication", "Optical fiber", "Postal systems", "Wireless technology", "Broadcasting"],
            "Agriculture": ["Farming", "Irrigation", "Agricultural machinery", "Agrochemicals", "Biotechnology", "Aquaculture", "Horticulture", "Forestry", "Livestock", "Sustainable agriculture"],
            "Construction": ["Building", "Architecture", "Civil works", "Structural engineering", "Urban planning", "Green building", "Project management", "Surveying", "Demolition", "Restoration"],
            "Military technology": ["Weapons", "Armor", "Naval technology", "Aerospace technology", "Cyberwarfare", "Surveillance", "Tanks", "Missiles", "Drones", "Robotics"],
            "Industrial processes": ["Manufacturing", "Automation", "Assembly lines", "Robotics", "3D printing", "Nanotechnology", "Textile industry", "Mining", "Chemical industry", "Food processing"],
            "Everyday technology": ["Home appliances", "Personal electronics", "Wearables", "Kitchen technology", "Entertainment technology", "Sports equipment", "Lighting", "Clothing technology", "Medical devices", "Cleaning technology"]
        }
    },
    "Mathematics": {
        "subcategories": {
            "Pure mathematics": ["Algebra", "Geometry", "Trigonometry", "Calculus", "Number theory", "Topology", "Mathematical logic", "Set theory", "Graph theory", "Combinatorics"],
            "Applied mathematics": ["Statistics", "Probability", "Mathematical physics", "Computational mathematics", "Operations research", "Control theory", "Game theory", "Information theory", "Numerical analysis", "Cryptography"],
            "History of mathematics": ["Ancient mathematics", "Greek mathematics", "Islamic mathematics", "Medieval mathematics", "Renaissance mathematics", "19th century mathematics", "20th century mathematics", "Famous mathematicians", "Mathematical texts", "Mathematical discoveries"],
            "Mathematical applications": ["Engineering", "Economics", "Biology", "Physics", "Chemistry", "Medicine", "Computer science", "Social sciences", "Finance", "Astronomy"],
            "Branches of mathematics": ["Arithmetic", "Algebra", "Geometry", "Calculus", "Probability", "Statistics", "Topology", "Logic", "Analysis", "Differential equations"]
        }
    },
    "Health": {
        "subcategories": {
            "Medicine": ["Cardiology", "Neurology", "Oncology", "Pediatrics", "Surgery", "Psychiatry", "Dermatology", "Endocrinology", "Orthopedics", "Radiology"],
            "Fitness": ["Exercise", "Yoga", "Pilates", "Strength training", "Aerobics", "Running", "Cycling", "Swimming", "Sports medicine", "Nutrition"],
            "Diseases": ["Infectious diseases", "Chronic diseases", "Genetic disorders", "Autoimmune diseases", "Cancers", "Mental disorders", "Cardiovascular diseases", "Respiratory diseases", "Neurological diseases", "Rare diseases"],
            "Public health": ["Epidemiology", "Global health", "Vaccination", "Health policy", "Health education", "Environmental health", "Maternal health", "Child health", "Occupational health", "Injury prevention"],
            "Healthcare systems": ["Hospitals", "Clinics", "Primary care", "Health insurance", "Telemedicine", "Nursing", "Pharmacy", "Emergency medicine", "Medical research", "Medical education"]
        }
    },
    "People": {
        "subcategories": {
            "Occupations": ["Doctors", "Engineers", "Teachers", "Artists", "Scientists", "Athletes", "Writers", "Politicians", "Entrepreneurs", "Musicians"],
            "Ethnic groups": ["Africans", "Europeans", "Asians", "Native Americans", "Australians", "Pacific Islanders", "Middle Easterners", "Latinos", "Indigenous peoples", "Mixed heritage groups"],
            "Demographics": ["Population", "Migration", "Fertility", "Mortality", "Aging", "Urbanization", "Education levels", "Income levels", "Health statistics", "Employment"],
            "Notable people": ["Leaders", "Philosophers", "Scientists", "Artists", "Athletes", "Writers", "Inventors", "Explorers", "Activists", "Entrepreneurs"]
        }
    },
    "Philosophy": {
        "subcategories": {
            "Branches": ["Metaphysics", "Epistemology", "Logic", "Ethics", "Aesthetics", "Political philosophy", "Philosophy of science", "Philosophy of mind", "Philosophy of language", "Existentialism"],
            "Philosophers": ["Plato", "Aristotle", "Descartes", "Kant", "Nietzsche", "Hegel", "Confucius", "Socrates", "Wittgenstein", "Heidegger"],
            "Philosophical movements": ["Idealism", "Realism", "Empiricism", "Rationalism", "Phenomenology", "Pragmatism", "Existentialism", "Structuralism", "Postmodernism", "Marxism"]
        }
    },
    "Religion and belief systems": {
        "subcategories": {
            "Major religions": [
                "Christianity", "Islam", "Judaism", "Hinduism", "Buddhism",
                "Sikhism", "Taoism", "Confucianism", "Shinto", "Zoroastrianism"
            ],
            "Religious texts": [
                "Bible", "Quran", "Torah", "Vedas", "Tripitaka",
                "Guru Granth Sahib", "Analects", "Kojiki", "Avesta", "Book of Mormon"
            ],
            "Belief systems": [
                "Atheism", "Agnosticism", "Deism", "Pantheism", "Animism",
                "Polytheism", "Monotheism", "Humanism", "New Age", "Spiritualism"
            ]
        }
    },
    "Society and social sciences": {
        "subcategories": {
            "Disciplines": [
                "Anthropology", "Sociology", "Political science", "Economics", "Linguistics",
                "Psychology", "Law", "Education", "Communication studies", "Human geography"
            ],
            "Societal issues": [
                "Poverty", "Inequality", "Crime", "Globalization", "Immigration",
                "Urbanization", "Gender studies", "Racism", "War", "Public health"
            ],
            "Institutions": [
                "Government", "Courts", "Education systems", "Healthcare systems", "Media organizations",
                "NGOs", "Corporations", "Trade unions", "Religious institutions", "Military"
            ]
        }
    },
    "Technology and applied sciences": {
        "subcategories": {
            "Engineering fields": [
                "Civil engineering", "Mechanical engineering", "Electrical engineering", "Chemical engineering", "Aerospace engineering",
                "Biomedical engineering", "Environmental engineering", "Computer engineering", "Industrial engineering", "Materials science"
            ],
            "Information technology": [
                "Software", "Hardware", "Networking", "Databases", "Artificial intelligence",
                "Cybersecurity", "Cloud computing", "Big data", "Blockchain", "Quantum computing"
            ],
            "Applied sciences": [
                "Medicine", "Agriculture", "Architecture", "Robotics", "Nanotechnology",
                "Pharmacology", "Forensic science", "Food science", "Veterinary medicine", "Energy technology"
            ]
        }
    },
    "General reference": {
        "subcategories": {
            "Reference works": [
                "Encyclopedias", "Dictionaries", "Atlases", "Gazetteers", "Almanacs",
                "Yearbooks", "Thesauri", "Bibliographies", "Chronologies", "Handbooks"
            ],
            "Knowledge organization": [
                "Libraries", "Archives", "Databases", "Indexes", "Catalogs",
                "Citation systems", "Knowledge graphs", "Ontologies", "Metadata", "Taxonomies"
            ],
            "Information sources": [
                "Newspapers", "Magazines", "Academic journals", "Websites", "Reports",
                "Manuscripts", "Government publications", "Research papers", "Maps", "Audio-visual media"
            ]
        }
    }
}

# prompt 模板
example_prompt_template = """
Generate a clean, well-structured Markdown table about {topic}.  
"""


def generate_table(topic):
    prompt = example_prompt_template.format(topic=topic)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def save_tables(base_dir="wiki_tables"):
    os.makedirs(base_dir, exist_ok=True)
    for category, data in wiki_categories.items():
        cat_dir = os.path.join(base_dir, category.replace(" ", "_"))
        os.makedirs(cat_dir, exist_ok=True)
        if "subcategories" in data:
            for subcat, topics in data["subcategories"].items():
                subcat_dir = os.path.join(cat_dir, subcat.replace(" ", "_"))
                os.makedirs(subcat_dir, exist_ok=True)
                for topic in topics:
                    table_md = generate_table(topic)
                    file_path = os.path.join(subcat_dir, f"{topic.replace(' ', '_')}.md")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(table_md)


if __name__ == "__main__":
    save_tables()
