import random
import psycopg2
from typing import List, Tuple

def generate_random_data(num_rows: int) -> Tuple[List[tuple]]:
    # Stadium data
    locations = ['Raith Rovers', 'Ayr United', 'East Fife', 'Queen\'s Park', 
                 'Stirling Albion', 'Arbroath', 'Alloa Athletic', 'Peterhead', 'Brechin City']
    stadium_names = ['Stark\'s Park', 'Somerset Park', 'Bayview Stadium', 'Hampden Park', 
                    'Forthbank Stadium', 'Gayfield Park', 'Recreation Park', 'Balmoor', 'Glebe Park']
    
    # Singer data
    singer_names = ['Joe Sharp', 'Timbaland', 'Justin Brown', 'Rose White', 'John Nizinik', 'Tribal King']
    countries = ['Netherlands', 'United States', 'France']
    songs = ['You', 'Dangerous', 'Hey Oh', 'Sun', 'Gentleman', 'Love']
    
    # Concert data
    concert_names = ['Auditions', 'Super bootcamp', 'Home Visits', 'Week 2']
    themes = ['Free choice', 'Bleeding Love', 'Wide Awake', 'Happy Tonight', 'Party All Night']

    stadium_data = []
    singer_data = []
    concert_data = []
    singer_concert_data = []

    # Generate stadium data
    for _ in range(num_rows):
        stadium_id = random.randint(11, 100)
        capacity = random.randint(2000, 60000)
        highest = random.randint(500, capacity)
        lowest = random.randint(100, highest)
        average = random.randint(lowest, highest)
        stadium_data.append((
            stadium_id,
            random.choice(locations),
            random.choice(stadium_names),
            capacity,
            highest,
            lowest,
            average
        ))

    # Generate singer data
    for _ in range(num_rows):
        singer_data.append((
            random.randint(7, 100),
            random.choice(singer_names),
            random.choice(countries),
            random.choice(songs),
            str(random.randint(1990, 2023)),
            random.randint(20, 60),
            random.choice([True, False])
        ))

    # Generate concert data
    for _ in range(10,num_rows):
        concert_data.append((
            _,
            random.choice(concert_names),
            random.choice(themes),
            random.randint(1, 7),
            str(random.randint(2010, 2023))
        ))

    # Generate singer_in_concert data
    for _ in range(num_rows):
        singer_concert_data.append((
            random.randint(1, 10),
            random.randint(1, 6)
        ))

    return stadium_data, singer_data, concert_data, singer_concert_data

def insert_to_postgres(data: Tuple[List[tuple]]) -> None:
    conn = psycopg2.connect(
        dbname="concert_singer",
        user="sqlbot",
        password="sqlbot_pw",
        host="localhost",
        port=5434,
    )
    cursor = conn.cursor()
    
    stadium_data, singer_data, concert_data, singer_concert_data = data
    
    try:        
        cursor.executemany("""
            INSERT INTO concert (concert_ID, concert_Name, Theme, Stadium_ID, Year)
            VALUES (%s, %s, %s, %s, %s)
        """, concert_data)        
        conn.commit()
    finally:
        cursor.close()
        conn.close()

# Generate and insert data
random_data = generate_random_data(500)  # Generate 5 rows for each table
insert_to_postgres(random_data)
