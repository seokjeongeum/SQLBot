DROP TABLE IF EXISTS singer_in_concert CASCADE;
DROP TABLE IF EXISTS concert CASCADE;
DROP TABLE IF EXISTS singer CASCADE;
DROP TABLE IF EXISTS stadium CASCADE;

CREATE TABLE stadium (
    Stadium_ID INTEGER PRIMARY KEY,
    Location TEXT,
    Name TEXT,
    Capacity INTEGER,
    Highest INTEGER,
    Lowest INTEGER,
    Average INTEGER
);

CREATE TABLE singer (
    Singer_ID INTEGER PRIMARY KEY,
    Name TEXT,
    Country TEXT,
    Song_Name TEXT,
    Song_release_year TEXT,
    Age INTEGER,
    Is_male BOOLEAN
);

CREATE TABLE concert (
    concert_ID INTEGER PRIMARY KEY,
    concert_Name TEXT,
    Theme TEXT,
    Stadium_ID INTEGER,
    Year TEXT,
    FOREIGN KEY (Stadium_ID) REFERENCES stadium(Stadium_ID)
);

CREATE TABLE singer_in_concert (
    concert_ID INTEGER,
    Singer_ID INTEGER,
    PRIMARY KEY (concert_ID, Singer_ID),
    FOREIGN KEY (concert_ID) REFERENCES concert(concert_ID),
    FOREIGN KEY (Singer_ID) REFERENCES singer(Singer_ID)
);

INSERT INTO stadium VALUES 
(1, 'Raith Rovers', 'Stark''s Park', 10104, 4812, 1294, 2106),
(2, 'Ayr United', 'Somerset Park', 11998, 2363, 1057, 1477),
(3, 'East Fife', 'Bayview Stadium', 2000, 1980, 533, 864),
(4, 'Queen''s Park', 'Hampden Park', 52500, 1763, 466, 730),
(5, 'Stirling Albion', 'Forthbank Stadium', 3808, 1125, 404, 642),
(6, 'Arbroath', 'Gayfield Park', 4125, 921, 411, 638),
(7, 'Alloa Athletic', 'Recreation Park', 3100, 1057, 331, 637),
(9, 'Peterhead', 'Balmoor', 4000, 837, 400, 615),
(10, 'Brechin City', 'Glebe Park', 3960, 780, 315, 552);

INSERT INTO singer VALUES 
(1, 'Joe Sharp', 'Netherlands', 'You', '1992', 52, FALSE),
(2, 'Timbaland', 'United States', 'Dangerous', '2008', 32, TRUE),
(3, 'Justin Brown', 'France', 'Hey Oh', '2013', 29, TRUE),
(4, 'Rose White', 'France', 'Sun', '2003', 41, FALSE),
(5, 'John Nizinik', 'France', 'Gentleman', '2014', 43, TRUE),
(6, 'Tribal King', 'France', 'Love', '2016', 25, TRUE);

INSERT INTO concert VALUES 
(1, 'Auditions', 'Free choice', 1, '2014'),
(2, 'Super bootcamp', 'Free choice 2', 2, '2014'),
(3, 'Home Visits', 'Bleeding Love', 2, '2015'),
(4, 'Week 1', 'Wide Awake', 10, '2014'),
(5, 'Week 1', 'Happy Tonight', 9, '2015'),
(6, 'Week 2', 'Party All Night', 7, '2015');

INSERT INTO singer_in_concert VALUES 
(1, 2), (1, 3), (1, 5), (2, 3), (2, 6),
(3, 5), (4, 4), (5, 6), (5, 3), (6, 2);
