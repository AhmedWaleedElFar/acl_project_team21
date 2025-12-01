import pandas as pd
from neo4j import GraphDatabase

"""
Fetching credentials from config.txt
"""
def get_config(filename="config.txt"):
    config = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip().lower()
                    value = value.strip().strip('"').strip("'") 
                    config[key] = value
                    """
                    config structure:
                    {
                    "uri": uri,
                    "username": username,
                    "password": password
                    }
                    """
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error reading config file: {e}")
        return None

"""
Reading csv file
"""
def prepare_data():
    print("\n--- Preparing DataFrames for Node Creation ---")
    try:
        airline_surveys_df = pd.read_csv("./Airline_surveys_sample.csv")
    except FileNotFoundError:
        print("Error: 'Airline_surveys_sample.csv' not found.")
        return None, None, None, None, None

    #Raw version of the dataset
    raw_df_list = airline_surveys_df.to_dict("records")

    #Selecting features related to each node
    #dropping duplicates to make sure that no node is created more than once.
    flight_data = airline_surveys_df[[
        "flight_number",
        "fleet_type_description"
        ]].drop_duplicates().to_dict('records')

    journey_data = airline_surveys_df[[
        "feedback_ID",
        "food_satisfaction_score",
        "arrival_delay_minutes",
        "actual_flown_miles",
        "number_of_legs",
        "passenger_class"
        ]].drop_duplicates().to_dict('records')

    passenger_data = airline_surveys_df[[
        "record_locator",
        "loyalty_program_level",
        "generation"
        ]].drop_duplicates().to_dict(orient="records")

    departure_airports = airline_surveys_df[['origin_station_code']].rename(columns={'origin_station_code': 'station_code'}).drop_duplicates()
    arrival_airports = airline_surveys_df[['destination_station_code']].rename(columns={'destination_station_code': 'station_code'}).drop_duplicates()
    airport_data = pd.concat([departure_airports, arrival_airports]).drop_duplicates().to_dict('records')

    print(f"Prepared {len(passenger_data)} Passengers, {len(journey_data)} Journeys, {len(flight_data)} Flights, {len(airport_data)} Airports.")

    return raw_df_list, flight_data, journey_data, passenger_data, airport_data

"""
KG Creation
"""
def create_constraints(driver):
    print("\n--- 1. Enforcing Constraints and Indexes ---")
    
    # Creating the primary keys using the "CONSTRAINT" keyword
    # CONSTRAINT makes sure that the data is unique and serves as an index
    # No primary keys were set for the Flight node because it only consists of two features and both uniquely identify the record
    constraints = [
        "CREATE CONSTRAINT passenger_locator IF NOT EXISTS FOR (p:Passenger) REQUIRE p.record_locator IS UNIQUE",
        "CREATE CONSTRAINT journey_feedback IF NOT EXISTS FOR (j:Journey) REQUIRE j.feedback_ID IS UNIQUE",
        "CREATE CONSTRAINT airport_code IF NOT EXISTS FOR (a:Airport) REQUIRE a.station_code IS UNIQUE",
    ]
    
    # Creating Indexes on both features for faster lookup
    # Data is already unique because we removed duplicates rom the airport_data
    indexes = [
        "CREATE INDEX flight_num IF NOT EXISTS FOR (f:Flight) ON (f.flight_number)",
        "CREATE INDEX flight_fleet IF NOT EXISTS FOR (f:Flight) ON (f.fleet_type_description)",
    ]
    
    for query in constraints + indexes:
        try:
            with driver.session(database="neo4j") as session:
                session.run(query).consume()
                print(f"✅ [DONE] {query}")
        except Exception as e:
            print(f"❌ [FAIL] {query[:50]}... Error: {e}")

# Creating nodes
def create_nodes(driver, flight_data, journey_data, passenger_data, airport_data):
    print("\n--- 2. Creating All Nodes ---")
    
    with driver.session(database="neo4j") as session:
        # Passenger Nodes
        passenger_query = """
            UNWIND $data AS row
            MERGE (p:Passenger {record_locator: row.record_locator})
            ON CREATE SET p.loyalty_program_level = row.loyalty_program_level, p.generation = row.generation
        """
        # "$data" is replaced by the passenger_data
        # Using ".consume" is to allow transaction finalization and release of connection and resources.
        session.run(passenger_query, data=passenger_data).consume() 
        print(f"✅ Created {len(passenger_data)} unique Passenger nodes.")

        # Flight Nodes
        flight_query = """
            UNWIND $data AS row
            MERGE (f:Flight {
                flight_number: row.flight_number,
                fleet_type_description: row.fleet_type_description
            })
        """
        session.run(flight_query, data=flight_data).consume() 
        print(f"✅ Created {len(flight_data)} unique Flight nodes.")

        # Journey Nodes
        journey_query = """
            UNWIND $data AS row
            MERGE (j:Journey {feedback_ID: row.feedback_ID})
            ON CREATE SET 
                j.food_satisfaction_score = toInteger(row.food_satisfaction_score),
                j.arrival_delay_minutes = toInteger(row.arrival_delay_minutes),
                j.actual_flown_miles = toInteger(row.actual_flown_miles),
                j.number_of_legs = toInteger(row.number_of_legs),
                j.passenger_class = row.passenger_class
        """
        session.run(journey_query, data=journey_data).consume() 
        print(f"✅ Created {len(journey_data)} unique Journey nodes.")

        # Airport Nodes
        airport_query = """
            UNWIND $data AS row
            MERGE (a:Airport {station_code: row.station_code})
        """
        session.run(airport_query, data=airport_data).consume() 
        print(f"✅ Created {len(airport_data)} unique Airport nodes.")

def create_relationships(driver, raw_df_list):
    print("\n--- 3. Creating All Relationships ---")

    with driver.session(database="neo4j") as session:
        
        # --- (Passenger)-[:TOOK]->(Journey) ---
        took_query = """
        UNWIND $data AS row
        MATCH (p:Passenger {record_locator: row.record_locator})
        MATCH (j:Journey {feedback_ID: row.feedback_ID})
        MERGE (p)-[:TOOK]->(j)
        """
        session.run(took_query, data=raw_df_list).consume()
        print("✅ Created [:TOOK] relationships.")

        # --- (Journey)-[:ON]->(Flight) ---
        on_query = """
        UNWIND $data AS row
        MATCH (j:Journey {feedback_ID: row.feedback_ID})
        MATCH (f:Flight {
            flight_number: row.flight_number,
            fleet_type_description: row.fleet_type_description
        })
        MERGE (j)-[:ON]->(f)
        """
        session.run(on_query, data=raw_df_list).consume()
        print("✅ Created [:ON] relationships.")

        # --- (Flight)-[:DEPARTS_FROM]->(Airport) & (Flight)-[:ARRIVES_AT]->(Airport) ---
        flight_airport_query = """
        UNWIND $data AS row
        MATCH (f:Flight {
            flight_number: row.flight_number,
            fleet_type_description: row.fleet_type_description
        })
        MATCH (a_dep:Airport {station_code: row.origin_station_code})
        MATCH (a_arr:Airport {station_code: row.destination_station_code})

        MERGE (f)-[:DEPARTS_FROM]->(a_dep)
        MERGE (f)-[:ARRIVES_AT]->(a_arr)
        """
        session.run(flight_airport_query, data=raw_df_list).consume()
        print("✅ Created [:DEPARTS_FROM] and [:ARRIVES_AT] relationships.")

def main():
    config_details = get_config()
    if not config_details:
        return

    # --- Data Preparation ---
    raw_df_list, flight_data, journey_data, passenger_data, airport_data = prepare_data()
    
    if not raw_df_list:
        print("Data preparation failed or returned an empty list.")
        return 

    # --- Database Connection and KG Creation ---
    URI = config_details.get("uri")
    USERNAME = config_details.get("username")
    PASSWORD = config_details.get("password")
    AUTH = (USERNAME, PASSWORD)
    
    print(f"Connecting to: {URI} with user: {USERNAME}")
    try:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            driver.verify_connectivity()
            
            with driver.session(database="neo4j") as session:
                result = session.run("RETURN 'Connection Successful' AS message")
                print(f"Connection established successfully and queried database: {result.single()['message']}")

            print("\n--- Starting Knowledge Graph Creation ---")
            create_constraints(driver)
            create_nodes(driver, flight_data, journey_data, passenger_data, airport_data)
            create_relationships(driver, raw_df_list)
            
            print("\n✨ Knowledge Graph Creation Complete! ✨")

    except Exception as e:
        print(f"❌ Connection failed! Error: {e}")
        
    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()