query1 = """
MATCH (origin:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(destination:Airport)
RETURN
    origin.station_code AS origin,
    destination.station_code AS destination,
    count(f) AS flight_count
ORDER BY
    flight_count DESC
LIMIT 5
"""

query2 = """
MATCH (j:Journey)-[:ON]->(f:Flight)
RETURN
    f.flight_number AS flight_id,
    count(j) AS passenger_feedback_count
ORDER BY
    passenger_feedback_count DESC
LIMIT 10
"""

query3 = """
MATCH (p:Passenger)-[:TOOK]->(j:Journey)
WHERE j.number_of_legs > 1
RETURN
    p.generation AS generation,
    count(j) AS multi_leg_count,
    avg(j.food_satisfaction_score) AS avg_score
ORDER BY
    multi_leg_count DESC
"""


query4 = """
MATCH (j:Journey)-[:ON]->(f:Flight)
RETURN
    f.flight_number AS flight_id,
    avg(j.arrival_delay_minutes) AS avg_arrival_delay
ORDER BY
    avg_arrival_delay ASC
LIMIT 10
"""

query5 = """
MATCH (p:Passenger)-[:TOOK]->(j:Journey)
RETURN
    p.loyalty_program_level AS loyalty_level,
    avg(j.actual_flown_miles) AS avg_actual_flown_miles
ORDER BY
    avg_actual_flown_miles DESC
"""

expected_results = {
    "query1": [
        {"origin": "LAX", "destination": "IAX", "flight_count": 21},
        {"origin": "LAX", "destination": "EWX", "flight_count": 17},
        {"origin": "IAX", "destination": "LAX", "flight_count": 17},
        {"origin": "SAX", "destination": "IAX", "flight_count": 15},
        {"origin": "IAX", "destination": "EWX", "flight_count": 14}
    ],
    "query2": [
        {"flight_id": 42, "feedback_count": 14},
        {"flight_id": 19, "feedback_count": 13},
        {"flight_id": 86, "feedback_count": 12},
        {"flight_id": 27, "feedback_count": 12},
        {"flight_id": 966, "feedback_count": 12},
        {"flight_id": 57, "feedback_count": 11},
        {"flight_id": 1686, "feedback_count": 11},
        {"flight_id": 219, "feedback_count": 11},
        {"flight_id": 819, "feedback_count": 9},
        {"flight_id": 991, "feedback_count": 9}
    ],
    "query3": [
        {"generation": "Boomer", "multi_leg_count": 498, "avg_score": 2.7911646586345387},
        {"generation": "Gen X", "multi_leg_count": 285, "avg_score": 2.999999999999999},
        {"generation": "Millennial", "multi_leg_count": 130, "avg_score": 2.738461538461538},
        {"generation": "Silent", "multi_leg_count": 48, "avg_score": 2.6874999999999996},
        {"generation": "Gen Z", "multi_leg_count": 18, "avg_score": 3.2777777777777777}
    ],
    "query4": [
        {"flight_id": 2442, "avg_arrival_delay": -99.0},
        {"flight_id": 274, "avg_arrival_delay": -60.0},
        {"flight_id": 425, "avg_arrival_delay": -59.0},
        {"flight_id": 982, "avg_arrival_delay": -46.0},
        {"flight_id": 120, "avg_arrival_delay": -45.0},
        {"flight_id": 1237, "avg_arrival_delay": -44.0},
        {"flight_id": 894, "avg_arrival_delay": -42.5},
        {"flight_id": 3546, "avg_arrival_delay": -42.0},
        {"flight_id": 942, "avg_arrival_delay": -41.333333333333336},
        {"flight_id": 828, "avg_arrival_delay": -41.0}
    ],
    "query5": [
        {"loyalty_level": "global services", "avg_actual_flown_miles": 2648.083333333333},
        {"loyalty_level": "premier gold", "avg_actual_flown_miles": 2461.018518518519},
        {"loyalty_level": "premier platinum", "avg_actual_flown_miles": 2420.5714285714294},
        {"loyalty_level": "non-elite", "avg_actual_flown_miles": 2254.072},
        {"loyalty_level": "premier silver", "avg_actual_flown_miles": 2068.673529411763},
        {"loyalty_level": "NBK", "avg_actual_flown_miles": 1989.0},
        {"loyalty_level": "premier 1k", "avg_actual_flown_miles": 1897.6666666666672}
    ]
}

def compare_results(query_name, actual_list, expected_list, tolerance=0.01):
    """Compares the actual query results with the expected results."""
    
    # 1. Compare size
    if len(actual_list) != len(expected_list):
        print(f"❌ Verification Failed for {query_name}: Expected {len(expected_list)} rows, but got {len(actual_list)}.")
        return False

    # Standardize result structure for comparison (e.g., convert record objects to dicts)
    actual_dicts = [dict(record) for record in actual_list]
    
    # 2. Iterate and compare row by row (assuming sorted lists as per ORDER BY)
    all_match = True
    for i, (actual, expected) in enumerate(zip(actual_dicts, expected_list)):
        row_match = True
        
        # 3. Compare keys and values
        if set(actual.keys()) != set(expected.keys()):
            print(f"❌ {query_name} Row {i+1} keys mismatch.")
            all_match = False
            break

        for key, expected_val in expected.items():
            actual_val = actual.get(key)
            
            # Special comparison for float values (allowing for small tolerance)
            if isinstance(expected_val, float):
                if not isinstance(actual_val, (float, int)) or abs(actual_val - expected_val) > tolerance:
                    print(f"❌ {query_name} Row {i+1} - Key '{key}' mismatch: Expected ~{expected_val:.2f}, Got {actual_val}.")
                    row_match = False
            
            # Direct comparison for other types (string, int)
            elif actual_val != expected_val:
                print(f"❌ {query_name} Row {i+1} - Key '{key}' mismatch: Expected '{expected_val}', Got '{actual_val}'.")
                row_match = False
        
        if not row_match:
            all_match = False
            # break # Don't break here, continue to show all mismatches if possible
            
    return all_match


def verify_kg_creation(driver: GraphDatabase.driver):
    """
    Executes a set of Cypher queries to verify the structure and content 
    of the newly created Knowledge Graph against known results.
    """
    print("\n--- 4. Verifying Knowledge Graph Data Integrity ---")
    
    verification_queries = {
        "query1": query1,
        "query2": query2,
        "query3": query3,
        "query4": query4,
        "query5": query5,
    }
    
    overall_success = True

    try:
        with driver.session(database="neo4j") as session:
            for name, query in verification_queries.items():
                print(f"-> Running {name}...")
                
                # Execute the query
                result = session.run(query)
                actual_records = [record for record in result]
                
                # Compare the results
                expected_data = expected_results[name]
                
                if compare_results(name, actual_records, expected_data):
                    print(f"✅ {name} verification passed.")
                else:
                    overall_success = False
                    print(f"❌ {name} verification failed. Check logs above.")
                    
    except Exception as e:
        print(f"❌ Error during KG verification: {e}")
        overall_success = False
        
    if overall_success:
        print("\n🎉 All data integrity checks passed! The KG is correctly constructed.")
    else:
        print("\n⚠️ One or more data integrity checks failed. Review the KG creation steps.")
        
    return overall_success

