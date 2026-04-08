import psycopg2
import requests
import time

conn = psycopg2.connect("dbname=Chokho_AI user=postgres password=postgres")
cur = conn.cursor()

regions = [
    (30.20, 30.45, 77.95, 78.20, 0.02),  # Dehradun
    (29.88, 30.05, 78.05, 78.25, 0.02),  # Haridwar
    (30.05, 30.20, 78.25, 78.35, 0.02),  # Rishikesh
]

headers = {"User-Agent": "Chokho/1.0"}

for (min_lat, max_lat, min_lon, max_lon, step) in regions:
    lat = min_lat
    while lat <= max_lat:
        lon = min_lon
        while lon <= max_lon:
            try:
                url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
                r = requests.get(url, headers=headers, timeout=5)
                data = r.json()
                
                address = data.get("address", {})
                name = (
                    address.get("suburb") or
                    address.get("neighbourhood") or
                    address.get("village") or
                    address.get("town") or
                    address.get("city_district") or
                    address.get("county") or
                    "Unknown"
                )
                
                cur.execute("""
                    INSERT INTO areas (latitude, longitude, location_name, geom)
                    VALUES (%s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))
                """, (lat, lon, name, lon, lat))
                conn.commit()
                print(f"{lat},{lon} → {name}")
                
            except Exception as e:
                print(f"Error at {lat},{lon}: {e}")
            
            time.sleep(1)
            lon = round(lon + step, 6)
        lat = round(lat + step, 6)

cur.close()
conn.close()
print("Seeding done!")