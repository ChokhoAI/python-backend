DEPOT_COORDS = (30.3245, 78.0467)

def nearest_neighbor_tsp(complaints):
    unvisited = complaints.copy()
    
    nearest_to_depot = min(unvisited, key=lambda c: 
        (c['coords'][0] - DEPOT_COORDS[0])**2 + 
        (c['coords'][1] - DEPOT_COORDS[1])**2
    )
    unvisited.remove(nearest_to_depot)
    route = [nearest_to_depot]
    
    while unvisited:
        current = route[-1]
        nearest = min(unvisited, key=lambda c: 
            (c['coords'][0] - current['coords'][0])**2 + 
            (c['coords'][1] - current['coords'][1])**2
        )
        route.append(nearest)
        unvisited.remove(nearest)
    
    return [c['id'] for c in route]