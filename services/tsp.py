def nearest_neighbor_tsp(complaints):
    unvisited = complaints.copy()
    route = [unvisited.pop(0)] 
    
    while unvisited:
        current = route[-1]
        nearest = min(unvisited, key=lambda c: 
            (c['coords'][0] - current['coords'][0])**2 + 
            (c['coords'][1] - current['coords'][1])**2
        )
        route.append(nearest)
        unvisited.remove(nearest)
    
    return [c['id'] for c in route]