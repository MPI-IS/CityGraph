

class Location:

    BAR = "bar"
    RESTAURANT = "restaurant"
    HOUSE = "house"
    OFFICE = "office"

    TYPES = [BAR,RESTAURANT,HOUSE,OFFICE]

    id_count = 0
    
    def __init__(self, location_type, coordinates):
        assert(location_type in self.types)
        self._location_type = location_type
        self._coordinates = coordinates
        self._location_id = self._get_id()
        
    @classmethod
    def _get_id(self):
        self.__class__.id_count+=1
        return self.__class__.id_count
