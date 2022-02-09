import csv

#This class is responsible for loading a predefined CSV file mapping
#to coordinates of the rooms on the provided floorplan.png file

class Floorplan:
    
    def __init__(self, data_path):
        self.rooms = []
        self.__read_csv(data_path)

    def __read_csv(self, data_path):
        with open(data_path, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                self.rooms.append(Room(row[0]))

    def get_room(self, name):
        room = next((x for x in self.rooms if str(x.name) == str(name)), None)
        return room  

class Room:

    def __init__(self, data) -> None:
        self.points = []
        self.__parse_data(data)

    def __parse_data(self, data):
        split = data.split(";")
        self.name = split[0]
        for i in range(1, len(split)):
            if split[i]:
                coordinate_split = split[i].split("-")
                for j in coordinate_split:
                    self.points.append(int(j))