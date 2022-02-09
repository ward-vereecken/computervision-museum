import numpy as np
import pandas as pd

from utils.string_utils import StringUtils

class Localisation:

    def __init__(self, db):
        self.db = db
        self.history = pd.DataFrame()
        self.floor_plan = pd.read_excel('resources/floorplan_matrix.xlsx')
        self.transition_matrix = self.__calculate_transition_matrix()

    def __calculate_transition_matrix(self): 
        RomanNumbers = ['rI','II','III','IV','V']

        #index bepalen: 0-18, 19 - 37 = A - S, 38-42 = I - V
        states = []
        for i in range(19):
            states.append(i+1)
        for i in StringUtils.letter_range("a", "t"):
            states.append(i)
        for i in range(len(RomanNumbers)):
            states.append(RomanNumbers[i])

        #Verschillende Dataframes vullen/ berekenen    
        room_scores = pd.DataFrame(0, index = states, columns = ['Score'])
        self.__count(room_scores)
        self.floor_plan.columns = states
        self.floor_plan.index = states
        self.__connect_rooms(room_scores,self.floor_plan)

        #schilderijloze zalen uit dataframes halen
        self.floor_plan = self.floor_plan.loc[:, (self.floor_plan != 0).any(axis=0)]
        self.floor_plan = self.floor_plan.loc[(self.floor_plan != 0).any(axis=1), :]
        zaalScores = room_scores.loc[(room_scores != 0).any(axis=1), :]

        #transitieMatrix bepalen
        trans = pd.DataFrame(index = self.floor_plan.index, columns = self.floor_plan.columns)
        self.__last_column(zaalScores,self.floor_plan, trans)
        trans = trans.fillna(0)
        return trans

    #in sommige zalen hangen geen schilderijen, maar men kan er wel doorwandelen naar een andere zaal
    #deze functie zorgt ervoor dat deze schilderijloze schilderij, twee zalen worden verbonden    
    def __connect_rooms(self, zaalScores,grondPlan):
        zero_rows = []
        for x in np.argwhere(zaalScores['Score'].to_numpy() == 0).ravel():
            zero_rows.append(x)
            k = grondPlan[grondPlan.columns[x]].to_numpy().nonzero()[0]
            for i in k:
                for j in k:
                    grondPlan.loc[grondPlan.index[i]][grondPlan.columns[j]]=1
        for x in zero_rows:
            grondPlan.at[grondPlan.index[x],:]=0
            grondPlan.at[:,grondPlan.columns[x]]=0

    #vult transitiematrix aan de hand van de zaalScores en het grondPlan
    def __last_column(self, zaalScores,grondPlan,trans): 
        for row in grondPlan.index:
            result = grondPlan.loc[row].to_numpy().nonzero()
            som = 0
            for x in result[0]:
                if x==row:
                    som += zaalScores.iloc[x]['Score']-1
                else:
                    som += (zaalScores.iloc[x]['Score'])
            for y in result[0]:
                if x==row:
                    trans.loc[row, trans.columns[y]] = (zaalScores.loc[zaalScores.index[y]]['Score']-1)/som
                else:
                    trans.loc[row, trans.columns[y]] = (zaalScores.loc[zaalScores.index[y]]['Score'])/som

    #berekent hoeveel schilderijen er in elke zaal hangen
    def __count(self, zaalScores):
        for object in self.db:
            index = object.room
            try:
                index = int(index)
            except ValueError:
                index = str(index)
            zaalScores.loc[index] += 1

    #berekent observatie
    def __calculate_observation(self, matches ,observaties):
        k = 10   
        for i in range(k):
            index = matches[i].image.room
            try:
                index = int(index)
            except ValueError:
                index = str(index)
            if observaties.loc[index].aantal==0:
                waarde = (matches[i].matches_count - matches[k+1].matches_count)/matches[0].matches_count
                observaties.loc[index]['aantal'] = waarde
    
    #kans berekenen 
    def find_location(self, matches): 
        if self.history.empty:
            self.history = pd.DataFrame([1/len(self.transition_matrix.index)]*len(self.transition_matrix))
        nieuwe = [0]*len(self.transition_matrix.index)
        laatste = self.history.iloc[:,-1]
        observaties = pd.DataFrame(0.0, index = self.transition_matrix.index, columns = ['aantal'])
        self.__calculate_observation(matches,observaties)
        som = 0
        for i in range(len(nieuwe)):
            for j in range(len(laatste)):
                nieuwe[i] += laatste[j]*self.transition_matrix.iat[j,i]
            nieuwe[i] *= observaties.iloc[i][0]
            som += nieuwe[i]
        nieuwe[:]=[x/som for x in nieuwe]

        return pd.DataFrame(nieuwe,index = self.transition_matrix.index, columns=["chance"])