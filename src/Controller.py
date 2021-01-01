from src.database.DataController import DataController
from src.database.Query import Query
from src.LearningController import LearningController
from src.DecisionTreeAlgorithm import *

class Controller:
    def __init__(self):
        self.connectService = DataController()
        self.learning = LearningController()


    def getQueryAlternativeConditions(self, query, database, negation, rate):
        queryNew = Query(database, query)
        X, result, pos_ids = self.connectService.createLearningSets(queryNew, negation, rate)
        tags =np.array(self.connectService.tags)
        feature_names = self.connectService.field_names
        max_depth = 6
        results = self.learning.constructTree(feature_names, max_depth, X, tags)
        return(results, pos_ids)

    def executeQueryId(self, queryToExec, database):
        query = Query(database, queryToExec)
        return query.executeQueryOnlyId()



