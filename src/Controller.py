from src.database.DataController import DataController
from src.database.Query import Query
from src.decision_tree.LearningController import LearningController
import numpy as np

class Controller:
    def __init__(self):
        self.connectService = DataController()
        self.learning = LearningController()


    def getQueryAlternativeConditions(self, query, database, negation, rate):
        '''

        :param query: the initial query
        :param database: the database name on witch the query was written
        :param negation: 1/0 depending on the used negation algorithm,
        :param rate: the coefficient for the random negation
        :return: a list of the new conditions that reach a positive node along with the ids of the old positive tuples
        '''
        queryNew = Query(database, query)
        X, pos_ids = self.connectService.createLearningSets(queryNew, negation, rate)
        tags =np.array(self.connectService.tags)
        feature_names = self.connectService.field_names
        max_depth = 6
        results = self.learning.constructTree(feature_names, max_depth, X, tags)
        return(results, pos_ids)

    def executeQueryId(self, queryToExec, database):
        '''

        :param queryToExec:
        :param database:
        :return: the resulted tuples's ids of the executed query
        '''
        query = Query(database, queryToExec)
        return query.executeQueryOnlyId()



