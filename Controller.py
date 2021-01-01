from database.DataController import DataController
import numpy as np
from database.DatabaseMethods import DatabaseMethods
from database.Query import Query
from LearningController import LearningController
from DecisionTreeAlgorithm import *

class Controller:
    def __init__(self):
        self.connectService = DataController()
        self.databaseManager = DatabaseMethods()
        self.learning = LearningController()


    def getQueryAlternativeConditions(self, query, database, negation, rate):
        queryNew = Query(database, query)
        X, result, pos_ids = self.connectService.createLearningSets(queryNew, negation, rate)
        tags =np.array(self.connectService.tags)
        target_names = [0,1]
        feature_names = self.connectService.field_names
        clf = DecisionTreeAlgorithm(feature_names, X, 5)
        tree = clf.fit(X, tags)
        self.learning.parseTree(tree)
        print('cond', self.learning.conditions)
        # learningService = LearningController(X, tags, feature_names, target_names, np.array(result))
        # results = learningService.generateConditionsQuery()
        # return(results, pos_ids)


    def executeQuery(self, queryToExec, database):
        query = Query(database, queryToExec)
        return query.executeQuery(), query.field_names

    def executeQueryId(self, queryToExec, database):
        query = Query(database, queryToExec)
        return query.executeQueryOnlyId()

    def getTablesDatabase(self, database):
        return self.databaseManager.getTables(database)

    def getColumns(self, database, table):
        return self.databaseManager.getColumns(database, table)

    def getMinMaxColumn(self, database, table, column):
        return self.databaseManager.getMinMaxColumn(database, column, table)



