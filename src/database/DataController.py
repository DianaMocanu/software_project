import numpy as np

class DataController:

    def __init__(self):
        self.tags = []  # remember the corresponding tag/class(0,1) for each tuple from the learning dataset


    def _getSizeDatabase(self, database):
        switch = {
            'iris': 150,
            'htru': 17898,
            'air': 33176
        }
        return switch.get(database)


    def createLearningSets(self, query, negation, randomRate):
        '''
        Creates the learning dataset
        :param query:
        :param negation:
        :param randomRate:
        :return: The new learning dataset along with the positive tuples ids
        '''
        self.tags = []
        newQuery = query.constructQueryWithoutId()
        positive_result = np.array(newQuery.executeQuery())
        positive_length = len(positive_result)
        self.field_names = newQuery.field_names
        if negation == 1:
            negative_result = newQuery.negateQueryRandom(positive_length, randomRate,
                                                         self._getSizeDatabase(query.database))
        else:
            negative_result = newQuery.negateQueryCombinationsN(positive_length)

        positive_ids = query.executeQueryOnlyId()
        negative_length = len(negative_result)
        print("Negate size: " + str(negative_length) + " Positive size: " + str(positive_length))
        self.addToTag(positive_length, 0)
        self.addToTag(negative_length, 1)
        return np.concatenate((np.array(positive_result), np.array(negative_result))), positive_ids


    def addToTag(self, size, tag):
        '''
        add to the tags array a specific tag for size times
        :param size:
        :param tag:
        :return:
        '''
        tagsToAdd = [tag] * size
        self.tags += tagsToAdd

