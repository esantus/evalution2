#-*- coding: utf8 -*-
#
# Author: Enrico Santus <esantus@mit.edu>
# Description: connector to create the database.
#
# The database is structured as follows:
#   - Lemma(lemma, lang) many-to-many Entry_LemmaID(concreteness, termType, reliability, source)
#   - Lemma one-to-many RelataCorpus(freq, pos_dep, inflections, normalization, 2_grams, 3_grams)
#   - Entry_LemmaID many-to-many Domain(score, source)
#   - Entry_LemmaID many-to-many Definition(source)
#   - Entry_LemmaID many-to-many MainSense/WordSense(source)
#   - Entry_LemmaID many-to-many Audio/MainImage/AllImages(source)
#   - Entry_LemmaID many-to-many Contributor(source)
#   - Entry_LemmaID many-to-many Relation(weight, relata_Entry_Lemma_ID, source)
#   - Relation one-to-many Pattern(weight, pattern, source)
#   - Source: many-to-many to all.

import sys, re
import pymysql.cursors
import pandas as pd

reload(sys)
sys.setdefaultencoding('utf8')


# Creating the tables
EVALution_creation_Tables = {  "CREATE_LEMMA_LANG" : "CREATE TABLE Lemma(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            lemma VARCHAR(250),\
                            lang VARCHAR(250),\
                            PRIMARY KEY (id)) DEFAULT CHARSET=utf8",

                        "CREATE_CORPUS" : "CREATE TABLE Corpus(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            freq NUMERIC,\
                            PosDep TEXT,\
                            inflections TEXT,\
                            normalization TEXT,\
                            PRIMARY KEY (id)) DEFAULT CHARSET=utf8",

                        "CREATE_COLLOCATION" : "CREATE TABLE Collocation(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            collocation TEXT,\
                            PRIMARY KEY (id)) DEFAULT CHARSET=utf8",

                        "CREATE_PATTERN" : "CREATE TABLE Pattern(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            pattern TEXT,\
                            PRIMARY KEY (id)) DEFAULT CHARSET=utf8",

                        "CREATE_SYNSET" : "CREATE TABLE Synset(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            entryID VARCHAR(100),\
                            wordID VARCHAR(100),\
                            concreteness FLOAT,\
                            reliability FLOAT,\
                            termType VARCHAR(50),\
                            contributor VARCHAR(100),\
                            PRIMARY KEY (id)) DEFAULT CHARSET=utf8",

                        "CREATE_RELATION" : "CREATE TABLE Relation(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            relation VARCHAR(100),\
                            PRIMARY KEY (id)) DEFAULT CHARSET=utf8",

                        "CREATE_SOURCE" : "CREATE TABLE Source(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            source VARCHAR(100),\
                            PRIMARY KEY (id)) DEFAULT CHARSET=utf8",

                        "CREATE_DOMAIN": "CREATE TABLE Domain(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            domain VARCHAR(250),\
                            score FLOAT,\
                            PRIMARY KEY(id)) DEFAULT CHARSET=utf8",

                        "CREATE_DEFINITION": "CREATE TABLE Definition(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            definition VARCHAR(10000),\
                            PRIMARY KEY(id)) DEFAULT CHARSET=utf8",

                        "CREATE_MAIN_SENSE": "CREATE TABLE Sense(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            sense VARCHAR(1000),\
                            main enum(\"True\", \"False\"),\
                            PRIMARY KEY(id)) DEFAULT CHARSET=utf8",

                        "CREATE_AUDIO": "CREATE TABLE Audio(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            link VARCHAR(1000),\
                            description VARCHAR(1000),\
                            PRIMARY KEY(id)) DEFAULT CHARSET=utf8",

                        "CREATE_MAIN_IMAGE": "CREATE TABLE Image(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            link VARCHAR(1000),\
                            description VARCHAR(1000),\
                            main enum(\"True\", \"False\"),\
                            PRIMARY KEY(id)) DEFAULT CHARSET=utf8"}



# Connecting the tables
EVALution_creation_Relations = {"CREATE_SYNSET2LEMMA" : "CREATE TABLE Synset2Lemma(\
                            lemmaID INTEGER NOT NULL,\
                            synsetID INTEGER NOT NULL,\
                            PRIMARY KEY(synsetID, lemmaID),\
                            constraint fk_S2L__Lemma\
                                foreign key(lemmaID) references Lemma(id),\
                            constraint fk_S2L__Synset\
                                foreign key(synsetID) references Synset(id))",

                            "CREATE_SYNSET2RELATUM" : "CREATE TABLE Synset2Synset(\
                                relatumID INTEGER NOT NULL,\
                                targetID INTEGER NOT NULL,\
                                relationID INTEGER NOT NULL,\
                                PRIMARY KEY(relatumID, targetID, relationID),\
                                constraint fk_L2R__Relatum\
                                    foreign key(relatumID) references Synset(id),\
                                constraint fk_L2R__Target\
                                    foreign key(targetID) references Synset(id),\
                                constraint fk_L2R__Relation\
                                    foreign key(relationID) references Relation(id))",

                            "CREATE_RELATION2PATTERN" : "CREATE TABLE Relation2Pattern(\
                                relationID INTEGER NOT NULL,\
                                patternID INTEGER NOT NULL,\
                                PRIMARY KEY(relationID, patternID),\
                                constraint fk_R2P__Relation\
                                    foreign key(relationID) references Relation(id),\
                                constraint fk_R2P__Pattern\
                                    foreign key(patternID) references Pattern(id))",

                            "CREATE_SYNSET2DOMAIN" : "CREATE TABLE Synset2Domain(\
                                domainID INTEGER NOT NULL,\
                                synsetID INTEGER NOT NULL,\
                                PRIMARY KEY(synsetID, domainID),\
                                constraint fk_S2Dom__Domain\
                                    foreign key(domainID) references Domain(id),\
                                constraint fk_S2Dom__Synset\
                                    foreign key(synsetID) references Synset(id))",

                            "CREATE_SYNSET2DEFINITION" : "CREATE TABLE Synset2Definition(\
                                definitionID INTEGER NOT NULL,\
                                synsetID INTEGER NOT NULL,\
                                PRIMARY KEY(synsetID, definitionID),\
                                constraint fk_S2Def__Definition\
                                    foreign key(definitionID) references Definition(id),\
                                constraint fk_S2Def__Synset\
                                    foreign key(synsetID) references Synset(id))",

                            "CREATE_SYNSET2SENSE" : "CREATE TABLE Synset2Sense(\
                                senseID INTEGER NOT NULL,\
                                synsetID INTEGER NOT NULL,\
                                PRIMARY KEY(synsetID, senseID),\
                                constraint fk_S2S__Sense\
                                    foreign key(senseID) references Sense(id),\
                                constraint fk_S2S__Synset\
                                    foreign key(synsetID) references Synset(id))",

                            "CREATE_SYNSET2AUDIO" : "CREATE TABLE Synset2Audio(\
                                audioID INTEGER NOT NULL,\
                                synsetID INTEGER NOT NULL,\
                                PRIMARY KEY(synsetID, audioID),\
                                constraint fk_S2A__Audio\
                                    foreign key(audioID) references Audio(id),\
                                constraint fk_S2A__Synset\
                                    foreign key(synsetID) references Synset(id))",

                            "CREATE_SYNSET2IMAGE" : "CREATE TABLE Synset2Image(\
                                imageID INTEGER NOT NULL,\
                                synsetID INTEGER NOT NULL,\
                                PRIMARY KEY(synsetID, imageID),\
                                constraint fk_S2I__Image\
                                    foreign key(imageID) references Image(id),\
                                constraint fk_S2I__Synset\
                                    foreign key(synsetID) references Synset(id))",

                            "CREATE_SYNSET2SOURCE" : "CREATE TABLE Synset2Source(\
                                sourceID INTEGER NOT NULL,\
                                synsetID INTEGER NOT NULL,\
                                PRIMARY KEY(synsetID, sourceID),\
                                constraint fk_S2Source__Source\
                                    foreign key(sourceID) references Source(id),\
                                constraint fk_S2Source__Synset\
                                    foreign key(synsetID) references Synset(id))",

                            "CREATE_LEMMA2SOURCE" : "CREATE TABLE Lemma2Source(\
                                sourceID INTEGER NOT NULL,\
                                lemmaID INTEGER NOT NULL,\
                                PRIMARY KEY(lemmaID, sourceID),\
                                constraint fk_L2Source__Source\
                                    foreign key(sourceID) references Source(id),\
                                constraint fk_L2Source__Lemma\
                                    foreign key(lemmaID) references Lemma(id))",

                            "CREATE_DOMAIN2SOURCE" : "CREATE TABLE Domain2Source(\
                                sourceID INTEGER NOT NULL,\
                                domainID INTEGER NOT NULL,\
                                PRIMARY KEY(domainID, sourceID),\
                                constraint fk_Dom2Source__Source\
                                    foreign key(sourceID) references Source(id),\
                                constraint fk_Dom2Source__Domain\
                                    foreign key(domainID) references Domain(id))",

                            "CREATE_LEMMA2SOURCE" : "CREATE TABLE Definition2Source(\
                                sourceID INTEGER NOT NULL,\
                                definitionID INTEGER NOT NULL,\
                                PRIMARY KEY(definitionID, sourceID),\
                                constraint fk_Def2Source__Source\
                                    foreign key(sourceID) references Source(id),\
                                constraint fk_Def2Source__Definition\
                                    foreign key(definitionID) references Definition(id))",

                            "CREATE_SENSE2SOURCE" : "CREATE TABLE Sense2Source(\
                                sourceID INTEGER NOT NULL,\
                                senseID INTEGER NOT NULL,\
                                PRIMARY KEY(senseID, sourceID),\
                                constraint fk_Sense2Source__Source\
                                    foreign key(sourceID) references Source(id),\
                                constraint fk_Sense2Source__Sense\
                                    foreign key(senseID) references Sense(id))"}





def create_EVALution(cursor):
    """
    create_EVALution takes a cursor as an argument and it creates all the tables
    and the relation tables (e.g. many-to-many tables) saved in two dictionaries
    that are initialized in the global environment. It prints errors if the
    tables already exist.

    Args:
        cursor (cursor): Database cursor
    Returns:
        nothing
    """

    try:
        # Create the tables
        for key in EVALution_creation_Tables:
            try:
                print("Creating Tables: ", key)
                cursor.execute(EVALution_creation_Tables[key])
            except Exception as error:
                print(error)
                continue

        # Create the relation many-to-many tables
        for key in EVALution_creation_Relations:
            try:
                print("Creating Relations: ", key)
                cursor.execute(EVALution_creation_Relations[key])
            except Exception as error:
                print(error)
                continue

    except Exception as error:
        print(error)




# To create the insertion for Source
def insert_source(field, cursor, column1, table):
    value1 = field[column1].replace("\"", "\\\"").replace("\'", "\\\'").encode('utf8')

    sources = value1.split(",")

    for source in sources:
        cursor.execute("INSERT INTO %s (%s)\
                    SELECT \"%s\"\
                    WHERE NOT EXISTS (SELECT * FROM %s\
                    WHERE %s = \"%s\")\
                    LIMIT 1" % (table, column1, source, table,
                    column1, source))




# To create the insertion for Definition and Domain (no score)
def insert_one(field, cursor, column1, table):
    value1 = field[column1].replace("\"", "\\\"").replace("\'", "\\\'").encode('utf8')

    definitions = []

    regex = re.compile("(IATE|TERMIUM)[\-0-9]+\)")

    start = 0
    for match in regex.finditer(value1):
        definitions.append(value1[start:match.end()])
        start = match.end() + 2

    for definition in definitions:
        cursor.execute("INSERT INTO %s (%s)\
                    SELECT \"%s\"\
                    WHERE NOT EXISTS (SELECT * FROM %s\
                    WHERE %s = \"%s\")\
                    LIMIT 1" % (table, column1, definition, table,
                    column1, definition))



# To create the insertion for Lemma, ...
def insert_two(field, column1, column2, table):
    value1 = field[column1].replace("\"", "\\\"").replace("\'", "\\\'").encode('utf8')
    value2 = field[column2].replace("\"", "\\\"").replace("\'", "\\\'").encode('utf8')

    return "INSERT INTO %s (%s, %s)\
                    SELECT \"%s\", \"%s\"\
                    WHERE NOT EXISTS (SELECT * FROM %s\
                    WHERE %s = \"%s\" AND %s = \"%s\")\
                    LIMIT 1" % (table, column1, column2, value1, value2, table,
                    column1, value1, column2, value2)



# To create the insertion to fill Synset
def insert_five(field, column1, column2, column3, column4, column5, table):
    value1 = field[column1].replace("\"", "\\\"").replace("\'", "\\\'").encode('utf8')
    value2 = field[column2].replace("\"", "\\\"").replace("\'", "\\\'").encode('utf8')
    value3 = field[column3].replace("\"", "\\\"").replace("\'", "\\\'").encode('utf8')
    value4 = field[column4].replace("\"", "\\\"").replace("\'", "\\\'").encode('utf8')
    value5 = field[column5].replace("\"", "\\\"").replace("\'", "\\\'").encode('utf8')

    value3 = ("NULL" if value3.lower() == "nan" else value3)
    #value4 = ("" if value4.lower() == "nan" else value4)
    #value5 = ("" if value5.lower() == "nan" else value5)

    return "INSERT INTO %s (%s, %s, %s, %s, %s)\
                    SELECT \"%s\", \"%s\", %s, \"%s\", \"%s\"\
                    WHERE NOT EXISTS (SELECT * FROM %s\
                    WHERE %s = \"%s\" AND %s = \"%s\")\
                    LIMIT 1" % (table, column1, column2, column3, column4, column5,
                    value1, value2, value3, value4, value5, table,
                    column1, value1, column2, value2)




# To connect Synset and Source
def connect_source(field, cursor, column1, table1, column2, column3, table2, column4, column5, table3):
    value1 = field[column1].replace("\"", "\\\"").replace("\'", "\\\'").encode('utf8')
    value2 = field[column2].replace("\"", "\\\"").replace("\'", "\\\'").encode('utf8')
    value3 = field[column3].replace("\"", "\\\"").replace("\'", "\\\'").encode('utf8')

    sources = value1.split(",")

    for source in sources:
        cursor.execute("INSERT INTO %s (%s, %s)\
	               (SELECT %s.id, %s.id FROM %s, %s\
                   WHERE %s.%s = \"%s\" and %s.%s = \"%s\" and %s.%s = \"%s\")" % (table3, column4,
                   column5, table1, table2, table1, table2, table1, column1,
                   source, table2, column2, value2, table2, column3, value3))




# To connect Synset and Domain/Definition
def connect_one(field, cursor, column1, table1, column2, column3, table2, column4, column5, table3):
    value1 = field[column1].replace("\"", "\\\"").replace("\'", "\\\'").encode('utf8')
    value2 = field[column2].replace("\"", "\\\"").replace("\'", "\\\'").encode('utf8')
    value3 = field[column3].replace("\"", "\\\"").replace("\'", "\\\'").encode('utf8')

    definitions = []

    regex = re.compile("(IATE|TERMIUM)[\-0-9]+\)")

    start = 0
    for match in regex.finditer(value1):
        definitions.append(value1[start:match.end()])
        start = match.end() + 2

    for definition in definitions:
        cursor.execute("INSERT INTO %s (%s, %s)\
	               (SELECT %s.id, %s.id FROM %s, %s\
                   WHERE %s.%s = \"%s\" and %s.%s = \"%s\" and %s.%s = \"%s\")" % (table3, column4,
                   column5, table1, table2, table1, table2, table1, column1,
                   definition, table2, column2, value2, table2, column3, value3))



# To connect Synset and Lemma
def connect_two(field, column1, column2, table1, column3, column4, table2, column5, column6, table3):
    value1 = field[column1].replace("\"", "\\\"").replace("\'", "\\\'").encode('utf8')
    value2 = field[column2].replace("\"", "\\\"").replace("\'", "\\\'").encode('utf8')
    value3 = field[column3].replace("\"", "\\\"").replace("\'", "\\\'").encode('utf8')
    value4 = field[column4].replace("\"", "\\\"").replace("\'", "\\\'").encode('utf8')

    return "INSERT INTO %s (%s, %s)\
	               (SELECT %s.id, %s.id FROM %s, %s\
                   WHERE %s.%s = \"%s\" and %s.%s = \"%s\"\
                    and %s.%s = \"%s\" and %s.%s = \"%s\")" % (table3, column5, column6, table1, table2,
                    table1, table2, table1, column1, value1, table1, column2, value2,
                    table2, column3, value3, table2, column4, value4)



def add_to_dataset(field, indices, cursor):

    try:
        print("Adding information for: ", unicode(field["entryID"]), unicode(field["lemma"]))

        # Adding Lemmas
        #cursor.execute(insert_two(field, "lemma", "lang", "Lemma"))

        # Adding IDs
        print("SYNSET")
        cursor.execute(insert_five(field, "entryID", "wordID", "concreteness",
        "termType", "contributor", "Synset"))

        # Connecting IDs and Lemmas
        print("CONNECTING SYSNSET")
        cursor.execute(connect_two(field, "lemma", "lang", "Lemma", "entryID",
        "wordID", "Synset", "lemmaID", "synsetID", "Synset2Lemma"))

        # Creating Definition
        #cursor.execute(
        print("\n\nDEFINITION")
        print(insert_one(field, cursor, "definition", "Definition"))

        # Creating Domain
        #cursor.execute(
        print("\n\nDOMAIN")
        print(insert_one(field, cursor, "domain", "Domain"))

        # Creating Source
        #cursor.execute(
        print("\n\nSOURCE")
        print(insert_source(field, cursor, "source", "Source"))

        # Connecting Definition to Synset2Audio
        print("\n\nCONNECT_DEFINITION")
        print(connect_one(field, cursor, "definition", "Definition", "entryID", "wordID", "Synset", "definitionID", "synsetID", "Synset2Definition"))

        print("\n\nCONNECT_DOMAIN")
        # Connecting Domain to Synset2Audio
        print(connect_one(field, cursor, "domain", "Domain", "entryID", "wordID", "Synset", "domainID", "synsetID", "Synset2Domain"))

        print("\n\nCONNECT_SOURCE")
        # Connecting Domain to Synset2Audio
        print(connect_source(field, cursor, "source", "Source", "entryID", "wordID", "Synset", "sourceID", "synsetID", "Synset2Source"))

    except Exception as error:
        print(error)



def main():
    """
    This function takes in input the file name of the file to be used to populate
    the dabase. It creates all tables and relation tables and then start the
    population.
    """

    if len(sys.argv) == 1:
        print("Please, make sure you insert the name of the file to use to "\
        "populate the database")
        return False

    # Connect to the Database
    try:
        connection = pymysql.connect(host='localhost',user='root',passwd='eval20',\
        db='EVALution2.0',charset='utf8',autocommit=True)

        # Create the tables in the Database
        try:
            cursor = connection.cursor()
            create_EVALution(cursor)
        except Exception as error:
            # If the tables already exist, print an error
            print(error)

        # Open the txt files, parse them and populate the tables
        try:

            # Dictionary with columns as keys, columns in a list for saving indices
            field = {}
            indices = []

            # Opening the input file
            with open(sys.argv[1], "r") as f_in:

                # All files have header, so the first row should be used for the keys
                header = True

                # For every line except the first, parse and add to the database
                for line in f_in:
                    columns = line.strip().split("||")

                    # First line: save the dictionary keys and the index.
                    if header == True:
                        for column in columns:
                            field[column] = ""
                            indices.append(column)
                        header = False
                    # Otherwise...
                    else:
                        # For every element in the index, update the dictionary
                        for i, index in enumerate(indices):
                            field[index] = columns[i]

                        # Add the columns to the dataset
                        add_to_dataset(field, indices, cursor)

        except Exception as error:
            print(error)

    except Exception as error:
        print(error)


main()
