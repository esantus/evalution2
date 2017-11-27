#-*- coding: utf8 -*-
#
# Author: Enrico Santus <esantus@mit.edu>
# Description: connector to create the database.
#
# The database is structured as follows:
#
#   TABLES
#   - Synset(entryID, contributor[CN])
#   - Lemma(id, wordID[CN], lemma, concreteness, termType, frequency[CO], reliability[BN])
#   - Synset2Lemma(lemma)
#   - Definition(id, definition)
#   - Source(id, source)
#   - Domain(id, domain, score)
#   - Sense[B](id, sense)
#
#   CONNECTIONS
#   - Synset2Lemma(lemmaID, lemmaID, synsetID)
#   - Definition2Synset(Synset2LemmaID, DefinitionID)



import sys, re, os
import pymysql.cursors
import pandas as pd
import pickle

reload(sys)
sys.setdefaultencoding('utf8')


# Creating the tables
EVALution_Table_Creation = [
                        # Table containing the Synset(id[BN,CN], contributor[CN])
                        "CREATE TABLE Synset(\
                            id INT NOT NULL,\
                            entryID VARCHAR(100),\
                            contributor VARCHAR(100),\
                            PRIMARY KEY (id)) DEFAULT CHARSET=utf8",

                        # Table containing Lemma(id, wordID[C], lemma, concreteness,
                        # termType, frequency[CO], reliability[B])
                        "CREATE TABLE Lemma(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            wordID VARCHAR(100),\
                            lemma VARCHAR(250),\
                            language VARCHAR(100),\
                            concreteness FLOAT,\
                            termType VARCHAR(50),\
                            frequency INT,\
                            reliability FLOAT,\
                            PRIMARY KEY (id)) DEFAULT CHARSET=utf8",

                        # Table containing Relation(id, relation, relatum, target,
                        # fk_Source, fk_Target)
                        "CREATE TABLE Relation(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            relation VARCHAR(100),\
                            weight FLOAT,\
                            relatumID INTEGER NOT NULL,\
                            targetID INTEGER NOT NULL,\
                            PRIMARY KEY (id),\
                            constraint fk_Relatum\
                                foreign key(relatumID) references Synset(id),\
                            constraint fk_Target\
                                foreign key(targetID) references Synset(id)) DEFAULT CHARSET=utf8",

                        # Table containing POS(id, POS[CO])
                        "CREATE TABLE POS(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            POS VARCHAR(100),\
                            PRIMARY KEY (id)) DEFAULT CHARSET=utf8",

                        # Table containing DEP(id, dep[CO])
                        "CREATE TABLE Dep(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            dep VARCHAR(100),\
                            PRIMARY KEY (id)) DEFAULT CHARSET=utf8",

                        # Table containing Inflection(id, inflection[CO], frequency[CO])
                        "CREATE TABLE Inflection(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            inflection VARCHAR(100),\
                            PRIMARY KEY (id)) DEFAULT CHARSET=utf8",

                        # Table containing Collocation(id, collocation[CO], frequency[CO])
                        "CREATE TABLE Collocation(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            collocation TEXT,\
                            frequency INTEGER,\
                            PRIMARY KEY (id)) DEFAULT CHARSET=utf8",

                        # Table containing Pattern(id, pattern[CO], frequency[CO])
                        "CREATE TABLE Pattern(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            pattern TEXT,\
                            PRIMARY KEY (id)) DEFAULT CHARSET=utf8",

                        # Table containing Source(id, source)
                        "CREATE TABLE Source(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            source VARCHAR(100),\
                            PRIMARY KEY (id)) DEFAULT CHARSET=utf8",

                        # Table contaoining Domain(id, domain, score[BN])
                        "CREATE TABLE Domain(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            domain VARCHAR(250),\
                            score FLOAT,\
                            PRIMARY KEY(id)) DEFAULT CHARSET=utf8",

                        # Table containing Definition(id, definition)
                        "CREATE TABLE Definition(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            definition VARCHAR(10000),\
                            PRIMARY KEY(id)) DEFAULT CHARSET=utf8",

                        # Table containing Sense(id, sense[B])
                        "CREATE TABLE Sense(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            sense VARCHAR(1000),\
                            PRIMARY KEY(id)) DEFAULT CHARSET=utf8",

                        # Table containing Image(id, image[B])
                        "CREATE TABLE Image(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            link VARCHAR(1000),\
                            PRIMARY KEY(id)) DEFAULT CHARSET=utf8",

                            # Table connecting Synset(id), Synset2Lemma, Lemma(id)
                        "CREATE TABLE Lemma2Synset(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            lemmaID INT NOT NULL,\
                            synsetID INT NOT NULL,\
                            definitionID INT,\
                            sourceID INT,\
                            PRIMARY KEY (id),\
                            constraint fk_Lemma2Synset_Lemma\
                                foreign key(lemmaID) references Lemma(id),\
                            constraint fk_Lemma2Synset_Synset\
                                foreign key(synsetID) references Synset(id),\
                            constraint fk_Lemma2Synset_Definition\
                                foreign key(definitionID) references Definition(id),\
                            constraint fk_Lemma2Source_Source\
                                foreign key(sourceID) references Source(id)) DEFAULT CHARSET=utf8",

                        # Table connecting Pattern(id), Relation(id)
                        "CREATE TABLE Pattern2Relation(\
                            id INTEGER NOT NULL,\
                            relationID INTEGER NOT NULL,\
                            patternID INTEGER NOT NULL,\
                            frequency INTEGER,\
                            PRIMARY KEY(id),\
                            constraint fk_Pattern2Relation__Pattern\
                                foreign key(patternID) references Pattern(id),\
                            constraint fk_Pattern2Relation_Relation\
                                foreign key(relationID) references Relation(id))\
                            DEFAULT CHARSET=utf8",

                        # Table connecting Domain(id), Synset(id)
                        "CREATE TABLE Domain2Synset(\
                            domainID INTEGER NOT NULL,\
                            synsetID INTEGER NOT NULL,\
                            PRIMARY KEY(domainID, synsetID),\
                            constraint fk_Domain2Synset_Domain\
                                foreign key(domainID) references Domain(id),\
                            constraint fk_Domain2Synset_Lemma\
                                foreign key(synsetID) references Synset(id)) DEFAULT CHARSET=utf8",

                        # Table connecting Sense(id), Synset(id) with MAIN Feature
                        "CREATE TABLE Sense2Synset(\
                            senseID INTEGER NOT NULL,\
                            synsetID INTEGER NOT NULL,\
                            main BIT,\
                            PRIMARY KEY(senseID, synsetID),\
                            constraint fk_Sense2Synset_Sense\
                                foreign key(senseID) references Sense(id),\
                            constraint fk_Sense2Synset_Synset\
                                foreign key(synsetID) references Synset(id)) DEFAULT CHARSET=utf8",

                        # Table connecting Image(id), Synset(id) with MAIN Feature
                        "CREATE TABLE Image2Synset(\
                            imageID INTEGER NOT NULL,\
                            synsetID INTEGER NOT NULL,\
                            main BIT,\
                            PRIMARY KEY(synsetID, imageID),\
                            constraint fk_Image2Synset_Image\
                                foreign key(imageID) references Image(id),\
                            constraint fk_Image2Synset_Synset\
                                foreign key(synsetID) references Synset(id)) DEFAULT CHARSET=utf8",

                        # Table connecting Domain(id), Source(id)
                        "CREATE TABLE Domain2Source(\
                            domainID INTEGER NOT NULL,\
                            sourceID INTEGER NOT NULL,\
                            PRIMARY KEY(domainID, sourceID),\
                            constraint fk_Domain2Source_Domain\
                                foreign key(domainID) references Domain(id),\
                            constraint fk_Domain2Source_Source\
                                foreign key(sourceID) references Source(id)) DEFAULT CHARSET=utf8",

                        # Table connecting Definition(id), Source(id)
                        "CREATE TABLE Definition2Source(\
                            definitionID INTEGER NOT NULL,\
                            sourceID INTEGER NOT NULL,\
                            PRIMARY KEY(definitionID, sourceID),\
                            constraint fk_Definition2Source_Definition\
                                foreign key(definitionID) references Definition(id),\
                            constraint fk_Definition2Source_Source\
                                foreign key(sourceID) references Source(id)) DEFAULT CHARSET=utf8",

                        # Table connecting POS(id), Synset2Lemma(id)
                        "CREATE TABLE POS2Lemma(\
                            POSID INTEGER NOT NULL,\
                            lemmaID INTEGER NOT NULL,\
                            frequency FLOAT,\
                            PRIMARY KEY(POSID, lemmaID),\
                            constraint fk_POS2Lemma_POS\
                                foreign key(POSID) references POS(id),\
                            constraint fk_POS2Lemma_Synset2Lemma\
                                foreign key(lemmaID) references Lemma(id)) DEFAULT CHARSET=utf8",

                        "CREATE TABLE Dep2Lemma(\
                            depID INTEGER NOT NULL,\
                            lemmaID INTEGER NOT NULL,\
                            frequency FLOAT,\
                            PRIMARY KEY(depID, lemmaID),\
                            constraint fk_Dep2Lemma_dep\
                                foreign key(depID) references Dep(id),\
                            constraint fk_Dep2Lemma_Lemma\
                                foreign key(lemmaID) references Lemma(id)) DEFAULT CHARSET=utf8",

                        "CREATE TABLE Inflection2Lemma(\
                            inflectionID INTEGER NOT NULL,\
                            lemmaID INTEGER NOT NULL,\
                            frequency FLOAT,\
                            PRIMARY KEY(inflectionID, lemmaID),\
                            constraint fk_Inflection2Lemma_Inflection\
                                foreign key(inflectionID) references Inflection(id),\
                            constraint fk_Inflection2Lemma_Lemma\
                                foreign key(lemmaID) references Lemma(id)) DEFAULT CHARSET=utf8",

                        # Table containing Normalization(id, FirstCap[CO], None[CO],
                        # All[CO], Others[CO], fk_Synset2Lemma)
                        "CREATE TABLE Normalization(\
                            id INTEGER NOT NULL AUTO_INCREMENT,\
                            First INTEGER,\
                            None INTEGER,\
                            Every INTEGER,\
                            Others INTEGER,\
                            lemmaID INTEGER,\
                            PRIMARY KEY (id),\
                            constraint fk_Normalization2Lemma_Lemma\
                                foreign key(lemmaID) references Lemma(id)) DEFAULT CHARSET=utf8"
                        ]





def create_EVALution_Tables(cursor):
    """
    create_EVALution_Tables takes a cursor as an argument and it creates all the tables
    and the relation tables (e.g. many-to-many tables) saved in two dictionaries
    that are initialized in the global environment. It prints errors if the
    tables already exist.

    Args:
        cursor (cursor): Database cursor
    Returns:
        Nothing. It creates the tables.
    """

    try:
        # Create the tables
        i = 0
        for item in EVALution_Table_Creation:
            try:
                print("Creating table %d..." % i)
                i += 1
                cursor.execute(item)
            except Exception as error:
                print(error)
                continue
    except Exception as error:
        print(error)




def save_column_values(filename):
    """
    Saves the single values for each column in a dictionary of sets, each of which
    with the particular column name. It is relevant to mention that "definition",
    "source" and "domain" columns are parsed, so that the content is separated
    by the source, which is instead added to "source"
    """

    prefixes = ["BN", "CN"]
    # Dictionary saving the set of values for each column
    fields = {}

    # Regular expression to split the source from content
    regex = re.compile("(IATE|TERMIUM)[\-0-9]+\)")

    for prefix in prefixes:

        with open(prefix + "_" + filename, "r") as f_in:

            print("Working on " + prefix + "_" + filename)

            header = True
            n_line = 0
            SOURCE = False
            for line in f_in:

                n_line += 1
                #print("Line %d" % n_line)

                items = line.strip().split("||")

                if header == True:
                    columns = [field.strip() for field in items]

                    for column in columns:
                        if column not in fields:
                            fields[column] = set()

                    print ("Now fields contain the following columns: " + ", ".join(fields.keys()))
                    print ("Currentl columns are: " + ", ".join(columns))
                    header = False

                else:
                    for i in range(0, len(columns)):

                        temp = []
                        sources = []

                        if columns[i] == "definition" or columns[i] == "domain" or columns[i] == "definitions" or columns[i] == "domains":

                            start = 0
                            for match in regex.finditer(items[i]):
                                temp.append((items[i][start:(match.start()-1)]).strip())
                                sources.append((items[i][match.start():(match.end()-1)]).strip())
                                SOURCE = True
                                start = match.end() + 1

                        elif columns[i] == "source" or columns[i] == "sources":

                            temp = [match.strip() for match in items[i].split(",")]

                        else:
                            temp.append(items[i].strip())

                        for element in temp:
                            fields[columns[i]].add(element.strip())

                        if SOURCE == True:
                            #print sources
                            for element in sources:
                                fields["source"].add(element)
                            SOURCE = False

    print("Let's print the set of values in separate files...")

    for column in fields:
        fields[column] = {i:value for i, value in enumerate(fields[column])}

    pickle.dump(fields, open(filename[:-4] + "_dict_of_single_values.pkl", "wb"))

    return fields




def turn_csv_into_matrix(fields, prefix, filename):
    """
    Given the file names, this function turns the CSV into a matrix of IDs, merging
    synsets from CN and BN into the same record.
    """

    # Dictionary saving the set of values for each column
    inv_fields = {}
    mapping = []

    for column in fields:
        inv_fields[column] = {value : i for i, value in fields[column].iteritems()}

    # Regular expression to split the source from content
    regex = re.compile("(IATE|TERMIUM)[\-0-9]+\)")

    if prefix != "":

        with open(prefix + "_" + filename, "r") as f_in:

            print("Working on " + prefix + "_" + filename)

            header = True
            n_line = 0
            for line in f_in:

                linea = []
                n_line += 1
                #print("Line %d" % n_line)

                items = line.strip().split("||")

                if header == True:
                    columns = items #[field.strip() for field in items]
                    linea.append(columns)
                    header = False
                else:
                    for i in range(0, len(columns)):
                        temp = []

                        if columns[i] == "definition" or columns[i] == "domain" or columns[i] == "definitions" or columns[i] == "domains":
                            start = 0
                            for match in regex.finditer(items[i]):
                                temp.append((inv_fields[columns[i]][items[i][start:(match.start()-1)].strip()],\
                                    inv_fields["source"][(items[i][match.start():(match.end()-1)]).strip()]))
                                start = match.end() + 1
                        elif columns[i] == "source" or columns[i] == "sources":
                            temp = [inv_fields[columns[i]][match.strip()] for match in items[i].split(",")]
                        else:
                            temp.append(inv_fields[columns[i]][items[i].strip()])

                        linea.append(temp)

                mapping.append(linea)

    pickle.dump(mapping, open(prefix + "_" + filename[:-4] + "_mapped.pkl", "wb"))

    return mapping



def print_mapping_to_text(mapping, fields):

    header = True
    for line in mapping:

        if header == True:
            columns = line[0]
            print ("||".join(columns))
            header = False
        else:
            print line
            if "source_entryID" in columns:
                #for i in range(0, len(columns)):
                print("||".join([fields[columns[i]][line[i][0]] for i in range(0, len(columns))]))
            else:
                string = ""
                for i in range(0, len(columns)):
                    for j in range(0, len(line[i])):
                        try:
                            if j == 0:
                                string = string + fields[columns[i]][line[i][j]]
                            else:
                                string = string + ", " + fields[columns[i]][line[i][j]]
                        except:
                            if j < (len(line[i])-1):
                                string = string + fields[columns[i]][line[i][j][0]] + " (" + fields["source"][line[i][j][1]] + "), "
                            else:
                                string = string + fields[columns[i]][line[i][j][0]] + " (" + fields["source"][line[i][j][1]] + ")"

                    if i < (len(columns)-1):
                        string = string + "||"

                print(string)




def print_mapping(mapping):
    for line in mapping:
        line = [str(item).strip("[]") for item in line]

        print "||".join(line)



def print_single_values(filename, fields):
    """
    Prints fields (dictionary of sets, for each column) in textual form, with a
    tab separated index, which can work as incremental ID
    """
    for column in fields:

        col_id = 0

        if column != "id":
            with open(filename[:-4] + "_" + column + ".txt", "w") as f_out:

                for item in fields[column]:
                    f_out.write(str(col_id) + "\t" + item + "\n")
                    col_id += 1




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
            print("Let's create the tables...")

            cursor = connection.cursor()
            create_EVALution_Tables(cursor)

        except Exception as error:
            # If the tables already exist, print an error
            print(error)

        # Dictionary of values
        if not os.path.isfile(sys.argv[1][:-4] + "_dict_of_single_values.pkl"):
            # This saves files, each of which the set of values
            print("Let's collect the set of values...")
            fields = save_column_values(sys.argv[1])
        else:
            print("Let's load the set of values...")
            fields = pickle.load(open(sys.argv[1][:-4] + "_dict_of_single_values.pkl", "rb"))

        print("Fields contains the following columns: " + ", ".join(fields.keys()))


        # Mapping
        if not os.path.isfile("BN_" + sys.argv[1][:-4] + "_mapped.pkl"):
            print("Let's build the mapping for BN...")
            BN_mapping = turn_csv_into_matrix(fields, "BN", sys.argv[1])
            print("These are the dimensions of mapping: ", len(BN_mapping), " x ", len(BN_mapping[0]))

        if not os.path.isfile("CN_" + sys.argv[1][:-4] + "_mapped.pkl"):
            print("Let's build the mapping for CN...")
            CN_mapping = turn_csv_into_matrix(fields, "CN", sys.argv[1])
            print("These are the dimensions of mapping: ", len(CN_mapping), " x ", len(CN_mapping[0]))

        if os.path.isfile("BN_" + sys.argv[1][:-4] + "_mapped.pkl") and os.path.isfile("CN_" + sys.argv[1][:-4] + "_mapped.pkl"):
            print("Let's load the mapping...")
            BN_mapping = pickle.load(open("BN_" + sys.argv[1][:-4] + "_mapped.pkl", "rb"))
            #CN_mapping = pickle.load(open("CN_" + sys.argv[1][:-4] + "_mapped.pkl", "rb"))

        #print_mapping(CN_mapping)
        print_mapping_to_text(BN_mapping, fields)


        # Relata can be duplicated in BN and CN, so we need to merge them.
        # The only way to do so is to verify if they share one or more sources.
        # Merging consists in adding one column to the fields dictionary
        # for which, given an entryID, returns the id of the BN entryID, if
        # this exists, or CN otherwise. This will allow to save all info related
        # to duplicates into the BN entryID

        #mapping = turn_matrix_into_csv(fields, sys.argv[1])

    except Exception as error:
        print(error)


main()
