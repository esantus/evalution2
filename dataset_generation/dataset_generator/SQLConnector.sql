DROP TABLE IF EXISTS word2Synset, word2Source, definition2SynsetLanguage, domain2Synset, sense2Synset, image2Synset, allWordSenses, synsetRelations;
DROP TABLE IF EXISTS source, word, language, wordType, image, wordPOS, sense, synsetDomain, definition, relationName, synsetID;

CREATE TABLE synsetID(
synset_id INT NOT NULL AUTO_INCREMENT,
synset_value VARCHAR(255) NOT NULL,
PRIMARY KEY(synset_id)
);

CREATE TABLE source(
source_id INT NOT NULL AUTO_INCREMENT,
source_value VARCHAR(255) NOT NULL,
UNIQUE KEY(source_value),
PRIMARY KEY(source_id)
);

CREATE TABLE word(
word_id INT NOT NULL AUTO_INCREMENT,
word_value VARCHAR(255) NOT NULL,
UNIQUE KEY(word_value),
PRIMARY KEY(word_id)
);

CREATE TABLE language( 
language_id INT NOT NULL AUTO_INCREMENT,
language_value VARCHAR(255) NOT NULL,
UNIQUE KEY(language_value),
PRIMARY KEY(language_id)
);

CREATE TABLE wordType(
wordType_id INT NOT NULL AUTO_INCREMENT,
wordType_value VARCHAR(255) NOT NULL,
UNIQUE KEY(wordType_value),
PRIMARY KEY(wordType_id)
);


CREATE TABLE image(
image_id INT NOT NULL AUTO_INCREMENT,
image_value LONGTEXT NOT NULL,
PRIMARY KEY(image_id)
);

CREATE TABLE wordPOS(
wordPOS_id INT NOT NULL AUTO_INCREMENT,
wordPOS_value VARCHAR(255) NOT NULL,
UNIQUE KEY(wordPOS_value),
PRIMARY KEY(wordPOS_id)
);

CREATE TABLE sense(
sense_id INT NOT NULL AUTO_INCREMENT,
sense_value VARCHAR(500) NOT NULL,
UNIQUE KEY(sense_value),
PRIMARY KEY(sense_id)
);

CREATE TABLE synsetDomain(
synsetDomain_id INT NOT NULL AUTO_INCREMENT,
synsetDomain_value VARCHAR(500) NOT NULL,
UNIQUE KEY(synsetDomain_value),
PRIMARY KEY(synsetDomain_id)
);

CREATE TABLE definition(
definition_id INT NOT NULL AUTO_INCREMENT,
definition_value LONGTEXT NOT NULL,
PRIMARY KEY(definition_id)
);

CREATE TABLE relationName(
relationName_id INT NOT NULL AUTO_INCREMENT,
relationName_value VARCHAR(255) NOT NULL,
UNIQUE KEY(relationName_value),
PRIMARY KEY(relationName_id)
);

CREATE TABLE word2Synset(
synset_id INT,
word_id INT,
wordSense_id INT,
mainWord BOOLEAN,
pos_id INT,
language_id INT,
wordType_id INT,
wordConcreteness DOUBLE,
PRIMARY KEY(synset_id, word_id, language_id),
FOREIGN KEY(synset_id) REFERENCES synsetID(synset_id),
FOREIGN KEY(word_id) REFERENCES word(word_id),
FOREIGN KEY(wordSense_id) REFERENCES sense(sense_id),
FOREIGN KEY(pos_id) REFERENCES wordPOS(wordPOS_id),
FOREIGN KEY(language_id) REFERENCES language(language_id),
FOREIGN KEY(wordType_id) REFERENCES wordType(wordType_id)
);

CREATE TABLE word2Source(
synset_id INT,
word_id INT,
language_id INT,
source_id INT,
PRIMARY KEY(synset_id, word_id, language_id, source_id),
FOREIGN KEY(synset_id) REFERENCES synsetID(synset_id),
FOREIGN KEY(word_id) REFERENCES word(word_id),
FOREIGN KEY(language_id) REFERENCES language(language_id),
FOREIGN KEY(source_id) REFERENCES source(source_id)
);

CREATE TABLE domain2Synset(
synset_id INT,
domain_id INT,
score DOUBLE,
source_id INT,
PRIMARY KEY(synset_id, domain_id),
FOREIGN KEY(synset_id) REFERENCES synsetID(synset_id),
FOREIGN KEY(domain_id) REFERENCES synsetDomain(synsetDomain_id),
FOREIGN KEY(source_id) REFERENCES source(source_id)
);

CREATE TABLE sense2Synset(
synset_id INT,
main BOOLEAN,
synsetSense_id INT,
PRIMARY KEY(synset_id, synsetSense_id),
FOREIGN KEY(synset_id) REFERENCES synsetID(synset_id),
FOREIGN KEY(synsetSense_id) REFERENCES sense(sense_id)
);

CREATE TABLE image2Synset(
synset_id INT,
image_id INT,
source_id INT,
main BOOLEAN,
PRIMARY KEY(synset_id, image_id),
FOREIGN KEY(synset_id) REFERENCES synsetID(synset_id),
FOREIGN KEY(image_id) REFERENCES image(image_id),
FOREIGN KEY(source_id) REFERENCES source(source_id)
);

CREATE TABLE definition2SynsetLanguage(
synset_id INT,
definition_id INT,
language_id INT,
source_id INT,
PRIMARY KEY(synset_id, definition_id, language_id),
FOREIGN KEY(synset_id) REFERENCES synsetID(synset_id),
FOREIGN KEY(definition_id) REFERENCES definition(definition_id),
FOREIGN KEY(language_id) REFERENCES language(language_id),
FOREIGN KEY(source_id) REFERENCES source(source_id)
);

CREATE TABLE allWordSenses(
word_id INT,
language_id INT,
wordSense_id INT,
source_id INT,
PRIMARY KEY(word_id, language_id, wordSense_id),
FOREIGN KEY(word_id) REFERENCES word(word_id),
FOREIGN KEY(wordSense_id) REFERENCES sense(sense_id),
FOREIGN KEY (language_id) REFERENCES language(language_id),
FOREIGN KEY(source_id) REFERENCES source(source_id)
);

CREATE TABLE synsetRelations(
sourceSynset_id INT,
relation_id INT,
targetSynset_id INT,
relationWeight DOUBLE,
relationSurfaceText VARCHAR(255),
PRIMARY KEY(sourceSynset_id, relation_id, targetSynset_id),
FOREIGN KEY(sourceSynset_id) REFERENCES synsetID(synset_id),
FOREIGN KEY(relation_id) REFERENCES relationName(relationName_id),
FOREIGN KEY(targetSynset_id) REFERENCES synsetID(synset_id)
);