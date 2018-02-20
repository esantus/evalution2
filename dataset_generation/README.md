# DatasetGeneration EVALution 2
This Java project allows for the comparison and merging of synsets from BabelNet with those of conceptNet, IATE, and Termium. It compares synsets first based on their English words and then based on their multilingual words only keeping those with more than 3 words in common across all languages. In order to use it, a local copy of BabelNet is needed or the interface to BabelNet needs to be changed to the API. Furthermore, the terminologies to compare BabelNet synsets with are required. 

<p>The folder "dataset_generator" contains the Java project for comparing and merging the synsets of the four resources listed below. The folder "filter_input" provides basic scripts to first transform the CSV/TBX input files into a format that can easily be imported to a MySQL or other database. The dataset_generator presumes the existence of a table for each resource other than BabelNet, where the BabelNet offline API is queried directly.</p>

## Datasets used in this project and their download path
* BabelNet offline API v3.7
* [IATE](http://iate.europa.eu/tbxPageDownload.do)
* [Termium](https://open.canada.ca/data/en/dataset/94fc74d6-9b9a-4c2e-9c6c-45a5092453aa)
* [conceptNet](https://github.com/commonsense/conceptnet5/wiki/Downloads)

## Dependencies
* babelnet-api-3.7.1.jar (and all related libraries)
* hibernate-core-5.2.6.Final.jar
* json-simple-1.1.jar
* slf4j-api.jar
* slf4j-simple.jar
* mysql-connector-java-5.1.45-bin.jar

## References 
If you use this code please cite the following paper: PAPER