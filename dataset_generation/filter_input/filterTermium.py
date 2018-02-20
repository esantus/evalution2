# -*- coding: utf-8 -*-
"""
Created in February 2018
@author: Dagmar Gromann 
file: Method to filter termium from the csv files to represent as dataframe
"""

import pandas as pd
import glob
 
INPUT = "path to CSV files"
OUTPUT = "path to output file that should then be loaded to a MySQL table"
outputFile = open("outputFile path and name", 'w')

def loadTermiumToDB():
	counter = 0
	outputFile.write("entryID"+"\t"+"domain"+"\t"+"term"+"\t"+"termLang"+"\t"+"termType"+"\t"+"definition"+"\n")
	for filename in glob.iglob(INPUT, recursive=True):
		print(filename)
		df = pd.read_csv(filename)

		for index, row in df.iterrows():
			counter += 1
			entryID = "TERMIUM"+str(counter)
			domain = row[0]
			writeToResults(row, entryID, 'TERM_EN', 'ABBREVIATION_EN', 'SYNONYMS_EN', domain, "en", 'TEXTUAL_SUPPORT_1_EN')
			writeToResults(row, entryID, 'TERME_FR', 'ABBREVIATION_FR', 'SYNONYMES_FR', domain, "fr", 'JUSTIFICATION_1_FR')
			writeToResults(row, entryID, 'TERME_TERM_ES', 'ABBR_ES', 'SYNO_ES', domain, "es", 'JUST_TEXTSUPP_1_ES')

	outputFile.close()

def writeToResults(row, entryID, term, abbreviation, synonyms, domain, language, definition): 
	
	if not pd.isnull(row[term]):
		outputFile.write(entryID+"\t"+domain+"\t"+str(row[term])+"\t"+language+"\t"+"fullForm"+"\t"+str(row[definition])+"\n")
		if not pd.isnull(row[abbreviation]):
			outputFile.write(entryID+"\t"+domain+"\t"+str(row[abbreviation])+"\t"+language+"\t"+"abbreviation"+"\t"+str(row[definition])+"\n")
		if not pd.isnull(row[synonyms]):
			for synonym in row[synonyms].split(";"):	
				outputFile.write(entryID+"\t"+domain+"\t"+synonym+"\t"+language+"\t"+"fullForm"+"\t"+str(row[definition])+"\n")
	
	#return results
				
def main():
	loadTermiumToDB()

if __name__ == '__main__':
	main()
