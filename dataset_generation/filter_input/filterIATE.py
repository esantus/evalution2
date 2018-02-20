# -*- coding: utf-8 -*-
"""
Created in February 2018
@author: Dagmar Gromann 
file: Method to filter iate from the tbx file to represent as dataframe
"""

import pandas as pd

from xml.etree import ElementTree as ET

SEP = "\t"

INPUT = "path to TBX input"
OUTPUT = "path to tsv output file"

IATE_DOMAINS = "path to tsv file containing iate domains (id and name listing)"

outputFile = open(OUTPUT, 'w')

def filterIATEFromTBX():
	dictionary = {}
	reliabilityCode = 0
	iateDomains = pd.read_csv(IATE_DOMAINS, sep=SEP)

	outputFile.write("entryID"+"\t"+"domain"+"\t"+"term"+"\t"+"termLang"+"\t"+"termType"+"\t"+"reliability"+"\n")

	'''Flexible method to parse large xml files iteratively'''
	parser = ET.iterparse(INPUT)

	'''For loop to extract individual elements in each termEntry'''
	for event, element in parser:
		if element.tag == 'termEntry':
			entryID = ""
			entryID = element.get('id')
			
			for child in element:
				
				if child.tag == "descripGrp":
					domain = ""
					domainElement = child.find('descrip')
					domainID = domainElement.text

					for ident in domainID.split(", "):
						if ident != "00" and len(ident) > 0:
							try:
								domainName = iateDomains.loc[(iateDomains['domainID'] == int(ident))]
							except ValueError: 
								print(ident, domainID, domainID.split(", "), "invalid int with base ten")

							if not domain and not domainName.empty: 
								domain = domainName['name'].iloc[0]
							else:
								if not domainName.empty:
									domain = domain+", "+domainName['name'].iloc[0]
				
				if child.tag == "langSet":
					language = child.get('{http://www.w3.org/XML/1998/namespace}lang')
					
					for termTag in child:
						term = termTag.find('term').text
						termType = termTag.find('termNote').text
						reliabilityCode = termTag.find('descrip').text
						
						if entryID and language and term:
							outputFile.write(entryID+"\t"+domain+"\t"+term+"\t"+language+"\t"+termType+"\t"+reliabilityCode+"\n")
							term, termType = "", ""

	outputFile.close()

def main():
	filterIATEFromTBX()

if __name__ == '__main__':
	main()