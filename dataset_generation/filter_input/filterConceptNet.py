# -*- coding: utf-8 -*-
"""
Created in February 2018
file: Method to filter conceptNet from the csv file to look for permitted relations and English only main 
entries and create a tsv file 
@author: Dagmar Gromann
"""

import pandas as pd
import json
import requests

WORKING_DIRECTORY = "main directory"
CONCEPTNET = "path to csv file"
OUTPUT = "path to output file"

SEP = "\t"
permittedRelations = ['IsA', 'Antonym', 'Synonym', 'HasProperty', 'HasA', 'PartOf', 'MemberOf', 'MadeOf', 'Entails']


"""Extremely slow thanks to API so alternative just to get it from id"""
def getLabel(cnid):
	obj1 = requests.get('http://api.conceptnet.io/'+cnid)
	if obj1 != None:
		obj1 = obj1.json()
	label = obj1['edges'][0]['start']['label']
	return label

def getLabelFromCNID(cnid):
	term = cnid[6:]
	if "/" in term: 
		term = term[:-2]
	term = term.replace("_", " ")
	#print(cnid, term)
	return term

def filterConceptNet():
	file = open(CONCEPTNET)

	cnids = list()
	terms = list()
	relations = list()
	relatumids = list()
	relatums = list()
	relatumLang = list()
	datasources = list()
	contributors = list()
	surfaceText = list()

	for elem in file:
		line = elem.split(SEP)
		relation = line[1].replace("/r/", "")
		if relation in permittedRelations and line[2][3:6] == "en/" and not line[2][6:7].isdigit():
			cnids.append(line[2])
			term = line[2][7:] 
			terms.append(getLabelFromCNID(line[2]))
			relations.append(relation)
			relatumids.append(line[3])
			relatums.append(getLabelFromCNID(line[3]))
			relatumLang.append(line[3][3:5])

			documentation = json.loads(line[4])
			datasources.append(documentation['dataset'])
			contributors.append(documentation['sources'][0]['contributor'])	
			if "surfaceText" in documentation:
				surfaceText.append(documentation['surfaceText'])
			else:
				surfaceText.append("nan")

	df = pd.DataFrame(dict(cnids=cnids, terms=terms, relations=relations, relatumids=relatumids, 
						relatums=relatums, relatumLang=relatumLang,
						datasources=datasources, contributors=contributors, surfaceText=surfaceText))

	df1 = df.drop_duplicates()

	df1.sort_values(by=['cnids', 'relations']).to_csv(OUTPUT, sep=SEP)

	file.close()

def main():
	filterConceptNet()

if __name__ == '__main__':
	main()