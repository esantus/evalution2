package dbReader;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.sql.PreparedStatement;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import configuration.Configuration;
import objects.ConceptNetEntry;
import objects.TermEntry;

public class LoadTermBase {	
	
	String DBname = Configuration.dataSource.getProperty("DBName");
	String usfDB = Configuration.dataSource.getProperty("usfDB");
	String conceptNetDB = Configuration.dataSource.getProperty("conceptNetDB");
	String taggerModel = Configuration.dataSource.getProperty("taggerModel");
	Connection conn; 
	
	String conceptNetFile = Configuration.dataSource.getProperty("conceptNet");
	HashMap<String, String> domainList = loadDomains();
	HashMap<String, Double> concretenessList = loadConcreteness();
	
	/**
	 * Establishes connection to the database
	 */
	public void openDataSource() {
		String dbConnection = "jdbc:mysql://localhost:3306/"+DBname+"?autoReconnect=true&useSSL=false&useUnicode=true&characterEncoding=utf8"; 
		try {
			conn = DriverManager.getConnection(dbConnection, "your user", "your password");

		} catch (SQLException e) {
			System.out.println("Some problem with opening the database");
			e.printStackTrace();
		}
	}
	
	/**
	 * Loads only the entryIDs of a terminological Database when given an table name of
	 * an SQL database
	 * @param dbName
	 * @return list of entry IDs
	 */
	public ArrayList<String> loadEntryIDs(String tableName){
		System.out.println("Loading a list of all entry identifiers...");
		ArrayList<String> entryIDs = new ArrayList<String>();
		Statement stmt;
		try {
			stmt = conn.createStatement();
			ResultSet rs = stmt.executeQuery("SELECT DISTINCT(entryID) FROM "+tableName);
			while (rs.next()){
				entryIDs.add(rs.getString("entryID"));
					
			}
		stmt.close();
		System.out.println("All entry ids in BabelNetExtraction "+entryIDs.size());
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return entryIDs;
	}
	
	/**
	 * Loads the set of identifier and domainName of IATE
	 * @return list of IATE domains
	 */
	public HashMap<String, String> loadDomains(){
		openDataSource();
		HashMap<String, String> domainList = new HashMap<String, String>();
		Statement stmt1;
		ResultSet rs;
		try {
			stmt1 = conn.createStatement();
			rs = stmt1.executeQuery("SELECT * FROM iate_domain");
			while(rs.next()){
				domainList.put(rs.getString("domain_id"), rs.getString("name"));
			}
			rs.close();
			stmt1.close();
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return domainList;
	}

	/**
	 * Auxiliary method to map the domainID to the domainName in the IATE DB 
	 * @param domainID: id associated with domain in IATE
	 * @return domainName: returns the domain name that is associated with this IATE domain id
	 */
	public String getIateDomainName(String domainID){
		String domainName = "";
		if (domainID.contains(",")){
			String[] domains = domainID.split(", ");
			for (String item: domains){
					if(!item.equals("00")){ 
						if(item.length() > 0 && item.substring(0,1).equals("0")){
							item = item.substring(1,item.length());
						}
						if(domainList.containsKey(item)){ 
							if(domainName.isEmpty()){ domainName = domainList.get(item);}
							else{ domainID += ", "+domainList.get(item);}
						}
					}
				}
		}
		else{
			domainName =  domainList.get(domainID);
		}
		return domainName;
	}
	
	/**
	 * This method allows to load one specific IATE entry when given the table name of the SQL table 
	 * and the entry ID
	 * @param entryID
	 * @param tableName
	 * @return IateEntry
	 */
	public HashMap<String, Set<String>> getEntryTerms(String entryID, String tableName){
		System.out.println("Loading entry "+entryID+" from IATE");
		Statement stmt2;
		Set<String> terms = new HashSet<String>();
		HashMap<String,Set<String>> entryTerms = new HashMap<String, Set<String>>();
		try {
			stmt2 = conn.createStatement();
			ResultSet rs = stmt2.executeQuery("SELECT term, termLang FROM "+ tableName +" WHERE entryID=\""+entryID+"\"");	
			while (rs.next()){
				if (!entryTerms.containsKey(rs.getString("termLang"))){
					terms = new HashSet<String>();
					terms.add(rs.getString("term"));
					entryTerms.put(rs.getString("termLang"), terms);
				}
				else{
					terms = entryTerms.get(rs.getString("termLang"));
					terms.add(rs.getString("term"));
					entryTerms.put(rs.getString("termLang"), terms);
				}
			}
		rs.close();
		stmt2.close();
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return entryTerms;
	}

	/**
	 * Retrieves all English words from a given table in the DB where the terminological resource and conceptNet are stored
	 * @param tableName: name of the DB table of the terminological resource from the properties file
	 * @return terms: returns a list of English words from the terminological resource
	 */
	public HashMap<String, Set<String>> getEnglishRows(String tableName){
		System.out.println("Loading English words of "+tableName+" ...");
		HashMap<String, Set<String>> terms = new HashMap<String, Set<String>>();
		Statement stmt4;
		try {
			stmt4 = conn.createStatement();
			ResultSet rs = stmt4.executeQuery("SELECT entryID, term FROM "+tableName +" WHERE termLang = \"en\"");
			while (rs.next()){
				//check whether term is not shorter than three (e.g. "Z" or "M"), does not contain any numbers (e.g. 4G), and does not only consist of capital letters (e.g. CAT as the tool)
				//those terms need to be matched based on their synonyms because considerably too ambiguous otherwise
				int countuc = rs.getString("term").split("(?=[A-Z])").length;
				if(rs.getString("term").length() > 2 && !rs.getString("term").matches(".*\\d+.*") && countuc != rs.getString("term").length()){
					//BabelNet Java API ignores case when queried for term. Thus the most efficient 
					// way to compare all entries is by means of term ignoring case
					if (terms.containsKey(rs.getString("term").toLowerCase())){
						Set<String> ids = terms.get(rs.getString("term").toLowerCase());
						ids.add(rs.getString("entryID"));
						terms.put(rs.getString("term").toLowerCase(), ids);
					}
					else{
						Set<String> ids = new HashSet<String>();
						ids.add(rs.getString("entryID"));
						terms.put(rs.getString("term").toLowerCase(), ids);
					}
				}
			}
		rs.close();
		stmt4.close();
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return terms;	
	}
	
	
//	public HashMap<String, List<String>> getBNIATEOverlap(){
//		HashMap<String, List<String>> results = new HashMap<String, List<String>>();
//		List<String> iateIds = new ArrayList<String>();
//		Statement stmtOverl;
//		String query = "SELECT bnID, iateID FROM overlapBNIATE";
//		
//		ResultSet rsOverl;
//		try {
//			stmtOverl = conn.createStatement();
//			rsOverl = stmtOverl.executeQuery(query);
//			while(rsOverl.next()){
//				if (!results.containsKey(rsOverl.getString("bnID"))){
//					iateIds.add(rsOverl.getString("iateID"));
//					results.put(rsOverl.getString("bnID"), iateIds);
//				}
//				else{
//					iateIds = results.get(rsOverl.getString("bnID"));
//					iateIds.add(rsOverl.getString("iateID"));
//					results.put(rsOverl.getString("bnID"), iateIds);
//				}	
//			}
//			
//			rsOverl.close();
//			conn.close();
//		} catch (SQLException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//	return results;
//	}
	
//	public Set<String> loadResultRelatums(){
//		Set<String> entryIDs = new HashSet<String>();
//		Statement stmt5;
//		try {
//			stmt5 = conn.createStatement();
//			ResultSet rs = stmt5.executeQuery("SELECT entryID FROM babelNetExtraction WHERE lang=\"en\"");
//			while (rs.next()){
//				if (!entryIDs.contains(rs.getString("entryID"))){
//					entryIDs.add(rs.getString("entryID"));
//				}
//			}
//		stmt5.close();
//		} catch (SQLException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//		return entryIDs;
//	}
	
//	public void replaceDomainIDsIate(String iateTable){
//		try {
//			Statement stmtDomains = conn.createStatement();
//			ResultSet rsDomain = stmtDomains.executeQuery("SELECT entryID, domain FROM iateAll");
//			while (rsDomain.next()){
//				
//			}
//		}catch (SQLException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//		
//	}
	
	/**
	 * Retrieves all synsets from a terminological resource in the DB based on a list of IDs
	 * @param idList: list of terminological synset IDs
	 * @param tableName: name of the DB table of the terminological resource obtained from properties file
	 * @return entries: returns all the synsets obtained from the terminological DB
	 */
	public HashMap<String, Set<TermEntry>> getEntriesFromList(Set<String> idList, String tableName){
		HashMap<String, Set<TermEntry>> entries = new HashMap<String, Set<TermEntry>>();
		ResultSet rs;
		StringBuilder builder = new StringBuilder();
		for( int i = 0 ; i < idList.size(); i++ ) {
		    builder.append("?,");
		}
		
		String sql = "SELECT * FROM "+tableName+" WHERE entryID IN ("+builder.deleteCharAt( builder.length() -1).toString()+")";
		try {
			PreparedStatement stm = conn.prepareStatement(sql.toString());
			
			int i = 1;
			for (String id : idList){
				stm.setString(i++, id);
			}
			
			rs = stm.executeQuery();
					
			while (rs.next()){
				TermEntry entry = new TermEntry();
				entry.setEntryID(rs.getString("entryID"));
				if(tableName.equals("termium")){
					if(!rs.getString("definition").equals("nan")){
						entry.setDefinition(rs.getString("definition"));
					}
				}
				if (tableName.equals("iateAll")){ entry.setDomain(getIateDomainName(rs.getString("domain"))); }
				else{ entry.setDomain(rs.getString("domain"));}
				entry.setLanguage(rs.getString("termLang"));
				if(tableName.equals("iateAll")){
					entry.setReliability(rs.getString("reliability"));
				}
				entry.setSource(rs.getString("entryID"));
				entry.setTerm(rs.getString("term").replace("\"", ""));
				entry.setTermType(rs.getString("termType"));
				
				if(entries.containsKey(rs.getString("entryID"))){
					Set<TermEntry> helper = entries.get(rs.getString("entryID"));
					helper.add(entry);
					entries.put(rs.getString("entryID"), helper);
				}
				else{
					Set<TermEntry> helper = new HashSet<TermEntry>();
					helper.add(entry);
					entries.put(rs.getString("entryID"), helper);
					
				}
				
			}
			rs.close();
			stm.close();
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return entries;
	}
	
	/**
	 * Load a list of words and their associated concreteness from the USF DB
	 * @return concreteness: hashmap of words and their associated concreteness values
	 */
	public HashMap<String, Double> loadConcreteness(){
		HashMap<String, Double> concreteness = new HashMap<String, Double>();
		try {
			Statement stmtConcrete = conn.createStatement();
			ResultSet rsConcrete = stmtConcrete.executeQuery("SELECT Cues, Targets, Cue_Concreteness_1_to_7, Target_Concreteness_1_to_7 FROM "+usfDB);
			while(rsConcrete.next()){
				if (rsConcrete.getString("Cues") != null && rsConcrete.getDouble("Cue_Concreteness_1_to_7") > 0.0 ){concreteness.put(rsConcrete.getString("Cues").toLowerCase(), rsConcrete.getDouble("Cue_Concreteness_1_to_7"));}
				if(rsConcrete.getString("Targets") != null && rsConcrete.getDouble("Target_Concreteness_1_to_7") > 0.0 ){concreteness.put(rsConcrete.getString("Targets"), rsConcrete.getDouble("Target_Concreteness_1_to_7"));}
			}
			rsConcrete.close();
			stmtConcrete.close();
		} catch (SQLException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		return concreteness;
	}
	
	/**
	 * Auxiliary method to query the previously generated concreteness list from the USF
	 * @param term: term for which the concreteness should be obtained
	 * @return concreteness: returns the concreteness value of this input term
	 */
	public Double getConcreteness(String term){
		Double concretes = null;
		
		//There are no abbreviations in USF so they need to be filtered out since 
		//abbreviations like CAT would retrieve a concreteness for cat
		int countuc = term.split("(?=[A-Z])").length;
		if (countuc != term.length()){
			if (concretenessList.containsKey(term.toLowerCase())){ concretes = concretenessList.get(term.toLowerCase()); }
		}
		return concretes;
	}
	
	/**
	 * Obtain the English word list of coneptNet
	 * @return words: all English words contained in conceptNet
	 */
	public HashMap<String, Set<String>> getConceptNetEnglishWordList(){
		System.out.println("Loading all English words of conceptNet...");
		HashMap<String, Set<String>> words = new HashMap<String, Set<String>>();		
		try{
			Statement stmtCNWords = conn.createStatement();
			ResultSet rsCNWords = stmtCNWords.executeQuery("SELECT source, target FROM "+conceptNetDB+" WHERE source LIKE \'%/c/en/%\' OR target LIKE \'%/c/en/%\'");
			while(rsCNWords.next()){
				String source = rsCNWords.getString("source");
				String target = rsCNWords.getString("target");
				if (source.contains("/c/en/")){
					if (words.containsKey(getCNLabel(source))){
						words.get(getCNLabel(source)).add(source);
					}
					else{	
						Set<String> ids = new HashSet<String>();
						ids.add(source);
						words.put(getCNLabel(source), ids);
					}
				}
				if(target.contains("/c/en/")){
					if (words.containsKey(getCNLabel(target))){
						words.get(getCNLabel(target)).add(target);
					}
					else{	
						Set<String> ids = new HashSet<String>();
						ids.add(target);
						words.put(getCNLabel(target), ids);
					}
				}
			}
			rsCNWords.close();
			stmtCNWords.close();
		} catch (SQLException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		return words;
	}
	
	/**
	 * Creates a synset for conceptNet based on the synonym relation for a specific conceptNet synset
	 * @param idList: conceptNet synset IDs
	 * @return entries: returns a set of conceptNet synsets created based on the synonym relation in conceptNet
	 */
	public HashMap<String, Set<TermEntry>> getCNSynonyms(Set<String> idList){
		HashMap<String, Set<TermEntry>> cnEntries = new HashMap<String, Set<TermEntry>>();
		ResultSet rsCN1;
		StringBuilder builderCN = new StringBuilder();
		for( int i = 0 ; i < idList.size(); i++ ) {
		    builderCN.append("?,");
		}
		
		StringBuilder builderCN1 = new StringBuilder();
		for( int i = 0 ; i < idList.size(); i++ ) {
		    builderCN1.append("?,");
		}
		
		String sqlCN = "SELECT source, target FROM "+conceptNetDB+" WHERE (source IN ("+builderCN.deleteCharAt( builderCN.length() -1).toString()+") OR target IN ("+builderCN1.deleteCharAt( builderCN1.length() -1).toString()+")) AND relation=\"/r/Synonym\"";
		try {
			PreparedStatement stmCN = conn.prepareStatement(sqlCN.toString());
			
			int i = 1;
			int j = idList.size()+1;
			for (String id : idList){
				stmCN.setString(i++, id);
				stmCN.setString(j++, id);
			}
			
			rsCN1 = stmCN.executeQuery();
			
			while (rsCN1.next()){
				TermEntry entry = new TermEntry();
				String source; 
				String target;
				String originalID;
				if (idList.contains(rsCN1.getString("source"))){
					source = rsCN1.getString("source");
					target = rsCN1.getString("target");
					originalID = rsCN1.getString("source");
				} else{
					target = rsCN1.getString("source");
					source = rsCN1.getString("target");
					originalID = rsCN1.getString("target");
				}
				if(!source.equals(target)){
					if(getPOS(source).isEmpty()){ source = checkPOS(source, target, idList); }
					entry.setEntryID(source);
					entry.setTerm(getCNLabel(target));
					entry.setSource(target);
					entry.setOriginalEntryID(originalID);
					entry.setLanguage(getCNLanguage(target));
					if (cnEntries.containsKey(source)){
						if (!cnEntries.get(source).contains(entry)){
							cnEntries.get(source).add(entry);
						}
					}
					else{
						Set<TermEntry> terms = new HashSet<TermEntry>();
						terms.add(entry);
						cnEntries.put(source, terms);
						entry = new TermEntry();
						entry.setEntryID(source);
						entry.setLanguage(getCNLanguage(source));
						entry.setSource(originalID);
						entry.setOriginalEntryID(originalID);
						entry.setTerm(getCNLabel(source));
						cnEntries.get(source).add(entry);
						}
					}
			}
			rsCN1.close();	
			stmCN.close();
		} catch (SQLException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		return cnEntries;	
	}
	
	/**
	 * Retrieves all conceptNet synonyms based on their synonym relation and returns at least the input conceptNet id and associated
	 * words as result
	 * @param idList: list of IDs from conceptNet
	 * @return entries: synsets created based on the conceptNet synset relation for a specific cnid
	 */
	public HashMap<String, Set<TermEntry>> getCNSynonymsPlus(Set<String> idList){
		HashMap<String, Set<TermEntry>> cnEntries = new HashMap<String, Set<TermEntry>>();
		Set<String> newIDList = new HashSet<String>();
		
		ResultSet rsCN;
		for (String tempID : idList){
			if (getPOS(tempID).isEmpty()){
				newIDList.add(tempID+"\n");
			}
			else{
				newIDList.add(tempID);
			}
		}
		StringBuilder builderCN = new StringBuilder();
		for( int i = 0 ; i < newIDList.size(); i++ ) {
		    builderCN.append("?,");
		}
		
		StringBuilder builderCN1 = new StringBuilder();
		for( int i = 0 ; i < newIDList.size(); i++ ) {
		    builderCN1.append("?,");
		}
		
		String sqlCN = "SELECT source, target FROM "+conceptNetDB+" WHERE (source IN ("+builderCN.deleteCharAt( builderCN.length() -1).toString()+") OR target IN ("+builderCN1.deleteCharAt( builderCN1.length() -1).toString()+")) AND relation=\"/r/Synonym\"";
		try {
			PreparedStatement stmCN = conn.prepareStatement(sqlCN.toString());
			
			int i = 1;
			int j = newIDList.size()+1;
			for (String id : newIDList){
				stmCN.setString(i++, id);
				stmCN.setString(j++, id);
			}
			
			rsCN = stmCN.executeQuery();
			
			while (rsCN.next()){
				TermEntry entry = new TermEntry();
				String source; 
				String target;
				String originalID;
				if (idList.contains(rsCN.getString("source"))){
					source = rsCN.getString("source");
					target = rsCN.getString("target");
					originalID = rsCN.getString("source");
				} else{
					target = rsCN.getString("source");
					source = rsCN.getString("target");
					originalID = rsCN.getString("target");
				}
				if(!source.equals(target)){
					if (!idList.contains(source) && idList.contains(source.substring(0, source.length()-2))){ source = source.substring(0, source.length()-2);}
					entry.setEntryID(source);
					entry.setOriginalEntryID(originalID);
					entry.setTerm(getCNLabel(target));
					entry.setSource(target);
					entry.setLanguage(getCNLanguage(target));
					if (cnEntries.containsKey(source)){
						if (!cnEntries.get(source).contains(entry)){
							cnEntries.get(source).add(entry);
						}
					}
					else{
						Set<TermEntry> terms = new HashSet<TermEntry>();
						terms.add(entry);
						cnEntries.put(source, terms);
						entry = new TermEntry();
						entry.setEntryID(source);
						entry.setOriginalEntryID(originalID);
						entry.setLanguage(getCNLanguage(source));
						entry.setTerm(getCNLabel(source));
						entry.setSource(source);
						cnEntries.get(source).add(entry);
						}
					}
			}
			rsCN.close();	
			stmCN.close();	
		} catch (SQLException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		for (String cnid : idList) {
			if (!cnEntries.containsKey(cnid)) {
				TermEntry ent = new TermEntry();
				ent.setOriginalEntryID(cnid);
				ent.setEntryID(cnid);
				ent.setLanguage(getCNLanguage(cnid));
				ent.setTerm(getCNLabel(cnid));
				ent.setOriginalEntryID(cnid);
				ent.setSource(cnid);
				if (!cnEntries.containsKey(cnid)) {
					Set<TermEntry> entries = new HashSet<TermEntry>();
					entries.add(ent);
					cnEntries.put(cnid, entries);
				}
				else {
					cnEntries.get(cnid).add(ent);
				}
			}
		}
		return cnEntries;
		
	}

	/**
	 * Get all the relations of a specific ConceptNet id
	 * @param cnid: input ID of ConceptNet
	 * @param unwantedRelations: list of relations we wish to exclude from the final DB
	 * @return relations: returns the source and target IDs and their relation as well as the relation weight and surfaceText
	 */
	public Set<ConceptNetEntry> getCNRelations(Set<String> cnid, Set<String> unwantedRelations){
		Set<ConceptNetEntry> relations = new HashSet<ConceptNetEntry>();
		JSONParser parser = new JSONParser();
		
		StringBuilder builderRel = new StringBuilder();
		for( int i = 0 ; i < cnid.size(); i++ ) {
			builderRel.append("?,");
		}
		
		ResultSet rsRels;
		
		try {
			String sqlQuery = "SELECT source, target, relation, metadata FROM "+conceptNetDB+" WHERE source IN ("+builderRel.deleteCharAt( builderRel.length() -1).toString()+") AND relation NOT IN (\"/r/ExternalURL\", \"/r/RelatedTo\", \"/r/Synonym\", \"/r/FormOf\", \"/r/EtymologicallyRelatedTo\", \"/r/DerivedFrom\", \"/r/HasContext\") AND target LIKE \"%/c/en/%\"";
			PreparedStatement stmRel = conn.prepareStatement(sqlQuery.toString());
			
			int i = 1;
			for (String id : cnid){
				stmRel.setString(i++, id);
			}
			
			rsRels = stmRel.executeQuery();
			
			while(rsRels.next()) {
				if(!unwantedRelations.contains(rsRels.getString("relation"))){
					String source = "";
					String target = "";
					if(getPOS(rsRels.getString("source")).isEmpty()) {source = checkPOS(rsRels.getString("source"), rsRels.getString("target"), cnid);}
					else { source = rsRels.getString("source");}
					if(getPOS(rsRels.getString("target")).isEmpty()) { target = checkPOS(rsRels.getString("target"), rsRels.getString("source"), cnid); }
					else { target = rsRels.getString("target");}	
					ConceptNetEntry rel = new ConceptNetEntry();
					rel.setSource(source);
					rel.setRelation(rsRels.getString("relation").replace("/r/", ""));
					rel.setTarget(target);
					JSONObject jsObject = (JSONObject) parser.parse(rsRels.getString("metadata"));
					rel.setWeight((Double) jsObject.get("weight"));
					rel.setSurfaceText((String) jsObject.get("surfaceText"));
					relations.add(rel);
				}
			}
			stmRel.close();
		} catch (SQLException | ParseException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return relations;
	}
	
	/**
	 * Auxiliary method to return only the language and words of a specific synset from a terminological synset
	 * to facilitate the multilingual comparison of sysnets
	 * @param entries: individual rows from the DB that contain the ID, word, and metadata 
	 * @return termsByLanguage: returns a hash map with all languages and their associated words for a given synset ID
	 */
	public HashMap<String, Set<String>> getTermsByLanguage(Set<TermEntry> entries){
		HashMap<String, Set<String>> termsByLanguage = new HashMap<String, Set<String>>();
		for (TermEntry entry : entries){
			int countuc = entry.getTerm().split("(?=[A-Z])").length;
			if(entry.getTerm().length() > 2 && !entry.getTerm().matches(".*\\d+.*") && countuc != entry.getTerm().length()){
				if (termsByLanguage.containsKey(entry.getLanguage())){ termsByLanguage.get(entry.getLanguage()).add(entry.getTerm().toLowerCase()); }
				else{
					Set<String> terms = new HashSet<String>(Arrays.asList(entry.getTerm().toLowerCase()));
					termsByLanguage.put(entry.getLanguage(), terms);
				}
			}
		}
		return termsByLanguage;
	}

	/**
	 * Auxiliary method to retrieve the word from the conceptNet id
	 * @param cnid: input id from conceptNet
	 * @return label: returns the word extracted from the input id
	 */
	public String getCNLabel(String cnid){
		return cnid.split("/")[3].replace("_", " ");
	}
	
	/**
	 * Extracts the language of a word from the input conceptNet id
	 * @param cnid: input id from conceptNet
	 * @return language: language extracted from the input id
	 */
	public String getCNLanguage(String cnid){
		if (cnid.split("/").length<2){
			System.out.println("Here language "+cnid);
		}
		return cnid.split("/")[2].replace("_", " ");
	}
	
	/**
	 * Extracts the part-of-speech information contained in the conceptNet id
	 * @param cnid: input id from conceptNet
	 * @return pos: part of speech tag from the input id
	 */
	public String getPOS(String cnid){
		String[] splitted = cnid.split("/");
		if (splitted.length > 4){
			return splitted[4];
		}
		return "";
	}
	
	/**
	 * Checks if any other similar ids or relation targets can be utilized to obtain pos information 
	 * for a specific conceptNet id
	 * @param cnid: conceptNet id that has no pos information
	 * @param cnid2: target conceptNet id that has a relation to cnid
	 * @param idList: all IDs that have been retrieved 
	 * @return pos: returns the id with a pos tag if it could be obtained
	 */
	public String checkPOS(String cnid, String cnid2, Set<String> idList){
		Boolean noun = false;
		Boolean verb = false;
		Boolean adj = false;
		Boolean r = false;
		Boolean posRetrieved = false;
		if (idList.contains(cnid+"/n")){ noun = true; }
		if(idList.contains(cnid+"/v")){ verb = true; }
		if(idList.contains(cnid+"/a")){ adj = true; }
		if(idList.contains(cnid+"/r")){ r = true; }
		if (noun && !verb && !adj){
			cnid = cnid +"/n";
			posRetrieved = true;
		}
		if(!noun & verb & !adj){
			cnid = cnid +"/v";
			posRetrieved = true;
		}
		if (!noun & !verb & adj){
			cnid = cnid +"/a";
			posRetrieved = true;
		}
		if (r & !noun & !verb & !adj) {
			cnid = cnid +"/r";
			posRetrieved = true;
		}
		if (!posRetrieved){
			String pos = getPOS(cnid2);
			if(!pos.isEmpty()){
				cnid = cnid+"/"+pos;
			}
		}
		return cnid;
	}
		
}
