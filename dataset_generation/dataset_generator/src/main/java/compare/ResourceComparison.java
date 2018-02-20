package compare;

import java.io.IOException;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

import it.uniroma1.lcl.babelnet.BabelImage;
import it.uniroma1.lcl.babelnet.BabelSense;
import it.uniroma1.lcl.babelnet.BabelSynset;
import it.uniroma1.lcl.babelnet.BabelSynsetID;
import it.uniroma1.lcl.babelnet.BabelSynsetIDRelation;
import it.uniroma1.lcl.babelnet.InvalidBabelSynsetIDException;
import it.uniroma1.lcl.babelnet.data.BabelDomain;
import it.uniroma1.lcl.jlt.util.Language;

import org.hibernate.SessionFactory;

import com.google.common.collect.Sets;

import net.codejava.hibernate.AllWordSensesManager;
import net.codejava.hibernate.Definition2SynsetManager;
import net.codejava.hibernate.DefinitionManager;
import net.codejava.hibernate.Domain2SynsetManager;
import net.codejava.hibernate.HibernateInitializer;
import net.codejava.hibernate.Image2SynsetManager;
import net.codejava.hibernate.ImageManager;
import net.codejava.hibernate.LanguageManager;
import net.codejava.hibernate.RelationNameManager;
import net.codejava.hibernate.Sense2SynsetManager;
import net.codejava.hibernate.SenseManager;
import net.codejava.hibernate.SourceManager;
import net.codejava.hibernate.SynsetDomainManager;
import net.codejava.hibernate.SynsetIDManager;
import net.codejava.hibernate.SynsetRelationsManager;
import net.codejava.hibernate.Word2SourceManager;
import net.codejava.hibernate.Word2SynsetManager;
import net.codejava.hibernate.WordManager;
import net.codejava.hibernate.WordPOSManager;
import net.codejava.hibernate.WordTypeManager;

import objects.ConceptNetEntry;
import objects.SynsetMetadata;
import objects.TermEntry;

import configuration.Configuration;
import dbReader.LoadTermBase;
import extractor.BabelNetExtractor;

public class ResourceComparison {
	String iateTable = Configuration.dataSource.getProperty("iateTableName");
	String termiumTable = Configuration.dataSource.getProperty("termiumTableName");
	
	//All the variables needed as DB connection using hibernate
	protected HibernateInitializer hbnInit;
	protected SessionFactory sessionFactory;
	
	protected AllWordSensesManager allWordSenses;
	protected DefinitionManager definition;
	protected Definition2SynsetManager def2syn;
	protected Domain2SynsetManager dom2syn;
	protected ImageManager image;
	protected Image2SynsetManager im2syn;
	protected LanguageManager language; 
	protected RelationNameManager relationName;
	protected SenseManager sense;
	protected Sense2SynsetManager sen2syn;
	protected SourceManager source; 
	protected SynsetDomainManager domain;
	protected SynsetIDManager synsetIDs;
	protected SynsetRelationsManager synsetRelation;
	protected WordManager word;
	protected Word2SourceManager word2source;
	protected Word2SynsetManager word2syn;
	protected WordPOSManager wordPos;
	protected WordTypeManager wordType;
	
	//Classes used to load and process synsets from all four resources
	LoadTermBase loader; 
	BabelNetExtractor babelEx; 
	
	//Variables for overall statistics
	Set<BabelSynsetID> allMatchingIDs = new HashSet<BabelSynsetID>();
	
	int notInBabelNet;
	int highestvalue;
	
	//Relations that are very frequent but contain little information in BN and CN
	Set<String> omittedBNRelations;	
	Set<String> omittedCNRelations;
	
	//Auxiliary variables that serve as lookup tables
	Set<BabelSynsetID> missingBNTargetIDs;
	Set<String> missingCNTargetIDs;
	HashMap<BabelSynsetID, String> bnID2synsetID;
	HashMap<String, String> cnID2synsetID;
	Set<String> cnIDs;
	Set<String> cnBNrelations;
	SynsetMetadata metadata;
	
	public ResourceComparison() {
		hbnInit = new HibernateInitializer();
		sessionFactory = hbnInit.setup();
		loader = new LoadTermBase();
		babelEx = new BabelNetExtractor();
		notInBabelNet = 0;
		highestvalue = 0;
		omittedBNRelations = new HashSet<String>(Arrays.asList("semantically_related_form", "gloss_related_form_(disambiguated)", "gloss_related_form_(monosemous)", "derivationally_related_form"));
		omittedCNRelations = new HashSet<String>(Arrays.asList("/r/ExternalURL","/r/RelatedTo","/r/DerivedFrom", "/r/EtymologicallyRelatedTo", "/r/FormOf", "/r/Synonym", "/r/HasContext"));

		missingBNTargetIDs = new HashSet<BabelSynsetID>();
		missingCNTargetIDs = new HashSet<String>();
		bnID2synsetID = new HashMap<BabelSynsetID, String>();
		cnID2synsetID = new HashMap<String, String>();
		cnIDs = new HashSet<String>();
		cnBNrelations = new HashSet<String>();
		metadata = new SynsetMetadata();
		
		allWordSenses = new AllWordSensesManager();
		definition = new DefinitionManager();
		def2syn = new Definition2SynsetManager();
		dom2syn = new Domain2SynsetManager();
		image = new ImageManager();
		im2syn = new Image2SynsetManager();
		language = new LanguageManager(); 
		relationName = new RelationNameManager();
		sense = new SenseManager();
		sen2syn = new Sense2SynsetManager();
		source = new SourceManager(); 
		domain = new SynsetDomainManager();
		synsetIDs = new SynsetIDManager();
		synsetRelation = new SynsetRelationsManager();
		word = new WordManager();
		word2source = new Word2SourceManager();
		word2syn = new Word2SynsetManager();
		wordPos = new WordPOSManager();
		wordType = new WordTypeManager();
	}

	/***
	 * Main method to run through BabelNet, IATE, Termium, and ConceptNet and create a merged version based on
	 * a multilingual comparison of their synset entries
	 */
	public void resourceAligner(){
		
		//Retrieve English word lists of IATE, Termium, and ConceptNet for an initial comparison
		HashMap<String, Set<String>> iateEnglish = loader.getEnglishRows(iateTable);
		HashMap<String, Set<String>> termiumEnglish = loader.getEnglishRows(termiumTable);
		HashMap<String, Set<String>> conceptNetEnglish = loader.getConceptNetEnglishWordList();
		
		//Print out some statistics
		System.out.println("All English words "+iateTable+" "+iateEnglish.size()); 
		System.out.println("All English words "+termiumTable+" "+termiumEnglish.size());
		System.out.println("All English words conceptNet "+	conceptNetEnglish.size());
		
		HashMap<String, List<BabelSynset>> babelSynsets = new HashMap<String, List<BabelSynset>>();
		Set<String> iateIDs = new HashSet<String>();
		Set<String> termiumIDs = new HashSet<String>();
		Set<String> cnIDsInBN = new HashSet<String>();
			
		HashMap<String, Set<TermEntry>> iateTB = new HashMap<String, Set<TermEntry>>();
		HashMap<String, Set<TermEntry>> termiumTB = new HashMap<String, Set<TermEntry>>();
		HashMap<String, Set<TermEntry>> conceptNetTB = new HashMap<String, Set<TermEntry>>();
		
		Set<String> inBoth = new HashSet<String>(iateEnglish.keySet());
		inBoth.retainAll(termiumEnglish.keySet());
		
		Set<String> inCN = new HashSet<String>(conceptNetEnglish.keySet());
		inCN.retainAll(inBoth);
		
		//Number of terms that overlap in both TermBases
		System.out.println("Matching English words in Termium and IATE: "+inBoth.size());
		System.out.println("Matching English words in Termium, IATE, and ConceptNet "+inCN.size());
		
		//Counters to allow for batch processing of the whole datasets to speed up processing
		int batchSize = (int)Math.ceil( (inBoth.size()) / 100);
		int lastBatchSize = inBoth.size() - (99 * batchSize);
		int listSize = inBoth.size();
		int batchCounter = 0;
		int counter = 0;
			
		//Run through the list of overlapping English words in the two terminologies
		for(String term : inBoth){
			counter += 1;
			try {
				List<BabelSynset> synsets = babelEx.getSynsets(term, "EN");
				if(!synsets.isEmpty()){
					babelSynsets.put(term, synsets);
					
					//Variables to batch load the corresponding overlapping terminological synsets
					iateIDs.addAll(iateEnglish.get(term));
					termiumIDs.addAll(termiumEnglish.get(term));
					if (inCN.contains(term)){ 
						
						//Variable to bacht load the CN subset that matches with BabelNet
						cnIDsInBN.addAll(conceptNetEnglish.get(term));
					}
				}
				else{
					if (inCN.contains(term)){
						
						//Variable to track all CN ids that do not match with BabelNet but with the two terminologies
						cnIDs.addAll(conceptNetEnglish.get(term));
					}
					notInBabelNet += 1;
				}
			} catch (IOException | InvalidBabelSynsetIDException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			//Batch processing to speed up and avoid heap exceptions
			if (counter == batchSize){
				batchCounter += 1;
				System.out.println(batchSize+" Number "+batchCounter);
				listSize = listSize - batchSize;
				if(listSize < (batchSize*2)){
					batchSize = lastBatchSize;
				}
				
				//Batch loading of all IDs stored in the ID variable so far
				iateTB = loader.getEntriesFromList(iateIDs, iateTable);
				termiumTB = loader.getEntriesFromList(termiumIDs, termiumTable);
				conceptNetTB = loader.getCNSynonyms(cnIDsInBN);

				//Calculate and store the overlap between the BabelNet synsets and the other resoruces 
				getBabelNetOverlap(babelSynsets, iateTB, termiumTB, conceptNetTB, iateEnglish, termiumEnglish, conceptNetEnglish);
				
				//Reinitialize all the variables for the next batch to process
				babelSynsets = new HashMap<String, List<BabelSynset>>();
				iateIDs = new HashSet<String>();
				termiumIDs = new HashSet<String>();
				cnIDsInBN = new HashSet<String>();
				counter = 0;

				iateTB = new HashMap<String, Set<TermEntry>>();
				termiumTB = new HashMap<String, Set<TermEntry>>();
//				if (batchCounter == 3){
//					break;
//				}
				break;
			}
		}
	
		//Combine the relation targets that have not automatically been stored as well to the DB to ensure that 
		//each relation has relatums in the DB and check again whether those do not overlap with at least 
		//one of the TermBase resources
		writeMissingBNTargets(iateEnglish, termiumEnglish, conceptNetEnglish);

		//Run through all conceptNet IDs that do not overlap with BabelNet and check 
		//whether they overlap with IATE or Termium and write the overlapping synsets
		//to the final DB
		getConceptNetOverlap(iateEnglish, termiumEnglish, conceptNetEnglish);
		
		//Write all relations of conceptNet synsets that overlap with a BabelNet synset 
		// to the DB - in the above process only the conceptNet IDs are added to the created
		// merged synsets, but not their relations and relation targets - this is done here
		for (String conceptNetID : cnBNrelations) {
			getCNRelations(conceptNetID, conceptNetEnglish);
		}	
		
		//All relation targets of the above two processing steps of conceptNet are now in a final
		//step stored to the final output DB
		writeMissingCNTargets(iateEnglish, termiumEnglish);
		
		//Some control variables at the very end to check functioning
		System.out.println("Not in BabelNet "+notInBabelNet);
		hbnInit.exit();
	}
	

	/**
	 * Method that compares the retrieved BabelNet synsets to IATE, Termium and conceptNet. If the BabelNet synset overlaps in more than three terms across languages with 
	 * the Termium and IATE synsets, we check the relations of the BabelNet sysnet. If it has relations other than the ones we wish to omit (see list at top) this method
	 * merges the synsets (also ConceptNet if available) and writes the merged entry to our final DB
	 * @param babelSynsets: all retrieved BabelNet synset of this batch
	 * @param iateTB: all IATE synsets of this batch
	 * @param termiumTB: all Termium synsets of this batch
	 * @param conceptNetTB: all conceptNet synsets of this batch
	 * @param iateEnglish: all English words in IATE
	 * @param termiumEnglish: all English words in Termium
	 * @param conceptNetEnglish: all English words in conceptNet
	 * @param missingBNTargetIDs: target synsets of relations that have not been written to the final DB
	 * @param allMatchingIDs: variable to keep track of already included BabelNet synsets and avoid writing them twice to the DB (if synonyms are queried later)
	 */	
	public void getBabelNetOverlap(HashMap<String, List<BabelSynset>> babelSynsets, HashMap<String, Set<TermEntry>> iateTB, HashMap<String, Set<TermEntry>> termiumTB, HashMap<String, 
		Set<TermEntry>> conceptNetTB, HashMap<String, Set<String>> iateEnglish, HashMap<String, Set<String>> termiumEnglish, HashMap<String, Set<String>> conceptNetEnglish) {
		
		for(Entry<String, List<BabelSynset>> babelEntry : babelSynsets.entrySet()){
			HashMap<String, Set<TermEntry>> iateSubset = new HashMap<String, Set<TermEntry>>();
			HashMap<String, Set<TermEntry>> termiumSubset = new HashMap<String, Set<TermEntry>>();
			HashMap<String, Set<TermEntry>> conceptNetSubset = new HashMap<String, Set<TermEntry>>();
			
			//Get all synsets that have a word in common with this specific BabelNet synset
			for (String iateID: iateEnglish.get(babelEntry.getKey())) { iateSubset.put(iateID, iateTB.get(iateID));}
			for (String termiumID : termiumEnglish.get(babelEntry.getKey())) { termiumSubset.put(termiumID, termiumTB.get(termiumID));}
			if (conceptNetEnglish.containsKey(babelEntry.getKey().toLowerCase())){
				for(String conceptNetID: conceptNetEnglish.get(babelEntry.getKey())){ 
					if (conceptNetTB.containsKey(conceptNetID)){ conceptNetSubset.put(conceptNetID, conceptNetTB.get(conceptNetID));}
				}
			}
			
			//Calculate the overlap between this BabelNet synsets and the synsets of all other resources
			HashMap<BabelSynset, Set<String>> overlapIate = calculateMultilingualOverlap(babelEntry.getValue(), iateSubset);
			HashMap<BabelSynset, Set<String>> overlapTermium = calculateMultilingualOverlap(babelEntry.getValue(), termiumSubset);
			HashMap<BabelSynset, Set<String>> overlapCNBN = calculateMultilingualOverlap(babelEntry.getValue(), conceptNetSubset);
			
			//Only keep those BabelSynsets that have more than three words in common with a sysnet of both terminological resources
			Set<BabelSynset> overlapBoth = Sets.intersection(overlapIate.keySet(), overlapTermium.keySet());
			
			//Only if the BabelNet synset overlaps in more than three words across languages with IATE and Termium is it written to the final DB
			if (!overlapBoth.isEmpty()){
				for (BabelSynset synset : overlapBoth){
					//Where both terminologies overlap AND there are valid relations in the corresponding BabelNet entry
					if(!allMatchingIDs.contains(synset.getId()) && checkBNRelations(synset)){
						allMatchingIDs.add(synset.getId());
						
						Set<TermEntry> cnSub = new HashSet<TermEntry>();
						Set<TermEntry> iateSub = new HashSet<TermEntry>();
						Set<TermEntry> termiumSub = new HashSet<TermEntry>();
						
						if(overlapCNBN.containsKey(synset)){ for (String cnid : overlapCNBN.get(synset)) { cnSub.addAll(conceptNetTB.get(cnid)); } }
						for(String iateID : overlapIate.get(synset)){ iateSub.addAll(iateTB.get(iateID));}
						for(String termiumID: overlapTermium.get(synset)){ termiumSub.addAll(termiumTB.get(termiumID));}
						combineEntries(synset, iateSub, termiumSub, cnSub);
						getBNRelations(synset);		
						//Since we write BN synsets to DB here, remove all synset ids that were previously detected as missing target synset of a relation
						//all remaining missing BabelNet synsets are in the end written to the final DB 
						if (missingBNTargetIDs.contains(synset.getId())){
							missingBNTargetIDs.remove(synset.getId());
						}
					}
				}
			}
		}
	}
	
	/**
	 * Method to calculate overlap of all conceptNet synsets that were not merged with BabelNet synsets and check whether they overlap with IATE and Termium. 
	 * Only if they overlap, they are written to the final DB.  
	 * @param iateEnglish: list of English words in IATE
	 * @param termiumEnglish: list of English words in Termium
	 * @param conceptNetEnglish: list of English words in conceptNet
	 */
	public void getConceptNetOverlap(HashMap<String, Set<String>> iateEnglish, HashMap<String, Set<String>> termiumEnglish, HashMap<String, Set<String>> conceptNetEnglish) {
		System.out.println("Size of remaining ConceptNet IDs to be matched with terminologies: "+cnIDs.size());
		
		HashMap<String, Set<String>> overlapCNIate = new HashMap<String, Set<String>>();
		HashMap<String, Set<String>> overlapCNTermium = new HashMap<String, Set<String>>();
		HashMap<String, Set<TermEntry>> conceptNetTB = loader.getCNSynonyms(cnIDs);
		
		HashMap<String, Set<TermEntry>> iateTB = new HashMap<String, Set<TermEntry>>();
		HashMap<String, Set<TermEntry>> termiumTB = new HashMap<String, Set<TermEntry>>();
		
		int CounterCn = cnIDs.size();
		for (String cnid : cnIDs) {
			CounterCn -= 1;
			System.out.println(CounterCn);
			if (conceptNetTB.containsKey(cnid) && checkCNRelations(cnid)) {
				String word = loader.getCNLabel(cnid).trim();
			
				iateTB = loader.getEntriesFromList(iateEnglish.get(word), iateTable);
				termiumTB = loader.getEntriesFromList(termiumEnglish.get(word), termiumTable);
			
				HashMap<String, Set<TermEntry>> iateSubset = new HashMap<String, Set<TermEntry>>();
				HashMap<String, Set<TermEntry>> termiumSubset = new HashMap<String, Set<TermEntry>>();
			
				for (String iateID : iateEnglish.get(word)) { iateSubset.put(iateID, iateTB.get(iateID)); }
				for (String termiumID : termiumEnglish.get(word)) { termiumSubset.put(termiumID, termiumTB.get(termiumID));}
				
				//Calculate the overlap between ConceptNet and the entries from the two terminologies
				if(!iateSubset.isEmpty()){ overlapCNIate = calculateMultilingualOverlapConceptNet(conceptNetTB.get(cnid), iateSubset);}
				if (!termiumSubset.isEmpty()){ overlapCNTermium = calculateMultilingualOverlapConceptNet(conceptNetTB.get(cnid), termiumSubset);}

				//Calculate the number of BabelNet synset that overlap with terminological entries across at least two terms in both TBs
				Set<String> overlapCNBoth = Sets.intersection(overlapCNIate.keySet(), overlapCNTermium.keySet());
			
				if(!overlapCNBoth.isEmpty()) {	
					Set<TermEntry> iateSub = new HashSet<TermEntry>();
					Set<TermEntry> termiumSub = new HashSet<TermEntry>();
					
					//Retrieve the subset of terminology synsets that overlap with the conceptNet synset 
					for (String iateID : overlapCNIate.get(cnid)) { iateSub.addAll(iateTB.get(iateID));} 
					for (String termiumID : overlapCNTermium.get(cnid)) { termiumSub.addAll(termiumTB.get(termiumID)); }
					
					combineCNWithTerminologies(conceptNetTB.get(cnid), iateSub, termiumSub);
					getCNRelations(cnid, conceptNetEnglish);	
					if (missingCNTargetIDs.contains(cnid)){
						missingCNTargetIDs.remove(cnid);
					}
				}
				iateTB = new HashMap<String, Set<TermEntry>>();
				termiumTB = new HashMap<String, Set<TermEntry>>();
				conceptNetTB = new HashMap<String, Set<TermEntry>>();
			
				overlapCNTermium = new HashMap<String, Set<String>>();
				overlapCNIate = new HashMap<String, Set<String>>();
			}
		}

	}

	/**
	 * Method that checks the multilingual overlap of several synsets and terminological entries from TermBases that have been previously matched based on their 
	 * English words 
	 * @param synsets: set of BabelNet synsets containing that specific English word
	 * @param terminology: set of the terminological entries from either IATE or Termium containing that specific English word
	 * @param results: result map that is returned containing the overlapping synset and entry IDs
	 * @return the results map with synsets across all resources that overlap in more than 3 words
	 */
	public HashMap<BabelSynset, Set<String>> calculateMultilingualOverlap(List<BabelSynset> synsets, HashMap<String, Set<TermEntry>> terminology){
		HashMap<BabelSynset, Set<String>> results = new HashMap<>();
		try {	
		for (BabelSynset synset: synsets){
			HashMap<String, Integer> overlapping = new HashMap<String, Integer>();
			HashMap<String, Set<String>> babelEntry = babelEx.getMainTerms(synset);
			for (String entryID : terminology.keySet()){
				HashMap<String, Set<String>> termsByLanguage = loader.getTermsByLanguage(terminology.get(entryID));
				Set<String> languages = Sets.intersection(termsByLanguage.keySet(), babelEntry.keySet());
				for (String language : languages){
					Set<String> intersection = Sets.intersection(termsByLanguage.get(language), babelEntry.get(language));
					if (!intersection.isEmpty()){
						if (overlapping.containsKey(entryID)) { overlapping.put(entryID, overlapping.get(entryID)+intersection.size());}
						else { overlapping.put(entryID, intersection.size());}
					}
				}	
			}
			if (!overlapping.isEmpty()){
				int value = overlapping.entrySet().stream().max((entry1, entry2) -> entry1.getValue() > entry2.getValue() ? 1 : -1).get().getValue();
				if(value > highestvalue){
					highestvalue = value;
					System.out.println("highest value "+highestvalue+" "+overlapping+" "+synset.getId().toString());
				}
				if (value > 3){
					Set<String> entryIDs = overlapping.entrySet().stream().filter(entry -> Objects.equals(entry.getValue(), value)).map(Map.Entry::getKey).collect(Collectors.toSet());
					if (results.containsKey(synset)) { results.get(synset).addAll(entryIDs); } 
					else { results.put(synset, entryIDs); }
				}
			}
		}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return results;
	}
	
	/**
	 * Method that checks the multilingual overlap of several synsets and terminological entries from TermBases that have been previously matched based on their 
	 * English words 
	 * @param synsets: set of BabelNet synsets containing that specific English word
	 * @param terminology: set of the terminological entries from either IATE or Termium containing that specific English word
	 * @param results: result map that is returned containing the overlapping synset and entry IDs
	 * @return the results map with synsets across all resources that overlap in more than 3 words
	 */
	public HashMap<String, Set<String>> calculateMultilingualOverlapConceptNet(Set<TermEntry> conceptNet, HashMap<String, Set<TermEntry>> terminology){
		HashMap<String, Set<String>> results = new HashMap<String, Set<String>>();
		HashMap<String, Integer> overlapping = new HashMap<String, Integer>();
		
		HashMap<String, Set<String>> cnTermsByLanguage = loader.getTermsByLanguage(conceptNet);
		for (String entryID : terminology.keySet()){
			overlapping = new HashMap<String, Integer>();
			HashMap<String, Set<String>> termsByLanguage = loader.getTermsByLanguage(terminology.get(entryID));
			Set<String> languages = Sets.intersection(termsByLanguage.keySet(), cnTermsByLanguage.keySet());
			for (String language : languages){
				Set<String> intersection = Sets.intersection(termsByLanguage.get(language), cnTermsByLanguage.get(language));
				if (!intersection.isEmpty()){
					if (overlapping.containsKey(entryID)) { overlapping.put(entryID, overlapping.get(entryID)+intersection.size());}
					else { overlapping.put(entryID, intersection.size());}
				}
			}
		}
		if (!overlapping.isEmpty()){
			int value = overlapping.entrySet().stream().max((entry1, entry2) -> entry1.getValue() > entry2.getValue() ? 1 : -1).get().getValue();
			if(value > highestvalue){
			highestvalue = value;
			System.out.println("highest value "+highestvalue+" "+overlapping);
			}
			if (value > 3){
				Set<String> entryIDs = overlapping.entrySet().stream().filter(entry -> Objects.equals(entry.getValue(), value)).map(Map.Entry::getKey).collect(Collectors.toSet());
				if (results.containsKey(conceptNet.iterator().next().getEntryID())) { results.get(conceptNet.iterator().next().getEntryID()).addAll(entryIDs); } 
				else { results.put(conceptNet.iterator().next().getEntryID(), entryIDs); }
			}
		}
		return results;
	}
	
	/**
	 * First writes all BabelNet data and metadata to the EVALution synset and then merges it with the synsets of the other resources
	 * @param synset: BabelNet synset to be merged with other entries because it overlaps in more than three words
	 * @param iateEntries: IATE synsets that are to be merged 
	 * @param termiumEntries: termium synsets that are to be merged
	 * @param conceptNetEntries: conceptNet entries that are to be merged
	 */
	public void combineEntries(BabelSynset synset, Set<TermEntry> iateEntries,  Set<TermEntry> termiumEntries, Set<TermEntry> conceptNetEntries){	
		try {	
			//Generate random UUID for this EVALution synset and write all BabelNet data of this synset to our final DB
			Long synsetID;
			String generatedID = "EVAL"+UUID.randomUUID().toString();
			if (bnID2synsetID.containsKey(synset.getId())){ 
				synsetID = synsetIDs.create(bnID2synsetID.get(synset.getId()), sessionFactory);
				generatedID = bnID2synsetID.get(synset.getId());
			}
			else{
				synsetID = synsetIDs.create(generatedID, sessionFactory);
				bnID2synsetID.put(synset.getId(), generatedID);
			}
			
			//Write source and images to this EVALution synset
			Long sourceID = source.create(synset.getId().toString(), sessionFactory);
			if(synset.getImage() != null) { 
				Long imageID = image.create(synset.getImage().toString(), sessionFactory);
				im2syn.create(synsetID, imageID, sourceID, true, sessionFactory);
			}
			
			if (synset.getImages().size() > 0) {
				for (BabelImage babelImage : synset.getImages()){
					if (!babelImage.equals(synset.getImage())){
						Long imagesID = image.create(babelImage.toString(), sessionFactory);
						im2syn.create(synsetID, imagesID, sourceID, false, sessionFactory);
					}
				}	
			}
			
			//Write the BabelNet domains to metadata
			if (!synset.getDomains().isEmpty()){
				for (Map.Entry<BabelDomain, Double> domainEntry : synset.getDomains().entrySet()){
					Long domainID = domain.create(domainEntry.getKey().toString(), sessionFactory);
					dom2syn.create(synsetID, domainID, domainEntry.getValue(), sourceID, sessionFactory);
				}
			}
			
			Long senseID = sense.create(synset.getMainSense(Language.EN).toString(), sessionFactory);
			sen2syn.create(synsetID, senseID, true, sessionFactory);
			
			for (BabelSense babelSense : synset.getSenses()){
					Long notMainSense = sense.create(babelSense.toString(), sessionFactory);
					Long senseLang = language.create(babelSense.getLanguage().toString().toLowerCase(), sessionFactory); 
					
					Long definitionID = null;
					if(synset.getMainGloss(Language.valueOf(babelSense.getLanguage().toString())) != null){ definitionID = definition.create(synset.getMainGloss(babelSense.getLanguage()).toString(), sessionFactory); }
					if (definitionID != null) { def2syn.create(synsetID, definitionID, senseLang, sourceID, sessionFactory); }
					 
					Set<String> terms = new HashSet<String>();
					String mainTerm = babelSense.getLemma().replace("_", " ").replaceAll("\"", "");
					terms.add(mainTerm);				
				
					for(String term : terms){
						if (term.contains("’")) { term = term.replace("’", "'"); }
						Long wordID = word.create(term, sessionFactory);
						Boolean mainWord = false; 
						if (term == mainTerm){ mainWord = true; }
						Long posID = wordPos.create(synset.getPOS().toString(), sessionFactory);
						Long wordTypeID = null;
						Double wordConcreteness = null;
						if (babelSense.getLanguage().toString() == "EN"){ wordConcreteness = loader.getConcreteness(term); }
						
						word2syn.create(synsetID, wordID, senseLang, notMainSense, mainWord, posID, wordTypeID, wordConcreteness, sessionFactory);
						word2source.create(synsetID, wordID, senseLang, sourceID, sessionFactory);
						writeAllWordSenses(term, babelSense.getLanguage().toString().toLowerCase(), babelSense.toString(), synset, wordID, senseLang, sourceID);
					}
			}
			
			//Add data and metadata of all other resources to this EVALution synset
			if(!termiumEntries.isEmpty()){
				mergeEntries(termiumEntries, synsetID, null, false);
			}
			
			if(!iateEntries.isEmpty()) {
				mergeEntries(iateEntries, synsetID, null, false);
			}
	
			if (!conceptNetEntries.isEmpty()) {
				mergeEntries(conceptNetEntries, synsetID, generatedID, true);
			}
		} catch (IOException | SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	/**
	 * ConceptNet entries that are to be merged with IATE and Termium because it overlaps in more than three words across langauges
	 * @param conceptNetEntries: conceptNet synsets to be merged
	 * @param iateEntries: IATE synsets to be merged
	 * @param termiumEntries: Termium synsets to be merged
	 */
	public void combineCNWithTerminologies(Set<TermEntry> conceptNetEntries, Set<TermEntry> iateEntries,  Set<TermEntry> termiumEntries){	

		//Generate random unique identifier for the merged synset
		Long synsetID;
		String generatedID = "EVAL"+UUID.randomUUID().toString();
		if (cnID2synsetID.containsKey(conceptNetEntries.iterator().next().getEntryID())){
			synsetID = synsetIDs.create(cnID2synsetID.get(conceptNetEntries.iterator().next().getEntryID()), sessionFactory);
			generatedID = cnID2synsetID.get(conceptNetEntries.iterator().next().getEntryID());
		}
		else{
			synsetID = synsetIDs.create(generatedID, sessionFactory);
			System.out.println(conceptNetEntries.iterator().next().getEntryID());
			cnID2synsetID.put(conceptNetEntries.iterator().next().getEntryID(), generatedID);
		}

			
		if (!conceptNetEntries.isEmpty()) {
			mergeEntries(conceptNetEntries, synsetID, generatedID, true);
		}

		if(!termiumEntries.isEmpty()){
			mergeEntries(termiumEntries, synsetID, null, false);
		}
			
		if(!iateEntries.isEmpty()) {
			mergeEntries(iateEntries, synsetID, null, false);
		}
	}
	
	/**
	 * Auxiliary method that merges the BabelNet synset that has been written to DB with the terminological synset
	 * @param terminology: synset of a terminological resource
	 * @param synsetID: EVALution synset id assigned to the BabelNet synset
	 * @param generatedID: EVALution generated UUID
	 * @param conceptNet: Boolean that tells the method whether the synsets are from conceptNet or not
	 */
	public void mergeEntries(Set<TermEntry> terminology, Long synsetID, String generatedID, Boolean conceptNet){
		for(TermEntry termEntry : terminology){	
			if (conceptNet & cnIDs.contains(termEntry.getOriginalEntryID())) { cnIDs.remove(termEntry.getOriginalEntryID()); }
			if (conceptNet & cnIDs.contains(termEntry.getEntryID())) { cnIDs.remove(termEntry.getEntryID()); }
			if(generatedID != null & !cnID2synsetID.containsKey(termEntry.getOriginalEntryID())) { 
				cnID2synsetID.put(termEntry.getOriginalEntryID(), generatedID);
				if(!cnID2synsetID.containsKey(termEntry.getEntryID())){ cnID2synsetID.put(termEntry.getEntryID(), generatedID); }
				if (conceptNet) { cnBNrelations.add(termEntry.getOriginalEntryID()); }
			}
			Long sourceID;
			if (conceptNet) { sourceID = source.create(termEntry.getSource(), sessionFactory); }
			else { sourceID = source.create(termEntry.getEntryID(), sessionFactory);}
			Long wordID = word.create(termEntry.getTerm(), sessionFactory);
			Long languageID = language.create(termEntry.getLanguage(), sessionFactory);
			Double wordConcreteness = null;
			if (termEntry.getLanguage().equals("en")){  wordConcreteness = loader.getConcreteness(termEntry.getTerm()); }
			Long wordTypeID = null;
			if (termEntry.getTermType() != null){ wordTypeID = wordType.create(termEntry.getTermType(), sessionFactory);}
			Long domainID = null;
			if(termEntry.getDomain() != null){ domainID = domain.create(termEntry.getDomain(), sessionFactory); }
			Long definitionID = null;
			if(termEntry.getDefinition() != null){definitionID = definition.create(termEntry.getDefinition(), sessionFactory); }
			Long wordSenseID = null; 
			Long posID = null;
			
			word2syn.create(synsetID, wordID, languageID, wordSenseID, false, posID, wordTypeID, wordConcreteness, sessionFactory);
			word2source.create(synsetID, wordID, languageID, sourceID, sessionFactory);
			writeAllWordSenses(termEntry.getTerm(), termEntry.getLanguage(), termEntry.getEntryID(), null, wordID, languageID, sourceID);
			if(domainID != null){ dom2syn.create(synsetID, domainID, null, sourceID, sessionFactory); }
			if(definitionID != null){ def2syn.create(synsetID, definitionID, languageID, sourceID, sessionFactory); }
		}
	}
	
	/**
	 * Method that retrieves all BabelNet senses for a specific term and stores them for reference in our final DB
	 * @param term: term to query senses in BabelNet
	 * @param lang: language of the term to enable query
	 * @param wordSense: source wordSense of synset from which we obtained the term
	 * @param synset: source synset from which we took the term
	 * @param wordID: the id associated with the term in our final DB
	 * @param languageID: the id associated with the language of the term in our final DB
	 * @param sourceID: the id associated with the source (BabelSynset id) in our final DB
	 */
	public void writeAllWordSenses(String term, String lang, String wordSense, BabelSynset synset, Long wordID, Long languageID, Long sourceID){
		try{
			List<BabelSynset> synsets = babelEx.getSynsets(term, lang.toUpperCase());
			if (synset != null && !synsets.contains(synset)){ 
				Long senseID = sense.create(wordSense, sessionFactory);
				allWordSenses.create(wordID, languageID, senseID, sourceID, sessionFactory);
			}
			if (!synsets.isEmpty()){
				for (BabelSynset syn : synsets){
					Long senseID = sense.create(syn.getMainSense(Language.valueOf(lang.toUpperCase())).toString(), sessionFactory);
					allWordSenses.create(wordID, languageID, senseID, sourceID, sessionFactory);
				}
			}
		} catch (IOException | InvalidBabelSynsetIDException | SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	/**
	 * Checks whether a BabelNet synset has any relations that are not in the list of relations we wish to omit
	 * Stops if one such relation is found and returns true
	 * @param synset
	 * @return true if relation is found, false if synset has no valid relation
	 */
	public Boolean checkBNRelations(BabelSynset synset) {
		Boolean hasValidRelations = false;
		for(BabelSynsetIDRelation edge : synset.getEdges()){
			if(!omittedBNRelations.contains(edge.getPointer().toString())){
				hasValidRelations = true;
				break;
			}
		}
		return hasValidRelations;
	}
	
	/**
	 * Checks whether a conceptNet id has any relations that are not in the list of relations we wish to omit
	 * Retrieves all associated and valid relations from the Database and returns true if relations are found
	 * @param cnid
	 * @return true if valid relation is found, false if no relation is found
	 */
	public Boolean checkCNRelations(String cnid) {
		Boolean hasValidRelations = false;
		Set<String> tempIDs = new HashSet<String>(Arrays.asList(cnid));
		Set<ConceptNetEntry> tempRels = loader.getCNRelations(tempIDs, omittedCNRelations);
		if(!tempRels.isEmpty()) {
			hasValidRelations = true;
		}
		return hasValidRelations;
	}
	
	/**
	 * Retrieves all relations of a BabelNet synset, checks if the target of the relation is already stored 
	 * If the target synset is not yet written to our final resulting DB, this method assigns a unique id to 
	 * the target synset, writes it to the DB and stores the id to a temporary field for later processing 
	 * called "missingBNTargetIDs" and the related generated id to "bnID2synsetID" 
	 * @param synset: source synset of relations
	 * @param omittedBNRelations: relations that we do not wish to include in the final database (see list at the top)
	 * @param bnID2synsetID: all BabelNet synset IDs and their mapping to a reference ID in our final DB
	 * @param missingBNTargetIDs: all BabelNet synset ids that are target synsets of relations but not yet in the final DB
	 */
	public void getBNRelations(BabelSynset synset){
		for(BabelSynsetIDRelation edge : synset.getEdges()){
			if(!omittedBNRelations.contains(edge.getPointer().toString())){
				Long sourceSynset_id = synsetIDs.create(bnID2synsetID.get(synset.getId()), sessionFactory);
				Long relName = relationName.create(edge.getPointer().toString(), sessionFactory);
				Long targetSynset_id;
				if(bnID2synsetID.containsKey(edge.getBabelSynsetIDTarget())){
					targetSynset_id = synsetIDs.create(bnID2synsetID.get(edge.getBabelSynsetIDTarget()), sessionFactory);
				}
				else{
					String generatedID = "EVAL"+UUID.randomUUID().toString();			
					targetSynset_id = synsetIDs.create(generatedID, sessionFactory);
					bnID2synsetID.put(edge.getBabelSynsetIDTarget(), generatedID);
					missingBNTargetIDs.add(edge.getBabelSynsetIDTarget());
				}
				synsetRelation.create(sourceSynset_id, targetSynset_id, relName, edge.getWeight(), null, sessionFactory);
			}
		}
	}
	
	/**
	 * Retrieves all relations of a conceptNet id and checks if the target synset has already been stored in 
	 * our resulting Database. If not, the method assigns a unique id to the target synset, stores it in our
	 * DB and in a temporary field called "missingCNTargetIDs" for later processing as well as the created 
	 * associated id in the field cnID2synsetIDs
	 * @param omittedCNRelations: relations that we do not wish to include in the final database (see list at the top)
	 * @param cnid: the source cnid for which all relations are checked
	 * @param cnID2synsetID: map to store all cnids and their related reference IDs in our final resource
	 * @param missingCNTargetIDs: all conceptNet ids that are target synsets of relations but not yet in the final DB
	 */
	public void getCNRelations(String cnid, HashMap<String, Set<String>> conceptNetEnglish){
		Set<String> helperCNID = new HashSet<String>(Arrays.asList(cnid));
		Set<ConceptNetEntry> tempRels = loader.getCNRelations(helperCNID, omittedCNRelations);
		
		for(ConceptNetEntry tempEntry : tempRels) {
			Long sourceID;
			if (cnID2synsetID.get(tempEntry.getSource()) == null){
				sourceID = synsetIDs.create(cnID2synsetID.get(cnid), sessionFactory);
			}
			else {
				sourceID = synsetIDs.create(cnID2synsetID.get(tempEntry.getSource()), sessionFactory);
			}
			Long relationID = relationName.create(tempEntry.getRelation().toLowerCase(), sessionFactory);
			Long targetID = null;
			System.out.println(cnID2synsetID.keySet());
			if(cnID2synsetID.containsKey(tempEntry.getTarget())) { 
				targetID = synsetIDs.create(cnID2synsetID.get(tempEntry.getTarget()), sessionFactory);
			}
			else {
				String generatedID = "EVAL"+UUID.randomUUID().toString();			
				targetID = synsetIDs.create(generatedID, sessionFactory);
				cnID2synsetID.put(tempEntry.getTarget(), generatedID);
				missingCNTargetIDs.add(tempEntry.getTarget());
			}
			synsetRelation.create(sourceID, targetID, relationID, tempEntry.getWeight(), tempEntry.getSurfaceText(), sessionFactory);
		}
		
	}
	
	/**
	 * Runs over all missing target synsets in the list missingBNTargetIDs. They cannot overlap with both terminologies, because then 
	 * they would have been included in the first processing of the resources. Thus, we check whether the target synset overlaps with one 
	 * of the resources and conceptNet and write the synset to the final DB in any case 
	 * @param missingBNTargetIDs: list of BabelNet IDs that were target synsets of relations but not yet written to the DB
	 * @param iateEnglish: list of English words in IATE
	 * @param termiumEnglish: list of English words in Termium
	 * @param conceptNetEnglish: list of English words in ConceptNet
	 */
	private void writeMissingBNTargets(HashMap<String, Set<String>> iateEnglish, HashMap<String, Set<String>> termiumEnglish, HashMap<String, Set<String>> conceptNetEnglish){
		System.out.println("Missing targets written to DB..."+ missingBNTargetIDs.size());

		if (!missingBNTargetIDs.isEmpty()){
			Set<BabelSynset> synsets = new HashSet<BabelSynset>();

			HashMap<String, Set<TermEntry>> iateTB = new HashMap<String, Set<TermEntry>>();
			HashMap<String, Set<TermEntry>> termiumTB = new HashMap<String, Set<TermEntry>>();
			HashMap<String, Set<TermEntry>> conceptNetTB = new  HashMap<String, Set<TermEntry>>();
			
			Set<String> iateIDs = new HashSet<String>();
			Set<String> termiumIDs = new HashSet<String>();
			Set<String> conceptNetIDHelper = new HashSet<String>();
			
			int counter = missingBNTargetIDs.size();
			for (BabelSynsetID bnID : missingBNTargetIDs) {
				BabelSynset sn = babelEx.getSynset(bnID);
				synsets.add(sn);
							
				for (BabelSense sense: sn.getMainSenses(Language.EN)){
					String bnWord = sense.getLemma().replace("_", " ").replace("\"", "");
					
					if (iateEnglish.containsKey(bnWord)){  iateIDs.addAll(iateEnglish.get(bnWord)); }
					if(termiumEnglish.containsKey(bnWord)) { termiumIDs.addAll(termiumEnglish.get(bnWord)); }
					if(conceptNetEnglish.containsKey(bnWord)) { conceptNetIDHelper.addAll(conceptNetEnglish.get(bnWord));}
				}
				
				//Batch processing of missing IDs to speed up
				if (synsets.size() > 100 || counter == synsets.size()){
					System.out.println("missing bn targets: "+counter);
					counter = counter - synsets.size();
					
					if (!iateIDs.isEmpty()) { iateTB = loader.getEntriesFromList(iateIDs, iateTable); }
					if (!termiumIDs.isEmpty()) { termiumTB = loader.getEntriesFromList(termiumIDs, termiumTable); }
					if (!conceptNetIDHelper.isEmpty()) { conceptNetTB = loader.getCNSynonyms(conceptNetIDHelper); }
					
					for (BabelSynset synset1 : synsets){
						String bnWord1 = synset1.getMainSense(Language.EN).getLemma().replace("_", " ").replace("\"", "");
						
						HashMap<String, Set<TermEntry>> iateSubset = new HashMap<String, Set<TermEntry>>();
						HashMap<String, Set<TermEntry>> termiumSubset = new HashMap<String, Set<TermEntry>>();
						HashMap<String, Set<TermEntry>> conceptNetSubset = new HashMap<String, Set<TermEntry>>();
						
						HashMap<BabelSynset, Set<String>> overlapIate =  new HashMap<BabelSynset, Set<String>>();
						HashMap<BabelSynset, Set<String>> overlapTermium = new HashMap<BabelSynset, Set<String>>();
						HashMap<BabelSynset, Set<String>> overlapConceptNet = new HashMap<BabelSynset, Set<String>>();
						
						//Retrieve all synsets from DBs
						if(iateEnglish.containsKey(bnWord1)) { for(String iateID : iateEnglish.get(bnWord1)) { iateSubset.put(iateID, iateTB.get(iateID)); } }
						if(termiumEnglish.containsKey(bnWord1)) { for(String termiumID : termiumEnglish.get(bnWord1)) {termiumSubset.put(termiumID, termiumTB.get(termiumID)); } }
						if (conceptNetEnglish.containsKey(bnWord1)){
							for(String conceptNetID: conceptNetEnglish.get(bnWord1)){ 
								if (conceptNetTB.containsKey(conceptNetID)){ conceptNetSubset.put(conceptNetID, conceptNetTB.get(conceptNetID));}
							}
						}
					
						List<BabelSynset> sns = new ArrayList<>();
						sns.add(synset1);
						
						//Calculate overlap between synsets of other resources of the same word and BabelNet synset
						if (!iateSubset.isEmpty()) { overlapIate = calculateMultilingualOverlap(sns, iateSubset); }
						if (!termiumSubset.isEmpty()) { overlapTermium = calculateMultilingualOverlap(sns, termiumSubset);}
						if (!conceptNetSubset.isEmpty()) {overlapConceptNet = calculateMultilingualOverlap(sns, conceptNetSubset);}
						
						Set<TermEntry> iateSub = new HashSet<TermEntry>();
						Set<TermEntry> termiumSub = new HashSet<TermEntry>();
						Set<TermEntry> conceptNetSub = new HashSet<TermEntry>();
						
						//Retrieve only those synsets from all of a specific word that overlap with the BabelNet synset
						if (overlapIate.containsKey(synset1)) {for(String iateID : overlapIate.get(synset1)){ iateSub.addAll(iateTB.get(iateID));}}
						if (overlapTermium.containsKey(synset1)) {for(String termiumID: overlapTermium.get(synset1)){ termiumSub.addAll(termiumTB.get(termiumID));}}
						if(overlapConceptNet.containsKey(synset1)){ for (String cnid : overlapConceptNet.get(synset1)) { conceptNetSub.addAll(conceptNetTB.get(cnid)); } }
						
						//Merge the BabelNet synset with all its overlapping synsets from other resources
			
						combineEntries(synset1, iateSub, termiumSub, conceptNetSub);
					}
					
					iateIDs = new HashSet<String>();
					termiumIDs = new HashSet<String>();
					conceptNetIDHelper = new HashSet<String>();
					
					iateTB = new HashMap<String, Set<TermEntry>>();
					termiumTB = new HashMap<String, Set<TermEntry>>();
					conceptNetTB = new  HashMap<String, Set<TermEntry>>();
					
					synsets = new HashSet<BabelSynset>();	
				}		
			}
		}
	}

	/**
	 * Runs over all missing target synsets in the list missingCNTargetIDs. Checks whether they overlap with one of the two 
	 * terminologies (overlaps with BabelNet would have been considered before) and writes them to the final DB in any case
	 * @param missingCNTargetIDs: list of conceptNet IDs that were target synsets of relations but not yet written to the DB
	 * @param iateEnglish: list of English words in IATE
	 * @param termiumEnglish: list of English words in Termium
	 */
	private void writeMissingCNTargets(HashMap<String, Set<String>> iateEnglish, HashMap<String, Set<String>> termiumEnglish){
		System.out.println("Missing CN targets written to DB..."+ missingCNTargetIDs.size());
		
		
		if (!missingCNTargetIDs.isEmpty()){
			HashMap<String, Set<TermEntry>> conceptNetTB = loader.getCNSynonymsPlus(missingCNTargetIDs);
			
			Set<String> iateIDs = new HashSet<String>();
			Set<String> termiumIDs = new HashSet<String>();
			
			for(String cnid : missingCNTargetIDs){
				String word = loader.getCNLabel(cnid);
				
				if (iateEnglish.containsKey(word)){  iateIDs.addAll(iateEnglish.get(word)); }
				if(termiumEnglish.containsKey(word)) { termiumIDs.addAll(termiumEnglish.get(word)); }
			}
			
			HashMap<String, Set<TermEntry>>iateTB = loader.getEntriesFromList(iateIDs, iateTable);
			HashMap<String, Set<TermEntry>> termiumTB = loader.getEntriesFromList(termiumIDs, termiumTable);
			
			for (String cnid : missingCNTargetIDs) {
				String cnword = loader.getCNLabel(cnid);
				
				HashMap<String, Set<TermEntry>> iateSubset = new HashMap<String, Set<TermEntry>>();
				HashMap<String, Set<TermEntry>> termiumSubset = new HashMap<String, Set<TermEntry>>();
				
				if(iateEnglish.containsKey(cnword)) { for(String iateID : iateEnglish.get(cnword)) { iateSubset.put(iateID, iateTB.get(iateID));} }
				if(termiumEnglish.containsKey(cnword)) { for(String termiumID : termiumEnglish.get(cnword)) {termiumSubset.put(termiumID, termiumTB.get(termiumID)); } }
				
				HashMap<String, Set<String>> overlapIate = new HashMap<String, Set<String>>(); 
				HashMap<String, Set<String>> overlapTermium = new HashMap<String, Set<String>>(); 
				
				if(!iateSubset.isEmpty()){ overlapIate = calculateMultilingualOverlapConceptNet(conceptNetTB.get(cnid), iateSubset);}
				if (!termiumSubset.isEmpty()) { overlapTermium = calculateMultilingualOverlapConceptNet(conceptNetTB.get(cnid), termiumSubset);}
				
				Set<TermEntry> iateSub = new HashSet<TermEntry>();
				Set<TermEntry> termiumSub = new HashSet<TermEntry>();
				
				if (overlapIate.containsKey(cnid)) {for(String iateID : overlapIate.get(cnid)){ iateSub.addAll(iateTB.get(iateID));}}
				if (overlapTermium.containsKey(cnid)) {for(String termiumID: overlapTermium.get(cnid)){ termiumSub.addAll(termiumTB.get(termiumID));}}
				
				combineCNWithTerminologies(conceptNetTB.get(cnid), iateSub, termiumSub); 
			}
		}
	}
}
