package extractor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import it.uniroma1.lcl.babelnet.BabelNet;
import it.uniroma1.lcl.babelnet.BabelSense;
import it.uniroma1.lcl.babelnet.BabelSynset;
import it.uniroma1.lcl.babelnet.BabelSynsetID;
import it.uniroma1.lcl.babelnet.InvalidBabelSynsetIDException;
import it.uniroma1.lcl.babelnet.data.BabelAudio;
import it.uniroma1.lcl.babelnet.data.BabelSensePhonetics;
import it.uniroma1.lcl.jlt.util.Language;

public class BabelNetExtractor {
		
	BabelNet bn = BabelNet.getInstance();
	Set<String> bnLanguages = new HashSet<String>(); 		
	
	
	public BabelNetExtractor() {
		for(Language lang : Language.values()) {
			bnLanguages.add(lang.toString());
		}
	}

	/**
	 * Retrieves synsets based on an input term and language
	 * @throws IOException
	 * @throws InvalidBabelSynsetIDException 
	 */
	public List<BabelSynset> getSynsets(String term, String language) throws IOException, InvalidBabelSynsetIDException{
		List<BabelSynset> synsets = new ArrayList<BabelSynset>();
		try{
			if (bnLanguages.contains(language)) {
				synsets = bn.getSynsets(term, Language.valueOf(language));
			}
		} catch (NullPointerException e){
			System.out.println(term+" "+language);
		}
		return synsets;
	}
	
	/**
	 * Method to retrieve BabelNet synset from an identifier
	 * @param identifier: BabelNet synset ID
	 * @return synset: returns the retrieved BabelNet synset based on the input identifier
	 */
	public BabelSynset getSynset(BabelSynsetID identifier){
		BabelSynset synset = null;
		try {
			synset = bn.getSynset(identifier);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return synset;
	}
	
	/**
	 * Retrieves all the languages of a specific BabelNet synset
	 * @param synset: input synset
	 * @return languages: list of languages in this specific synset
	 */
	public Set<String> getAllLanguages(BabelSynset synset){
		Set<String> languages = new HashSet<String>();
		for(BabelSense sense : synset){
			languages.add(sense.getLanguage().toString());
		}
		return languages;
	}
	
	/**
	 * Retrieve all audio files associated with specific BabelNet synset
	 * @param synset: input synset
	 * @return audios: String of all concatenated audio URLs associated with this input synset
	 */
	public String getAllAudio(BabelSynset synset){
		Set<String> audios = new HashSet<String>();
		for(BabelSense sense : synset.getSenses()){
			BabelSensePhonetics phonetic = sense.getPronunciations();
            for (BabelAudio audio : phonetic.getAudioItems()) {
            	audios.add(audio.getValidatedUrl().toString());
            }
		}
		if(!audios.isEmpty()){
			return audios.toString();
		}
		else{
			return null;
		}
	}
	
	/**
	 * Retrieve all words of a BabelNet synset and return them in a HashMap that sorts 
	 * the words by language
	 * @param synset: input synset
	 * @return entry: HashMap of synset languages and their associated synset words
	 * @throws IOException
	 */
	public HashMap<String,Set<String>> getEntry(BabelSynset synset) throws IOException{		
		HashMap<String, Set<String>> entry = new HashMap<String, Set<String>>();
		for (BabelSense sense : synset){
			String language = sense.getLanguage().toString().toLowerCase();
			Set<String> terms = new HashSet<String>();
			terms.add(sense.getLemma().replace("_", " "));
			for (String term : synset.getOtherForms(sense.getLanguage())){
				terms.add(term.replace("_", " "));
			}
			if (entry.containsKey(language)){
				Set<String> termsHelper = entry.get(language);
				termsHelper.addAll(terms);
				entry.put(language, termsHelper);
			}
			else{
				entry.put(language, terms);
			}
		}
		return entry;
	}
	
	/**
	 * Retrieves only the main terms of a BabelNet synset and not all variants 
	 * @param synset: input synset
	 * @return entry: HashMap of languages and their associated main words of this synset
	 * @throws IOException
	 */
	public HashMap<String,Set<String>> getMainTerms(BabelSynset synset) throws IOException{		
		HashMap<String, Set<String>> entry = new HashMap<String, Set<String>>();
		Set<String> terms = new HashSet<String>();
		terms.add(synset.getMainSense(Language.EN).getLemma().replaceAll("_", " ").replaceAll("\"", "").toLowerCase());
		entry.put("en", terms);
		for (BabelSense sense : synset.getTranslations().keySet()){
			String language = sense.getLanguage().toString().toLowerCase();
			terms = new HashSet<String>();
			terms.add(sense.getLemma().replaceAll("_", " ").replaceAll("\"", "").toLowerCase());
			if (entry.containsKey(language)){
				entry.get(language).addAll(terms);
			}
			else {
				entry.put(language, terms);
			}
		}
		return entry;
	}
	
	/**
	 * Retrieve all senses associated with this specific synset
	 * @param synset: input synset
	 * @return senses: list of all senses associated with this synset
	 */
	public String getSenses(BabelSynset synset){
		String senseResult = synset.getMainSense(Language.valueOf("EN")).toString();
		List<BabelSense> senses = synset.getSenses();
		if (!senseResult.contains(senses.get(0).getSynset().toString())){
			senseResult += ", "+senses.get(0).getSynset().toString();
		}
		System.out.println(senseResult);
		return senseResult;
	}	

	/**
	 * Retrieve all audio URLs associated with a specific sense
	 * @param sense: BabelNet sense
	 * @return audio: String of all audio URLs for this specific sense
	 */
	public String getAudioFromSense(BabelSense sense){
		Set<String> audios = new HashSet<String>();
		BabelSensePhonetics phonetic = sense.getPronunciations();
        for (BabelAudio audio : phonetic.getAudioItems()) {
        	audios.add(audio.getUrl().toString());
		}
		if(!audios.isEmpty()){
			return audios.toString();
		}
		else{
			return null;
		}
	}
	

}
