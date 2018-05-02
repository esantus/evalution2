package net.codejava.hibernate;

import org.hibernate.Session;
import org.hibernate.SessionFactory;

public class Word2SynsetManager {
 
    public void create(Long synsetID, Long wordID, Long languageID, Long wordSenseID, Boolean mainWord, Long posID, Long wordTypeID, Double wordConcreteness, SessionFactory sessionFactory) {
    	Word2SynsetID w2sID = new Word2SynsetID();
    	Word2Synset w2s = new Word2Synset();
    	
    	w2sID.setSynsetID(synsetID);
    	w2sID.setWordID(wordID);
    	w2sID.setLanguageID(languageID);
    	
    	w2s.setId(w2sID);
    	w2s.setWordSense_id(wordSenseID);
    	w2s.setMainWord(mainWord);
    	w2s.setPos_id(posID);
    	w2s.setWordType_id(wordTypeID);
    	w2s.setWordConcreteness(wordConcreteness);

    	Session session = sessionFactory.openSession();
        session.beginTransaction();
        if (session.get(Word2Synset.class, w2sID) == null){
        	session.save(w2s);
        }
        else{
        	  Word2Synset w2sUpdate = session.get(Word2Synset.class, w2sID);
        	  w2sUpdate.setWordType_id(wordTypeID);
        	  session.update(w2sUpdate);
        }
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void read(Long synsetID, Long wordID, Long languageID, SessionFactory sessionFactory) {
        // code to get a book
        Session session = sessionFactory.openSession();
        Word2SynsetID w2sID = new Word2SynsetID();
        w2sID.setSynsetID(synsetID);
        w2sID.setWordID(wordID);
        w2sID.setLanguageID(languageID);
        
        Word2Synset w2syn = session.get(Word2Synset.class, w2sID);
     
        System.out.println("Id: " + w2syn.getId());
        System.out.println("Sense: " + w2syn.getWordSense_id());
        System.out.println("POS: " + w2syn.getPos_id());
        System.out.println("Type "+ w2syn.getWordType_id());
     
        session.close();
    }
 
    public void update(Long synsetID, Long wordID, Long languageID, Long wordSenseID, Boolean mainWord, Long posID, Long wordTypeID, Double wordConcreteness, SessionFactory sessionFactory) {
        // code to modify a book
    	Word2SynsetID w2sID = new Word2SynsetID();
    	Word2Synset w2s = new Word2Synset();
    	
    	w2sID.setSynsetID(synsetID);
    	w2sID.setWordID(wordID);
    	w2sID.setLanguageID(languageID);
    	
    	w2s.setId(w2sID);
    	w2s.setWordSense_id(wordSenseID);
    	w2s.setMainWord(mainWord);
    	w2s.setPos_id(posID);
    	w2s.setWordType_id(wordTypeID);
    	w2s.setWordConcreteness(wordConcreteness);
    	
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.update(w2s);
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void delete(Long synsetID, Long wordID, Long languageID, SessionFactory sessionFactory) {
        // code to remove a book
    	Word2SynsetID w2sID = new Word2SynsetID();
    	Word2Synset w2s = new Word2Synset();
    	
    	w2sID.setSynsetID(synsetID);
    	w2sID.setWordID(wordID);
    	w2sID.setLanguageID(languageID);
    	w2s.setId(w2sID);
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.delete(w2s);
     
        session.getTransaction().commit();
        session.close();
    }
}