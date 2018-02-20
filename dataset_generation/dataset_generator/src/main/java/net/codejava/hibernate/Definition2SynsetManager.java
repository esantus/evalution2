package net.codejava.hibernate;

import org.hibernate.Session;
import org.hibernate.SessionFactory;

public class Definition2SynsetManager {
 
    public void create(Long synsetID, Long definitionID, Long languageID, Long sourceID, SessionFactory sessionFactory) {
    	Definition2SynsetID defID = new Definition2SynsetID();
    	Definition2SynsetLanguage d2s = new Definition2SynsetLanguage();
    	
    	defID.setSynset_id(synsetID);
    	defID.setLanguage_id(languageID);
    	defID.setDefinition_id(definitionID);
    	
    	d2s.setId(defID);
    	d2s.setSource_id(sourceID);
    	
    	Session session = sessionFactory.openSession();
        session.beginTransaction();
        
        if (session.get(Definition2SynsetLanguage.class, defID) == null){
        	session.save(d2s);
        }
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void read(Long synsetID, Long definitionID, Long languageID, SessionFactory sessionFactory) {
        // code to get a book
        Session session = sessionFactory.openSession();
        Definition2SynsetID defID = new Definition2SynsetID();
        defID.setSynset_id(synsetID);
        defID.setLanguage_id(languageID);
        defID.setDefinition_id(definitionID);
        
        Definition2SynsetLanguage d2syn = session.get(Definition2SynsetLanguage.class, defID);
     
        System.out.println("Id: " + d2syn.getId());
        System.out.println("Source: " + d2syn.getSource_id());
        
        session.close();
    }
 
    public void update(Long synsetID, Long definitionID, Long languageID, Long sourceID, SessionFactory sessionFactory){
    	Definition2SynsetID defID = new Definition2SynsetID();
    	Definition2SynsetLanguage d2s = new Definition2SynsetLanguage();
    	
    	defID.setSynset_id(synsetID);
    	defID.setLanguage_id(languageID);
    	defID.setDefinition_id(definitionID);
    	
    	d2s.setId(defID);
    	d2s.setSource_id(sourceID);
    	
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.update(d2s);
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void delete(Long synsetID, Long definitionID, Long languageID, SessionFactory sessionFactory) {
        // code to remove a book
    	Definition2SynsetID defID = new Definition2SynsetID();
    	Definition2SynsetLanguage d2s = new Definition2SynsetLanguage();
    	
    	defID.setSynset_id(synsetID);
    	defID.setLanguage_id(languageID);
    	defID.setDefinition_id(definitionID);
     
    	d2s.setId(defID);
    	
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.delete(d2s);
     
        session.getTransaction().commit();
        session.close();
    }
}