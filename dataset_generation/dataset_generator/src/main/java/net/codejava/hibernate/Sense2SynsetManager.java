package net.codejava.hibernate;

import org.hibernate.Session;
import org.hibernate.SessionFactory;

public class Sense2SynsetManager {
 
    public void create(Long synset_id, Long synsetSense_id, Boolean main, SessionFactory sessionFactory) {
    	Sense2SynsetID s2sID = new Sense2SynsetID();
    	Sense2Synset s2s = new Sense2Synset();
    	
    	s2sID.setSynset_id(synset_id);
    	s2sID.setSynsetSense_id(synsetSense_id);
    	s2s.setId(s2sID);
    	s2s.setMain(main);
    	
    	Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        if(session.get(Sense2Synset.class, s2sID) == null){
        	session.save(s2s);
        }
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void read(Long synset_id, Long synsetSense_id, SessionFactory sessionFactory) {
        // code to get a book
        Session session = sessionFactory.openSession();
        Sense2SynsetID s2sID = new Sense2SynsetID();
        s2sID.setSynset_id(synset_id);
        s2sID.setSynsetSense_id(synsetSense_id);
        
        Sense2Synset s2syn = session.get(Sense2Synset.class, s2sID);
     
        System.out.println("Id: " + s2syn.getId());
        System.out.println("Sense: " + s2syn.isMain());
        
        session.close();
    }
 
    public void update(Long synset_id, Long synsetSense_id, Boolean main, SessionFactory sessionFactory){
    	Sense2SynsetID s2sID = new Sense2SynsetID();
    	Sense2Synset s2s = new Sense2Synset();
    	
    	s2sID.setSynset_id(synset_id);
    	s2sID.setSynsetSense_id(synsetSense_id);
    	s2s.setId(s2sID);
    	s2s.setMain(main);
    	
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.update(s2s);
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void delete(Long synset_id, Long synsetSense_id, SessionFactory sessionFactory) {
        // code to remove a book
    	Sense2SynsetID s2sID = new Sense2SynsetID();
    	Sense2Synset s2s = new Sense2Synset();
    	
    	s2sID.setSynset_id(synset_id);
    	s2sID.setSynsetSense_id(synsetSense_id);
    	s2s.setId(s2sID);
    	
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.delete(s2s);
     
        session.getTransaction().commit();
        session.close();
    }
}