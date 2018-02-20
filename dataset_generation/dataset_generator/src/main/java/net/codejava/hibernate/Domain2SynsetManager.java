package net.codejava.hibernate;

import org.hibernate.Session;
import org.hibernate.SessionFactory;


public class Domain2SynsetManager {
 
    public void create(Long synsetID, Long domainID, Double score, Long sourceID, SessionFactory sessionFactory) {
    	Domain2SynsetID d2sID = new Domain2SynsetID();
    	Domain2Synset d2s = new Domain2Synset();
    	
    	d2sID.setSynset_id(synsetID);
    	d2sID.setDomain_id(domainID);
    	
    	d2s.setId(d2sID);
    	d2s.setScore(score);
    	d2s.setSource_id(sourceID);
    	
    	Session session = sessionFactory.openSession();
        session.beginTransaction();
        
        if(session.get(Domain2Synset.class, d2sID) == null){
        	session.save(d2s);
        }
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void read(Long synsetID, Long domainID, SessionFactory sessionFactory) {
        // code to get a book
        Session session = sessionFactory.openSession();
        Domain2SynsetID d2sID = new Domain2SynsetID();
        
        d2sID.setSynset_id(synsetID);
        d2sID.setDomain_id(domainID);
        
        Domain2Synset dom2syn = session.get(Domain2Synset.class, d2sID);
     
        System.out.println("Id: " + dom2syn.getId());
        System.out.println("Sense: " + dom2syn.getSource_id());
        
        session.close();
    }
 
    public void update(Long synsetID, Long domainID, Double score, Long sourceID, SessionFactory sessionFactory) {
    	Domain2SynsetID d2sID = new Domain2SynsetID();
    	Domain2Synset d2s = new Domain2Synset();
    	
    	d2sID.setSynset_id(synsetID);
    	d2sID.setDomain_id(domainID);
    	
    	d2s.setId(d2sID);
    	d2s.setScore(score);
    	d2s.setSource_id(sourceID);
    	   
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.update(d2s);
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void delete(Long synsetID, Long domainID, SessionFactory sessionFactory) {
    	Domain2SynsetID d2sID = new Domain2SynsetID();
    	Domain2Synset d2s = new Domain2Synset();
    	
    	d2sID.setSynset_id(synsetID);
    	d2sID.setDomain_id(domainID);
    	d2s.setId(d2sID);
    	
    	Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.delete(d2s);
     
        session.getTransaction().commit();
        session.close();
    }
}