package net.codejava.hibernate;

import java.util.List;

import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.query.Query;

public class SynsetIDManager {
 
    public Long create(String value, SessionFactory sessionFactory) {
        SynsetID synset = new SynsetID();
    	synset.setSynset_value(value);
    	
        Session session = sessionFactory.openSession();
        session.beginTransaction();
        String hq1 = "SELECT synset_id FROM SynsetID WHERE synset_value = :synset_value";
        Query<?> query = session.createQuery(hq1);
        query.setParameter("synset_value", value);
        List<?> queryResults = query.getResultList();
        Long id;
        if (!queryResults.isEmpty()){
        	id = (Long) queryResults.get(0);
        }
        else{
           	id = (Long) session.save(synset);
        	session.getTransaction().commit();
        }
        session.close();
        
        return id; 
    }
 
    public void read(Long synsetID, SessionFactory sessionFactory) {
        // code to get a book
        Session session = sessionFactory.openSession();
        
        SynsetID synset = session.get(SynsetID.class, synsetID);
     
        System.out.println("Id: " + synset.getSynset_id());
        System.out.println("Synset: " + synset.getSynset_value());
     
        session.close();
    }
 
    public void update(Long synsetID, String value, SessionFactory sessionFactory) {
        // code to modify a book
    	SynsetID synset = new SynsetID();
    	synset.setSynset_id(synsetID);
    	synset.setSynset_value(value);
    	
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.update(synset);
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void delete(Long synsetID, SessionFactory sessionFactory) {
        // code to remove a book
    	SynsetID synset = new SynsetID();
    	synset.setSynset_id(synsetID);
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.delete(synset);
     
        session.getTransaction().commit();
        session.close();
    }
}