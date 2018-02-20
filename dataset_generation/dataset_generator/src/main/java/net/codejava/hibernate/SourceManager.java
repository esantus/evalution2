package net.codejava.hibernate;

import java.util.List;

import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.query.Query;

public class SourceManager {
 
    public Long create(String value, SessionFactory sessionFactory) {
        // code to save synsetID to DB
    	Source source = new Source();
    	source.setSource_value(value);
    	
        Session session = sessionFactory.openSession();
        session.beginTransaction();
        
        String hq1 = "SELECT source_id FROM Source WHERE source_value = :source_value";
        Query<?> query = session.createQuery(hq1);
        query.setParameter("source_value", value);
        List<?> queryResults = query.getResultList();
        Long id;
        if (!queryResults.isEmpty()){
        	id = (Long) queryResults.get(0);
        }
        else{
           	id = (Long) session.save(source);
        	session.getTransaction().commit();
        }
        session.close();
        
        return id; 
    }
 
    public void read(Long sourceID, SessionFactory sessionFactory) {
        // code to get a book
        Session session = sessionFactory.openSession();
        
        Source source = session.get(Source.class, sourceID);
     
        System.out.println("Id: " + source.getSource_id());
        System.out.println("Source: " + source.getSource_value());
     
        session.close();
    }
 
    public void update(Long sourceID, String value, SessionFactory sessionFactory) {
        // code to modify a book
    	Source source = new Source();
    	source.setSource_id(sourceID);
    	source.setSource_value(value);
    	
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.update(source);
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void delete(Long sourceID, SessionFactory sessionFactory) {
        // code to remove a book
    	Source source = new Source();
    	source.setSource_id(sourceID);
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.delete(source);
     
        session.getTransaction().commit();
        session.close();
    }
}