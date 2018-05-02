package net.codejava.hibernate;

import java.util.List;

import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.query.Query;

public class WordTypeManager {
	
    public Long create(String value, SessionFactory sessionFactory) {
    	WordType type = new WordType();
    	type.setWordType_value(value);
    	
        Session session = sessionFactory.openSession();
        session.beginTransaction();
        
        String hq1 = "SELECT wordType_id FROM WordType WHERE wordType_value = :wordType_value";
        Query<?> query = session.createQuery(hq1);
        query.setParameter("wordType_value", value);
        List<?> queryResults = query.getResultList();
        Long id;
        if (!queryResults.isEmpty()){
        	id = (Long) queryResults.get(0);
        }
        else{
           	id = (Long) session.save(type);
        	session.getTransaction().commit();
        }
        session.close();
        
        return id; 
    }
 
    public void read(Long wordTypeID, SessionFactory sessionFactory) {
        // code to get a book
        Session session = sessionFactory.openSession();
        
        WordType type = session.get(WordType.class, wordTypeID);
     
        System.out.println("Id: " + type.getWordType_id());
        System.out.println("WordType: " + type.getWordType_value());
     
        session.close();
    }
 
    public void update(Long wordTypeID, String value, SessionFactory sessionFactory) {
        // code to modify a book
    	WordType type = new WordType();
    	type.setWordType_id(wordTypeID);
    	type.setWordType_value(value);
    	
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.update(type);
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void delete(Long wordTypeID, SessionFactory sessionFactory) {
        // code to remove a book
    	WordType type = new WordType();
    	type.setWordType_id(wordTypeID);
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.delete(type);
     
        session.getTransaction().commit();
        session.close();
    }
}