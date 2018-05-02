package net.codejava.hibernate;

import java.util.List;

import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.query.Query;

public class WordPOSManager {
	
    public Long create(String value,  SessionFactory sessionFactory) {
        WordPOS pos = new WordPOS();
    	pos.setWordPOS_value(value);
    	
        Session session = sessionFactory.openSession();
        session.beginTransaction();
        
        String hq1 = "SELECT wordPOS_id FROM WordPOS WHERE wordPOS_value = :wordPOS_value";
        Query<?> query = session.createQuery(hq1);
        query.setParameter("wordPOS_value", value);
        List<?> queryResults = query.getResultList();
        Long id;
        if (!queryResults.isEmpty()){
        	id = (Long) queryResults.get(0);
        }
        else{
           	id = (Long) session.save(pos);
        	session.getTransaction().commit();
        }
        session.close();
        
        return id; 
    }
 
    public void read(Long posID,  SessionFactory sessionFactory) {
        // code to get a book
        Session session = sessionFactory.openSession();
        
        WordPOS pos = session.get(WordPOS.class, posID);
     
        System.out.println("Id: " + pos.getWordPOS_id());
        System.out.println("WordPOS: " + pos.getWordPOS_value());
     
        session.close();
    }
 
    public void update(Long posID, String value,  SessionFactory sessionFactory) {
        // code to modify a book
       	WordPOS pos = new WordPOS();
       	pos.setWordPOS_id(posID);
    	pos.setWordPOS_value(value);
    	
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.update(pos);
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void delete(Long posID,  SessionFactory sessionFactory) {
        // code to remove a book   	
    	WordPOS pos = new WordPOS();
    	pos.setWordPOS_id(posID);
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.delete(pos);
     
        session.getTransaction().commit();
        session.close();
    }
}