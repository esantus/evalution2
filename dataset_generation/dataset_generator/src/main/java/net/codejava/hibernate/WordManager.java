package net.codejava.hibernate;

import java.util.List;

import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.query.Query;

public class WordManager {
 
    public Long create(String value, SessionFactory sessionFactory) {
    	Word word = new Word();
    	word.setWord_value(value);
    	
        Session session = sessionFactory.openSession();
        session.beginTransaction();
      
        String hq1 = "SELECT word_id FROM Word WHERE word_value = :word_value";
        Query<?> query = session.createQuery(hq1);
        query.setParameter("word_value", value);
        List<?> queryResults = query.getResultList();
        Long id;
        if (!queryResults.isEmpty()){
        	id = (Long) queryResults.get(0);
        }
        else{
           	id = (Long) session.save(word);
        	session.getTransaction().commit();
        }
        session.close(); 
        return id; 
    }
 
    public void read(Long wordID, SessionFactory sessionFactory) {
        // code to get a book
        Session session = sessionFactory.openSession();
        
        Word word = session.get(Word.class, wordID);
     
        System.out.println("Id: " + word.getWord_id());
        System.out.println("Word: " + word.getWord_value());
     
        session.close();
    }
 
    public void update(Long wordID, String value, SessionFactory sessionFactory) {
        // code to modify a book
    	Word word = new Word();
    	word.setWord_id(wordID);
    	word.setWord_value(value);
    	
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.update(word);
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void delete(Long wordID, SessionFactory sessionFactory) {
        // code to remove a book
    	Word word = new Word();
    	word.setWord_id(wordID);
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.delete(word);
     
        session.getTransaction().commit();
        session.close();
    }
}